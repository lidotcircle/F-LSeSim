import numpy as np
import torch

from models import losses
from .augment  import AugmentPipe
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from util import computeModelGradientsNorm1, computeModelParametersNorm1
import util.util as util
import os


class AngleLoss(torch.nn.Module):
    def __init__(self, eps=1e-5):
        super(AngleLoss, self).__init__()
        self.eps = eps

    def forward(self, f1, f2):
        f1xf2 = (f1 * f2).sum(dim=1)
        f1_ = f1.pow(2).sum(dim=1).pow(0.5)
        f2_ = f2.pow(2).sum(dim=1).pow(0.5)
        cos_angle = f1xf2 / (f1_ * f2_ + self.eps)
        loss = 1 - cos_angle
        return loss.mean()


class DiscriminatorStats:
    def __init__(self):
        self.loss_d_val = []
        self.loss_g_adv = []
        self.loss_d_train = []

    def report_validation_loss(self, value):
        assert isinstance(value, float)
        self.loss_d_val.append(value)

    def report_train_loss(self, value):
        assert isinstance(value, float)
        self.loss_d_train.append(value)

    def report_generated_loss(self, value):
        assert isinstance(value, float)
        self.loss_g_adv.append(value)
    
    def get_r_v(self):
        assert len(self.loss_d_train) > 0
        assert len(self.loss_g_adv) > 0
        assert len(self.loss_d_train) > 0
        l_d_val = np.mean(self.loss_d_val)
        l_g_adv = np.mean(self.loss_g_adv)
        l_d_train = np.mean(self.loss_d_train)
        self.loss_d_val = []
        self.loss_g_adv = []
        self.loss_d_train = []
        return (l_d_val - l_d_train) / max(np.abs(l_g_adv - l_d_train), 1e-5)


class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        # adaptive discriminator augmentation
        parser.add_argument('--ada', action='store_true', help='adaptive discriminator augmentation')
        parser.add_argument('--ada_target', type=float, default=0.1, help='threshold of r_v')
        parser.add_argument('--ada_speed', type=int, default=500, help='ADA speed')
        parser.add_argument('--ada_cuda', action='store_true', help='ADA pipe ops implementation')
        parser.add_argument('--ada_augment_p', type=float, default=0.0, help='initial augment p')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        self.enable_ADA = opt.ada
        self.ada_speed = opt.ada_speed
        self.ada_target = opt.ada_target
        self.ada_r_v = 0
        self.dis_stats = DiscriminatorStats()
        os.environ["ADA_CUDA"] = "YES" if opt.ada_cuda else "NO"

        if self.enable_ADA:
            self.augment_p = opt.ada_augment_p
            self.augment_pipe_dis = AugmentPipe(
                rotate=1,
                brightness=1, contrast=1, lumaflip=1, hue=1, saturation=1,
                imgfilter=1,
                noise=1
            ).train().requires_grad_(False).to(self.device)
            self.augment_pipe_dis.p.copy_(torch.as_tensor(self.augment_p))

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        self.loss_G_GAN = 0
        self.loss_D_real = 0
        self.loss_D_fake = 0
        self.loss_G = 0
        self.loss_NEC = 0
        self.loss_NCE_Y = 0

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            discriminator_lr = opt.lr
            if opt.netD == 'convtrans' or opt.netD == 'transdis':
                discriminator_lr *= 5

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=discriminator_lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()

        if self.opt.param_stats:
            self.d_param_norm, self.d_num_params = computeModelParametersNorm1(self.netD)
            self.d_grad_norm, d_num_grad_params = computeModelGradientsNorm1(self.netD)
            self.d_param_norm = self.d_param_norm.item()
            self.d_grad_norm = self.d_grad_norm.item()
            self.d_param_norm_avg = self.d_param_norm / self.d_num_params
            self.d_grad_norm_avg = self.d_grad_norm / d_num_grad_params

        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()

        if self.opt.param_stats:
            self.g_param_norm, self.g_num_params = computeModelParametersNorm1(self.netG)
            self.g_grad_norm, g_num_grad_params = computeModelGradientsNorm1(self.netG)
            self.g_param_norm = self.g_param_norm.item()
            self.g_grad_norm = self.g_grad_norm.item()
            self.g_param_norm_avg = self.g_param_norm / self.g_num_params
            self.g_grad_norm_avg = self.g_grad_norm / g_num_grad_params

        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample' and self.opt.lambda_NCE > 0.0:
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        cat_AB = self.opt.nce_idt and self.opt.isTrain
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if cat_AB else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        if hasattr(self, 'idt_B'):
            del self.idt_B

        self.fake, _ = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.real.size(0) > self.real_A.size(0):
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pred_fake = self.netD(self.augment_pipe_dis(self.fake_B) if self.enable_ADA else self.fake_B)
        pred_real = self.netD(self.augment_pipe_dis(self.real_B) if self.enable_ADA else self.real_B)
        self.validation_loss_fake = self.criterionGAN(pred_fake, False).mean().item()
        self.validation_loss_real = self.criterionGAN(pred_real, True).mean().item()
        self.dis_stats.report_validation_loss(self.validation_loss_real)
        self.adjust_augment_p()

    def adjust_augment_p(self):
        if not self.enable_ADA:
            return
        self.ada_r_v = self.dis_stats.get_r_v()
        adjust = np.sign(self.ada_r_v - self.ada_target) * (self.opt.display_freq) / (self.ada_speed * 1000)
        self.augment_p = min(max(self.augment_pipe_dis.p.item() + adjust, 0), 1)
        self.augment_pipe_dis.p.copy_(torch.as_tensor(self.augment_p))

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        real = self.real_B
        if self.enable_ADA:
            fake = self.augment_pipe_dis(fake)
            real = self.augment_pipe_dis(real)

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(real)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()
        self.dis_stats.report_train_loss(self.loss_D_real.item())

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        if self.opt.gan_mode == 'wgangp':
            self.loss_D_gp, gradients = networks.cal_gradient_penalty(self.netD, self.real_B, fake, self.device, lambda_gp=1)
            self.dis_grad_norm = torch.norm(gradients).item()
            self.loss_D = self.loss_D + self.loss_D_gp * 10
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        if self.enable_ADA:
            fake = self.augment_pipe_dis(fake)

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
            self.dis_stats.report_generated_loss(self.loss_G_GAN.item())
        else:
            self.loss_G_GAN = 0.0
            self.dis_stats.report_train_loss(0)

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both * self.opt.lambda_NCE
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        _, feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        _, feat_k = self.netG(src, self.nce_layers, encode_only=True)

        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k)
            total_nce_loss += loss.mean()
        
        return total_nce_loss / n_layers

    def compute_visuals(self):
        super().compute_visuals()
        if hasattr(self, 'netD'):
            pred_fake_B = self.netD(self.fake_B)
            pred_real_B = self.netD(self.real_B)
            self.val_loss_G = self.criterionGAN(pred_fake_B, True)
            self.val_loss_D_real = self.criterionGAN(pred_real_B, True)
            self.val_loss_D_fake = self.criterionGAN(pred_fake_B, False)