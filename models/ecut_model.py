import torch
from models import losses
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
from util import computeModelGradientsNorm1, computeModelParametersNorm1


class ECUTModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', action='store_true', help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='4,7,9', help='compute NCE loss on which layers')
        parser.add_argument('--feature_net', type=str, default='vgg16', choices=['vgg16', 'efficientnet_lite', 'learned'], help='network for extracting features')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch', action='store_true', help='includes all negatives from minibatch')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.feature_net == "vgg16":
            parser.set_defaults(nce_layers="4,7,9")
        elif opt.feature_net == "efficientnet_lite":
            parser.set_defaults(nce_layers="2,4,6")
        elif opt.feature_net == "learned":
            parser.set_defaults(nce_layers="0,4,8,12,16")
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
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

        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            if opt.feature_net == 'efficientnet_lite':
                self.netPre = losses.EfficientNetLite().to(self.device)
            elif opt.feature_net == 'vgg16':
                self.netPre = losses.VGG16().to(self.device)
            elif opt.feature_net == 'learned':
                self.netPre = self.netG
            else:
                raise NotImplemented(opt.feature_net)

            # define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
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

        if hasattr(self, 'idt_B'):
            del self.idt_B

        out = self.netG(self.real)
        if isinstance(out, tuple):
            self.fake, _ = out
        else:
            self.fake = out
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.real.size(0) > self.real_A.size(0):
            self.idt_B = self.fake[self.real_A.size(0):]

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pred_fake = self.netD(self.fake_B)
        pred_real = self.netD(self.real_B)
        self.validation_loss_fake = self.criterionGAN(pred_fake, False).mean().item()
        self.validation_loss_real = self.criterionGAN(pred_real, True).mean().item()

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        real = self.real_B

        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(real)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

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

        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.netPre, self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.netPre, self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        self.loss_G = self.loss_G_GAN + loss_NCE_both * self.opt.lambda_NCE
        return self.loss_G

    def calculate_NCE_loss(self, feat_net: torch.nn.Module, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = feat_net(tgt, self.nce_layers, encode_only=True)
        feat_k = feat_net(src, self.nce_layers, encode_only=True)
        if isinstance(feat_q, tuple):
            feat_q = feat_q[1]
        if isinstance(feat_k, tuple):
            feat_k = feat_k[1]

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