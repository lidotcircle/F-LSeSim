from collections import UserDict
import torch
import itertools
from util.image_pool import ImagePool
from util.util import flatten_list
from .base_model import BaseModel
from . import losses
from imaginaire.utils.misc import random_shift
from imaginaire.losses import GaussianKLLoss, PerceptualLoss
from imaginaire.generators.munit import Generator
from imaginaire.discriminators.munit import Discriminator
from imaginaire.utils.diff_aug import apply_diff_aug


class MUNITModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            parser.add_argument('--aug_policy', type=str, default='', help='diff_aug policy list of [color,translation,translation_scale,cutout] separated by comma')
            parser.add_argument('--gan_recon', action='store_true', help='discriminate identity reconstruction images')
            parser.add_argument('--within_latent_recon', action='store_true', help='returns reconstructed latent code during within-domain reconstruction.')
            parser.add_argument('--lambda_cycle', type=float, default=10.0, help='weight for cycle loss (A -> B -> A) and (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=10, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
            parser.add_argument('--lambda_adv', type=float, default=1.0, help='adversarial loss for generator')
            parser.add_argument('--lambda_style_recon', type=float, default=1.0, help='style reconstruction loss weight')
            parser.add_argument('--lambda_content_recon', type=float, default=1.0, help='content reconstruction loss weight')
            parser.add_argument('--lambda_perc', type=float, default=0, help='perceptual loss weight')
            parser.add_argument('--lambda_KL', type=float, default=1, help='KLD weight')
            parser.add_argument('--lambda_consistency_reg', type=float, default=0, help='consistency regularizer')

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load Gs
            self.model_names = ['G']
            
        self.within_latent_recon = self.opt.within_latent_recon
        self.netG = Generator(UserDict(
            num_image_channels=opt.input_nc, num_filters=opt.ngf, latent_dim=8, 
            num_res_blocks=4, num_mlp_blocks=2, 
            content_norm_type='instance', style_norm_type='', decoder_norm_type='instance', weight_norm_type='',
            num_downsamples_style=4, num_downsamples_content=2), UserDict())

        if self.isTrain:
            self.netD = Discriminator(UserDict(
                patch_wise=False,
                image_channels=opt.input_nc,
                num_filters=64,
                max_num_filters=512,
                num_layers=4,
                padding_mode='zeros',
                weight_norm_type='',
            ), UserDict())

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionKL = GaussianKLLoss()
            if self.opt.lambda_perc > 0:
                self.criterionPerceptual = PerceptualLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

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
        self.data = UserDict(images_a=self.real_A, images_b=self.real_B)

    def forward(self):
        self.G_output = self.netG(
            self.data,
            random_style=True, image_recon=True, latent_recon=True, cycle_recon=True,
            within_latent_recon=self.within_latent_recon)

        self.idt_A = self.G_output['images_aa']
        self.idt_B = self.G_output['images_bb']
        self.fake_B = self.G_output['images_ab']
        self.fake_A = self.G_output['images_ba']
        self.rec_A = self.G_output['images_aba']
        self.rec_B = self.G_output['images_bab']

    def backward_D(self):
        with torch.no_grad():
            net_G_output = self.netG(self.data,
                                        image_recon=self.opt.gan_recon,
                                        latent_recon=False,
                                        cycle_recon=False,
                                        within_latent_recon=False)
        net_G_output['images_ba'].requires_grad = True
        net_G_output['images_ab'].requires_grad = True

        # Differentiable augmentation.
        keys_fake = ['images_ab', 'images_ba']
        if self.opt.gan_recon:
            keys_fake += ['images_aa', 'images_bb']
        keys_real = ['images_a', 'images_b']

        net_D_output = self.netD(
            apply_diff_aug(self.data, keys_real, self.opt.aug_policy),
            apply_diff_aug(net_G_output, keys_fake, self.opt.aug_policy),
            gan_recon=self.opt.gan_recon)

        dis_losses = {}

        # GAN loss.
        dis_losses['gan_a'] = \
            self.criterionGAN(net_D_output['out_a'], True) + \
            self.criterionGAN(net_D_output['out_ba'], False)
        dis_losses['gan_b'] = \
            self.criterionGAN(net_D_output['out_b'], True) + \
            self.criterionGAN(net_D_output['out_ab'], False)
        dis_losses['gan'] = dis_losses['gan_a'] + dis_losses['gan_b']
        loss = dis_losses['gan']
        self.loss_D_A = dis_losses['gan_a'].item()
        self.loss_D_B = dis_losses['gan_b'].item()
        self.loss_names.append('D_A')
        self.loss_names.append('D_B')

        # Consistency regularization.
        if self.opt.lambda_consistency_reg > 0:
            data_aug, net_G_output_aug = {}, {}
            data_aug['images_a'] = random_shift(self.data['images_a'].flip(-1))
            data_aug['images_b'] = random_shift(self.data['images_b'].flip(-1))
            net_G_output_aug['images_ab'] = \
                random_shift(net_G_output['images_ab'].flip(-1))
            net_G_output_aug['images_ba'] = \
                random_shift(net_G_output['images_ba'].flip(-1))
            net_D_output_aug = self.netD(data_aug, net_G_output_aug)
            feature_names = ['fea_ba', 'fea_ab',
                             'fea_a', 'fea_b']
            for feature_name in feature_names:
                feature_aug = [net_D_output_aug[feature_name]]
                feature = [net_D_output[feature_name]]
                feature_aug = list(map(lambda ts: ts.view(ts.size(0), -1), flatten_list(feature_aug)))
                feature = list(map(lambda ts: ts.view(ts.size(0), -1), flatten_list(feature)))
                feature_aug = torch.cat(feature_aug, dim=1)
                feature = torch.cat(feature, dim=1)
                dis_losses['consistency_reg'] = \
                    torch.pow(feature_aug - feature, 2).mean()
            self.loss_consistency_reg = dis_losses['consistency_reg']
            loss = loss + self.loss_consistency_reg * self.opt.lambda_consistency_reg
            self.loss_names.append('consistency_reg')

        loss.backward()

    def _get_G_total_loss(self, G_loss: dict):
        loss = 0
        losses = dict(
            gan=('adv', self.opt.lambda_adv), perceptual=('perc', self.opt.lambda_perc),
            cycle_recon=('cycle', self.opt.lambda_cycle), image_recon=('recon', self.opt.lambda_identity),
            style_recon=('style_recon', self.opt.lambda_style_recon), content_recon=('content_recon', self.opt.lambda_content_recon),
            style_recon_within=('', self.opt.lambda_style_recon), content_recon_within=('', self.opt.lambda_content_recon), kl=('KL', self.opt.lambda_KL))
        for name in losses:
            if name in G_loss:
                lname, weight = losses[name]
                val = G_loss[name]
                loss = loss + val * weight
                if lname != '':
                    setattr(self, f'loss_{lname}', val.item())
                    self.loss_names.append(lname)
        return loss

    def backward_G(self):
        # Differentiable augmentation.
        keys = ['images_ab', 'images_ba']
        if self.opt.gan_recon:
            keys += ['images_aa', 'images_bb']
        net_D_output = self.netD(self.data,
                                apply_diff_aug(
                                    self.G_output, keys, self.opt.aug_policy),
                                real=False,
                                gan_recon=self.opt.gan_recon)
        gen_losses = {}

        # GAN loss
        if self.opt.gan_recon:
            gen_losses['gan_a'] = \
                0.5 * (self.criterionGAN(net_D_output['out_ba'], True) +
                       self.criterionGAN(net_D_output['out_aa'], True))
            gen_losses['gan_b'] = \
                0.5 * (self.criterionGAN(net_D_output['out_ab'], True) +
                       self.criterionGAN(net_D_output['out_bb'], True))
        else:
            gen_losses['gan_a'] = self.criterionGAN(net_D_output['out_ba'], True)
            gen_losses['gan_b'] = self.criterionGAN(net_D_output['out_ab'], True)
        gen_losses['gan'] = gen_losses['gan_a'] + gen_losses['gan_b']

        # Perceptual loss
        if self.opt.lambda_perc > 0:
            gen_losses['perceptual_a'] = \
                self.criterionPerceptual(self.G_output['images_ab'], self.data['images_a'])
            gen_losses['perceptual_b'] = \
                self.criterionPerceptual(self.G_output['images_ba'], self.data['images_b'])
            gen_losses['perceptual'] = gen_losses['perceptual_a'] + gen_losses['perceptual_b']

        # Image reconstruction loss
        if self.opt.lambda_identity:
            gen_losses['image_recon'] = \
                self.criterionIdt(self.G_output['images_aa'], self.data['images_a']) + \
                self.criterionIdt(self.G_output['images_bb'], self.data['images_b'])

        # Style reconstruction loss
        gen_losses['style_recon_a'] = torch.abs(
            self.G_output['style_ba'] -
            self.G_output['style_a_rand']).mean()
        gen_losses['style_recon_b'] = torch.abs(
            self.G_output['style_ab'] -
            self.G_output['style_b_rand']).mean()
        gen_losses['style_recon'] = gen_losses['style_recon_a'] + gen_losses['style_recon_b']

        if self.within_latent_recon:
            gen_losses['style_recon_aa'] = torch.abs(
                self.G_output['style_aa'] -
                self.G_output['style_a'].detach()).mean()
            gen_losses['style_recon_bb'] = torch.abs(
                self.G_output['style_bb'] -
                self.G_output['style_b'].detach()).mean()
            gen_losses['style_recon_within'] = gen_losses['style_recon_aa'] + gen_losses['style_recon_bb']

        # Content reconstruction loss
        gen_losses['content_recon_a'] = torch.abs(
            self.G_output['content_ab'] -
            self.G_output['content_a'].detach()).mean()
        gen_losses['content_recon_b'] = torch.abs(
            self.G_output['content_ba'] -
            self.G_output['content_b'].detach()).mean()
        gen_losses['content_recon'] = gen_losses['content_recon_a'] + gen_losses['content_recon_b']

        if self.within_latent_recon:
            gen_losses['content_recon_aa'] = torch.abs(
                self.G_output['content_aa'] -
                self.G_output['content_a'].detach()).mean()
            gen_losses['content_recon_bb'] = torch.abs(
                self.G_output['content_bb'] -
                self.G_output['content_b'].detach()).mean()
            gen_losses['content_recon_within'] = gen_losses['content_recon_aa'] + gen_losses['content_recon_bb']

        # KL loss
        gen_losses['kl'] = \
            self.criterionKL(self.G_output['style_a']) + \
            self.criterionKL(self.G_output['style_b'])

        # Cycle reconstruction loss
        gen_losses['cycle_recon'] = \
            torch.abs(self.G_output['images_aba'] -
                      self.data['images_a']).mean() + \
            torch.abs(self.G_output['images_bab'] -
                      self.data['images_b']).mean()

        # Compute total loss
        total_loss = self._get_G_total_loss(gen_losses)
        total_loss.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        self.loss_names = []
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # netG
        self.set_requires_grad([self.netD], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # netD
        self.set_requires_grad([self.netD], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate gradients for D_A
        self.optimizer_D.step()  # update D_A and D_B's weights

    def compute_visuals(self):
        super().compute_visuals()
        if hasattr(self, 'netD'):
            pass
