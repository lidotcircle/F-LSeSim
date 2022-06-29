import itertools
import torch
import torch.nn as nn
from models import losses
from models.base_model import BaseModel
from util.image_pool import ImagePool
from .utils import bhw2heatmap, image_blend_normal
from .ugatit_networks import ResnetGenerator, Discriminator, RhoClipper


class UGATITModel(BaseModel) :
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--light', action='store_true', help='[U-GAT-IT full version / U-GAT-IT light version]')

        parser.add_argument('--lambda_adv', type=int, default=1, help='Weight for GAN')
        parser.add_argument('--lambda_cycle', type=int, default=10, help='Weight for Cycle')
        parser.add_argument('--lambda_identity', type=int, default=10, help='Weight for Identity')
        parser.add_argument('--lambda_cam', type=int, default=1000, help='Weight for CAM')

        parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
        parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
        parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')

        parser.add_argument('--img_size', type=int, default=256, help='The size of image')
        parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
        return parser
    
    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = [
            'DG_A', 'DL_A', 'DG_cam_A', 'DL_cam_A', 'G_A', 'cycle_A', 'idt_A', 
            'DG_B', 'DL_B', 'DG_cam_B', 'DL_cam_B', 'G_B', 'cycle_B', 'idt_B'
        ]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        visual_names_A = ['real_A', 'fake_B', 'rec_A', 'heatmap_A', 'heatmap_DL_A', 'heatmap_DG_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B', 'heatmap_B', 'heatmap_DL_B', 'heatmap_DG_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_A(B)
            visual_names_A.append('idt_B')
            visual_names_B.append('idt_A')
        self.visual_names = visual_names_A + visual_names_B

        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'DG_A', 'DG_B', 'DL_A', 'DL_B']
        else:  # during test time, only load Gs
            self.model_names = ['G_A', 'G_B']

        self.light = opt.light
        self.ch = opt.ch

        """ Weight """
        self.lambda_adv = opt.lambda_adv
        self.lambda_cycle = opt.lambda_cycle
        self.lambda_identity = opt.lambda_identity
        self.lambda_cam = opt.lambda_cam
        assert self.lambda_identity > 0

        """" Image Shape """
        self.img_size = opt.img_size
        self.img_ch = opt.img_ch

        """ Define Generator, Discriminator """
        self.n_res = opt.n_res
        self.netG_A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)
        self.netG_B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light).to(self.device)

        """ Trainer """
        if self.isTrain:
            self.n_dis = opt.n_dis
            """ Global Discriminator """
            self.netDG_A = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
            self.netDG_B = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
            """ Local Discriminator """
            self.netDL_A = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)
            self.netDL_B = Discriminator(input_nc=3, ndf=self.ch, n_layers=5).to(self.device)

            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                assert(opt.input_nc == opt.output_nc)
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images

            # define loss functions
            assert opt.gan_mode == 'lsgan'
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netDG_A.parameters(), self.netDG_B.parameters(), self.netDL_A.parameters(), self.netDL_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)
    
    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B, self.real_A_cam_logit, self.real_A_heatmap = self.netG_A(self.real_A)  # G_A(A)
        self.rec_A, self.fake_B_cam_logit, _ = self.netG_B(self.fake_B)   # G_B(G_A(A))
        self.fake_A, self.real_B_cam_logit, self.real_B_heatmap = self.netG_B(self.real_B)  # G_B(B)
        self.rec_B, self.fake_A_cam_logit, _ = self.netG_A(self.fake_A)   # G_A(G_B(B))

        self.idt_A, self.fake_A2A_cam_logit, _ = self.netG_B(self.real_A)
        self.idt_B, self.fake_B2B_cam_logit, _ = self.netG_A(self.real_B)

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.

        # DG_A and DG_B   DL_A and DL_B
        self.set_requires_grad([self.netDG_A, self.netDG_B, self.netDL_A, self.netDL_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D()      # calculate graidents for Global Discriminator and Local Discriminator
        self.optimizer_D.step()  # update D_A and D_B's weights

        # G_A and G_B
        self.set_requires_grad([self.netDG_A, self.netDG_B, self.netDL_A, self.netDL_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights

    def backward_G(self):
        fake_GA_logit, fake_GA_cam_logit, _ = self.netDG_A(self.fake_A)
        fake_LA_logit, fake_LA_cam_logit, _ = self.netDL_A(self.fake_A)
        fake_GB_logit, fake_GB_cam_logit, _ = self.netDG_B(self.fake_B)
        fake_LB_logit, fake_LB_cam_logit, _ = self.netDL_B(self.fake_B)

        G_ad_loss_GA = self.criterionGAN(fake_GA_logit, True)
        G_ad_cam_loss_GA = self.criterionGAN(fake_GA_cam_logit, True)
        G_ad_loss_LA = self.criterionGAN(fake_LA_logit, True)
        G_ad_cam_loss_LA = self.criterionGAN(fake_LA_cam_logit, True)
        G_ad_loss_GB = self.criterionGAN(fake_GB_logit, True)
        G_ad_cam_loss_GB = self.criterionGAN(fake_GB_cam_logit, True)
        G_ad_loss_LB = self.criterionGAN(fake_LB_logit, True)
        G_ad_cam_loss_LB = self.criterionGAN(fake_LB_cam_logit, True)

        self.loss_cycle_A = self.criterionCycle(self.real_A, self.rec_A)
        self.loss_cycle_B = self.criterionCycle(self.real_B, self.rec_B)

        self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_A)
        self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_B)

        G_cam_loss_A = self.BCE_loss(self.real_B_cam_logit, torch.ones_like(self.real_B_cam_logit).to(self.device)) + self.BCE_loss(self.fake_A2A_cam_logit, torch.zeros_like(self.fake_A2A_cam_logit).to(self.device))
        G_cam_loss_B = self.BCE_loss(self.real_A_cam_logit, torch.ones_like(self.real_A_cam_logit).to(self.device)) + self.BCE_loss(self.fake_B2B_cam_logit, torch.zeros_like(self.fake_B2B_cam_logit).to(self.device))

        self.loss_G_A =  self.lambda_adv * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.lambda_cycle * self.loss_cycle_A + self.lambda_identity * self.loss_idt_A + self.lambda_cam * G_cam_loss_A
        self.loss_G_B = self.lambda_adv * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.lambda_cycle * self.loss_cycle_B + self.lambda_identity * self.loss_idt_B + self.lambda_cam * G_cam_loss_B
        self.loss_G = self.loss_G_A + self.loss_G_B
        self.loss_G.backward()
        self.netG_A.apply(self.Rho_clipper)
        self.netG_B.apply(self.Rho_clipper)

    def backward_D_basic(self, netD: nn.Module, real: torch.Tensor, fake: torch.Tensor):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real, real_cam_logit, _ = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        loss_D_real_cam = self.criterionGAN(real_cam_logit, True)
        # Fake
        pred_fake, fake_cam_logit, _ = netD(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        loss_D_fake_cam = self.criterionGAN(fake_cam_logit, True)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D_cam = (loss_D_real_cam + loss_D_fake_cam) * self.lambda_cam
        loss_total = loss_D + loss_D_cam
        loss_total.backward()
        return loss_D, loss_D_cam

    def backward_D(self):
        fake_B = self.fake_B_pool.query(self.fake_B.detach())
        fake_A = self.fake_A_pool.query(self.fake_A.detach())
        self.loss_DG_A, self.loss_DG_cam_A = self.backward_D_basic(self.netDG_A, self.real_A, fake_A)
        self.loss_DL_A, self.loss_DL_cam_A = self.backward_D_basic(self.netDL_A, self.real_A, fake_A)
        self.loss_DG_B, self.loss_DG_cam_B = self.backward_D_basic(self.netDG_B, self.real_B, fake_B)
        self.loss_DL_B, self.loss_DL_cam_B = self.backward_D_basic(self.netDL_B, self.real_B, fake_B)

    def compute_visuals(self):
        super().compute_visuals()
        if hasattr(self, 'netDL_B'):
            pred_fake_B, _, _ = self.netDL_B(self.fake_B)
            pred_real_B, _, DL_real_B_heatmap = self.netDL_B(self.real_B)
            self.val_loss_DL_A_real = self.criterionGAN(pred_real_B, True)
            self.val_loss_DL_A_fake = self.criterionGAN(pred_fake_B, False)
        if hasattr(self, 'netDL_A'):
            pred_fake_A, _, _ = self.netDL_A(self.fake_A)
            pred_real_A, _, DL_real_A_heatmap = self.netDL_A(self.real_A)
            self.val_loss_DL_B_real = self.criterionGAN(pred_real_A, True)
            self.val_loss_DL_B_fake = self.criterionGAN(pred_fake_A, False)
        if hasattr(self, 'netDG_B'):
            pred_fake_B, _, _ = self.netDG_B(self.fake_B)
            pred_real_B, _, DG_real_B_heatmap = self.netDG_B(self.real_B)
            self.val_loss_DG_A_real = self.criterionGAN(pred_real_B, True)
            self.val_loss_DG_A_fake = self.criterionGAN(pred_fake_B, False)
        if hasattr(self, 'netDG_A'):
            pred_fake_A, _, _ = self.netDG_A(self.fake_A)
            pred_real_A, _, DG_real_A_heatmap = self.netDG_A(self.real_A)
            self.val_loss_DG_B_real = self.criterionGAN(pred_real_A, True)
            self.val_loss_DG_B_fake = self.criterionGAN(pred_fake_A, False)

        self.heatmap_A = image_blend_normal(bhw2heatmap(self.real_A_heatmap), self.real_A, 0.3)
        self.heatmap_B = image_blend_normal(bhw2heatmap(self.real_B_heatmap), self.real_B, 0.3)
        self.heatmap_DL_A = image_blend_normal(bhw2heatmap(DL_real_A_heatmap), self.real_A, 0.3)
        self.heatmap_DG_A = image_blend_normal(bhw2heatmap(DG_real_A_heatmap), self.real_A, 0.3)
        self.heatmap_DL_B = image_blend_normal(bhw2heatmap(DL_real_B_heatmap), self.real_B, 0.3)
        self.heatmap_DG_B = image_blend_normal(bhw2heatmap(DG_real_B_heatmap), self.real_B, 0.3)