import torch
import itertools

import dnnlib
import legacy
from models.simple_resnet import ResNet18
from .scg_networks import Encoder
from .base_model import BaseModel
from . import losses


class SCGModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        :param parser: original options parser
        :return: the modified parser
        """
        parser.set_defaults(no_dropout=True)

        parser.add_argument('--gen_network', type=str, help='generator snapshot')
        parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
        parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
        parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
        parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')
        parser.add_argument('--vae_mode', action='store_true', help='variational autoencoder, gaussian posterior')
        parser.add_argument('--lambda_KLD', type=float, default=1.0, help='weight for regularize latent')
        parser.add_argument('--lambda_reg', type=float, default=0, help='weight for regularize latent')
        parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
        parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping')
        parser.add_argument('--lambda_spatial_idt', type=float, default=0.0, help='weight for idt spatial loss')

        return parser

    def gaussian_reparameterization(self, mu, logvar):
        mu_shape = mu.shape
        mu = mu.view(mu.size(0), -1)
        logvar = logvar.view(logvar.size(0), -1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        ans = mu + eps * std
        return ans.view(mu_shape)

    def __init__(self, opt):
        """
        Initialize the translation losses
        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out
        self.loss_names = ['G_s', 'reg']
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'fake_B' , 'real_B']
        # specify the models you want to save to the disk
        self.model_names = ['G']
        # define the networks TODO
        # self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout,
        #                        opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netG = Encoder(num_outputs=256 * 2 if opt.vae_mode else 256).to(self.device)
        with dnnlib.util.open_url(opt.gen_network) as f:
            self.decoder = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            self.set_requires_grad([self.decoder], False)

        # define the training process
        if self.isTrain:
            self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
            if opt.lambda_spatial_idt > 0.0 or opt.lambda_identity:
                # only works when input and output images have the same number of channels
                self.visual_names.append('idt_B')
                if opt.lambda_spatial_idt > 0.0:
                    self.loss_names.append('G_s_idt_B')
                if opt.lambda_identity > 0.0:
                    self.loss_names.append('idt_B')
                assert (opt.input_nc == opt.output_nc)
            if opt.vae_mode:
                self.loss_names.append('KLD')
            # define the loss function
            self.netPre = losses.VGG16().to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionSpatial = losses.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    False, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)
            self.normalization = losses.Normalization(self.device)
            self.set_requires_grad([self.netPre], False)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)

    def data_dependent_initialize(self, data):
        """
        The learnable spatially-correlative map is defined in terms of the shape of the intermediate, extracted features
        of a given network (encoder or pretrained VGG16). Because of this, the weights of spatial are initialized at the
        first feedforward pass with some input images
        :return:
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()
        if self.isTrain:
            self.backward_G()
            self.optimizer_G.zero_grad()

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps
        :param input: include the data itself and its metadata information
        :return:
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass"""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain else self.real_A
        if self.opt.vae_mode:
            ans = self.netG(self.real)
            self.real_mu, self.real_logvar = ans[:, :ans.size(1)//2], ans[:, ans.size(1)//2:]
            self.latent = self.gaussian_reparameterization(self.real_mu, self.real_logvar)
        else:
            self.latent: torch.Tensor = self.netG(self.real)
        self.fake = self.decoder(self.latent, c=None)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if (self.opt.lambda_identity + self.opt.lambda_spatial_idt > 0) and self.opt.isTrain:
            self.idt_B = self.fake[self.real_A.size(0):]

    def backward_G(self):
        """Calculate the loss for generator G_A"""
        l_sptial = self.opt.lambda_spatial
        l_idt = self.opt.lambda_identity
        l_spatial_idt = self.opt.lambda_spatial_idt
        l_reg = self.opt.lambda_reg
        l_kld = self.opt.lambda_KLD
        norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        norm_real_B = self.normalization((self.real_B + 1) * 0.5)
        self.loss_G_s = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None) * l_sptial if l_sptial > 0 else 0
        # identity loss
        if l_spatial_idt > 0:
            norm_fake_idt_B = self.normalization((self.idt_B + 1) * 0.5)
            self.loss_G_s_idt_B = self.Spatial_Loss(self.netPre, norm_real_B, norm_fake_idt_B, None) * l_spatial_idt
        else:
            self.loss_G_s_idt_B = 0
        self.loss_idt_B = self.criterionIdt(self.real_B, self.idt_B) * l_idt if l_idt > 0 else 0
        self.loss_KLD = torch.mean(-0.5 * torch.sum(1 + self.real_logvar - self.real_mu.pow(2) - self.real_logvar.exp(), dim=1)) * l_kld if l_kld > 0 and self.opt.vae_mode else 0

        # TODO
        self.loss_reg = self.latent.abs().mean() * l_reg if l_reg > 0 else self.latent.abs().mean().item()
        self.loss_G = self.loss_G_s + self.loss_G_s_idt_B + self.loss_idt_B + self.loss_reg + self.loss_KLD
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights"""
        # forward
        self.forward()
        # G_A
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers
    
    def compute_visuals(self):
        super().compute_visuals()