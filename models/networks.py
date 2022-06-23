import torch
from torch.optim import lr_scheduler
from . import cyclegan_networks, stylegan_networks, cut_networks
from .spatchgan_discriminator_pytorch import SPatchDiscriminator
from .transtyle import Transtyle, TransDiscriminator


##################################################################################
# Networks
##################################################################################
def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal',
             init_gain=0.02, no_antialias=False, no_antialias_up=False, gpu_ids=[], opt=None):
    """
    Create a generator
    :param input_nc: the number of channels in input images
    :param output_nc: the number of channels in output images
    :param ngf: the number of filters in the first conv layer
    :param netG: the architecture's name: resnet_9blocks | munit | stylegan2
    :param norm: the name of normalization layers used in the network: batch | instance | none
    :param use_dropout: if use dropout layers.
    :param init_type: the name of our initialization method.
    :param init_gain: scaling factor for normal, xavier and orthogonal.
    :param no_antialias: use learned down sampling layer or not
    :param no_antialias_up: use learned up sampling layer or not
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :param opt: options
    :return:
    """
    norm_value = cyclegan_networks.get_norm_layer(norm)

    if netG == 'resnet_9blocks':
        net = cyclegan_networks.ResnetGenerator(input_nc, output_nc, ngf, norm_value, use_dropout, n_blocks=9, no_antialias=no_antialias, no_antialias_up=no_antialias_up, opt=opt)
    elif netG == 'transtyle':
        net = Transtyle(input_nc, output_nc, ngf, use_dropout=use_dropout, no_antialias=no_antialias, no_antialias_up=no_antialias_up, n_blocks=6, opt=opt)
    elif netG == 'stylegan2':
        net = stylegan_networks.StyleGAN2Generator(input_nc, output_nc, ngf, opt=opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return cyclegan_networks.init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    """
    Create a discriminator
    :param input_nc: the number of channels in input images
    :param ndf: the number of filters in the first conv layer
    :param netD: the architecture's name
    :param n_layers_D: the number of conv layers in the discriminator; effective when netD=='n_layers'
    :param norm: the type of normalization layers used in the network
    :param init_type: the name of the initialization method
    :param init_gain: scaling factor for normal, xavier and orthogonal
    :param no_antialias: use learned down sampling layer or not
    :param gpu_ids: which GPUs the network runs on: e.g., 0,1,2
    :param opt: options
    :return:
    """
    norm_value = cyclegan_networks.get_norm_layer(norm)
    if netD == 'basic':
        net = cyclegan_networks.NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_value, no_antialias)
    elif netD == 'bimulti':
        net = cyclegan_networks.D_NLayersMulti(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_value, num_D=2)
    elif netD == 'spatch':
        net = SPatchDiscriminator(stats=['mean', 'max','stddev'])
    elif netD == 'transdis':
        net = TransDiscriminator()
    elif 'stylegan2' in netD:
        net = stylegan_networks.StyleGAN2Discriminator(input_nc, ndf, opt=opt)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return cyclegan_networks.init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netD))


def define_F(input_nc, netF, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, no_antialias=False, gpu_ids=[], opt=None):
    if netF == 'global_pool':
        net = cut_networks.PoolingF()
    elif netF == 'reshape':
        net = cut_networks.ReshapeF()
    elif netF == 'sample':
        net = cut_networks.PatchSampleF(use_mlp=False, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'mlp_sample':
        net = cut_networks.PatchSampleF(use_mlp=True, init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids, nc=opt.netF_nc)
    elif netF == 'strided_conv':
        net = cut_networks.StridedConvF(init_type=init_type, init_gain=init_gain, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('projection model name [%s] is not recognized' % netF)
    return cyclegan_networks.init_net(net, init_type, init_gain, gpu_ids)


###############################################################################
# Helper Functions
###############################################################################
def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None
