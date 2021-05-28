import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from .utils import check_or_create_dir


class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class Identity(nn.Module):
    def __init__(self, inplace=True):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class OnePlus(nn.Module):
    def __init__(self, inplace=True):
        super(Identity, self).__init__()

    def forward(self, x):
        return F.softplus(x, beta=1)


class BWtoRGB(nn.Module):
    def __init__(self):
        super(BWtoRGB, self).__init__()

    def forward(self, x):
        assert len(list(x.size())) == 4
        chans = x.size(1)
        if chans < 3:
            return torch.cat([x, x, x], 1)
        else:
            return x


class MaskedConv2d(nn.Conv2d):
    ''' from jmtomczak's github '''
    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        assert mask_type in {'A', 'B'}
        self.register_buffer('mask', self.weight.data.clone())
        _, _, kH, kW = self.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedConv2d, self).forward(x)


class GatedConv2d(nn.Module):
    '''from jmtomczak's github '''
    def __init__(self, input_channels, output_channels, kernel_size,
                 stride, padding=0, dilation=1, activation=None):
        super(GatedConv2d, self).__init__()

        self.activation = activation
        self.sigmoid    = nn.Sigmoid()

        self.h = nn.Conv2d(input_channels, output_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)
        self.g = nn.Conv2d(input_channels, output_channels, kernel_size,
                           stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g


class GatedConvTranspose2d(nn.Module):
    ''' from jmtomczak's github'''
    def __init__(self, input_channels, output_channels, kernel_size,
                 stride, padding=0, dilation=1, activation=None):
        super(GatedConvTranspose2d, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.ConvTranspose2d(input_channels, output_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation)
        self.g = nn.ConvTranspose2d(input_channels, output_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        if self.activation is None:
            h = self.h(x)
        else:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g


class EarlyStopping(object):
    def __init__(self, model, max_steps=10, burn_in_interval=None, save_best=True):
        self.max_steps = max_steps
        self.model     = model
        self.save_best = save_best
        self.burn_in_interval = burn_in_interval

        self.loss          = 0.0
        self.iteration     = 0
        self.stopping_step = 0
        self.best_loss     = np.inf

    def restore(self):
        self.model.load()

    def __call__(self, loss):
        if self.burn_in_interval is not None and self.iteration < self.burn_in_interval:
            ''' don't save the model until the burn-in-interval has been exceeded'''
            self.iteration += 1
            return False

        if (loss < self.best_loss):
            self.stopping_step = 0
            self.best_loss = loss
            if self.save_best:
                self.model.save(overwrite=True)
        else:
            self.stopping_step += 1

        is_early_stop = False
        if self.stopping_step >= self.max_steps:
            print("Early stopping is triggered;  current_loss:{} --> best_loss:{} | iter: {}".format(
                loss, self.best_loss, self.iteration))
            is_early_stop = True

        self.iteration += 1
        return is_early_stop

def flatten_layers(model, base_index=0):
    layers = []
    for l in model.children():
        if isinstance(l, nn.Sequential):
            sub_layers, base_index = flatten_layers(l, base_index)
            layers.extend(sub_layers)
        else:
            layers.append(('layer_%d'%base_index, l))
            base_index += 1

    return layers, base_index


def init_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            print("initializing ", m, " with xavier init")
            nn.init.xavier_uniform(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                print("initial bias from ", m, " with zeros")
                nn.init.constant(m.bias, 0.0)
        elif isinstance(m, nn.Sequential):
            for mod in m:
                init_weights(mod)

    return module


class IsotropicGaussian(nn.Module):
    def __init__(self, mu, logvar, use_cuda=False):
        super(IsotropicGaussian, self).__init__()
        self.mu       = mu
        self.dims     = self.mu.size()[1]
        self.logvar   = logvar
        self.use_cuda = use_cuda

    def mean(self):
        return self.mu

    def log_var(self):
        return self.logvar

    def var(self):
        return self.logvar.exp()

    def update_add(self, mu_update, logvar_update, eps=1e-9):
        self.mu    += mu_update
        self.logvar = torch.log(self.logvar.exp() * logvar_update.exp() + eps)

    def sample(self, mu, logvar):
        eps = Variable(float_type(self.use_cuda)(self.logvar.size()).normal_())
        return mu + logvar.exp() * eps

    def forward(self, logits, return_mean=False):
        ''' If return_mean is true then returns mean instead of sample '''
        mu, var = _divide_logits(logits)
        logvar  = F.softplus(var)
        self.update_add(mu, logvar)
        return self.mu if return_mean else self.sample(self.mu, self.logvar)


class Convolutional(nn.Module):
    def __init__(self, input_size, layer_maps=[32, 64, 128, 64, 32],
                 kernel_sizes=[1, 1, 1, 1, 1],
                 activation_fn=nn.ELU, use_dropout=False, use_bn=False,
                 use_in=False, use_wn=False, ngpu=1):
        super(Convolutional, self).__init__()
        ''' input_size = 2d or 3d'''

        assert len(kernel_sizes) == len(layer_maps)

        # Parameters pass into layer
        self.input_size   = input_size
        self.is_color     = input_size[-1] > 1
        self.layer_maps   = [3 if self.is_color else 1] + layer_maps + [3 if self.is_color else 1]
        self.kernel_sizes = [1] + kernel_sizes + [1]
        self.activation   = activation_fn
        self.use_bn = use_bn
        self.use_in = use_in
        self.use_wn = use_wn
        self.ngpu   = ngpu
        self.use_dropout = use_dropout

        # Are we using a normalization scheme?
        self.use_norm = bool(use_bn or use_in)

        # Build our model as a sequential layer
        self.net = self._build_layers()
        self.add_module('network', self.net)
        #self = init_weights(self)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        return output

    def cuda(self, device_id=None):
        super(Convolutional, self).cuda(device_id)
        self.net = self.net.cuda()
        return self

    def get_info(self):
        return self.net.modules()

    def get_sizing(self):
        return str(self.sizes)

    def _add_normalization(self, num_features):
        if self.use_bn:
            return nn.BatchNorm2d(num_features)
        elif self.use_in:
            return nn.InstanceNorm2d(num_features)

    def _add_dropout(self):
        if self.use_dropout:
            return nn.AlphaDropout()

    def _build_layers(self):
        '''Conv/FC --> BN --> Activ --> Dropout'''
        'Conv2d maps (N, C_{in}, H, W) --> (N, C_{out}, H_{out}, W_{out})'
        layers = []

        for i, (in_channels, out_channels) in enumerate(zip(self.layer_maps[0:-1], self.layer_maps[1:-1])):
            l = nn.Conv2d(in_channels, out_channels, self.kernel_sizes[i], padding=0)
            if self.use_wn:
                layers.append(('conv_{}'.format(i), nn.utils.weight_norm(l)))
            else:
                layers.append(('conv_{}'.format(i), l))

            if self.use_norm:  # add normalization
                layers.append(('norm_{}'.format(i), self._add_normalization(out_channels)))

            layers.append(('activ_{}'.format(i), self.activation()))

            if self.use_dropout:  # add dropout
                layers.append(('dropout_{}'.format(i), self._add_dropout()))

        l_f = nn.Conv2d(self.layer_maps[-2], self.layer_maps[-1], self.kernel_sizes[-1], padding=0)
        layers.append(('conv_proj', l_f))
        return nn.Sequential(OrderedDict(layers))


class UpsampleConvLayer(torch.nn.Module):
    '''Upsamples the input and then does a convolution.
    This method gives better results compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/ '''

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample, mode='bilinear')

        reflection_padding  = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d         = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)

        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class Dense(nn.Module):
    def __init__(self, input_size, latent_size, layer_sizes,
                 activation_fn, use_dropout=False, use_bn=False,
                 use_in=False, use_wn=False, ngpu=1):
        super(Dense, self).__init__()

        # Parameters pass into layer
        self.input_size  = input_size
        self.latent_size = latent_size
        self.layer_sizes = [input_size] + layer_sizes
        self.activation  = activation_fn
        self.use_bn = use_bn
        self.use_in = use_in
        self.use_wn = use_wn
        self.ngpu   = ngpu
        self.use_dropout = use_dropout

        # Are we using a normalization scheme?
        self.use_norm = bool(use_bn or use_in)

        # Build our model as a sequential layer
        self.net = self._build_layers()
        self.add_module('network', self.net)
        #self = init_weights(self)

    def forward(self, x):
        if isinstance(x.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)

        return output

    def cuda(self, device_id=None):
        super(Dense, self).cuda(device_id)
        self.net = self.net.cuda()
        return self

    def get_info(self):
        return self.net.modules()

    def get_sizing(self):
        return str(self.sizes)

    def _add_normalization(self, num_features):
        if self.use_bn:
            return nn.BatchNorm1d(num_features)
        elif self.use_in:
            return nn.InstanceNorm1d(num_features)

    def _add_dropout(self):
        if self.use_dropout:
            return nn.AlphaDropout()

    def _build_layers(self):
        '''Conv/FC --> BN --> Activ --> Dropout'''
        layers = [('flatten', View([-1, self.input_size]))]
        for i, (input_size, output_size) in enumerate(
                zip(self.layer_sizes, self.layer_sizes[1:])):
            if self.use_wn:
                layers.append((
                    'linear_{}'.format(i),
                    nn.utils.weight_norm(nn.Linear(input_size, output_size))
                ))
            else:
                layers.append((
                    'linear_{}'.format(i),
                    nn.Linear(input_size, output_size)
                ))

            if self.use_norm:  # add normalization
                layers.append(('norm_{}'.format(i), self._add_normalization(output_size)))

            layers.append(('activ_{}'.format(i), self.activation()))

            if self.use_dropout:  # add dropout
                layers.append(('dropout_{}'.format(i), self._add_dropout()))

        layers.append(('linear_proj', nn.Linear(self.layer_sizes[-1], self.latent_size)))
        return nn.Sequential(OrderedDict(layers))


def str_to_activ_module(str_activ):
    ''' Helper to return a tf activation given a str'''
    str_activ = str_activ.strip().lower()
    activ_map = {
        'identity'   : Identity,
        'elu'        : nn.ELU,
        'sigmoid'    : nn.Sigmoid,
        'log_sigmoid': nn.LogSigmoid,
        'tanh'       : nn.Tanh,
        'oneplus'    : OnePlus,
        'softmax'    : nn.Softmax,
        'log_softmax': nn.LogSoftmax,
        'selu'       : nn.SELU,
        'relu'       : nn.ReLU,
        'softplus'   : nn.Softplus,
        'hardtanh'   : nn.Hardtanh,
        'leaky_relu' : nn.LeakyReLU,
        'softsign'   : nn.Softsign
    }

    assert str_activ in activ_map, "unknown activation requested"
    return activ_map[str_activ]


def str_to_activ(str_activ):
    ''' Helper to return a tf activation given a str'''
    str_activ = str_activ.strip().lower()
    activ_map = {
        'identity'   : lambda x: x,
        'elu'        : F.elu,
        'sigmoid'    : F.sigmoid,
        'tanh'       : F.tanh,
        'oneplus'    : oneplus,
        'softmax'    : F.softmax,
        'log_softmax': F.log_softmax,
        'selu'       : F.selu,
        'relu'       : F.relu,
        'softplus'   : F.softplus,
        'hardtanh'   : F.hardtanh,
        'leaky_relu' : F.leaky_relu,
        'softsign'   : F.softsign
    }

    assert str_activ in activ_map, "unknown activation requested"
    return activ_map[str_activ]


def build_image_downsampler(img_shp, new_shp, stride=[3, 3], padding=[0, 0]):
    '''Takes a tensor and returns a downsampling operator'''
    equality_test = np.asarray(img_shp) == np.asarray(new_shp)
    if equality_test.all():
        return Identity()

    height     = img_shp[0]
    width      = img_shp[1]
    new_height = new_shp[0]
    new_width  = new_shp[1]

    # calculate the width and height by inverting the equations from:
    # http://pytorch.org/docs/master/nn.html?highlight=avgpool2d#torch.nn.AvgPool2d
    kernel_width  = -1 * ((new_width  - 1) * stride[1] - width  - 2 * padding[1])
    kernel_height = -1 * ((new_height - 1) * stride[0] - height - 2 * padding[0])
    print('kernel = ', kernel_height, 'x', kernel_width)
    assert kernel_height > 0
    assert kernel_width  > 0

    return  nn.AvgPool2d(kernel_size=(kernel_height, kernel_width), stride=stride, padding=padding)


def build_relational_conv_encoder(input_shape, filter_depth=32,
                                  activation_fn=nn.ELU, bilinear_size=(32, 32)):
    upsampler = nn.Upsample(size=bilinear_size, mode='bilinear')
    chans     = input_shape[0]
    return nn.Sequential(
        upsampler if input_shape[1:] != bilinear_size else Identity(),
        # input dim: num_channels x 32 x 32
        nn.Conv2d(chans, filter_depth, 5, stride=1, bias=True),
        nn.BatchNorm2d(filter_depth),
        activation_fn(inplace=True),
        # state dim: 32 x 28 x 28
        nn.Conv2d(filter_depth, filter_depth*2, 4, stride=2, bias=True),
        nn.BatchNorm2d(filter_depth*2),
        activation_fn(inplace=True),
        # state dim: 64 x 13 x 13
        nn.Conv2d(filter_depth*2, filter_depth*4, 4, stride=1, bias=True),
        nn.BatchNorm2d(filter_depth*4),
        activation_fn(inplace=True),
        # state dim: 128 x 10 x 10
        nn.Conv2d(filter_depth*4, filter_depth*8, 4, stride=2, bias=True)
        # state dim: 256 x 4 x 4
    )



def build_pixelcnn_decoder(input_size, output_shape, filter_depth=32,
                           activation_fn=nn.ELU, bilinear_size=(32, 32),
                           normalization_str="none"):
    ''' from jmtomczak's github '''

    chans = output_shape[0]
    act   = nn.ReLU(True)
    return nn.Sequential(
        MaskedConv2d('A', input_size, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        MaskedConv2d('B', 64, 64, 3, 1, 1, bias=False),
        add_normalization(normalization_str, 2, 64, num_groups=32), act,
        nn.Conv2d(64, chans, 1, 1, 0, dilation=1, bias=True)
    )


def add_normalization(normalization_str, ndims, nfeatures, **kwargs):
    norm_map = {
        'batchnorm': {
            1: nn.BatchNorm1d(nfeatures),
            2: nn.BatchNorm2d(nfeatures),
            3: nn.BatchNorm3d(nfeatures)
        },
        'groupnorm': {
            2: nn.GroupNorm(kwargs['num_groups'], nfeatures)
        },
        'instancenorm': {
            1: nn.InstanceNorm1d(nfeatures),
            2: nn.InstanceNorm2d(nfeatures),
            3: nn.InstanceNorm3d(nfeatures)
        },
        'none': {
            1: Identity(),
            2: Identity(),
            3: Identity()
        }
    }

    if normalization_str == 'groupnorm':
        assert 'num_groups' in kwargs, "need to specify groups for GN"
        assert ndims > 1, "group norm needs channels to operate"

    return norm_map[normalization_str][ndims]


def build_gated_conv_encoder(input_shape, output_size, filter_depth=32,
                             activation_fn=Identity, bilinear_size=(32, 32),
                             normalization_str="none"):
    print('building gated conv encoder...')
    upsampler = nn.Upsample(size=bilinear_size, mode='bilinear')
    chans     = input_shape[0]
    return nn.Sequential(
        upsampler if input_shape[1:] != bilinear_size else Identity(),
        # input dim: num_channels x 32 x 32
        GatedConv2d(chans, filter_depth, 5, stride=1),
        add_normalization(normalization_str, 2, filter_depth, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 32 x 28 x 28
        GatedConv2d(filter_depth, filter_depth*2, 4, stride=2),
        add_normalization(normalization_str, 2, filter_depth*2, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 64 x 13 x 13
        GatedConv2d(filter_depth*2, filter_depth*4, 4, stride=1),
        add_normalization(normalization_str, 2, filter_depth*4, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 128 x 10 x 10
        GatedConv2d(filter_depth*4, filter_depth*8, 4, stride=2),
        add_normalization(normalization_str, 2, filter_depth*8, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 256 x 4 x 4
        GatedConv2d(filter_depth*8, filter_depth*16, 4, stride=1),
        add_normalization(normalization_str, 2, filter_depth*16, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 512 x 1 x 1
        GatedConv2d(filter_depth*16, filter_depth*16, 1, stride=1),
        add_normalization(normalization_str, 2, filter_depth*16, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 512 x 1 x 1
        GatedConv2d(filter_depth*16, output_size, 1, stride=1),
        # output dim: opt.z_dim x 1 x 1
        View([-1, output_size])
    )


def build_gated_conv_decoder(input_size, output_shape, filter_depth=32,
                             activation_fn=Identity, bilinear_size=(32, 32),
                             normalization_str="none"):
    print('building gated conv decoder...')
    chans     = output_shape[0]
    upsampler = nn.Upsample(size=output_shape[1:], mode='bilinear')
    return nn.Sequential(
        View([-1, input_size, 1, 1]),
        # input dim: z_dim x 1 x 1
        GatedConvTranspose2d(input_size, filter_depth*8, 4, stride=1),
        add_normalization(normalization_str, 2, filter_depth*8, num_groups=32),
        activation_fn(inplace=True),
        # state dim:   256 x 4 x 4
        GatedConvTranspose2d(filter_depth*8, filter_depth*4, 4, stride=2),
        add_normalization(normalization_str, 2, filter_depth*4, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 128 x 10 x 10
        GatedConvTranspose2d(filter_depth*4, filter_depth*2, 4, stride=1),
        add_normalization(normalization_str, 2, filter_depth*2, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 64 x 13 x 13
        GatedConvTranspose2d(filter_depth*2, filter_depth, 4, stride=2),
        add_normalization(normalization_str, 2, filter_depth, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 32 x 28 x 28
        GatedConvTranspose2d(filter_depth, filter_depth, 5, stride=1),
        add_normalization(normalization_str, 2, filter_depth, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 32 x 32 x 32
        nn.Conv2d(filter_depth, chans, 1, stride=1),
        # output dim: num_channels x 32 x 32
        upsampler if output_shape[1:] != bilinear_size else Identity()
    )


def build_conv_decoder(input_size, output_shape, filter_depth=32,
                       activation_fn=nn.ELU, bilinear_size=(32, 32),
                       normalization_str='none'):
    chans = output_shape[0]
    upsampler = nn.Upsample(size=output_shape[1:], mode='bilinear')
    return nn.Sequential(
        View([-1, input_size, 1, 1]),
        # input dim: z_dim x 1 x 1
        nn.ConvTranspose2d(input_size, filter_depth*8, 4, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth*8, num_groups=32),
        activation_fn(inplace=True),
        # state dim:   256 x 4 x 4
        nn.ConvTranspose2d(filter_depth*8, filter_depth*4, 4, stride=2, bias=True),
        add_normalization(normalization_str, 2, filter_depth*4, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 128 x 10 x 10
        nn.ConvTranspose2d(filter_depth*4, filter_depth*2, 4, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth*2, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 64 x 13 x 13
        nn.ConvTranspose2d(filter_depth*2, filter_depth, 4, stride=2, bias=True),
        add_normalization(normalization_str, 2, filter_depth, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 32 x 28 x 28
        nn.ConvTranspose2d(filter_depth, filter_depth, 5, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 32 x 32 x 32
        nn.Conv2d(filter_depth, chans, 1, stride=1, bias=True),
        # output dim: num_channels x 32 x 32
        upsampler if output_shape[1:] != bilinear_size else Identity()
    )



def build_conv_encoder(input_shape, output_size, filter_depth=32,
                       activation_fn=nn.ELU, bilinear_size=(32, 32),
                       normalization_str="none"):
    upsampler = nn.Upsample(size=bilinear_size, mode='bilinear')
    chans     = input_shape[0]
    return nn.Sequential(
        upsampler if input_shape[1:] != bilinear_size else Identity(),
        # input dim: num_channels x 32 x 32
        nn.Conv2d(chans, filter_depth, 5, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 32 x 28 x 28
        nn.Conv2d(filter_depth, filter_depth*2, 4, stride=2, bias=True),
        add_normalization(normalization_str, 2, filter_depth*2, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 64 x 13 x 13
        nn.Conv2d(filter_depth*2, filter_depth*4, 4, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth*4, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 128 x 10 x 10
        nn.Conv2d(filter_depth*4, filter_depth*8, 4, stride=2, bias=True),
        add_normalization(normalization_str, 2, filter_depth*8, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 256 x 4 x 4
        nn.Conv2d(filter_depth*8, filter_depth*16, 4, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth*16, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 512 x 1 x 1
        nn.Conv2d(filter_depth*16, filter_depth*16, 1, stride=1, bias=True),
        add_normalization(normalization_str, 2, filter_depth*16, num_groups=32),
        activation_fn(inplace=True),
        # state dim: 512 x 1 x 1
        nn.Conv2d(filter_depth*16, output_size, 1, stride=1, bias=True),
        # output dim: opt.z_dim x 1 x 1
        View([-1, output_size])
    )

def build_dense_encoder(input_shape, output_size, latent_size=512, activation_fn=nn.ELU, normalization_str="none"):
    input_flat  = int(np.prod(input_shape))
    output_flat = int(np.prod(output_size))
    output_size = [output_size] if not isinstance(output_size, list) else output_size
    return nn.Sequential(
        View([-1, input_flat]),
        nn.Linear(input_flat, latent_size),
        add_normalization(normalization_str, 1, latent_size, num_groups=32),
        activation_fn(),
        nn.Linear(latent_size, output_flat),
        View([-1, *output_size])
    )

def build_dense_decoder(input_size, output_shape, latent_size=512, activation_fn=nn.ELU, normalization_str="none"):
    input_flat = int(np.prod(input_size))
    return nn.Sequential(
        View([-1, input_flat]),
        nn.Linear(input_flat, latent_size),
        add_normalization(normalization_str, 1, latent_size, num_groups=32),
        activation_fn(),
        nn.Linear(latent_size, latent_size),
        add_normalization(normalization_str, 1, latent_size, num_groups=32),
        activation_fn(),
        nn.Linear(latent_size, int(np.prod(output_shape))),
        View([-1] + output_shape)
    )
