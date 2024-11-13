import torch
import torch.nn as nn
import functools
from functools import partial
from models.networks import get_norm_layer
from models.networks_one_to_one_rep import ResnetBlock, AttnBlock
from models.networks_custom_func import SigLinear

##############################################################################
# Model for intrinsic_decomposition
##############################################################################
def define_net_intrinsic_decomposition(input_nc, output_nc, ngf, norm='batch', use_dropout=False,
                                       net_intrinsic = "unet"):
    """Create a generator for intrinsic_decomposition

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a generator
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    flag_add_nonlocal = "nonlocal" in net_intrinsic
    flag_InfRange = "InfRange" in net_intrinsic
    if "resnet" in net_intrinsic:
        number_res_blocks = int(net_intrinsic.split("_")[0].strip("resnet"))
        net = ResnetGeneratorIntrinsic(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                       n_blocks=number_res_blocks, add_nonlocal=flag_add_nonlocal,
                                       InfRange=flag_InfRange)
    elif "unet" in net_intrinsic:
        net = Unet_intrinsic_decomposition(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                                           use_dropout=use_dropout, InfRange=flag_InfRange)
    else:
        raise Exception("Error in parameter net_intrinsic")

    # return init_net(net, init_type, init_gain, gpu_ids, parallel_method)
    return net



class Unet_intrinsic_decomposition(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 InfRange=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(Unet_intrinsic_decomposition, self).__init__()
        # construct unet structure
        skip = partial(UnetSkipConnectionBlock_intrinsic_decomposition, InfRange=InfRange)
        unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=None,
                                                   norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                                                       norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = skip(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        unet_block = skip(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        unet_block = skip(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer)
        self.model = skip(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                                                   outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, x_image):
        """Standard forward"""
        return self.model(x_image)


class UnetSkipConnectionBlock_intrinsic_decomposition(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
        |                               -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 InfRange=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetDoubleSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock_intrinsic_decomposition, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # except the outermost layer, input_nc should be equal to outer_nc
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu1 = nn.ReLU(False)
        upnorm1 = norm_layer(outer_nc)
        uprelu2 = nn.ReLU(False)
        upnorm2 = norm_layer(outer_nc)

        if outermost:
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            upconv2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [downconv]
            up1 = [uprelu1, upconv1]
            up2 = [uprelu2, upconv2]
            if not InfRange:
                up1 += [nn.Sigmoid()]
                up2 += [nn.Sigmoid()]
            else:
                up1 += [nn.Sigmoid()]
                up2 += [SigLinear()]
        elif innermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            upconv2 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up1 = [uprelu1, upconv1, upnorm1]
            up2 = [uprelu2, upconv2, upnorm2]
        else:
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            upconv2 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up1 = [uprelu1, upconv1, upnorm1]
            up2 = [uprelu2, upconv2, upnorm2]

            if use_dropout:
                updrop1 = nn.Dropout(0.5)
                up1 = up1 + [updrop1]
                updrop2 = nn.Dropout(0.5)
                up2 = up2 + [updrop2]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up1 = nn.Sequential(*up1)
        self.up2 = nn.Sequential(*up2)

    def forward(self, x_input):
        x_down = self.down(x_input)

        if self.innermost:
            y_up1 = self.up1(x_down)
            y_up2 = self.up2(x_down)
        else:
            y_up1_sub, y_up2_sub = self.submodule(x_down)
            y_up1 = self.up1(y_up1_sub)
            y_up2 = self.up2(y_up2_sub)
        if self.outermost:
            return y_up1, y_up2
        else:
            # add skip connections
            return torch.cat([x_input, y_up1], 1), torch.cat([x_input, y_up2], 1)


##############################################################################
# Resnet for intrinsic decomposition
##############################################################################
class ResnetGeneratorIntrinsic(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', add_nonlocal=False, InfRange=False):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGeneratorIntrinsic, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        # downsampling part
        model_down = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_down += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                           norm_layer(ngf * mult * 2),
                           nn.ReLU(True)]

        # Resnet part
        mult = 2 ** n_downsampling
        n_blocks_left = int(n_blocks / 2)
        n_blocks_right = n_blocks - n_blocks_left
        model_res_left = []
        for i in range(n_blocks_left):       # add ResNet blocks
            if add_nonlocal:
                if i == (n_blocks_left - 1):
                    model_res_left += [AttnBlock(ngf * mult)]
            model_res_left += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                           use_dropout=use_dropout, use_bias=use_bias)]
        n_ch_1 = int(ngf * mult / 2)
        n_ch_2 = ngf * mult - n_ch_1
        self.split_ch = n_ch_1
        model_res_right_1 = []
        model_res_right_2 = []
        for i in range(n_blocks_right):
            if add_nonlocal:
                if i == 1:
                    model_res_right_1 += [AttnBlock(n_ch_1)]
                    model_res_right_2 += [AttnBlock(n_ch_2)]
            model_res_right_1 += [ResnetBlock(n_ch_1, padding_type=padding_type, norm_layer=norm_layer,
                                              use_dropout=use_dropout, use_bias=use_bias)]
            model_res_right_2 += [ResnetBlock(n_ch_2, padding_type=padding_type, norm_layer=norm_layer,
                                              use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling part
        model_up1 = []
        model_up2 = []
        # split for relighting
        for i in range(n_downsampling):  # add upsampling layers
            model_up1 += [nn.ConvTranspose2d(n_ch_1, int(n_ch_1 / 2), kernel_size=3, stride=2, padding=1,
                                            output_padding=1, bias=use_bias),
                         norm_layer(int(n_ch_1 / 2)),
                         nn.ReLU(True)]
            n_ch_1 = int(n_ch_1 / 2)
            model_up2 += [nn.ConvTranspose2d(n_ch_2, int(n_ch_2 / 2), kernel_size=3, stride=2, padding=1,
                                             output_padding=1, bias=use_bias),
                          norm_layer(int(n_ch_2 / 2)),
                          nn.ReLU(True)]
            n_ch_2 = int(n_ch_2 / 2)
        model_up1 += [nn.ReflectionPad2d(3)]
        model_up1 += [nn.Conv2d(int(ngf / 2), output_nc, kernel_size=7, padding=0)]
        model_up2 += [nn.ReflectionPad2d(3)]
        model_up2 += [nn.Conv2d(int(ngf / 2), output_nc, kernel_size=7, padding=0)]
        if not InfRange:
            model_up1 += [nn.Sigmoid()]
            model_up2 += [nn.Sigmoid()]
        else:
            model_up1 += [nn.Sigmoid()]
            model_up2 += [SigLinear()]
        # model += [nn.Tanh()]

        self.model_down = nn.Sequential(*model_down)
        self.model_res_left = nn.Sequential(*model_res_left)
        self.model_res_right_1 = nn.Sequential(*model_res_right_1)
        self.model_res_right_2 = nn.Sequential(*model_res_right_2)
        self.model_up1 = nn.Sequential(*model_up1)
        self.model_up2 = nn.Sequential(*model_up2)

    def forward(self, x_image):
        """Standard forward"""
        out = self.model_down(x_image)
        out = self.model_res_left(out)
        out1 = out[:, :self.split_ch, :, :]
        out2 = out[:, self.split_ch:, :, :]
        out1 = self.model_res_right_1(out1)
        out2 = self.model_res_right_2(out2)
        out1 = self.model_up1(out1)
        out2 = self.model_up2(out2)
        return out1, out2

