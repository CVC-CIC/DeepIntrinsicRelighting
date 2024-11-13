import torch
import torch.nn as nn
import functools
from functools import partial
from models.networks import get_norm_layer
from models.networks_custom_func import SigLinear


##############################################################################
# Model for one-to-one Unet with new light condition
##############################################################################
def define_net_one_to_one_new_light(input_nc, output_nc, ngf, norm='batch', use_dropout=False,
                                    light_type="pan_tilt_color", light_prediction=True, netG="unet"):
    """
    Define a Unet with one encoder and one decoder. The light is replaced in the bottleneck.
    Parameters are the same as above.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    flag_add_nonlocal = "nonlocal" in netG
    flag_InfRange = "InfRange" in netG
    if "resnet" in netG:
        number_res_blocks = int(netG.split("_")[0].strip("resnet"))
        net = ResnetGeneratorRelighting(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                                        n_blocks=number_res_blocks, light_type=light_type,
                                        light_prediction=light_prediction,
                                        add_nonlocal=flag_add_nonlocal, InfRange=flag_InfRange)
    elif "unet" in netG:
        net = Unet_one_to_one_new_light(input_nc, output_nc, 8, ngf, norm_layer=norm_layer,
                                        use_dropout=use_dropout,
                                        light_type=light_type, light_prediction=light_prediction,
                                        InfRange=flag_InfRange)
    else:
        raise Exception("Error in parameter netG")

    # return init_net(net, init_type, init_gain, gpu_ids, parallel_method)
    return net


class Unet_one_to_one_new_light(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 light_type="pan_tilt_color", light_prediction=True, InfRange=False):
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
        super(Unet_one_to_one_new_light, self).__init__()
        # construct unet structure
        unet_block = Bot_replace_light(ngf * 8, norm_layer=norm_layer, light_type=light_type,
                                       light_prediction=light_prediction)
        skip = partial(UnetSkipConnectionBlock_one_to_one_new_light, InfRange=InfRange)
        unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                          norm_layer=norm_layer, innermost=True,
                          light_prediction=light_prediction)  # add the innermost layer
        for i in range(num_downs - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = skip(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block,
                              norm_layer=norm_layer, use_dropout=use_dropout, light_prediction=light_prediction)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = skip(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer, light_prediction=light_prediction)
        unet_block = skip(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer, light_prediction=light_prediction)
        unet_block = skip(ngf, ngf * 2, input_nc=None, submodule=unet_block,
                                                   norm_layer=norm_layer, light_prediction=light_prediction)
        self.model = skip(output_nc, ngf, input_nc=input_nc, submodule=unet_block,
                          outermost=True, norm_layer=norm_layer, light_prediction=light_prediction)  # add the outermost layer

    def forward(self, x_image, x_new_light):
        """Standard forward"""
        return self.model(x_image, x_new_light)


class UnetSkipConnectionBlock_one_to_one_new_light(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
        |                               -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 light_prediction=True, InfRange=False):
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
        super(UnetSkipConnectionBlock_one_to_one_new_light, self).__init__()
        self.light_prediction = light_prediction
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

        if outermost:
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1)
            down = [downconv]
            up1 = [uprelu1, upconv1]
            if not InfRange:
                up1 += [nn.Sigmoid()]
            else:
                up1 += [SigLinear()]
        elif innermost:
            upconv1 = nn.ConvTranspose2d(inner_nc, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up1 = [uprelu1, upconv1, upnorm1]
        else:
            upconv1 = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                         kernel_size=4, stride=2,
                                         padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up1 = [uprelu1, upconv1, upnorm1]

            if use_dropout:
                updrop1 = nn.Dropout(0.5)
                up1 = up1 + [updrop1]

        self.down = nn.Sequential(*down)
        self.submodule = submodule
        self.up1 = nn.Sequential(*up1)

    def forward(self, x_input, x_new_light):
        x_down = self.down(x_input)
        if self.light_prediction:
            y_ori_light, y_up1_sub = self.submodule(x_down, x_new_light)
        else:
            y_up1_sub = self.submodule(x_down, x_new_light)
        y_up1 = self.up1(y_up1_sub)
        if self.outermost:
            if self.light_prediction:
                return y_ori_light, y_up1
            else:
                return y_up1
        else:
            # add skip connections
            if self.light_prediction:
                return y_ori_light, torch.cat([x_input, y_up1], 1)
            else:
                return torch.cat([x_input, y_up1], 1)


class Bot_replace_light(nn.Module):
    """Defines the center submodule of Unet to replace the light condition.
            X -------------------identity----------------------
            |-- downsampling -- original light condition                  |
            |                        new light condition  -- upsampling --|
    """
    def __init__(self, nc_two_head, norm_layer=nn.BatchNorm2d, light_type="pan_tilt_color", light_prediction=True):
        super(Bot_replace_light, self).__init__()
        self.light_type = light_type
        self.light_prediction = light_prediction
        input_nc = nc_two_head
        output_nc = nc_two_head
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        conv_down = [nn.LeakyReLU(0.2, False), nn.Conv2d(input_nc, input_nc // 2, kernel_size=1,
                                                         stride=1, padding=0, bias=use_bias), nn.LeakyReLU(0.2, False)]
        self.conv_down = nn.Sequential(*conv_down)
        if light_type == "probes":
            conv_up_probe = [nn.ReLU(False), nn.ConvTranspose2d(input_nc, input_nc,
                                                          kernel_size=1, stride=1, padding=0), norm_layer(output_nc)]
            self.conv_up_probe = nn.Sequential(*conv_up_probe)
        else:
            conv_up = [nn.ReLU(False), nn.ConvTranspose2d(input_nc // 2, input_nc,
                                                          kernel_size=1, stride=1, padding=0), norm_layer(output_nc)]
            self.conv_up = nn.Sequential(*conv_up)

        if light_type == "Spherical_harmonic":
            if not light_prediction:
                raise Exception("This setting is not implemented.")
            fc_down_sh = [nn.Linear(256, 128), nn.PReLU(), nn.Linear(128, 9)]
            fc_up_sh = [nn.Linear(9*2, 128), nn.PReLU(), nn.Linear(128, 256)]
            self.fc_down_sh = nn.Sequential(*fc_down_sh)
            self.fc_up_sh = nn.Sequential(*fc_up_sh)
        elif light_type == "pan_tilt_color":
            if not light_prediction:
                raise Exception("This setting is not implemented.")
            fc_down = [nn.Linear(256, 7), nn.Sigmoid()]
            fc_up = [nn.Linear(7 * 2, 256)]
            self.fc_down = nn.Sequential(*fc_down)
            self.fc_up = nn.Sequential(*fc_up)
        elif light_type == "probes":
            if light_prediction:
                raise Exception("This setting is not implemented.")
            light_channel = 6
            probe_in = []
            channel_in = light_channel
            for i in range(3):
                channel_out = (4**i) * 16
                probe_in += [
                    nn.Conv2d(channel_in, channel_out, kernel_size=4, stride=4, padding=0, bias=use_bias),
                    norm_layer(channel_out),
                    nn.ReLU(True)]
                channel_in = channel_out
            self.probe_in = nn.Sequential(*probe_in)
        else:
            raise Exception("light_type is not implemented.")

    def forward(self, x_down, x_new_light):
        y_ori_light = self.conv_down(x_down)
        if self.light_type in ["Spherical_harmonic", "pan_tilt_color"]:
            y_ori_light = y_ori_light.view(y_ori_light.size(0), -1)
            if self.light_type == "Spherical_harmonic":
                y_ori_light = self.fc_down_sh(y_ori_light)
                y_up = self.fc_up_sh(torch.cat([y_ori_light, x_new_light], 1))
            elif self.light_type == "pan_tilt_color":
                y_ori_light = self.fc_down(y_ori_light)
                y_up = self.fc_up(torch.cat([y_ori_light, x_new_light], 1))
            else:
                raise Exception("light_type is not implemented.")
            y_up = y_up.unsqueeze(2)
            y_up = y_up.unsqueeze(3)
            y_up = self.conv_up(y_up)
        elif self.light_type == "probes":
            y_new_light = self.probe_in(x_new_light)
            y_up = torch.cat([y_ori_light, y_new_light], 1)
            y_up = self.conv_up_probe(y_up)
        else:
            raise Exception("light_type is not implemented.")

        if self.light_prediction:
            return y_ori_light, y_up
        else:
            return y_up


##############################################################################
# Resnet for relighting
##############################################################################
class ResnetGeneratorRelighting(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 light_type="pan_tilt_color", light_prediction=True, padding_type='reflect', add_nonlocal=False,
                 InfRange=False):
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
        super(ResnetGeneratorRelighting, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.norm_layer = norm_layer
        self.use_bias = use_bias

        self.light_type = light_type
        self.light_prediction = light_prediction
        # for light condition
        light_merge = 16
        if light_type == "probes":
            light_channel = 6
            probe_in = [nn.ReflectionPad2d(3),
                        nn.Conv2d(light_channel, light_merge, kernel_size=7, padding=0, bias=use_bias),
                        norm_layer(light_merge),
                        nn.ReLU(True)]
            probe_in += [ResnetBlock(light_merge, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=use_bias)]
            probe_in += [ResnetBlock(light_merge, padding_type=padding_type, norm_layer=norm_layer,
                                     use_dropout=use_dropout, use_bias=use_bias)]
            self.probe_in = nn.Sequential(*probe_in)
            if self.light_prediction:
                probe_out = [ResnetBlock(light_merge, padding_type=padding_type, norm_layer=norm_layer,
                                         use_dropout=use_dropout, use_bias=use_bias)]
                probe_out += [ResnetBlock(light_merge, padding_type=padding_type, norm_layer=norm_layer,
                                          use_dropout=use_dropout, use_bias=use_bias)]
                probe_out += [nn.ReflectionPad2d(3),
                              nn.Conv2d(light_merge, light_channel, kernel_size=7, padding=0),
                              nn.Sigmoid()]
                self.probe_out = nn.Sequential(*probe_out)
        elif light_type == "pan_tilt_color":
            light_modules = self.light_vector_module(light_channel=7, light_merge=light_merge)
            self.light_up = nn.Sequential(*light_modules[1])
            self.fc_in_light = nn.Sequential(*light_modules[0])
            if self.light_prediction:
                self.light_down = nn.Sequential(*light_modules[2])
                self.fc_out_light = nn.Sequential(*light_modules[3])
        elif light_type == "Spherical_harmonic":
            light_modules = self.light_vector_module(light_channel=9, light_merge=light_merge)
            self.SH_in_light = nn.Sequential(*light_modules[0])
            self.SH_light_up = nn.Sequential(*light_modules[1])
            if self.light_prediction:
                self.SH_light_down = nn.Sequential(*light_modules[2])
                self.SH_out_light = nn.Sequential(*light_modules[3])
        else:
            raise Exception("light_type is wrong!")

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
        model_res_right = []
        for i in range(n_blocks_right):  # add ResNet blocks
            if add_nonlocal:
                if i == 1:
                    model_res_right += [AttnBlock(ngf * mult)]
            model_res_right += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer,
                                            use_dropout=use_dropout, use_bias=use_bias)]

        # Upsampling part
        model_up = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_up += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                             kernel_size=3, stride=2,
                                             padding=1, output_padding=1,
                                             bias=use_bias),
                         norm_layer(int(ngf * mult / 2)),
                         nn.ReLU(True)]
        model_up += [nn.ReflectionPad2d(3)]
        model_up += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if not InfRange:
            model_up += [nn.Sigmoid()]
        else:
            model_up += [SigLinear()]
        # model += [nn.Tanh()]

        self.model_down = nn.Sequential(*model_down)
        self.model_res_left = nn.Sequential(*model_res_left)
        self.model_res_right = nn.Sequential(*model_res_right)
        self.model_up = nn.Sequential(*model_up)
        self.split_point = ngf * 2 ** n_downsampling - light_merge

    def light_vector_module(self, light_channel=7, light_merge=16):
        fc_in_light = [nn.Linear(light_channel, light_merge)]

        n_sample = 3
        light_up = []
        for i in range(n_sample):
            light_up += [nn.ConvTranspose2d(light_merge, light_merge, kernel_size=4, stride=4,
                                            padding=0, bias=self.use_bias),
                         self.norm_layer(light_merge),
                         nn.ReLU(True)]

        if self.light_prediction:
            light_down = []
            for i in range(n_sample):
                light_down += [
                    nn.Conv2d(light_merge, light_merge, kernel_size=4, stride=4, padding=0, bias=self.use_bias),
                    self.norm_layer(light_merge),
                    nn.ReLU(True)]

            fc_out_light = [nn.Linear(light_merge, light_channel), nn.Sigmoid()]

            return fc_in_light, light_up, light_down, fc_out_light
        else:
            return fc_in_light, light_up

    def forward(self, x_image, x_new_light):
        """Standard forward"""
        out = self.model_down(x_image)
        out = self.model_res_left(out)
        out1 = out[:, self.split_point:, :, :]
        out2 = out[:, :self.split_point, :, :]
        if self.light_type == "probes":
            if self.light_prediction:
                out1 = self.probe_out(out1)
            in_light = self.probe_in(x_new_light)
        elif self.light_type == "pan_tilt_color":
            if self.light_prediction:
                out1 = self.light_down(out1).squeeze(3).squeeze(2)
                out1 = self.fc_out_light(out1)
            in_light = self.fc_in_light(x_new_light)
            in_light = in_light.unsqueeze(2).unsqueeze(3)
            in_light = self.light_up(in_light)
        elif self.light_type == "Spherical_harmonic":
            if self.light_prediction:
                out1 = self.SH_light_down(out1).squeeze(3).squeeze(2)
                out1 = self.SH_out_light(out1)
            in_light = self.SH_in_light(x_new_light)
            in_light = in_light.unsqueeze(2).unsqueeze(3)
            in_light = self.SH_light_up(in_light)
        else:
            raise Exception("light_type is wrong!")

        out2 = torch.cat((out2, in_light), 1)
        out2 = self.model_res_right(out2)
        out2 = self.model_up(out2)
        if self.light_prediction:
            return out1, out2
        else:
            return out2


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_

