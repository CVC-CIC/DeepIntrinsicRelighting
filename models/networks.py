import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
from util.util import PARA_NOR
import lpips
from pytorch_msssim import SSIM
# from ignite.metrics import SSIM as SSIM_ignite
# from models.ssim_modify import SSIM
import kornia as K


###############################################################################
# Functions
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
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=opt.lr_decay_ratio)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.n_epochs, eta_min=0)
    elif opt.lr_policy == 'cyclic':
        scheduler = lr_scheduler.CyclicLR(optimizer, base_lr=opt.lr_cyclic_base, max_lr=opt.lr_cyclic_max,
                                          step_size_up=int(opt.cyclic_iter / 2), cycle_momentum=False)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        # elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
        elif classname.find('Norm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], parallel_method='DataParallel'):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        parallel_method (str) -- 'DataParallel', 'DistributedDataParallel'

    Return an initialized network.
    """
    init_weights(net, init_type, init_gain=init_gain)
    net = init_device(net, gpu_ids=gpu_ids, parallel_method=parallel_method)

    return net


def init_device(net, gpu_ids=[], parallel_method='DataParallel'):
    if parallel_method == 'DataParallel':
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    elif parallel_method == 'DistributedDataParallel':
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
        net.cuda(gpu_ids)
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[gpu_ids], broadcast_buffers=False)
    else:
        raise Exception("Parallel method type error!")
    return net


##############################################################################
# Losses
##############################################################################
class PanTiltLoss(nn.Module):
    # Pan and tilt should use the angular loss because pan can be any values when tilt = 0
    # The normalized angle is [cos(pan)/2+0.5, sin(pan)/2+0.5, cos(tilt), sin(tilt)]
    def __init__(self):
        super(PanTiltLoss, self).__init__()

    def forward(self, pan_tilt_pred, pan_tilt_target):
        def pan_tilt_to_vector(tri_nor):
            # tri_nor: [cos(pan) / 2 + 0.5, sin(pan) / 2 + 0.5, cos(tilt), sin(tilt)]
            tri_nor = torch.clamp(tri_nor, min=0.0, max=1.0)
            tri = torch.zeros(tri_nor.size()).cuda()
            tri[:, :2] = (tri_nor[:, :2] - PARA_NOR['pan_b']) / PARA_NOR['pan_a']
            tri[:, 2:] = (tri_nor[:, 2:] - PARA_NOR['tilt_b']) / PARA_NOR['tilt_a']
            # tri: [cos(pan), sin(pan), cos(tilt), sin(tilt)]
            vector = torch.zeros(tri.size()[0], 3).cuda()
            vector[:, 0] = torch.mul(tri[:, 3], tri[:, 0])
            vector[:, 1] = torch.mul(tri[:, 3], tri[:, 1])
            vector[:, 2] = tri[:, 2]
            return vector

        vector_pred = pan_tilt_to_vector(pan_tilt_pred)
        vector_target = pan_tilt_to_vector(pan_tilt_target)

        distance = torch.sqrt(F.mse_loss(vector_pred, vector_target))

        return distance


# class LPIPS_L1(nn.Module):
#     def __init__(self, gpu_ids):
#         super(LPIPS_L1, self).__init__()
#         net = lpips.LPIPS(net='alex')
#         if len(gpu_ids) > 0:
#             assert (torch.cuda.is_available())
#             net.to(gpu_ids[0])
#             net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#         self.loss_lpips = net
#         # self.loss_lpips = lpips.LPIPS(net='alex').cuda()
#         self.loss_l1 = torch.nn.L1Loss()
#
#     def forward(self, predict, target, weight):
#         distance_lpips = self.loss_lpips.forward(predict * 2 - 1, target * 2 - 1)  # change [0,+1] to [-1,+1]
#         distance_lpips = distance_lpips.mean()
#         distance_l1 = torch.mean(torch.abs(predict-target) * weight)
#         distance_total = distance_lpips + distance_l1
#         return distance_total


# class WeightedL1Loss(nn.Module):
#     def __init__(self):
#         super(WeightedL1Loss, self).__init__()
#
#     def forward(self, predict, target, weight):
#         return torch.mean(torch.abs(predict-target) * weight)


class LPIPS_LOSS(nn.Module):
    def __init__(self, gpu_ids):
        super(LPIPS_LOSS, self).__init__()
        net = lpips.LPIPS(net='alex')
        if isinstance(gpu_ids, list):
            assert (torch.cuda.is_available())
            net.to(gpu_ids[0])
            net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
        elif isinstance(gpu_ids, (int, str)):
            net.to(gpu_ids)
        self.loss_lpips = net

    def forward(self, predict, target):
        distance_lpips = self.loss_lpips.forward(predict * 2 - 1, target * 2 - 1)  # change [0,+1] to [-1,+1]
        distance_lpips = distance_lpips.mean()
        return distance_lpips


# class ORI_L1_LPIPS(nn.Module):
#     def __init__(self, gpu_ids):
#         super(ORI_L1_LPIPS, self).__init__()
#         net = lpips.LPIPS(net='alex')
#         if len(gpu_ids) > 0:
#             assert (torch.cuda.is_available())
#             net.to(gpu_ids[0])
#             net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#         self.loss_lpips = net
#         # self.loss_lpips = lpips.LPIPS(net='alex').cuda()
#         self.loss_l1 = torch.nn.L1Loss()
#
#     def forward(self, predict, target):
#         distance_lpips = self.loss_lpips.forward(predict * 2 - 1, target * 2 - 1)  # change [0,+1] to [-1,+1]
#         distance_lpips = distance_lpips.mean()
#         distance_l1 = self.loss_l1(predict, target)
#         distance_total = distance_lpips + distance_l1
#         return distance_total


class LPIPS_WL1_SSIM(nn.Module):
    def __init__(self, gpu_ids, use_amp=False):
        super(LPIPS_WL1_SSIM, self).__init__()
        self.loss_lpips = LPIPS_LOSS(gpu_ids)
        self.loss_l1 = torch.nn.L1Loss()
        self.loss_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.use_amp = use_amp

    def forward(self, predict, target, weight):
        distance_lpips = self.loss_lpips(predict, target)
        distance_l1 = torch.mean(torch.abs(predict-target) * weight)
        if self.use_amp:
            with autocast(enabled=False):
                distance_ssim = 1.0 - self.loss_ssim(predict.float(), target.float())
        else:
            distance_ssim = 1.0 - self.loss_ssim(predict, target)
        distance_total = distance_lpips + distance_l1 + distance_ssim
        return distance_total


# class LPIPS_L1_SSIM(nn.Module):
#     def __init__(self, gpu_ids):
#         super(LPIPS_L1_SSIM, self).__init__()
#         net = lpips.LPIPS(net='alex')
#         if len(gpu_ids) > 0:
#             assert (torch.cuda.is_available())
#             net.to(gpu_ids[0])
#             net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
#         self.loss_lpips = net
#         self.loss_l1 = torch.nn.L1Loss()
#         self.loss_ssim = SSIM(data_range=1.0, size_average=True, channel=3)
#
#     def forward(self, predict, target):
#         distance_lpips = self.loss_lpips.forward(predict * 2 - 1, target * 2 - 1)  # change [0,+1] to [-1,+1]
#         distance_lpips = distance_lpips.mean()
#         distance_l1 = self.loss_l1(predict, target)
#         distance_ssim = 1.0 - self.loss_ssim(predict, target)
#         distance_total = distance_lpips + distance_l1 + distance_ssim
#         return distance_total

# import matplotlib.pyplot as plt
# import kornia as K
# import torchvision
# import numpy as np
# def imshow(input: torch.Tensor):
#     out = torchvision.utils.make_grid(input, nrow=2, padding=5)
#     out_np: np.ndarray = K.utils.tensor_to_image(out)
#     out_np = out_np.astype('float32')
#     plt.imshow(out_np)
#     plt.axis('off')
#     plt.show()

def edge_attention_mask(img_tensor: torch.Tensor, dilation_kernel=5):
    if img_tensor.max() > 1.0 or img_tensor.min() < 0.0:
        raise ValueError("The range of input is wrong.")
    x_rgb = img_tensor.detach().clone()
    # imshow(x_rgb[0])
    x_gray = K.color.rgb_to_grayscale(x_rgb)
    x_canny = K.filters.canny(x_gray)[1]
    # dliation method
    # kernel = torch.ones(dilation_kernel, dilation_kernel).cuda()
    # mask = K.morphology.dilation(x_canny, kernel)
    # return mask.bool()
    mask = K.filters.gaussian_blur2d(x_canny, (21, 21), (8.0, 8.0))
    mask = (mask - mask.min()) / max(mask.max(), 1e-6)
    # imshow(mask[0])
    return mask


def edge_canny(img_tensor: torch.Tensor):
    if img_tensor.max() > 1.0 or img_tensor.min() < 0.0:
        raise ValueError("The range of input is wrong.")
    x_rgb = img_tensor
    x_gray = K.color.rgb_to_grayscale(x_rgb)
    x_canny = K.filters.canny(x_gray)[1]
    x_canny = x_canny.repeat(1, 3, 1, 1)
    return x_canny


class L1_DSSIM_LPIPS(nn.Module):
    def __init__(self, gpu_ids, flag_L1_DSSIM_LPIPS=None, use_amp=False,
                 divide_half=False, divide_para=None):
        super(L1_DSSIM_LPIPS, self).__init__()
        if flag_L1_DSSIM_LPIPS is None:
            flag_L1_DSSIM_LPIPS = [True, True, True]
        self.flag_L1_DSSIM_LPIPS = flag_L1_DSSIM_LPIPS
        if self.flag_L1_DSSIM_LPIPS[0]:
            self.loss_l1 = torch.nn.L1Loss()
        if self.flag_L1_DSSIM_LPIPS[1]:
            self.loss_ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3)
        if self.flag_L1_DSSIM_LPIPS[2]:
            self.loss_lpips = LPIPS_LOSS(gpu_ids)
        self.use_amp = use_amp
        # Different weights for the fast half and the last half
        self.divide_half = divide_half
        if divide_half:
            self.divide_para = divide_para

    def loss_dssim(self, predict, target):
        if self.use_amp:
            with autocast(enabled=False):
                distance_ssim = 1.0 - self.loss_ssim_metric(predict.float(), target.float())
        else:
            distance_ssim = 1.0 - self.loss_ssim_metric(predict, target)
        return distance_ssim

    def all_loss(self, predict, target):
        distance_list = []
        if self.flag_L1_DSSIM_LPIPS[0]:
            distance_list.append(self.loss_l1(predict, target))
        if self.flag_L1_DSSIM_LPIPS[1]:
            distance_list.append(self.loss_dssim(predict, target))
        if self.flag_L1_DSSIM_LPIPS[2]:
            distance_list.append(self.loss_lpips(predict, target))
        distance_total = torch.stack(distance_list).sum()
        return distance_total

    def forward(self, img1, img2, disable_divide=False):
        if self.divide_half and not disable_divide:
            half_length = int(img1.size()[0] / 2)
            distance_total = self.divide_para[0] * self.all_loss(img1[:half_length], img2[:half_length]) + \
                             self.divide_para[1] * self.all_loss(img1[half_length:], img2[half_length:])
        else:
            distance_total = self.all_loss(img1, img2)
        return distance_total

# class Atten_L1_SSIM_LPIPS(nn.Module):
#     def __init__(self, gpu_ids, flag_L1_DSSIM_LPIPS=[True, True, True], use_attention=True, attention_weight=None,
#                  attention_schduler_type=None, use_amp=False):
#         super(Atten_L1_SSIM_LPIPS, self).__init__()
#         self.loss_lpips = LPIPS_LOSS(gpu_ids)
#         self.loss_l1 = torch.nn.L1Loss()
#         self.loss_ssim_metric = SSIM(data_range=1.0, size_average=True, channel=3)
#         self.use_amp = use_amp
#         self.use_attention = use_attention
#         if use_attention:
#             self.schduler_type = attention_schduler_type
#             if self.schduler_type == "increase":
#                 self.attention_weight_last = attention_weight
#                 self.step_epoch_last = 50.0
#                 self.attention_weight = 0.0
#             elif self.schduler_type == "step":
#                 self.attention_weight_last = attention_weight
#                 self.step_epoch_last = 10.0
#                 self.attention_weight = 0.0
#             else:
#                 self.attention_weight = attention_weight
#
#     def update_attention_weight(self, epoch):
#         if self.schduler_type == "increase":
#             if epoch < self.step_epoch_last:
#                 self.attention_weight = (epoch / self.step_epoch_last) * self.attention_weight_last
#             else:
#                 self.attention_weight = self.attention_weight_last
#         elif self.schduler_type == "step":
#             self.attention_weight = 0.0 if epoch < self.step_epoch_last else self.attention_weight_last
#         else:
#             pass
#         print("Update the weight of edge attention to ", self.attention_weight)
#
#     def loss_ssim(self, predict, target):
#         if self.use_amp:
#             with autocast(enabled=False):
#                 distance_ssim = 1.0 - self.loss_ssim_metric(predict.float(), target.float())
#         else:
#             distance_ssim = 1.0 - self.loss_ssim_metric(predict, target)
#         return distance_ssim
#
#     def all_loss(self, predict, target):
#         distance_lpips = self.loss_lpips(predict, target)
#         distance_l1 = self.loss_l1(predict, target)
#         distance_ssim = self.loss_ssim(predict, target)
#         distance_total = distance_lpips + distance_l1 + distance_ssim
#         return distance_total
#
#     def all_loss_mask(self, predict, target):
#         distance_l1 = self.loss_l1(predict, target)
#         return distance_l1
#         # distance_lpips = self.loss_lpips(predict, target)
#         # distance_ssim = self.loss_ssim(predict, target)
#         # distance_total_mask = distance_l1 + distance_ssim + distance_lpips
#         # return distance_total_mask
#
#     def forward(self, img1, img2, atten_mask=None):
#         distance_wo_mask = self.all_loss(img1, img2)
#         if not self.use_attention or atten_mask is None:
#             return distance_wo_mask
#         mask_ch = atten_mask.repeat(1, 3, 1, 1)
#         distance_mask = self.all_loss_mask(img1 * mask_ch, img2 * mask_ch)
#         distance_total = distance_wo_mask + distance_mask * self.attention_weight
#         return distance_total


def sobel_lab_chromaticity(x, obtain_overall=False):
    """
    x should be BxCxHxW and in range [0, 1].
    """
    x_lab = K.color.rgb_to_lab(x)
    x_lab[:, 0, :, :] = x_lab[:, 0, :, :] / 100.0
    x_lab[:, 1, :, :] = (x_lab[:, 1, :, :] + 127.0) / 254.0
    x_lab[:, 2, :, :] = (x_lab[:, 2, :, :] + 127.0) / 254.0
    if x_lab.max() > 1.0 or x_lab.min() < 0.0:
        raise Exception("LAB range error!")
    x_lab_sobel = K.filters.sobel(x_lab)
    ab_sobel = x_lab_sobel[:, 1, :, :].mean() + x_lab_sobel[:, 2, :, :].mean()
    if not obtain_overall:
        return ab_sobel / 2.0
    else:
        overall_sobel = ab_sobel + x_lab_sobel[:, 0, :, :].mean()
        return ab_sobel / 2.0, overall_sobel / 3.0

def convert_rgb2opp(x):
    r = x[:, 0, :, :]
    g = x[:, 1, :, :]
    b = x[:, 2, :, :]
    o1 = (r + g + b - 1.5) / 1.5
    o2 = r - g
    o3 = (r + g - 2.0 * b) / 2.0
    opp = torch.stack([o1, o2, o3], dim=-3)
    return opp

def sobel_opp_chromaticity(x, obtain_overall=False, crop_edge=True):
    opp = convert_rgb2opp(x)
    opp_sobel = K.filters.sobel(opp)
    if crop_edge:
        # We observed that the edge will have some strange colors when using the regularization.
        # So, this is to crop the edge.
        opp_sobel = opp_sobel[:, :, 1:-1, 1:-1]
    c_sobel_mean = opp_sobel[:, 1, :, :].mean() + opp_sobel[:, 2, :, :].mean()
    if not obtain_overall:
        return c_sobel_mean / 2.0
    else:
        overall_sobel = c_sobel_mean + opp_sobel[:, 0, :, :].mean()
        return c_sobel_mean / 2.0, overall_sobel / 3.0

def scheduler_init_ref(epoch, para_init_ref):
    if para_init_ref['decay']:
        w_first = 1.0
        w_last = 0.01
        e_first = 1
        if 'decay_last_epoch' in para_init_ref:
            e_last = para_init_ref['decay_last_epoch']
        else:
            e_last = 50
        w_epoch = w_first - (w_first - w_last) * (epoch - e_first) / (e_last - e_first) if epoch < e_last else w_last
        return w_epoch
    else:
        return 1.0


