import torch
from .base_model import BaseModel
import os
from torch.autograd import Variable
from models.drn.networks import define_G, define_D, GANLoss, VGGLoss
import models.drn.pytorch_ssim as pytorch_ssim
from models.drn.image_pool import ImagePool
from torch import nn
import numpy as np

from models.networks import init_device
from thop import profile, clever_format


class DRNModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        if opt.use_discriminator:
            self.loss_names = ['G_rec', 'G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake']
        else:
            self.loss_names = ['G_rec', 'G_VGG']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['Image_input', 'Relighted_gt', 'Relighted_predict']
        if not opt.isTrain:
            if opt.light_type != "probes":
                self.visual_names.extend(['light_position_color_original', 'light_position_color_new'])
            if opt.special_test:
                self.visual_names = ['Image_input', 'Relighted_predict', 'light_position_color_predict',
                                     'light_position_color_new']
        if opt.light_type == "probes":
            self.visual_names.extend(['probe_gt_ori', 'probe_gt_new'])
        if opt.use_discriminator:
            self.model_names = ['G', 'D']
            self.optimizer_names = ['G', 'D']
        else:
            self.model_names = ['G']
            self.optimizer_names = ['G']
        # define networks (both generator and discriminator)
        self.netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG,
                             opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                             opt.n_blocks_local, opt.norm, light_type=opt.light_type,
                             light_prediction=opt.light_prediction, gpu_ids=self.gpu_ids)
        self.netG = init_device(self.netG, gpu_ids=self.gpu_ids, parallel_method=opt.parallel_method)

        if opt.use_discriminator:
            self.netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.n_layers_D, opt.norm, opt.use_sigmoid,
                                 opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)
            self.netD = init_device(self.netD, gpu_ids=self.gpu_ids, parallel_method=opt.parallel_method)

        # for plotting model
        self.expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if self.isTrain:
            self.fake_pool = ImagePool(opt.pool_size)
            # define loss functions
            if opt.use_discriminator:
                self.loss_weight_GAN = opt.loss_weight_GAN
                self.criterionGAN = GANLoss(use_lsgan=not opt.no_lsgan, tensor=torch.cuda.FloatTensor)
                self.criterionFeat = torch.nn.L1Loss()
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionSSIM = pytorch_ssim.SSIM()
            # self.criterionMSE = torch.nn.MSELoss()
            self.criterionLAP = LapLoss()
            self.criterionVGG = VGGLoss(self.gpu_ids)

            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if opt.use_discriminator:
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)

    def get_macs(self):
        x_image = torch.rand(1, 3, 256, 256).to(self.device)
        l_channel = 9 if self.opt.light_type == "Spherical_harmonic" else 7
        x_light = torch.rand(1, l_channel).to(self.device)
        macs, params = profile(self.netG, inputs=(x_image, x_light))
        macs, params = clever_format([macs, params], "%.3f")
        print("macs, params", macs, params)
        if self.opt.use_discriminator:
            # discriminator
            x_D = torch.rand(1, 6, 256, 256).to(self.device)
            macs, params = profile(self.netD, inputs=(x_D,))
            macs, params = clever_format([macs, params], "%.3f")
            print("macs, params", macs, params)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        ['Image_input', 'light_position_color_new', 'light_position_color_original', 'Reflectance_output',
        'Shading_output', 'Image_relighted', 'scene_label', 'num_imgs_in_one_batch',
        'Shading_ori']
        """
        self.input = {}
        # for key in ['Image_input', 'light_position_color_original', 'light_position_color_new', 'Image_relighted']:
        #     self.input[key] = input[key].to(self.device)
        # self.input['scene_label'] = input['scene_label']
        for key, data in input.items():
            if key != 'scene_label':
                self.input[key] = data.to(self.device)
            else:
                self.input[key] = data

        self.image_paths = input['scene_label']

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)

    def forward(self):
        img_in = self.input['Image_input']
        real_image = self.input['Image_relighted']
        input_label = img_in

        fake_image = self.netG.forward(img_in, self.input['light_position_color_new'])

        if self.opt.isTrain and self.opt.use_discriminator:
            # Fake Detection and Loss
            pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
            loss_D_fake = self.criterionGAN(pred_fake_pool, False)

            # Real Detection and Loss
            pred_real = self.discriminate(input_label, real_image)
            loss_D_real = self.criterionGAN(pred_real, True)

            # GAN loss (Fake Passability Loss)
            pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))
            loss_G_GAN = self.criterionGAN(pred_fake, True)

            # GAN feature matching loss
            loss_G_GAN_Feat = 0
            if not self.opt.no_ganFeat_loss:
                feat_weights = 4.0 / (self.opt.n_layers_D + 1)
                D_weights = 1.0 / self.opt.num_D
                for i in range(self.opt.num_D):
                    for j in range(len(pred_fake[i]) - 1):
                        loss_G_GAN_Feat += D_weights * feat_weights * \
                                           self.criterionFeat(pred_fake[i][j],
                                                              pred_real[i][j].detach()) * self.opt.lambda_feat

        if self.opt.isTrain:
            # VGG feature matching loss
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat * 0.01
            # loss_G_rec = (self.criterionL1(fake_image, real_image) + 0.1 * self.criterionLPIPS(fake_image,real_image)) \
            #              * self.opt.lambda_feat
            loss_G_rec = (self.criterionLAP(fake_image, real_image) + 0.1 * self.criterionSSIM(fake_image,
                                                                                               real_image)) * 0.1 + self.criterionL1(
                fake_image, real_image) * self.opt.lambda_feat

        self.Image_input = self.input['Image_input']
        if self.isTrain or not self.opt.special_test:
            self.Relighted_gt = self.input['Image_relighted']

        self.Relighted_predict = fake_image

        self.light_position_color_original = self.input['light_position_color_original']
        self.light_position_color_new = self.input['light_position_color_new']

        if self.opt.light_type == "probes":
            def probe_reshape(p):
                return torch.cat([p[:, :3, :, :], p[:, 3:, :, :]], dim=-1)
            self.probe_gt_ori = probe_reshape(self.input['light_position_color_original'])
            self.probe_gt_new = probe_reshape(self.input['light_position_color_new'])

        if self.opt.isTrain:
            self.loss_G_rec = loss_G_rec
            self.loss_G_VGG = loss_G_VGG
        if self.opt.use_discriminator:
            self.loss_G_GAN = loss_G_GAN * self.loss_weight_GAN
            self.loss_G_GAN_Feat = loss_G_GAN_Feat * self.loss_weight_GAN
            self.loss_D_real = loss_D_real * self.loss_weight_GAN
            self.loss_D_fake = loss_D_fake * self.loss_weight_GAN

    def backward_G(self):
        if self.opt.use_discriminator:
            loss_G = self.loss_G_rec + self.loss_G_GAN + self.loss_G_GAN_Feat + self.loss_G_VGG
        else:
            loss_G = self.loss_G_rec + self.loss_G_VGG
        self.loss_weighted_total = loss_G
        loss_G.backward()

    def backward_D(self):
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        loss_D.backward()

    def optimize_parameters(self, epoch=0, iter=0):
        # if epoch > self.opt.epoch_start_train_discriminator:
        #     self.weightGAN = self.opt.loss_weight_GAN
        # else:
        #     self.weightGAN = 0.0
        self.forward()  # compute fake images: G(A)
        self.optimizer_G.zero_grad()  # set G's gradients to zero
        self.backward_G()
        self.optimizer_G.step()  # udpate G's weights
        if self.opt.use_discriminator:
            self.optimizer_D.zero_grad()  # set D's gradients to zero
            self.backward_D()  # calculate gradients for D
            self.optimizer_D.step()  # update D's weights

        # check_unused_parameters(self.netG)

    def calculate_val_loss(self):
        if self.opt.use_discriminator:
            self.loss_relit_validation = self.loss_G_rec + self.loss_G_GAN + self.loss_G_GAN_Feat + self.loss_G_VGG
        else:
            self.loss_relit_validation = self.loss_G_rec + self.loss_G_VGG
        return self.loss_relit_validation


# def check_unused_parameters(model):
#     for name, param in model.named_parameters():
#         if param.grad is None:
#             print('Unused parameter:', name)


import torch.nn.functional as F

def build_gauss_kernel(size=5, sigma=1.0, n_channels=1, cuda=False):
    if size % 2 != 1:
        raise ValueError("kernel size must be uneven")
    grid = np.float32(np.mgrid[0:size, 0:size].T)

    def gaussian(x):
        return np.exp(
            (x - size // 2) ** 2 / (-2 * sigma ** 2)) ** 2

    kernel = np.sum(gaussian(grid), axis=2)
    kernel /= np.sum(kernel)
    # repeat same kernel across depth dimension
    kernel = np.tile(kernel, (n_channels, 1, 1))
    # conv weight should be (out_channels, groups/in_channels, h, w),
    # and since we have depth-separable conv we want the groups dimension to be 1
    kernel = torch.FloatTensor(kernel[:, None, :, :])
    if cuda:
        kernel = kernel.cuda()
    return Variable(kernel, requires_grad=False)


def conv_gauss(img, kernel):
    # conv img with a gaussian kernel that has been built with build_gauss_kernel
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def laplacian_pyramid(img, kernel, max_levels=5):
    current = img
    pyr = []

    for level in range(max_levels):
        filtered = conv_gauss(current, kernel)
        diff = current - filtered
        pyr.append(diff)
        current = F.avg_pool2d(filtered, 2)

    pyr.append(current)
    return pyr


# Lap_criterion = LapLoss(max_levels=5)
class LapLoss(nn.Module):
    def __init__(self, max_levels=5, k_size=5, sigma=2.0):
        super(LapLoss, self).__init__()
        self.max_levels = max_levels
        self.k_size = k_size
        self.sigma = sigma
        self._gauss_kernel = None
        self.L1_loss = nn.L1Loss()

    def forward(self, input, target):
        if self._gauss_kernel is None or self._gauss_kernel.shape[1] != input.shape[1]:
            self._gauss_kernel = build_gauss_kernel(size=self.k_size, sigma=self.sigma,
                                                    n_channels=input.shape[1], cuda=input.is_cuda)

        pyr_input = laplacian_pyramid(input, self._gauss_kernel, self.max_levels)
        pyr_target = laplacian_pyramid(target, self._gauss_kernel, self.max_levels)
        return sum(self.L1_loss(a, b) for a, b in zip(pyr_input, pyr_target))


