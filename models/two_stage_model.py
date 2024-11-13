import torch
from torch.cuda.amp import autocast
from .base_model import BaseModel
# from torchviz import make_dot
import os
from models.networks_intrinsic import define_net_intrinsic_decomposition
from models.networks_one_to_one_rep import define_net_one_to_one_new_light
from models.networks_discriminator import define_D, GANLoss
from models.networks import PanTiltLoss, L1_DSSIM_LPIPS
from models.networks import init_net
from models.networks import sobel_lab_chromaticity, scheduler_init_ref, sobel_opp_chromaticity, convert_rgb2opp
import torch.nn.functional as F
# from models.networks_custom_func import CustomClamp
from thop import profile, clever_format


class TwoStageModel(BaseModel):
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
        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        if self.isTrain:
            self.loss_names = ['relighted']
            if opt.two_stage:
                self.loss_names.extend(['reconstruct'])
            if opt.light_prediction:
                if opt.light_type == "pan_tilt_color":
                    self.loss_names.extend(['angular', 'color'])
                elif opt.light_type == "Spherical_harmonic":
                    self.loss_names.extend(['SHlight'])
                elif opt.light_type == "probes":
                    self.loss_names.extend(['probes_light'])
                else:
                    raise Exception("light_type is wrong!")
            if opt.constrain_intrinsic:
                self.loss_names.extend(['reflectance', 'shading_ori', 'shading_new'])
            if opt.cross_model:
                if opt.flag_ref_consistency:
                    self.loss_names.extend(['ref_consistency'])
                if opt.flag_sha_consistency:
                    self.loss_names.extend(['sha_consistency'])
            if opt.flag_sha_chromaticity_smooth:
                self.loss_names.extend(['sha_chromaticity_smooth'])
                if opt.loss_weight_sha_overall_smooth > 0.0:
                    self.loss_names.extend(['sha_overall_smooth'])
            if opt.flag_sha_ref_regression:
                if opt.method_sha_ref_regression == 'm1':
                    self.loss_names.extend(['sha_ref_regression_1'])
                    if opt.loss_weight_sha_ref_regression_2 > 0.0:
                        self.loss_names.extend(['sha_ref_regression_2'])
                elif opt.method_sha_ref_regression == 'm2':
                    para_reg = self.opt.para_sha_ref_regression
                    self.reg_filtered_keys = [key.replace('w_', '') for key in para_reg if
                                              key.startswith('w_') and para_reg[key] != 0]
                    self.loss_names.extend(['reg_' + x for x in self.reg_filtered_keys])
            if opt.flag_init_ref:
                self.loss_names.extend(['init_ref'])
            if opt.use_discriminator:
                self.loss_names.extend(['G_GAN', 'D_real', 'D_fake'])
        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['Image_input', 'Relighted_gt', 'Relighted_predict']
        if opt.two_stage:
            self.visual_names.extend(['Reconstruct', 'Reflectance_predict', 'Shading_ori_predict',
                                      'Shading_new_predict'])
        if opt.light_type == "probes":
            self.visual_names.extend(['probe_gt_ori', 'probe_gt_new'])
            if opt.light_prediction:
                self.visual_names.extend(['probe_predict'])
        if opt.constrain_intrinsic or opt.show_gt_intrinsic:
            self.visual_names.extend(['Reflectance_gt', 'Shading_ori_gt', 'Shading_new_gt'])
        if opt.cross_model:
            # if opt.flag_ref_consistency:
            self.visual_names.extend(['Ref_lower_half', 'Sha_ori_lower_half', 'Sha_new_lower_half'])
        if not opt.isTrain:
            if opt.light_type != "probes":
                self.visual_names.extend(['light_position_color_original', 'light_position_color_new'])
                if opt.light_prediction:
                    self.visual_names.extend(['light_position_color_predict'])
            if opt.special_test:
                self.visual_names = ['Image_input', 'Reconstruct', 'Reflectance_predict', 'Shading_ori_predict',
                                     'Shading_new_predict', 'Relighted_predict', 'light_position_color_predict',
                                     'light_position_color_new']
        self.visual_names.sort()
        # specify the models you want to save to the disk. The training/test scripts will call
        # <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and opt.use_discriminator:
            if opt.two_stage:
                self.model_names = ['G_1', 'G_2', 'D']
            else:
                self.model_names = ['G', 'D']
            self.optimizer_names = ['G', 'D']
        else:  # during test time, only load G
            if opt.two_stage:
                self.model_names = ['G_1', 'G_2']
            else:
                self.model_names = ['G']
            self.optimizer_names = ['G']

        # define networks (both generator and discriminator)
        if opt.two_stage:
            # Input: original image; output: ref and sha.
            self.netG_1 = define_net_intrinsic_decomposition(opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
                                                             not opt.no_dropout, opt.net_intrinsic)
            # init the model, send to GPUs.
            self.netG_1 = init_net(self.netG_1, opt.init_type, opt.init_gain, self.gpu_ids, opt.parallel_method)
            # Input: ref and sha; output: new shading
            nc_G_2 = opt.output_nc * 2 if opt.introduce_ref_G_2 else opt.output_nc
            self.netG_2 = define_net_one_to_one_new_light(nc_G_2, opt.output_nc, opt.ngf, norm=opt.norm,
                                                          use_dropout=not opt.no_dropout, light_type=opt.light_type,
                                                          light_prediction=opt.light_prediction, netG=opt.netG)
            self.netG_2 = init_net(self.netG_2, opt.init_type, opt.init_gain, self.gpu_ids, opt.parallel_method)
        else:
            self.netG = define_net_one_to_one_new_light(opt.input_nc, opt.output_nc, opt.ngf, norm=opt.norm,
                                                        use_dropout=not opt.no_dropout, light_type=opt.light_type,
                                                        light_prediction=opt.light_prediction, netG=opt.netG)
            self.netG = init_net(self.netG, opt.init_type, opt.init_gain, self.gpu_ids, opt.parallel_method)
        # define a discriminator; conditional GANs need to take both input and output images;
        # Therefore, channels for D is input_nc + output_nc
        if self.isTrain and opt.use_discriminator:
            self.embedded_light_D = opt.netD == 'embedded_light'
            self.netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                 opt.n_layers_D, opt.norm)
            self.netD = init_net(self.netD, opt.init_type, opt.init_gain, self.gpu_ids, opt.parallel_method)

        # for plotting model
        self.expr_dir = os.path.join(opt.checkpoints_dir, opt.name)

        if self.isTrain:
            # define loss functions
            if opt.main_loss_function == 'L1_DSSIM_LPIPS':
                if opt.cross_model:
                    main_loss_function = L1_DSSIM_LPIPS(opt.gpu_ids, flag_L1_DSSIM_LPIPS=opt.flag_L1_DSSIM_LPIPS,
                                                        use_amp=opt.use_amp, divide_half=opt.unbalanced,
                                                        divide_para=opt.unbalanced_para)
                else:
                    main_loss_function = L1_DSSIM_LPIPS(opt.gpu_ids, flag_L1_DSSIM_LPIPS=opt.flag_L1_DSSIM_LPIPS,
                                                        use_amp=opt.use_amp)
            else:
                raise Exception('main_loss_function error')
            self.main_loss_function = main_loss_function

            if opt.light_prediction:
                if opt.light_type == "pan_tilt_color":
                    self.criterionAngular = PanTiltLoss()
                    self.weight_angular = opt.loss_weight_angular
                    self.criterionColor = torch.nn.L1Loss()
                    self.weight_color = opt.loss_weight_color
                elif opt.light_type == "Spherical_harmonic":
                    self.criterionSHlight = torch.nn.L1Loss()
                    self.weight_SHlight = opt.loss_weight_SHlight
                elif opt.light_type == "probes":
                    self.criterionProbes = torch.nn.L1Loss()
                    self.weight_probes_light = opt.loss_weight_probes_light
                else:
                    raise Exception("light_type is wrong!")

            if opt.constrain_intrinsic:
                self.criterionReflectance = main_loss_function
                self.weight_reflectance = opt.loss_weight_reflectance
                self.criterionShading_ori = main_loss_function
                self.weight_shading_ori = opt.loss_weight_shading_ori
                self.criterionShading_new = main_loss_function
                self.weight_shading_new = opt.loss_weight_shading_new
            if opt.cross_model:
                if opt.flag_ref_consistency:
                    # self.criterionRefCons = main_loss_function
                    self.criterionRefCons = torch.nn.L1Loss()
                    self.weight_ref_consistency = opt.loss_weight_ref_consistency
                if opt.flag_sha_consistency:
                    self.criterionShaCons = torch.nn.L1Loss()
                    self.weight_sha_consistency = opt.loss_weight_sha_consistency

            self.criterionRelighted = main_loss_function
            self.weight_relighted = opt.loss_weight_relighted
            if self.opt.two_stage:
                self.criterionReconstruct = main_loss_function
                self.weight_reconstruct = opt.loss_weight_reconstruct
            if opt.flag_sha_chromaticity_smooth:
                self.weight_sha_chromaticity_smooth = opt.loss_weight_sha_chromaticity_smooth
                if opt.loss_weight_sha_overall_smooth > 0.0:
                    self.weight_sha_overall_smooth = opt.loss_weight_sha_overall_smooth
            if opt.flag_sha_ref_regression:
                if opt.method_sha_ref_regression == 'm1':
                    self.weight_sha_ref_regression_1 = opt.loss_weight_sha_ref_regression_1
                    if opt.loss_weight_sha_ref_regression_2 > 0.0:
                        self.weight_sha_ref_regression_2 = opt.loss_weight_sha_ref_regression_2
                elif opt.method_sha_ref_regression == 'm2':
                    for x in self.reg_filtered_keys:
                        setattr(self, 'weight_reg_' + x, self.opt.para_sha_ref_regression['w_'+x])

            if opt.flag_init_ref:
                if self.opt.para_init_ref['method'] == "OPP":
                    self.criterion_init_ref = torch.nn.L1Loss()
                elif self.opt.para_init_ref['method'] == "ORI":
                    # self.criterion_init_ref = main_loss_function
                    self.criterion_init_ref = torch.nn.L1Loss()
                else:
                    raise Exception("Error opt.para_init_ref['method']")
                self.weight_init_ref = opt.loss_weight_init_ref

            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if opt.two_stage:
                self.optimizer_G = torch.optim.Adam(list(self.netG_1.parameters()) + list(self.netG_2.parameters()),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            if self.opt.use_discriminator:
                # for GAN
                self.criterionGAN = GANLoss(opt.gan_mode).to(self.device)
                self.weightGAN = opt.loss_weight_GAN
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr_d, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            # for amp
            if opt.use_amp:
                self.scaler = torch.cuda.amp.GradScaler()

        # if opt.infinite_range_sha:
        #     self.custom_clamp = CustomClamp(min_value=0.0, max_value=1.0)

    # def plot_model(self):
    #     # generator
    #     x_image = torch.rand(12, 3, 256, 256).to(self.device)
    #     l_channel = 9 if self.opt.light_type == "Spherical_harmonic" else 7
    #     x_light = torch.rand(12, l_channel).to(self.device)
    #     ref, sha_ori = self.netG_1(x_image)
    #     sha_new = self.netG_2(torch.cat([ref, sha_ori], 1).detach(), x_light)
    #     g1 = make_dot((ref, sha_ori), params=dict(self.netG_1.named_parameters()))
    #     g1.render(filename='netG_1', directory=self.expr_dir, view=False)
    #     g2 = make_dot(sha_new, params=dict(self.netG_2.named_parameters()))
    #     g2.render(filename='netG_2', directory=self.expr_dir, view=False)
    #     # if self.opt.use_discriminator:
    #     #     # discriminator
    #     #     x_D = torch.cat((x1, y[1]), 1)
    #     #     y_D = self.netD(x_D.detach())
    #     #     g = make_dot(y_D, params=dict(self.netD.named_parameters()))
    #     #     g.render(filename='netD', directory=self.expr_dir, view=False)
    def get_macs(self):
        if self.opt.two_stage:
            x_image = torch.rand(1, 3, 256, 256).to(self.device)
            # Count the number of FLOPs
            macs, params = profile(self.netG_1, inputs=(x_image, ))
            macs, params = clever_format([macs, params], "%.3f")
            print("macs, params", macs, params)
            x_image2 = torch.rand(1, 3, 256, 256).to(self.device)
            l_channel = 9 if self.opt.light_type == "Spherical_harmonic" else 7
            x_light = torch.rand(1, l_channel).to(self.device)
            macs, params = profile(self.netG_2, inputs=(x_image2, x_light))
            macs, params = clever_format([macs, params], "%.3f")
            print("macs, params", macs, params)
        else:
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
        if self.isTrain and self.opt.cross_model:
            key = 'scene_label'
            self.half_length = self.input[key].__len__()
            r_copy = ['r_'+x for x in self.input[key]]
            self.input[key].extend(r_copy)
            repeat_list = ['Reflectance_output']
            cross_list = [
                ['light_position_color_original', 'light_position_color_new'],
                ['Shading_ori', 'Shading_output'],
                ['Image_input', 'Image_relighted'],
            ]
            for label in repeat_list:
                if label in self.input.keys():
                    duplication = self.input[label].clone()
                    self.input[label] = torch.cat([self.input[label], duplication], 0)
            for label1, label2 in cross_list:
                if label1 in self.input.keys() and label2 in self.input.keys():
                    version1 = torch.cat([self.input[label1], self.input[label2]], 0)
                    version2 = torch.cat([self.input[label2], self.input[label1]], 0)
                    self.input[label1] = version1
                    self.input[label2] = version2

        self.image_paths = input['scene_label']


    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # from tensorboardX import SummaryWriter
        # with SummaryWriter(comment='structure') as w:
        #     w.add_graph(self.netG.module, (self.input['Image_input'], self.input['light_position_color_new'],))
        self.predict = {}

        if self.opt.two_stage:
            self.predict['Reflectance_predict'], self.predict['Shading_ori_predict'], \
                = self.netG_1(self.input['Image_input'])
            # calculate edge attention mask after intrinsic decomposition
            if self.opt.light_prediction:
                if self.opt.introduce_ref_G_2:
                    self.predict['light_position_color_predict'], self.predict['Shading_new_predict'], \
                        = self.netG_2(torch.cat([self.predict['Reflectance_predict'], self.predict['Shading_ori_predict']], 1),
                                      self.input['light_position_color_new'])
                else:
                    self.predict['light_position_color_predict'], self.predict['Shading_new_predict'], \
                        = self.netG_2(self.predict['Shading_ori_predict'], self.input['light_position_color_new'])
            else:
                if self.opt.introduce_ref_G_2:
                    self.predict['Shading_new_predict'] \
                        = self.netG_2(torch.cat([self.predict['Reflectance_predict'], self.predict['Shading_ori_predict']], 1),
                                      self.input['light_position_color_new'])
                else:
                    self.predict['Shading_new_predict'] \
                        = self.netG_2(self.predict['Shading_ori_predict'], self.input['light_position_color_new'])

            self.predict['Relighted_predict'] = torch.mul(self.predict['Reflectance_predict'],
                                                          self.predict['Shading_new_predict'])
            self.predict['Reconstruct'] = torch.mul(self.predict['Reflectance_predict'],
                                                    self.predict['Shading_ori_predict'])
        else:
            if self.opt.light_prediction:
                self.predict['light_position_color_predict'], self.predict['Relighted_predict'], \
                    = self.netG(self.input['Image_input'], self.input['light_position_color_new'])
            else:
                self.predict['Relighted_predict'] \
                    = self.netG(self.input['Image_input'], self.input['light_position_color_new'])

        # if self.opt.infinite_range_sha:
        #     for key in ['Relighted_predict', 'Reconstruct', 'Reflectance_predict', 'Shading_ori_predict',
        #                 'Shading_new_predict']:
        #         self.predict[key] = self.custom_clamp(self.predict[key])

        # for visdom
        if self.opt.cross_model:
            # if self.opt.flag_ref_consistency:
            self.Ref_lower_half = self.predict['Reflectance_predict'][self.half_length:]
            self.Sha_ori_lower_half = self.predict['Shading_ori_predict'][self.half_length:]
            self.Sha_new_lower_half = self.predict['Shading_new_predict'][self.half_length:]
        self.Image_input = self.input['Image_input']
        if self.isTrain or not self.opt.special_test:
            self.Relighted_gt = self.input['Image_relighted']
            if self.opt.constrain_intrinsic or self.opt.show_gt_intrinsic:
                self.Reflectance_gt = self.input['Reflectance_output']
                self.Shading_ori_gt = self.input['Shading_ori']
                self.Shading_new_gt = self.input['Shading_output']
        if self.opt.two_stage:
            self.Reflectance_predict = self.predict['Reflectance_predict']
            self.Shading_ori_predict = self.predict['Shading_ori_predict']
            self.Shading_new_predict = self.predict['Shading_new_predict']
            self.Reconstruct = self.predict['Reconstruct']
        self.Relighted_predict = self.predict['Relighted_predict']

        if self.isTrain or not self.opt.special_test:
            self.light_position_color_original = self.input['light_position_color_original']
        if self.opt.light_prediction:
            self.light_position_color_predict = self.predict['light_position_color_predict']
        self.light_position_color_new = self.input['light_position_color_new']

        if self.opt.light_type == "probes":
            def probe_reshape(p):
                return torch.cat([p[:, :3, :, :], p[:, 3:, :, :]], dim=-1)
            self.probe_gt_ori = probe_reshape(self.input['light_position_color_original'])
            self.probe_gt_new = probe_reshape(self.input['light_position_color_new'])
            if self.opt.light_prediction:
                self.probe_predict = probe_reshape(self.predict['light_position_color_predict'])

        # for discriminator
        if self.isTrain and self.opt.use_discriminator and self.embedded_light_D and not self.opt.special_test:
            self.light_condition_ori_new = torch.cat((self.light_position_color_original,
                                                      self.light_position_color_new), 1)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # A to B, state what is real and fake.
        self.real_A = self.Image_input
        self.fake_B = self.Relighted_predict
        self.real_B = self.Relighted_gt
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B),
                            1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        if self.embedded_light_D:
            pred_fake = self.netD(fake_AB.detach(), self.light_condition_ori_new)
        else:
            pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False) * self.weightGAN
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        if self.embedded_light_D:
            pred_real = self.netD(real_AB, self.light_condition_ori_new)
        else:
            pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True) * self.weightGAN
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

    def backward_G(self):
        """Calculate all loss for the Unet generator"""
        if self.opt.use_discriminator:
            # First, G(A) should fake the discriminator
            fake_AB = torch.cat((self.real_A, self.fake_B), 1)
            if self.embedded_light_D:
                pred_fake = self.netD(fake_AB, self.light_condition_ori_new)
            else:
                pred_fake = self.netD(fake_AB)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True) * self.weightGAN
        # Second, G(A) = B
        if self.opt.light_prediction:
            if self.opt.light_type == "pan_tilt_color":
                self.loss_angular = self.criterionAngular(self.light_position_color_predict[:, :4],
                                                          self.light_position_color_original[:, :4])
                self.loss_color = self.criterionColor(self.light_position_color_predict[:, 4:],
                                                      self.light_position_color_original[:, 4:])
            elif self.opt.light_type == "Spherical_harmonic":
                self.loss_SHlight = self.criterionSHlight(self.light_position_color_predict,
                                                          self.light_position_color_original)
            elif self.opt.light_type == "probes":
                self.loss_probes_light = self.criterionProbes(self.probe_predict, self.probe_gt_ori)
            else:
                raise Exception("light_type is wrong!")

        # calculate the loss for images.
        if self.opt.constrain_intrinsic:
            self.loss_reflectance = self.criterionReflectance(self.Reflectance_predict,
                                                              self.Reflectance_gt)
            self.loss_shading_ori = self.criterionShading_ori(self.Shading_ori_predict,
                                                              self.Shading_ori_gt)
            self.loss_shading_new = self.criterionShading_new(self.Shading_new_predict,
                                                              self.Shading_new_gt)
        self.loss_relighted = self.criterionRelighted(self.Relighted_predict,
                                                      self.Relighted_gt)
        if self.opt.two_stage:
            self.loss_reconstruct = self.criterionReconstruct(self.Reconstruct,
                                                              self.Image_input)
        if self.opt.cross_model:
            if self.opt.flag_ref_consistency:
                self.loss_ref_consistency = \
                    self.criterionRefCons(self.Reflectance_predict[:self.half_length],
                                          self.Reflectance_predict[self.half_length:])
            if self.opt.flag_sha_consistency:
                self.loss_sha_consistency = \
                    self.criterionShaCons(self.Shading_ori_predict[:self.half_length],
                                          self.Shading_new_predict[self.half_length:]) \
                    + self.criterionShaCons(self.Shading_new_predict[:self.half_length],
                                            self.Shading_ori_predict[self.half_length:])

        if self.opt.flag_sha_chromaticity_smooth:
            if self.opt.method_sha_chromaticity_smooth == "OPP":
                if not self.opt.loss_weight_sha_overall_smooth > 0.0:
                    self.loss_sha_chromaticity_smooth = sobel_opp_chromaticity(self.Shading_ori_predict)
                else:
                    self.loss_sha_chromaticity_smooth, self.loss_sha_overall_smooth = \
                        sobel_opp_chromaticity(self.Shading_ori_predict, obtain_overall=True)
            elif self.opt.method_sha_chromaticity_smooth == "LAB":
                if not self.opt.loss_weight_sha_overall_smooth > 0.0:
                    self.loss_sha_chromaticity_smooth = sobel_lab_chromaticity(self.Shading_ori_predict)
                else:
                    self.loss_sha_chromaticity_smooth, self.loss_sha_overall_smooth = \
                        sobel_lab_chromaticity(self.Shading_ori_predict, obtain_overall=True)
            else:
                raise Exception("Error: method_sha_chromaticity_smooth")

        if self.opt.flag_sha_ref_regression:
            if self.opt.method_sha_ref_regression == 'm1':
                flag_obtain_overall = self.opt.loss_weight_sha_ref_regression_2 > 0.0
                sha_chromaticity_gradient, sha_overall_gradient = sobel_opp_chromaticity(self.Shading_ori_predict,
                                                                                         obtain_overall=flag_obtain_overall)
                ref_chromaticity_gradient, ref_overall_gradient = sobel_opp_chromaticity(self.Reflectance_predict,
                                                                                         obtain_overall=flag_obtain_overall)
                eposilon = torch.tensor(1e-8).to(self.device)
                self.loss_sha_ref_regression_1 = torch.square(
                    sha_chromaticity_gradient / torch.maximum(ref_chromaticity_gradient, eposilon) -
                    self.opt.sha_ref_regression_mean[0])
                if self.opt.loss_weight_sha_ref_regression_2 > 0.0:
                    self.loss_sha_ref_regression_2 = torch.square(
                        sha_overall_gradient / torch.maximum(ref_overall_gradient, eposilon) -
                        self.opt.sha_ref_regression_mean[1])
            elif self.opt.method_sha_ref_regression == 'm2':
                flag_obtain_overall = True
                reg_element = []
                for key in self.reg_filtered_keys:
                    reg_element.extend(key.split('_')[:2])
                reg_element = list(set(reg_element))
                reg_element_gradient = {}
                if 'S' in reg_element:
                    reg_element_gradient['S'] = \
                        sobel_opp_chromaticity(self.Shading_ori_predict, obtain_overall=flag_obtain_overall)
                if 'R' in reg_element:
                    reg_element_gradient['R'] = \
                        sobel_opp_chromaticity(self.Reflectance_predict, obtain_overall=flag_obtain_overall)
                if 'I' in reg_element:
                    reg_element_gradient['I'] = \
                        sobel_opp_chromaticity(self.Image_input, obtain_overall=flag_obtain_overall)
                eposilon = torch.tensor(1e-8).to(self.device)
                para_reg = self.opt.para_sha_ref_regression
                elu_loss = lambda x_elu: para_reg['elu_alpha'] + F.elu(x_elu + para_reg['elu_shift'],
                                                                       alpha=para_reg['elu_alpha'])
                for key in self.reg_filtered_keys:
                    key_split = key.split('_')
                    if key_split[2] == 'c':
                        idx = 0
                    elif key_split[2] == 'a':
                        idx = 1
                    else:
                        raise Exception("Should not happen")
                    element1 = reg_element_gradient[key_split[0]][idx]
                    element2 = reg_element_gradient[key_split[1]][idx]
                    result_divide = element1 / torch.maximum(element2, eposilon)
                    if key in ['R_I_c', 'R_I_a']:
                        value = elu_loss(para_reg[key] - result_divide)
                        setattr(self, 'loss_reg_' + key, value)
                    if key in ['S_I_c', 'S_I_a', 'S_R_c', 'S_R_a']:
                        value = elu_loss(-(para_reg[key] - result_divide))
                        setattr(self, 'loss_reg_' + key, value)

        if self.opt.flag_init_ref:
            if self.opt.para_init_ref['cross_ij']:
                if not self.opt.cross_model:
                    raise Exception("No cross input!")
                # reverse the input
                input_compare = torch.cat([self.Image_input[self.half_length:],
                                           self.Image_input[:self.half_length]], 0)
            else:
                input_compare = self.Image_input
            temporal_weight = scheduler_init_ref(self.epoch_count, self.opt.para_init_ref)
            if self.opt.para_init_ref['method'] == "OPP":
                raise Exception("not use")
                # c_ref = convert_rgb2opp(self.Reflectance_predict)[:, 1:3, :, :]
                # c_input = convert_rgb2opp(input_compare)[:, 1:3, :, :]
            elif self.opt.para_init_ref['method'] == "ORI":
                c_ref = self.Reflectance_predict
                c_input = input_compare
            else:
                raise Exception("Error opt.para_init_ref['method']")
            self.loss_init_ref = temporal_weight * self.criterion_init_ref(c_ref, c_input)

        # combine loss and calculate gradients
        self.loss_weighted_total = torch.tensor(0).to(self.device)
        for name in self.loss_names:
            if isinstance(name, str) and name not in ['G_GAN', 'D_real', 'D_fake']:
                setattr(self, 'loss_' + name, getattr(self, 'loss_' + name) * getattr(self, 'weight_' + name))
                self.loss_weighted_total = self.loss_weighted_total + getattr(self, 'loss_' + name)

        if self.opt.use_discriminator:
            self.loss_weighted_total = self.loss_weighted_total + self.loss_G_GAN

    def optimize_parameters(self, epoch=0, iter=0):
        self.epoch_count = epoch
        if self.opt.use_amp:
            with autocast():
                self.forward()  # compute fake images: G(A)
            if self.opt.use_discriminator:
                # update D
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                with autocast():
                    self.backward_D()  # calculate gradients for D
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.scaler.scale(self.loss_D).backward()
                self.scaler.step(self.optimizer_D)
                # update G
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            with autocast():
                self.backward_G()  # calculate graidents for G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.scaler.scale(self.loss_weighted_total).backward()
            self.scaler.step(self.optimizer_G)
            # Don't forget
            self.scaler.update()
        else:
            self.forward()  # compute fake images: G(A)
            if self.opt.use_discriminator:
                # update D
                self.set_requires_grad(self.netD, True)  # enable backprop for D
                self.optimizer_D.zero_grad()  # set D's gradients to zero
                self.backward_D()  # calculate gradients for D
                self.loss_D.backward()
                self.optimizer_D.step()  # update D's weights
                # update G
                self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
            self.optimizer_G.zero_grad()  # set G's gradients to zero
            self.backward_G()  # calculate graidents for G
            self.loss_weighted_total.backward()
            self.optimizer_G.step()  # udpate G's weights

    def calculate_val_loss(self):
        self.loss_relit_validation = self.weight_relighted * self.criterionRelighted(self.Relighted_predict,
                                                                                     self.Relighted_gt)
        return self.loss_relit_validation

    def compute_visuals(self):
        """
        Calculate images for metrics, visdom and HTML visualization.
        Since the prediction of tensors may exceed 1.0, we need to clamp them to [0.0, 1.0].
        """
        if self.opt.infinite_range_sha:
            for name in self.visual_names:
                if isinstance(name, str) and name in ['Relighted_predict', 'Reconstruct']:
                    setattr(self, name, torch.clamp(getattr(self, name), min=0.0, max=1.0))
                elif isinstance(name, str) and name in ['Shading_ori_predict', 'Shading_new_predict',
                                                        'Sha_ori_lower_half', 'Sha_new_lower_half']:
                    # setattr(self, name, getattr(self, name) / getattr(self, name).max())
                    setattr(self, name, torch.clamp(getattr(self, name), min=0.0, max=1.0))
                elif isinstance(name, str) and name in ['Reflectance_predict', 'Ref_lower_half']:
                    if getattr(self, name).max() > 1.0:
                        raise Exception("Range error")
                else:
                    pass
        pass
