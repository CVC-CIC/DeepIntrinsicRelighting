from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self, which_experiment):
        super(TrainOptions, self).__init__()

        self.which_experiment = which_experiment
        self.isTrain = True

        # Setting for dataset
        self.dataset_mode = 'relighting_single_image_rsr'  # name of the dataset
        self.dataset_rsr_type = "AnyLight"
        self.anno = 'data/anno_RSR/train.txt'  # the anno file from prepare_dataset.py
        self.anno_validation = 'data/anno_RSR/AnyLightval_pairs.txt'
        # Setting for GPU
        self.parallel_method = "DistributedDataParallel"  # "DataParallel" "DistributedDataParallel"
        if self.parallel_method == "DataParallel":
            number_gpus = 1
            self.gpu_ids = [i for i in range(number_gpus)]
        elif self.parallel_method == "DistributedDataParallel":
            self.world_size = None
            self.gpu_ids = None
        self.use_amp = False
        # parameters for batch
        self.batch_size = 6

        # Setting for the optimizer
        self.lr_policy = 'step'   #   learning rate policy. [linear | step | plateau | cosine]
        self.lr = 0.0001  # initial learning rate for adam
        self.lr_d = self.lr
        self.lr_decay_ratio = 0.5  # decay ratio in step scheduler.
        self.n_epochs = 150   #100 number of epochs with the initial learning rate
        self.n_epochs_decay = 0   #100 when using 'linear', number of epochs to linearly decay learning rate to zero
        self.lr_decay_iters = 100  # when using 'step', multiply by a gamma every lr_decay_iters iterations
        self.optimizer_type = 'Adam'   # 'Adam', 'SGD'
        self.beta1 = 0.5  # momentum term of adam
        self.adam_eps = 1e-8

        # Setting for continuing the training.
        self.continue_train = True  # continue training: load the latest model
        self.epoch = 'base_isr'  # default='latest', which epoch to load? set to latest to use latest cached model
        self.load_iter = 0  # default='0', which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]
        if self.continue_train:
            try:
                self.epoch_count = int(self.epoch)
            except:
                self.epoch_count = 1  # the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        else:
            self.epoch_count = 1

        self.model_modify_layer = []
        self.modify_layer = len(self.model_modify_layer) != 0

        # Setting for the model
        self.name = self.which_experiment  # name of the experiment. It decides where to store samples and models

        self.model_name = 'relighting_two_stage'  # ['relighting' | 'intrinsic_decomposition']
        self.two_stage = True
        self.no_dropout = False  # old option: no dropout for the model
        self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]
        self.light_type = "pan_tilt_color"  # ["pan_tilt_color" | "Spherical_harmonic"]
        self.light_prediction = True
        self.netG = "resnet9_nonlocal"
        self.net_intrinsic = "resnet9"
        self.introduce_ref_G_2 = False
        self.cross_model = True
        # range of sha
        self.infinite_range_sha = True
        if self.infinite_range_sha:
            self.net_intrinsic = self.net_intrinsic + "_InfRange"
            self.netG = self.netG + "_InfRange"
        # reflectance consistency
        self.flag_ref_consistency = True
        self.loss_weight_ref_consistency = 1.0
        self.flag_sha_consistency = True
        self.loss_weight_sha_consistency = 1.0
        # Regularizing chromaticity
        self.flag_sha_chromaticity_smooth = False
        # self.method_sha_chromaticity_smooth = "OPP"  # "OPP", "LAB"
        # self.loss_weight_sha_chromaticity_smooth = 75.0
        # self.loss_weight_sha_overall_smooth = 0.5
        self.flag_sha_ref_regression = True
        # self.method_sha_ref_regression = 'm1'
        # self.loss_weight_sha_ref_regression_1 = 1.0  # for chromaticity
        # self.loss_weight_sha_ref_regression_2 = 0.1  # for all channels
        # self.sha_ref_regression_mean = [0.43, 0.61]
        self.method_sha_ref_regression = 'm2'
        self.para_sha_ref_regression = {
            # 'R_I_c': 1.2119,
            # 'R_I_a': 1.1603,
            'S_I_c': 0.5254,
            'S_I_a': 0.7089,
            # 'S_R_c': 0.4336,
            # 'S_R_a': 0.6109,
            'elu_alpha': 0.1,
            'elu_shift': 0.0,
            # 'w_R_I_c': 0.0,
            # 'w_R_I_a': 0.0,
            'w_S_I_c': 2.0,
            'w_S_I_a': 0.1,
            # 'w_S_R_c': 0.0,
            # 'w_S_R_a': 0.0,
        }
        # Regularizing init_ref
        self.flag_init_ref = True
        self.para_init_ref = {
            'cross_ij': True,
            'decay': True,
            'method': "ORI"  # "OPP", "ORI"
        }
        self.loss_weight_init_ref = 1.0

        # # discriminator
        self.use_discriminator = True
        # self.epoch_start_train_discriminator = -1
        self.netD = 'n_layers'  # specify discriminator architecture [basic | n_layers | pixel].
        # The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator'
        self.n_layers_D = 3  # only used if netD==n_layers
        self.gan_mode = 'lsgan'  # 'the type of GAN objective. [vanilla| lsgan | wgangp].
        # vanilla GAN loss is the cross-entropy objective used in the original GAN paper.'
        self.ndf = 64  # number of discrim filters in the first conv layer
        self.loss_weight_GAN = 0.05

        # parameters for loss functions
        self.constrain_intrinsic = False
        self.show_gt_intrinsic = False
        self.main_loss_function = 'L1_DSSIM_LPIPS'   # choose using L2 or L1 during the training
        self.flag_L1_DSSIM_LPIPS = [True, True, True]
        if self.cross_model:
            self.unbalanced = False
            self.unbalanced_para = None
        # Weights of losses
        self.loss_weight_angular = 1.0
        self.loss_weight_color = 1.0
        self.loss_weight_reflectance = 1.0
        self.loss_weight_shading_ori = 1.0
        self.loss_weight_reconstruct = 1.0
        self.loss_weight_shading_new = 1.0
        self.loss_weight_relighted = 5.0

        # data augmentation
        self.preprocess = 'none'   # 'resize_and_crop'   # scaling and cropping of images at load time
        # [resize_and_crop | crop | scale_width | scale_width_and_crop | none]
        self.load_size = 256   # scale images to this size
        self.crop_size = 256   # then crop to this size
        self.no_flip = True   # if specified, do not flip the images for data augmentation

        # dataloader
        self.serial_batches = False   # if true, takes images in order to make batches, otherwise takes them randomly
        self.num_threads = 10   # threads for loading data

        # save model and output images
        self.save_epoch_freq = 50  # frequency of saving checkpoints at the end of epochs
        self.save_latest = False
        self.save_optimizer = True
        self.load_optimizer = False
        self.load_scaler = False

        # visdom and HTML visualization parameters
        self.display_env = self.name
        self.save_and_show_by_epoch = True
        self.display_freq = 4000  # frequency of showing training results on screen')
        self.display_ncols = 5  # if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.display_id = 1  # window id of the web display')
        self.display_server = "http://localhost"  # visdom server of the web display')

        self.display_port = 8097  # visdom port of the web display')
        self.update_html_freq = 4000  # frequency of saving training results to html')
        self.print_freq = 4000  # frequency of showing training results on console')
        self.no_html = False  # do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

