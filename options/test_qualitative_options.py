from .base_options import BaseOptions
import argparse

class TestOptions(BaseOptions):
    def __init__(self):
        super(TestOptions, self).__init__()

        # get para
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('--name', type=str)
        parser_value = parser.parse_args()
        self.name = parser_value.name

        # self.name = 'exp_isr'  # name of the experiment. It decides where to store samples and models
        # self.name = 'exp_rsr_ours_f'  # name of the experiment. It decides where to store samples and models
        # self.name = 'exp_vidit_ours_f'
        # self.name = 'exp_multilum_ours_f'

        self.special_test = False
        using_dataset = self.name.split('_')[1]
        if using_dataset == "isr":
            self.dataset_mode = 'relighting_single_image_test'  # name of the dataset
            self.anno = 'data/anno_ISR/test_qualitative_pairs.txt'  # the anno file from prepare_dataset.py
            self.preprocess = 'none'
            self.show_gt_intrinsic = True
            self.light_type = "pan_tilt_color"
        elif using_dataset == "rsr":
            self.dataset_mode = 'relighting_single_image_rsr'  # name of the dataset
            self.anno = 'data/anno_RSR/AnyLighttest_pairs_qualitative.txt'  # the anno file from prepare_dataset.py
            self.preprocess = 'none'
            self.show_gt_intrinsic = False
            self.light_type = "pan_tilt_color"
        elif using_dataset == "vidit":
            self.dataset_mode = 'relighting_single_image_vidit'  # name of the dataset
            self.anno = 'data/anno_VIDIT/any2any/AnyLight_test_pairs_qualitative.txt'  # the anno file from prepare_dataset.py
            self.preprocess = 'resize'
            self.show_gt_intrinsic = False
            self.light_type = "pan_tilt_color"
        elif using_dataset == "multilum":
            self.dataset_mode = 'relighting_single_image_multilum'
            self.dataroot_multilum = self.server_root + 'Multi_Illumination_small/test/'
            self.anno = 'data/multi_illumination/test_qualitative.txt'
            self.preprocess = 'none'
            self.show_gt_intrinsic = False
            self.light_type = "probes"
        elif using_dataset == "special":
            self.dataset_mode = 'relighting_single_image_special_test'  # name of the dataset
            self.preprocess = 'resize'
            self.show_gt_intrinsic = False
            self.light_type = "pan_tilt_color"
            self.special_test = True  # special test for pictures from other datasets.

        self.phase = 'test_' + using_dataset  # str, default='test', help='train, val, test, etc')
        #####
        if len(self.name.split('_')) > 2:
            using_model = self.name.split('_')[2]
        else:
            using_model = "ours"
        if using_model == "ours":
            self.model_name = 'relighting_two_stage'  # ['relighting_two_stage' | 'relighting_one_decoder']
            self.two_stage = True
            if using_dataset == "multilum":
                self.light_prediction = False
            else:
                self.light_prediction = True
            if self.two_stage and self.show_gt_intrinsic:
                self.metric_list = ['Relighted', 'Reflectance', 'Shading_ori', 'Shading_new',
                                    'Reconstruct']
            else:
                self.metric_list = ['Relighted']
            if self.light_prediction:
                self.metric_list.append('light_position_color')
            self.netG = "resnet9_nonlocal"
            self.net_intrinsic = "resnet9"
            # self.netG = "unet"
            # self.net_intrinsic = "unet"
            self.infinite_range_sha = True
            if self.infinite_range_sha:
                self.net_intrinsic = self.net_intrinsic + "_InfRange"
                self.netG = self.netG + "_InfRange"
            self.introduce_ref_G_2 = False
            self.no_dropout = False  # old option: no dropout for the model
            self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]
        elif using_model == "pix2pix":
            self.model_name = 'relighting_two_stage'  # ['relighting' | 'intrinsic_decomposition']
            self.two_stage = False
            if using_dataset == "multilum":
                self.light_prediction = False
            else:
                self.light_prediction = True
            self.no_dropout = False  # old option: no dropout for the model
            self.norm = 'batch'  # instance normalization or batch normalization [instance | batch | none]
            self.netG = "unet"
            self.infinite_range_sha = False
            self.metric_list = ['Relighted']
            if self.light_prediction:
                self.metric_list.append('light_position_color')
        elif using_model == "drn":
            self.model_name = 'drn'  # ['relighting' | 'intrinsic_decomposition']
            self.light_prediction = False
            self.netG = 'global'
            self.n_downsample_global = 3
            self.n_blocks_global = 9
            self.n_local_enhancers = 1
            self.n_blocks_local = 3
            self.norm = 'instance'
            self.n_layers_D = 1
            self.use_sigmoid = False
            self.num_D = 1
            self.no_ganFeat_loss = False
            self.no_lsgan = False
            self.pool_size = 0
            self.ndf = 4
            self.lambda_feat = 10.0
            self.metric_list = ['Relighted']
        elif using_model == "ian":
            self.model_name = 'IAN'
            self.metric_list = ['Relighted']
        else:
            raise Exception("Using_model not exist. ")

        # train parameter
        self.batch_size = 1  # batch size # 6
        # select which model to load, set continue_train = True to load the weight
        self.continue_train = True  # continue training: load the latest model
        self.epoch = 'save_best'  # default='latest', which epoch to load? set to latest to use latest cached model
        self.load_iter = 0  # default='0', which iteration to load?

        self.parallel_method = "DataParallel"
        self.use_amp = False
        self.use_discriminator = False
        self.cross_model = False

        self.model_modify_layer = []
        self.modify_layer = len(self.model_modify_layer) != 0
        self.constrain_intrinsic = False

        self.aspect_ratio = 1.0  # float, default=1.0, help='aspect ratio of result images')
        self.results_dir = './results/'  # str, default='./results/', help='saves results here.')
        self.isTrain = False
        self.gpu_ids = [0]

        # Dropout and Batchnorm has different behavioir during training and test.
        self.eval = True   # use eval mode during test time.
        self.num_test = 130   # how many test images to run
        # dataloader
        self.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        self.num_threads = 1   # test code only supports num_threads = 1
        # data augmentation
        self.crop_size = 256  # then crop to this size
        self.load_size = self.crop_size  # scale images to this size
        self.no_flip = True  # if specified, do not flip the images for data augmentation

        self.display_id = -1  # no visdom display; the test code saves the results to a HTML file.

