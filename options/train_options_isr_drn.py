from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def __init__(self, which_experiment):
        super(TrainOptions, self).__init__()

        self.which_experiment = which_experiment
        self.isTrain = True

        # Setting for dataset
        self.dataset_mode = 'relighting_single_image'  # name of the dataset
        self.anno = 'data/check_dataset/SID2_new_train.txt'  # the anno file from prepare_dataset.py
        self.anno_validation = 'data/check_dataset/SID2_new_val_pairs.txt'
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
        self.lr = 0.0002  # initial learning rate for adam
        self.lr_d = self.lr
        self.lr_decay_ratio = 0.5  # decay ratio in step scheduler.
        self.n_epochs = 150   #100 number of epochs with the initial learning rate
        self.n_epochs_decay = 0   #100 when using 'linear', number of epochs to linearly decay learning rate to zero
        self.lr_decay_iters = 100  # when using 'step', multiply by a gamma every lr_decay_iters iterations
        self.optimizer_type = 'Adam'  # 'Adam', 'SGD'
        self.beta1 = 0.5  # momentum term of adam
        self.adam_eps = 1e-8

        # Setting for continuing the training.
        self.continue_train = False  # continue training: load the latest model
        self.epoch = '75'  # default='latest', which epoch to load? set to latest to use latest cached model
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

        self.model_name = 'drn'  # ['relighting' | 'intrinsic_decomposition']
        self.light_type = "pan_tilt_color"  # ["pan_tilt_color" | "Spherical_harmonic"]
        self.light_prediction = False
        self.netG = 'global'
        self.use_discriminator = False
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

        self.loss_weight_GAN = 0.05

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
        self.load_optimizer = True
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

