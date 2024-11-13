import os
from util import util

class BaseOptions():
    def __init__(self):
        self.server_root = '/ghome/yyang/dataset/'
        self.dataroot = self.server_root + 'ISR/'  # path of the dataset
        self.dataroot_vidit = self.server_root + 'VIDIT_full/'
        self.dataroot_RSR = self.server_root + 'RSR_256/'
        self.dataroot_multilum = self.server_root + 'Multi_Illumination_small/train/'
        self.checkpoints_dir = './checkpoints/'   # models are saved here
        self.max_dataset_size = float("inf") #float("inf")   # Maximum number of samples allowed per dataset. If the dataset
        # directory contains more than max_dataset_size, only a subset is loaded.
        self.img_size = (256, 256)   # size of the image
        self.input_nc = 3   # number of input image channels
        self.output_nc = 3   # number of output image channels
        self.ngf = 64   # number of gen filters in first conv layer
        self.init_type = 'normal'   # network initialization [normal | xavier | kaiming | orthogonal]
        self.init_gain = 0.02  # scaling factor for normal, xavier and orthogonal.
        self.verbose = False   # if specified, print more debugging information

        self.normalization_type = '[0, 1]'   # '[0, 1]' or '[-1, 1]' if this is changed, the inverse normalization in
        # visualizer should also be changed manually.
        self.multiple_replace_image = True   # if specified, the Image type of the dataset will not be read from the
        # dataset, it will be replaced by the multiple of reflectance and shading.
        self.pre_read_data = False   # if specified, the dataset will be stored in memory before training.

        self.display_winsize = 256  # display window size for both visdom and HTML

    def parse(self, verbose=True):
        args = vars(self)

        if verbose:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.checkpoints_dir, self.name)
        util.mkdirs(expr_dir)
        if self.isTrain:
            options_name = 'options_train.txt'
        else:
            options_name = 'options_test.txt'
        file_name = os.path.join(expr_dir, options_name)
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self
