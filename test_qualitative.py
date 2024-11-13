
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from options.test_qualitative_options import TestOptions
from data import create_dataset
from models.models import create_model
from util.visualizer import save_images_one_batch
from util import html


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        # img_path should provide the image name of one batch in list format.
        img_path = model.get_image_paths()     # get image paths
        # img_path = [x for x in img_path[0]]
        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images_one_batch(webpage, visuals, img_path, opt.normalization_type, aspect_ratio=opt.aspect_ratio,
                              width=opt.display_winsize)
    webpage.save()  # save the HTML
