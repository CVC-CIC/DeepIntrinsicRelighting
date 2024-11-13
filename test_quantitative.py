"""
test script which is used to get quantitive results
"""
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4'
from options.test_quantitative_options import TestQuantitiveOptions
from data import create_dataset
from models.models import create_model
import torch
from tqdm import tqdm
from util.metric import calculate_all_metrics


if __name__ == '__main__':
    opt = TestQuantitiveOptions().parse()  # get test options
    print(opt.name)
    opt.pre_read_data = False

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    metric_results = {}
    metric_function = calculate_all_metrics()
    for key in opt.metric_list:
        metric_results[key] = []
    for i, data in tqdm(enumerate(dataset)):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        all_results = metric_function.run(visuals, metric_results.keys())
        for key in metric_results.keys():
            metric_results[key].append(all_results[key])
    for key in metric_results.keys():
        results = torch.tensor(metric_results[key])
        results_mean = torch.mean(results, 0)
        results_std = torch.std(results, 0)
        if key in ['Reflectance', 'Shading_ori', 'Shading_new', 'Relighted', 'Reconstruct', 'Input_and_relighted_gt']:
            print("{}: MPS, SSIM, LPIPS, PSNR, MSE"
                  " = ,{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, "
                  "".format(key, results_mean[4], results_mean[1], results_mean[3], results_mean[2], results_mean[0]))
            print("{}(std): MPS, SSIM, LPIPS, PSNR, MSE"
                  " = ,{:.4f}, {:.4f}, {:.4f}, {:.4f}, {:.4f}, "
                  "".format(key, results_std[4], results_std[1], results_std[3], results_std[2], results_std[0]))
        elif key == 'light_position_color':
            print("{}: light_position_angle_error, pan_error, tilt_error, "
                  "light_color_angle_error = ,{:.4f}, {:.4f}, {:.4f}, {:.4f}, ".format(key, results_mean[0],
                                                                                       results_mean[1],
                                                                                       results_mean[2],
                                                                                       results_mean[3]))
            print("{}(std): light_position_angle_error, pan_error, tilt_error, "
                  "light_color_angle_error = ,{:.4f}, {:.4f}, {:.4f}, {:.4f}, ".format(key, results_std[0],
                                                                                       results_std[1],
                                                                                       results_std[2],
                                                                                       results_std[3]))
        else:
            raise Exception("key error")


