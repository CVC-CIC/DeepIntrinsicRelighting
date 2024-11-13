
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6,7'

import time

from data import create_dataset
from models.models import create_model
from util.visualizer import Visualizer
import torch
import torch.distributed as dist
from util.metric import calculate_all_metrics

from tqdm import tqdm
import numpy as np
import shutil
import argparse
import importlib

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--local_rank')
    parser.add_argument('option_name', type=str)
    parser_value = parser.parse_args()
    option_name = parser_value.option_name
    # import package
    TrainOptions = getattr(importlib.import_module('options.train_options_{}'.format(option_name)), "TrainOptions")

    opt = TrainOptions('exp_'+option_name).parse()  # set CUDA_VISIBLE_DEVICES before import torch
    # print(opt.which_experiment)

    if opt.parallel_method == "DistributedDataParallel":
        rank = int(os.environ["RANK"])
        print(rank)
        world_size = int(os.environ['WORLD_SIZE'])
        opt.gpu_ids = rank
        opt.world_size = world_size
        torch.cuda.set_device(rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)

    # flag_master controls to show result and save checkpoint.
    flag_master = opt.parallel_method != "DistributedDataParallel" or (
                opt.parallel_method == "DistributedDataParallel" and rank == 0)

    model = create_model(opt)

    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_validation = create_dataset(opt, validation=True)
    dataset_size = len(dataset)  # get the number of images in the dataset.
    dataset_size_validation = len(dataset_validation)  # get the number of images in the dataset.
    if flag_master:
        print('The number of training images = %d' % dataset_size)
        print('The number of validation images = %d' % dataset_size_validation)

    model.setup(opt)  # regular setup: load and print networks; create schedulers
    # model.plot_model()   # plot the model
    # model.get_macs()   # plot the model
    if flag_master:
        visualizer = Visualizer(opt)  # create a visualizer that display/save images and plots
    total_iters = 0  # the total number of training iterations
    best_loss_relit_validation = float('inf')
    # define the metric for monitor during training
    metric_function = calculate_all_metrics()
    if hasattr(opt, 'val_keys'):
        val_keys = opt.val_keys
    else:
        val_keys = ['Relighted']

    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>

    for epoch in range(opt.epoch_count,
                       opt.n_epochs + opt.n_epochs_decay + 1):
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        if flag_master:
            visualizer.reset()  # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        print("Begin training")
        model.train()
        if opt.parallel_method == "DistributedDataParallel":
            dataset.dataloader.sampler.set_epoch(epoch)
        loss_epoch = []
        for i, data in tqdm(enumerate(dataset)):  # inner loop within one epoch
            # if i > 10:
            #     break
            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            total_iters += 1   # opt.batch_size
            epoch_iter += 1   # opt.batch_size
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters(epoch, i)  # calculate loss functions, get gradients, update network weights
            iter_data_time = time.time()
            # print training losses and save logging information to the disk
            current_loss = model.get_current_losses()
            current_loss['weighted_total'] = float(model.loss_weighted_total)
            loss_epoch.append(current_loss)
        losses = {}
        for key in current_loss:
            losses[key] = np.mean([x[key] for x in loss_epoch])

        # display images on visdom and save images to a HTML file
        save_result = epoch_iter % opt.update_html_freq == 0
        model.compute_visuals()
        if flag_master:
            visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

        # print("light_position_color_predict[0] = ", model.light_position_color_predict[0])
        # print("light_position_color_original[0] = ", model.light_position_color_original[0])

        print("Begin validation")
        loss_val = []
        metric_val = []
        model.eval()
        with torch.no_grad():
            # test the last batch of training
            model.test()
            visuals = model.get_current_visuals()  # get image results
            metric_train = metric_function.run(visuals, ['Relighted'])['Relighted']
            # test the validation dataset
            for i, data in tqdm(enumerate(dataset_validation)):  # inner loop within one epoch
                # if i > 10:
                #     break
                model.set_input(data)
                model.test()  # run inference
                visuals = model.get_current_visuals()  # get image results
                val_results = []
                for val_key in val_keys:
                    val_results.extend(metric_function.run(visuals, [val_key])[val_key])
                metric_val.append(torch.stack(val_results))
                loss_val.append(model.calculate_val_loss())
        # last batch of train
        metric_train = torch.stack(metric_train).unsqueeze(0)
        # vallidation
        metric_val = torch.stack(metric_val)
        loss_relit_validation = torch.stack(loss_val)
        if opt.parallel_method == "DistributedDataParallel":
            # print(losses)
            for key in losses:
                value = torch.tensor(losses[key]).cuda()
                dist.all_reduce(value)
                losses[key] = float(value) / float(world_size)
            # print(losses)
            # print(metric_train)
            dist.all_reduce(metric_train)
            # print("metric_train:", metric_train)
            dist.all_reduce(metric_val)
            dist.all_reduce(loss_relit_validation)
            torch.distributed.barrier()
            # all_reduce collects the sum of all GPUs results, so it needs to be averaged.
            metric_train = metric_train / float(world_size)
            metric_val = metric_val / float(world_size)
            loss_relit_validation = loss_relit_validation / float(world_size)
        # move to the cpu, otherwise Visdom cannot work.
        metric_train = torch.mean(metric_train, 0).cpu()
        metric_val_mean = torch.mean(metric_val, 0).cpu()
        loss_relit_validation = float(torch.mean(loss_relit_validation, 0))

        # add loss
        losses['relit_validation'] = loss_relit_validation
        # add metric
        for metric_index, metric_key in enumerate(['MSE', 'SSIM', 'PSNR', 'LPIPS', 'MPS']):
            losses['_'.join(['train', metric_key])] = metric_train[metric_index]
        val_count = 0
        for val_key in val_keys:
            for metric_key in ['MSE', 'SSIM', 'PSNR', 'LPIPS', 'MPS']:
                losses['_'.join(['val', val_key, metric_key])] = metric_val_mean[val_count]
                val_count = val_count + 1

        t_comp = (time.time() - iter_start_time) / opt.batch_size

        if flag_master:
            visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data,
                                            model.optimizers[0].param_groups[0]['lr'])
            if opt.display_id > 0:
                visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            # cache our latest model every <save_latest_freq> iterations
            if opt.save_latest:
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'latest'
                model.save_networks(save_suffix)

            if epoch % opt.save_epoch_freq == 0:  # cache our model every <save_epoch_freq> epochs
                print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks(epoch)
            if loss_relit_validation < best_loss_relit_validation:
                print('saving the best model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('best')
                best_loss_relit_validation = loss_relit_validation

        print('End of epoch %d / %d \t Time Taken: %d sec' % (
        epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

    if flag_master:
        # rename the best_* to save_best_*
        save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        for filename in os.listdir(save_dir):
            if "best" in filename:
                old_name = os.path.join(save_dir, filename)
                new_name = os.path.join(save_dir, "save_"+filename)
                shutil.copyfile(old_name, new_name)
        pass

