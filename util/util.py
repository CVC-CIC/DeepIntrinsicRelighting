"""This module contains simple helper functions """
from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


"""
Normalize the trigonometric function value of pan and tilt to [0, 1]
pan can be -180 ~ 180 deg, so cos(pan), sin(pan) are [-1, 1], which should be adapted.
tilt can be 0 ~ 90 deg, so cos(pan), sin(pan) are [0, 1], which should not be adapted.
Normalization: y = ax + b;
denormalization:  x = (y-b)/a
"""
PARA_NOR = {
    'pan_a': 0.5,
    'pan_b': 0.5,
    'tilt_a': 1.0,
    'tilt_b': 0.0,
}


def tensor2im(input_image, normalization_type, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        if normalization_type == '[-1, 1]':
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        elif normalization_type == '[0, 1]':
            if np.max(image_numpy) > 1.0:
                image_numpy = np.clip(image_numpy, a_min=0.0, a_max=1.0)
                raise Exception("A warning for possible exceeding. ")
                # image_numpy = image_numpy / np.max(image_numpy)
            image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0  # post-processing: tranpose and scaling
        else:
            raise Exception("rewrite inverse normalization here.")
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


# def tensor2pan_tilt_color(pan_tilt_color, num_type=np.int16):
#     pan_tilt_color_numpy = pan_tilt_color.cpu().detach().numpy()
#     pan_tilt_color_numpy = pan_tilt_color_numpy / [math.pi / 180.0, math.pi / 180.0, 0.001]
#     pan_tilt_color_numpy = np.around(pan_tilt_color_numpy)
#     return pan_tilt_color_numpy.astype(num_type)


def tensor2pan_tilt_color(data, num_type=np.int16):
    # This function transforms the normalized light condition into the real value.
    data = data.cpu().detach().numpy()
    data[:2] = (data[:2] - PARA_NOR['pan_b']) / PARA_NOR['pan_a']
    data[2:4] = (data[2:4] - PARA_NOR['tilt_b']) / PARA_NOR['tilt_a']
    pan = np.rad2deg(np.arctan2(data[1], data[0]))
    tilt = np.rad2deg(np.arctan2(data[3], data[2]))
    RGB = data[4:] * 255.0
    around_data = np.around(np.array([pan, tilt, RGB[0], RGB[1], RGB[2]]))
    return around_data.astype(num_type)


def diagnose_network(net, name='network'):
    """Calculate and print the mean of average absolute(gradients)

    Parameters:
        net (torch network) -- Torch network
        name (str) -- the name of the network
    """
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    """Print the mean, min, max, median, std, and size of a numpy array

    Parameters:
        val (bool) -- if print the values of the numpy array
        shp (bool) -- if print the shape of the numpy array
    """
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
