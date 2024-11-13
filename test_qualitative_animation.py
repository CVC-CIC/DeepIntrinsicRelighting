import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '5'
from options.test_qualitative_options import TestOptions
from models.models import create_model
from util.util import PARA_NOR
from util.util import tensor2im
from util.k_to_rgb import convert_K_to_RGB
import cv2
import numpy as np
import torch
from tqdm import tqdm
from data.base_dataset import get_params, get_transform
from PIL import Image
import imageio


def light_condition2tensor(pan_deg, tilt_deg, color, color_type = "temperature"):
    """
    transform pan, tilt, color into the tensors for input.
    :param pan: in deg
    :param tilt: in deg
    :param color: in temperature
    :return: tensor size(7)
    """
    factor_deg2rad = math.pi / 180.0
    pan = float(pan_deg) * factor_deg2rad
    tilt = float(tilt_deg) * factor_deg2rad

    # transform light position to cos and sin
    light_position = [math.cos(pan), math.sin(pan), math.cos(tilt), math.sin(tilt)]
    # normalize the light position to [0, 1]
    light_position[:2] = [x * PARA_NOR['pan_a'] + PARA_NOR['pan_b'] for x in light_position[:2]]
    light_position[2:] = [x * PARA_NOR['tilt_a'] + PARA_NOR['tilt_b'] for x in light_position[2:]]
    # transform light temperature to RGB, and normalize it.
    if color_type == "temperature":
        color_temp = int(color)
        light_color = list(map(lambda x: x / 255.0, convert_K_to_RGB(color_temp)))
    else:
        light_color = [x/255 for x in color]
    light_position_color = light_position + light_color
    return torch.tensor(light_position_color)



def read_image(img_name, opt):
    transform_params = get_params(opt, opt.img_size)
    img_transform = get_transform(opt, transform_params)
    if not os.path.exists(img_name):
        raise Exception("RelightingDataset __getitem__ error")
    img_component = Image.open(img_name).convert('RGB')
    aspect = img_component.size[1] / img_component.size[0]
    img_component = img_transform(img_component)
    return img_component.unsqueeze(0), aspect


class ImageDial():
    def __init__(self, dial_img_name):
        dial_img = Image.open(dial_img_name).convert('RGB')
        dial_img = np.array(dial_img)
        scale_ratio = 256 / 880
        self.dial_img = cv2.resize(dial_img, None, fx=scale_ratio, fy=scale_ratio,
                                   interpolation=cv2.INTER_CUBIC)
        self.dial_center = [int(523 * scale_ratio), int(1320 * scale_ratio)]
        self.radius = 400 * scale_ratio
        # original size is (281, 768, 3), we need to fill to (288, 768, 3) to satisfy macro_block_size=16 in imageio
        self.h_pad, self.w_pad = tuple([x if x % 16 == 0 else x + 16 - x % 16 for x in self.dial_img.shape[:2]])

    def insert_img(self, img_input, img_relit, pan, tilt):
        merged_img = np.copy(self.dial_img)
        # merged_img[-256:, :256, :] = cv2.cvtColor(img_input, cv2.COLOR_RGB2BGR)
        # merged_img[-256:, -256:, :] = cv2.cvtColor(img_relit, cv2.COLOR_RGB2BGR)
        merged_img[-256:, :256, :] = img_input
        merged_img[-256:, -256:, :] = img_relit
        # plot the point of pan and tilt.
        length = math.sin(tilt/180*math.pi) / math.sin(50/180*math.pi) * self.radius
        position = (int(self.dial_center[1] + math.sin(pan/180*math.pi) * length),
                    int(self.dial_center[0] + math.cos(pan/180*math.pi) * length))
        # cv2.circle(merged_img, position, 3, (0, 0, 255), -1)
        cv2.circle(merged_img, position, 3, (255, 0, 0), -1)
        # add padding
        white_padding = np.full((self.h_pad, self.w_pad, 3), 255, dtype=np.uint8)
        white_padding[:merged_img.shape[0], :merged_img.shape[1]] = merged_img
        return white_padding


def generate_path(points, steps, length):
    loop_path = []
    for i in range(len(points)-1):
        point_a = points[i]
        point_b = points[i+1]
        this_step = [step if point_b[k] > point_a[k] else -step for k, step in enumerate(steps)]
        lists = [np.arange(point_a[k], point_b[k], this_step[k]) for k in range(len(point_a))]
        for j in range(max([len(lst) for lst in lists])):
            loop_path.append([lists[k][j] if j < len(lists[k]) else point_b[k] for k in range(len(lists))])
    path = [loop_path[i % len(loop_path)] for i in range(length)]

    return path


def create_pan_tilt_temperature_seq(length, seq_type):
    # pan, tilt, temperature
    # default_steps = [2, 1, 100]
    default_start = [90, 30, 4100]
    if seq_type == "cycle_tilt":
        points = [[default_start[0], 40.0, default_start[2]],
                  [default_start[0], 0, default_start[2]],
                  [-default_start[0], 40, default_start[2]],
                  [-default_start[0], 0, default_start[2]],
                  [default_start[0], 40.0, default_start[2]]]
        steps = [float('inf'), 1, float('inf')]
    elif seq_type == "cycle_pan":
        points = [[0, default_start[1], default_start[2]],
                  [360, default_start[1], default_start[2]], ]
        steps = [2, float('inf'), float('inf')]
    elif seq_type == "cycle_temperature":
        points = [[default_start[0], default_start[1], 2300],
                  [default_start[0], default_start[1], 6400],]
        steps = [float('inf'), float('inf'), 100]
    else:
        raise Exception("seq_type wrong!")

    sequence = generate_path(points, steps, length)
    return sequence


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.special_test = True

    dial_img_name = "./util/pan_tilt_dial.png"
    img_dial = ImageDial(dial_img_name)

    data = {}
    img_name = "./202102_008_221_35_3200_108_00_Image_input.png"
    data['scene_label'] = img_name.split('/')[-1]
    data['Image_input'], _ = read_image(img_name, opt)

    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    frame_number = 320
    seq_type = "cycle_pan"
    seq_light = create_pan_tilt_temperature_seq(frame_number, seq_type=seq_type)
    suffix = '_' + seq_type

    out_dir = os.path.join(opt.results_dir, opt.name, opt.epoch)  # define the website directory
    input_name = os.path.splitext(data['scene_label'])[0]
    fix_tilt = True
    video_reso = (768, 281)
    video_name = '{}_{}'.format(out_dir, input_name)+suffix
    fps = 25
    # out = cv2.VideoWriter(video_name + '.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, video_reso)
    # Use MPEG-4 encoding
    # fourcc = cv2.VideoWriter_fourcc(*'avc1')
    # out = cv2.VideoWriter(video_name + '.mp4', fourcc, fps, video_reso)

    writer = imageio.get_writer(video_name + '.mp4', fps=fps, codec='libx264')

    print("Create video at {}".format(video_name + '.mp4'))

    for frame in tqdm(range(frame_number)):
        pan, tilt, temperature = tuple(seq_light[frame])

        data['light_position_color_new'] = light_condition2tensor(pan, tilt, temperature, color_type="temperature").unsqueeze(0)
        model.set_input(data)  # unpack data from data loader
        model.test()  # run inference
        visuals = model.get_current_visuals()  # get image results

        im_input = tensor2im(visuals['Image_input'][0].unsqueeze(0), opt.normalization_type)
        im_relit = tensor2im(visuals['Relighted_predict'][0].unsqueeze(0), opt.normalization_type)
        im = img_dial.insert_img(im_input, im_relit, pan, tilt)

        # out.write(im)
        writer.append_data(im)
    # out.release()
    writer.close()


