from data.base_dataset import BaseDataset, get_params, get_transform
from util.util import PARA_NOR
import math
import torch
from util.k_to_rgb import convert_K_to_RGB
from PIL import Image
import os.path
import random
DIRECTION = ['E', 'N', 'NE', 'NW', 'S', 'SE', 'SW', 'W']
TEMPERATURE_LIST = [2500, 3500, 4500, 5500, 6500]


def read_anno_pairs(anno_filename):
    # read lines from anno
    with open(anno_filename, 'r') as f:
        annos = []
        for line in f.readlines():
            line = line.strip('\n')
            this_pair = line.split(' ')
            annos.append(this_pair)
    return annos


def read_train_pairs(anno_filename):
    # read lines from anno
    with open(anno_filename, 'r') as f:
        annos = []
        for line in f.readlines():
            line = line.strip('\n')
            annos.append(["{}_6500_N.png".format(line), "{}_4500_E.png".format(line)])
    return annos


def read_anno_group(anno_filename, Type_select):
    scene_list = []
    with open(anno_filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            scene_list.append(line)

    groups = []
    for scene_id in scene_list:
        if Type_select == "LightColorOnly":
            for direction in DIRECTION:
                groups.append(
                    ['_'.join([scene_id, str(temperature), direction]) + '.png' for temperature in TEMPERATURE_LIST])
        elif Type_select == "LightPositionOnly":
            for temperature in TEMPERATURE_LIST:
                groups.append(
                    ['_'.join([scene_id, str(temperature), direction]) + '.png' for direction in DIRECTION])
        elif Type_select == "AnyLight":
            groups.append(
                ['_'.join([scene_id, str(temperature), direction]) + '.png' for temperature in TEMPERATURE_LIST for
                 direction in DIRECTION])
        else:
            raise Exception("No Type_select!")
    return groups, len(groups) * len(groups[0]), len(groups[0])


class RelightingDatasetSingleImageVidit(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image.
    """

    def __init__(self, opt, validation=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # for vidit dataset
        self.dataroot = self.opt.dataroot_vidit
        if validation:
            anno_file = opt.anno_validation
        else:
            anno_file = opt.anno
        self.fix_pair = validation or not opt.isTrain
        if self.fix_pair:
            self.pairs_list = read_anno_pairs(anno_file)
            self.length = self.pairs_list.__len__()
        elif self.opt.dataset_assignment_type == "6500_N_4500_E":
            self.pairs_list = read_train_pairs(anno_file)
            self.length = self.pairs_list.__len__()
            self.fix_pair = True
        else:
            self.group_type = self.opt.dataset_assignment_type.split('_')[0]
            # DCDP means different color and different position
            self.DCDP = 'DCDP' in self.opt.dataset_assignment_type
            self.groups_list, self.length, self.image_per_group = read_anno_group(anno_file, self.group_type)

    def __getitem__(self, index):
        # get parameters
        img_size = self.opt.img_size

        if self.fix_pair:
            pair = self.pairs_list[index]
            img_input = pair[0]
            img_target = pair[1]
        else:
            div, mod = divmod(index, self.image_per_group)
            group = self.groups_list[div]
            img_input = group[mod]
            if self.DCDP:
                color_input = img_input.split('.')[0].split('_')[1]
                position_input = img_input.split('.')[0].split('_')[2]
                filtered_group = []
                for img in group:
                    color, position = tuple(img.split('.')[0].split('_')[1:3])
                    if color != color_input and position != position_input:
                        filtered_group.append(img)
                img_target = random.choice(filtered_group)
            else:
                id_list = list(range(self.image_per_group))
                id_list.remove(mod)
                id_target = random.choice(id_list)
                img_target = group[id_target]

        # get the parameters of data augmentation
        transform_params = get_params(self.opt, img_size)
        self.img_transform = get_transform(self.opt, transform_params)

        data = {}
        data['scene_label'] = img_input
        data['light_position_color_original'] = self.get_light(img_input)
        data['light_position_color_new'] = self.get_light(img_target)

        data['Image_input'] = self.get_image(img_input)
        data['Image_relighted'] = self.get_image(img_target)

        return data

    def get_light(self, img_name):
        str2pan = {'E': 90,
                   'N': 180,
                   'NE': 135,
                   'NW': 225,
                   'S': 0,
                   'SE': 45,
                   'SW': 315,
                   'W': 270}
        img_name_split = img_name.split('.')[0].split('_')
        pan = str2pan[img_name_split[-1]]
        tilt = 45.0
        color_temp = int(img_name_split[-2])
        # transform light position to cos and sin
        light_position = [math.cos(pan), math.sin(pan), math.cos(tilt), math.sin(tilt)]
        # normalize the light position to [0, 1]
        light_position[:2] = [x * PARA_NOR['pan_a'] + PARA_NOR['pan_b'] for x in light_position[:2]]
        light_position[2:] = [x * PARA_NOR['tilt_a'] + PARA_NOR['tilt_b'] for x in light_position[2:]]
        # transform light temperature to RGB, and normalize it.
        light_color = list(map(lambda x: x / 255.0, convert_K_to_RGB(color_temp)))
        light_position_color = light_position + light_color
        return torch.tensor(light_position_color)

    def get_image(self, img_name):
        image_filename = os.path.join(self.dataroot, img_name)
        if not os.path.exists(image_filename):
            raise Exception("RelightingDataset __getitem__ error")

        img = Image.open(image_filename).convert('RGB')
        img_tensor = self.img_transform(img)
        return img_tensor

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length

