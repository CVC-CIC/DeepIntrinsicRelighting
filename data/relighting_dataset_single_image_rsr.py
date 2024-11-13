from data.base_dataset import BaseDataset, get_params, get_transform
import random
import os
from PIL import Image
import math
from util.util import PARA_NOR
import torch


def read_anno_pairs(anno_filename):
    # read lines from anno
    with open(anno_filename, 'r') as f:
        annos = []
        for line in f.readlines():
            line = line.strip('\n')
            this_pair = line.split(' ')
            annos.append(this_pair)
    return annos


def read_anno_group(anno_filename, data_root, Type_select):
    scene_list = []
    with open(anno_filename, 'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            scene_list.append(line)

    groups = []
    for scene_id in scene_list:
        scene_path = os.path.join(data_root, scene_id)
        file_list = os.listdir(scene_path)
        file_list.sort()
        if len(file_list) != 288:
            raise Exception("The number of files are wrong!")

        if Type_select == "LightColorOnly":
            for i in range(8):
                start_point = i * 9 * 4
                for j in range(9):
                    one_part = [file_list[start_point + j + 9 * k] for k in range(4)]
                    groups.append([scene_id, one_part])
        elif Type_select == "LightPositionOnly":
            slice = 9
            for i in range(32):
                one_part = file_list[i * slice:(i + 1) * slice]
                groups.append([scene_id, one_part])
        elif Type_select == "AnyLight":
            slice = 36
            for i in range(8):
                one_part = file_list[i * slice:(i + 1) * slice]
                groups.append([scene_id, one_part])
        else:
            raise Exception("No Type_select!")
    return groups, len(groups) * len(groups[0][1])



class RelightingDatasetSingleImageRSR(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image.
    """

    def __init__(self, opt, validation=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = self.opt.dataroot_RSR
        if validation:
            anno_file = opt.anno_validation
        else:
            anno_file = opt.anno
        self.fix_pair = validation or not opt.isTrain
        if self.fix_pair:
            self.pairs_list = read_anno_pairs(anno_file)
            self.length = self.pairs_list.__len__()
        else:
            self.group_type = self.opt.dataset_rsr_type
            self.groups_list, self.length = read_anno_group(anno_file, self.dataroot, self.group_type)
            self.image_per_group = len(self.groups_list[0][1])

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains
            'Image_input': ,
            'light_position_color_new': ,
            'light_position_color_original': ,
            'Reflectance_output': ,
            'Shading_output': ,
            'Shading_ori': ,
            'Image_relighted': ,
            'scene_label': ,
        """
        # get parameters
        img_size = self.opt.img_size

        if self.fix_pair:
            pair = self.pairs_list[index]
            folder_id = pair[0]
            img_input = pair[1]
            img_target = pair[2]
        else:
            div, mod = divmod(index, self.image_per_group)
            group = self.groups_list[div]
            folder_id = group[0]
            img_input = group[1][mod]
            id_list = list(range(self.image_per_group))
            id_list.remove(mod)
            id_target = random.choice(id_list)
            img_target = group[1][id_target]

        # get the parameters of data augmentation
        transform_params = get_params(self.opt, img_size)
        self.img_transform = get_transform(self.opt, transform_params)

        data = {}
        data['scene_label'] = img_input
        data['light_position_color_original'] = self.get_light_condition(img_input)
        data['light_position_color_new'] = self.get_light_condition(img_target)

        data['Image_input'] = self.get_image(folder_id, img_input)
        data['Image_relighted'] = self.get_image(folder_id, img_target)

        return data

    def get_light_condition(self, img_name):
        """
        Name of image:
        {Image index}_{Relighting scene index}_{Pan}_{tilt}_{R}_{G}_{B}_{Object scene index}_{rotation of platform}_
        {light color index}_{light position index}.jpg
        """
        factor_deg2rad = math.pi / 180.0
        names = os.path.splitext(img_name)[0].split('_')
        pan = float(names[2]) * factor_deg2rad
        tilt = float(names[3]) * factor_deg2rad
        # transform light position to cos and sin
        light_position = [math.cos(pan), math.sin(pan), math.cos(tilt), math.sin(tilt)]
        # normalize the light position to [0, 1]
        light_position[:2] = [x * PARA_NOR['pan_a'] + PARA_NOR['pan_b'] for x in light_position[:2]]
        light_position[2:] = [x * PARA_NOR['tilt_a'] + PARA_NOR['tilt_b'] for x in light_position[2:]]
        # transform light temperature to RGB, and normalize it.
        light_color = list(map(lambda x: int(x) / 255.0, names[4:7]))
        light_position_color = light_position + light_color
        return torch.tensor(light_position_color)

    def get_image(self, folder_name, img_name):
        image_filename = os.path.join(self.dataroot, folder_name, img_name)
        if not os.path.exists(image_filename):
            raise Exception("RelightingDataset __getitem__ error")

        img = Image.open(image_filename).convert('RGB')
        img_tensor = self.img_transform(img)
        return img_tensor

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length


