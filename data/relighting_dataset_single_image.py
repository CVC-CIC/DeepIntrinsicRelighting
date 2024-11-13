import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import torch
import random
import math
from util.k_to_rgb import convert_K_to_RGB
from util.util import PARA_NOR
LIGHT_INDEX_IN_NAME = 2


def read_anno_single_image(anno_filename):
    """Read the name of images from the anno file yielded by prepare_dataset.py.
       For each image, we must know which scene it belongs to.
    """
    # read lines from anno
    f = open(anno_filename, 'r')
    file_names = []
    for line in f.readlines():
        line = line.strip('\n')
        file_names.append(line)
    # Don't sort because the test and val sets need the determined order.
    scene_all = []
    scene_index = []   # record the scene index of each image
    last_scene = " "
    scene = []
    for x in file_names:
        x_scene = "{}_{}".format(x.split('_')[0], x.split('_')[1])
        if x_scene != last_scene:
            if len(scene) > 0:
                scene_all.append(scene)
            scene = []
        scene.append(x)
        scene_index.append(len(scene_all))
        last_scene = x_scene
    if len(scene) > 0:
        scene_all.append(scene)

    return file_names, scene_index, scene_all


def image_name2light_condition(img_name):
    factor_deg2rad = math.pi / 180.0
    names = os.path.splitext(img_name)[0].split('_')
    pan = float(names[LIGHT_INDEX_IN_NAME]) * factor_deg2rad
    tilt = float(names[LIGHT_INDEX_IN_NAME+1]) * factor_deg2rad
    color_temp = int(names[LIGHT_INDEX_IN_NAME+2])
    # transform light position to cos and sin
    light_position = [math.cos(pan), math.sin(pan), math.cos(tilt), math.sin(tilt)]
    # normalize the light position to [0, 1]
    light_position[:2] = [x * PARA_NOR['pan_a'] + PARA_NOR['pan_b'] for x in light_position[:2]]
    light_position[2:] = [x * PARA_NOR['tilt_a'] + PARA_NOR['tilt_b'] for x in light_position[2:]]
    # transform light temperature to RGB, and normalize it.
    light_color = list(map(lambda x: x / 255.0, convert_K_to_RGB(color_temp)))
    light_position_color = light_position + light_color
    return torch.tensor(light_position_color)


def read_component(dataroot, component, file_name, img_transform, r_pil=False):
    component_path = "{}{}/{}".format(dataroot, component, file_name)
    if not os.path.exists(component_path):
        raise Exception("RelightingDataset __getitem__ error")

    img_component = Image.open(component_path).convert('RGB')
    img_tensor = img_transform(img_component)
    if r_pil:
        return img_tensor, img_component
    return img_tensor


def get_data(file_name_input, file_name_output, dataroot, img_transform, multiple_replace_image):
    data = {}  # output dictionary

    data['scene_label'] = file_name_input
    data['light_position_color_original'] = image_name2light_condition(file_name_input)
    data['light_position_color_new'] = image_name2light_condition(file_name_output)

    # Reflectance_output
    data['Reflectance_output'] = read_component(dataroot, 'Reflectance', file_name_input, img_transform)
    data['Shading_ori'], s_ori = read_component(dataroot, 'Shading', file_name_input, img_transform, r_pil=True)
    data['Shading_output'], s_output = read_component(dataroot, 'Shading', file_name_output, img_transform, r_pil=True)
    if multiple_replace_image:
        data['Image_input'] = torch.mul(data['Reflectance_output'], data['Shading_ori'])
        data['Image_relighted'] = torch.mul(data['Reflectance_output'], data['Shading_output'])
    else:
        data['Image_input'] = read_component(dataroot, 'Image', file_name_input, img_transform)
        data['Image_relighted'] = read_component(dataroot, 'Image', file_name_output, img_transform)

    return data

class RelightingDatasetSingleImage(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image.
    """

    def __init__(self, opt, validation=False):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if validation:
            anno_file = opt.anno_validation
        else:
            anno_file = opt.anno
        self.file_names, self.scene_index, self.scenes_list = read_anno_single_image(anno_file)

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
        dataroot = self.dataroot
        img_size = self.opt.img_size
        multiple_replace_image = self.opt.multiple_replace_image

        index_file_names = index

        # get one image
        file_name_input = self.file_names[index_file_names]
        # get the scene_index
        scene_id = self.scene_index[index_file_names]
        scene = self.scenes_list[scene_id].copy()
        # remove the input image.
        scene.remove(file_name_input)
        id_in_scene = random.randrange(0, len(scene))
        file_name_output = scene[id_in_scene]

        # get the parameters of data augmentation
        transform_params = get_params(self.opt, img_size)
        img_transform = get_transform(self.opt, transform_params)

        data = get_data(file_name_input, file_name_output, dataroot, img_transform, multiple_replace_image)

        return data

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.file_names)


