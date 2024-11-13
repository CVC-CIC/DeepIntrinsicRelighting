import torch

from data.base_dataset import BaseDataset, get_params, get_transform, __make_power_2
import torchvision.transforms as transforms
import random
import os
from PIL import Image


def read_anno(anno_filename):
    """Read the name of images from the anno file yielded by prepare_dataset.py.
       For each image, we must know which scene it belongs to.
    """
    # read lines from anno
    f = open(anno_filename, 'r')
    annos = []
    for line in f.readlines():
        line = line.strip('\n')
        annos.append(line)
    return annos


class RelightingDatasetSingleImageMultilum(BaseDataset):
    """A dataset class for relighting dataset.
       This dataset read data image by image.
    """
    def __init__(self, opt, validation=False):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dataroot = self.opt.dataroot_multilum
        if validation:
            anno_file = opt.anno_validation
        else:
            anno_file = opt.anno

        self.fix_pairs = not (self.opt.isTrain and opt.dataset_setting == 'ALL' and not validation)

        if self.fix_pairs:
            self.pairs_list = read_anno(anno_file)
            self.length = self.pairs_list.__len__()
        else:
            self.scenes_list = read_anno(anno_file)
            self.image_per_scene = 25
            self.length = self.image_per_scene * self.scenes_list.__len__()
            # kingston_bigbathroom2/dir_3_mip2.jpg is wrong
            self.scenes_list.remove("kingston_bigbathroom2")
            self.scenes_list.append("kingston_bigbathroom2")
            self.length = self.length - 1
            self.special_index_kingston_bigbathroom2 = list(range(self.image_per_scene))
            self.special_index_kingston_bigbathroom2.remove(3)

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

        if self.fix_pairs:
            pair = self.pairs_list[index]
            pair_elements = pair.split()
            scene = pair_elements[0]
            input_id = pair_elements[1]
            target_id = pair_elements[2]
        else:
            div, mod = divmod(index, self.image_per_scene)
            scene = self.scenes_list[div]
            # random pairs
            if scene != "kingston_bigbathroom2":
                input_id = mod
                id_list = list(range(self.image_per_scene))
                id_list.remove(mod)
                target_id = random.choice(id_list)
            else:
                input_id = self.special_index_kingston_bigbathroom2[mod]
                filtered_list = [x for x in self.special_index_kingston_bigbathroom2 if x != input_id]
                target_id = random.choice(filtered_list)

        # get the parameters of data augmentation
        transform_params = get_params(self.opt, img_size)
        self.img_transform = get_transform(self.opt, transform_params)
        self.probes_transform = self.get_transform_probes()

        data = {}
        data['scene_label'] = scene + '_' + str(input_id) + '_' + str(target_id)
        data['Image_input'], data['light_position_color_original'] = self.get_image(scene, input_id)
        data['Image_relighted'], data['light_position_color_new'] = self.get_image(scene, target_id)
        return data

    def get_image(self, scene, img_id):
        # image
        image_filename = os.path.join(self.dataroot, scene, "dir_{}_mip2.jpg".format(str(img_id)))
        if not os.path.exists(image_filename):
            raise Exception("RelightingDataset __getitem__ error")
        img = Image.open(image_filename).convert('RGB')
        img_tensor = self.img_transform(img)
        # the corresponding probes.
        probe1_name = os.path.join(self.dataroot, scene, "probes/dir_{}_chrome.jpg".format(str(img_id)))
        probe1 = Image.open(probe1_name).convert('RGB')
        probe1_tensor = self.probes_transform(probe1)
        probe2_name = os.path.join(self.dataroot, scene, "probes/dir_{}_gray.jpg".format(str(img_id)))
        probe2 = Image.open(probe2_name).convert('RGB')
        probe2_tensor = self.probes_transform(probe2)
        probes_tensor = torch.cat([probe1_tensor, probe2_tensor], dim=0)
        return img_tensor, probes_tensor

    def get_transform_probes(self, grayscale=False):
        transform_list = []

        transform_list += [transforms.ToTensor()]
        if self.opt.normalization_type == '[-1, 1]':
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        elif self.opt.normalization_type == '[0, 1]':
            pass
        else:
            raise Exception("normalization_type error")
        return transforms.Compose(transform_list)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return self.length


