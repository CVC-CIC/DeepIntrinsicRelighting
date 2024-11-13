"""
This script is to crop and resize the images of Multi-illumination dataset
"""
import os
import cv2
from tqdm import tqdm

def transfer_img(ori_name, new_name):
    ori_image = cv2.imread(ori_name, cv2.IMREAD_UNCHANGED)
    shape = ori_image.shape
    height = shape[0]
    width = shape[1]
    left = int(0.5 * width - 0.5 * height)
    right = left + height
    crop_image = ori_image[:, left:right, :]
    resized_image = cv2.resize(crop_image, (256, 256), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite(new_name, resized_image)


def transfer_probes(ori_name, new_name):
    for suffix in ["chrome", "gray"]:
        name1 = ori_name + suffix + "256.jpg"
        name2 = new_name + suffix + ".jpg"
        ori_image = cv2.imread(name1, cv2.IMREAD_UNCHANGED)
        resized_image = cv2.resize(ori_image, (64, 64), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(name2, resized_image)


dataset_path_ori = '/home/yyang/dataset/Multi_Illumination/'
dataset_path_new = '/home/yyang/dataset/Multi_Illumination_small/'
num_img_scene = 25

for subset in os.listdir(dataset_path_ori):
    # subset: train, test
    if not os.path.exists(os.path.join(dataset_path_new, subset)):
        os.mkdir(os.path.join(dataset_path_new, subset))
    for scene in tqdm(os.listdir(os.path.join(dataset_path_ori, subset))):
        # scenes
        if not os.path.exists(os.path.join(dataset_path_new, subset, scene)):
            os.mkdir(os.path.join(dataset_path_new, subset, scene))
        if not os.path.exists(os.path.join(dataset_path_new, subset, scene, 'probes')):
            os.mkdir(os.path.join(dataset_path_new, subset, scene, 'probes'))
        current_path_ori = os.path.join(dataset_path_ori, subset, scene)
        current_path_new = os.path.join(dataset_path_new, subset, scene)
        for img_id in range(num_img_scene):
            img_name = "dir_{}_mip2.jpg".format(str(img_id))
            transfer_img(os.path.join(current_path_ori, img_name), os.path.join(current_path_new, img_name))
            probes_name = "probes/dir_{}_".format(str(img_id))
            transfer_probes(os.path.join(current_path_ori, probes_name), os.path.join(current_path_new, probes_name))


