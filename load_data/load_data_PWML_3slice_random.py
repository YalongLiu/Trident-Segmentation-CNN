# coding:utf-8
import os
from skimage.io import imread
import numpy as np
import random

from frames import utils
from frames import self_utils


class TargetDataset(utils.Dataset):
    """The same with load_data_PWML_3slice but randomly sample each .png
    """

    def __init__(self, config):
        super(TargetDataset, self).__init__()
        self.TEST_DIR = "gtFine/test"
        # self.ori_8b_path = '/media/james/Repository/datasets/PWML/ori/8b'
        self.ori_8b_path = 'F:/datasets/PWML/ori/8b'
        self.cord = []
        self.split_dataset(config)

    def split_dataset(self, config):
        # Split files to train/val/test
        folders = next(os.walk(config.GT_DIR))[1]  # load in folders
        target_pathes = []
        for folder in folders:
            files_in_one_folder = next(os.walk(config.GT_DIR + '/' + folder))[2]
            for file in files_in_one_folder:
                target_pathes.append([config.DATA_DIR, folder, file])

        len_samples = len(target_pathes)
        train_num = round(config.DATA_RATIO[0] * len_samples)
        # val_num = round(config.DATA_RATIO[1] * len_samples)

        random.seed(config.RANDOM_SEED)
        random.shuffle(target_pathes)

        self.train_pathes = target_pathes[:train_num]
        self.test_pathes = target_pathes[train_num:]

        # self.val_pathes = target_pathes[train_num:]
        # self.val_pathes = target_pathes[train_num: train_num + val_num]
        # self.test_pathes = target_pathes[train_num + val_num:]

    def compute_image_mean(self, config):
        images = []
        target_ids = next(os.walk(config.GT_DIR))[1]  # load in folders
        for sample_id in target_ids:
            mask_ids = next(os.walk(config.GT_DIR + '/' + sample_id))[2]
            for mask_id in mask_ids:
                image = imread(config.IMAGE_DIR + '/' + sample_id + '/' + mask_id)
                images.append(np.mean(image))
        mean_value = np.mean(images)
        return mean_value

    def load_samples(self, flag, config):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        # Add classes
        self.class_info = [{"source": "", "id": 0, "name": "BG"}]
        self.add_class("pwml_t1_data", 1, "pwml")

        # Add images
        if flag == 'train':
            self.sample_pathes = self.train_pathes
        if flag == 'val':
            self.sample_pathes = self.val_pathes
        if flag == 'test':
            self.sample_pathes = self.test_pathes
        i = 0
        for sample_path in self.sample_pathes:
            self.add_image("pwml_t1_data", image_id=i, path=sample_path, image=None, mask=None)
            i += 1

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        path_list = info['path']
        img_num = int(path_list[2][:-4])
        image_path = self.ori_8b_path + '/input/' + path_list[1] + '/'

        image_0 = imread(image_path + str(img_num - 1) + '.png')
        image_1 = imread(image_path + str(img_num) + '.png')
        image_2 = imread(image_path + str(img_num + 1) + '.png')

        x1, x2, y1, y2 = self_utils.crop_to_skull(image_1)
        self.cord = [x1, x2, y1, y2]
        image_0 = image_0[y1:y2 + 1, x1:x2 + 1]
        image_1 = image_1[y1:y2 + 1, x1:x2 + 1]
        image_2 = image_2[y1:y2 + 1, x1:x2 + 1]

        image_ori = np.stack([image_0, image_1, image_2], axis=-1)
        image_ori = np.divide(image_ori, 255, dtype=np.float32)
        return image_ori

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        path_list = info['path']
        path = self.ori_8b_path + '/gt'
        for single in path_list[1:]:
            path = path + '/' + single
        if path[-4:] == '.npy':
            mask = np.load(path)
        else:
            mask = imread(path)
        class_ids = np.ones([mask.shape[-1]], dtype=np.int32)  # Map class names to class IDs.
        mask = mask.astype(np.bool)
        mask = mask.astype(np.float32)

        x1, x2, y1, y2 = self.cord
        mask = mask[y1:y2 + 1, x1:x2 + 1]
        mask = np.expand_dims(mask, axis=-1)
        return mask, class_ids
