# coding:utf-8
import os
from skimage.measure import label
import numpy as np
import time
import sys
from skimage.io import imsave
from tqdm import tqdm
import random
from keras import backend as K
from keras.callbacks import Callback
from frames import utils

time_list = []


def time_show(epoch, logs):
    time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    time_list.append(time_now)
    print(time_now)


def time_lists():
    return time_list


def log(text, array=None):
    """Prints a text message. And, optionally, if a Numpy array is provided it
    prints it's shape, min, and max values.
    """
    if array is not None:
        text = text.ljust(25)
        text += ("shape: {:20}  min: {:10.5f}  max: {:10.5f}  {}".format(
            str(array.shape), array.min() if array.size else "", array.max() if array.size else "", array.dtype))
    print(text)


def crop_to_skull(image):
    '''
    Crop black space in the image
    :param image:
    :return:
    '''
    if len(image.shape) == 3:
        image = image[:, :, 0]
    shape_ori = image.shape
    y1 = int(shape_ori[0] / 2)
    y2 = int(shape_ori[0] / 2)
    x1 = int(shape_ori[1] / 2)
    x2 = int(shape_ori[1] / 2)
    cut_threshold = 1024
    for i in range(int(shape_ori[1] / 2)):
        if (np.sum(image[:, i]) > cut_threshold) & (x1 == shape_ori[1] / 2):
            x1 = i
        if (np.sum(image[:, shape_ori[1] - i - 1]) > cut_threshold) & (x2 == shape_ori[1] / 2):
            x2 = shape_ori[1] - i - 1

    for i in range(shape_ori[0]):
        if (np.sum(image[i, :]) > cut_threshold) & (y1 == shape_ori[0] / 2):
            y1 = i
        if (np.sum(image[shape_ori[0] - i - 1, :]) > cut_threshold) & (y2 == shape_ori[0] / 2):
            y2 = shape_ori[0] - i - 1
    return [x1, x2, y1, y2]


########################
#     WarmUp Policy    #
########################
class WarmUp(Callback):
    def __init__(self, warmup_epochs, config_lr):
        super(WarmUp, self).__init__()
        self.warmup_epochs = warmup_epochs
        self.config_lr = config_lr

    def on_batch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            # Learning rate turn from 0 to self.config_lr during self.warmup_epochs
            K.set_value(self.model.optimizer.lr, self.config_lr * (epoch + 1) / self.warmup_epochs)


def mask_pad(mask_ori, pad_size):
    tmp_expand_mask = np.zeros([mask_ori.shape[0], pad_size])
    mask_expand = np.hstack([tmp_expand_mask, mask_ori, tmp_expand_mask])
    tmp_expand_mask = np.zeros([pad_size, mask_expand.shape[1]])
    mask_expand = np.vstack([tmp_expand_mask, mask_expand, tmp_expand_mask])
    mask_expand = np.uint8(mask_expand)
    return mask_expand


def mask_pad_off(mask_ori, pad_size):
    mask_pad_off = mask_ori[pad_size:-pad_size, pad_size:-pad_size]
    return mask_pad_off


def generate_patches(image_ori=None, semantic_mask=None, patch_size=None):
    image_ori_shape = list(np.shape(image_ori))
    image_patchs = []
    mask_patchs = []
    if semantic_mask is not None:
        instance_mask = sementic2instance_mask(semantic_mask)
        bboxes = utils.extract_bboxes(instance_mask)
        for bbox in bboxes:
            y1, x1, y2, x2 = bbox
            x = int(round(x1 + x2) / 2)
            y = int(round(y1 + y2) / 2)
            shift_pos_x = random.randint(-int(round((x2 - x1) / 2)), int(round((x2 - x1) / 2)))
            shift_pos_y = random.randint(-int(round((y2 - y1) / 2)), int(round((y2 - y1) / 2)))
            new_x = x + shift_pos_x
            new_y = y + shift_pos_y
            if new_x >= image_ori_shape[1] - int(patch_size[1] / 2):
                x2 = image_ori_shape[1]
                x1 = image_ori_shape[1] - patch_size[1]
            elif new_x <= int(patch_size[1] / 2):
                x2 = patch_size[1]
                x1 = 0
            else:
                x1 = new_x - int(patch_size[1] / 2)
                x2 = new_x + int(patch_size[1] / 2)
            if new_y >= image_ori_shape[0] - int(patch_size[0] / 2):
                y2 = image_ori_shape[0]
                y1 = image_ori_shape[0] - patch_size[0]
            elif new_y <= int(patch_size[0] / 2):
                y2 = patch_size[0]
                y1 = 0
            else:
                y1 = new_y - int(patch_size[0] / 2)
                y2 = new_y + int(patch_size[0] / 2)
            image_patchs.append(image_ori[y1:y2, x1:x2])
            mask_patchs.append(semantic_mask[y1:y2, x1:x2])
        return image_patchs, mask_patchs
    else:
        bboxs = []
        n_y = int(image_ori_shape[0] / patch_size[0]) + 1
        n_x = int(image_ori_shape[1] / patch_size[1]) + 1
        delta_y = image_ori_shape[0] / n_y
        delta_x = image_ori_shape[1] / n_x
        pos_y = [delta_y * i for i in range(n_y)]
        pos_x = [delta_x * i for i in range(n_x)]
        for y in pos_y:
            for x in pos_x:
                if x + patch_size[1] > image_ori_shape[1]:
                    x1 = image_ori_shape[1] - patch_size[1]
                    x2 = image_ori_shape[1]
                else:
                    x1 = int(round(x))
                    x2 = int(round(x)) + patch_size[1]
                if y + patch_size[0] > image_ori_shape[0]:
                    y1 = image_ori_shape[0] - patch_size[0]
                    y2 = image_ori_shape[0]
                else:
                    y1 = int(round(y))
                    y2 = int(round(y)) + patch_size[0]
                bboxs.append([y1, x1, y2, x2])
                image_patchs.append(image_ori[y1:y2, x1:x2])
        return image_patchs, bboxs


def merge_patches(image, mask_patches, bboxs):
    mask = np.zeros(np.shape(image)[:2])
    for i in range(len(bboxs)):
        y1, x1, y2, x2 = bboxs[i]
        mask[y1:y2, x1:x2] = np.where(mask[y1:y2, x1:x2] == 0, mask_patches[i],
                                      [(mask[y1:y2, x1:x2] + mask_patches[i]) / 2])
    return mask


def sementic2instance_mask(semantic_mask):
    '''
    Convert semantic mask into instance mask
    :param semantic_mask:
    :return: instance mask
    '''
    mask_shape = list(np.shape(semantic_mask))
    mask_labeled, instance_num = label(semantic_mask, return_num=True)
    instance_mask = np.zeros([mask_shape[0], mask_shape[1], instance_num], dtype=np.uint8)
    for i in range(instance_num):
        instance_mask[:, :, i] = np.where(mask_labeled == i + 1, [True], [False])
    return instance_mask


###########################
#     Inference
###########################
def copy_folder_structor(input_path, output_path):
    '''
    Build a folder tree of output_path like the input_path
    :param input_path:
    :param output_path:
    :return:
    '''
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    folders = next(os.walk(input_path))[1]
    if len(folders) > 0:
        for folder in folders:
            if not os.path.exists(output_path + '/' + folder):
                os.mkdir(output_path + '/' + folder)
        for folder in folders:
            copy_folder_structor(input_path + '/' + folder, output_path + '/' + folder)


def make_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


def segmentation_post_process(dataset_test, model, config, pred_folder_name):
    '''
    data{'image_path', 'gt_path', 'save_path'}
    function:
    Detect
    if gt_path!=[]-->+Show
    if save_path!=[]-->+Save pred
    '''
    images = []
    len_data = len(dataset_test.image_ids)
    whole_path = []
    make_path(config.PRED_DIR)
    path = config.PRED_DIR + '/' + pred_folder_name
    make_path(path)
    for i, image_id in tqdm(enumerate(dataset_test.image_ids), total=len_data):
        # Load image and run detection
        info = dataset_test.image_info[image_id]
        path_list = info['path']
        make_path(path + '/' + path_list[1])
        save_path = path + '/' + path_list[1] + '/' + path_list[2]
        if not os.path.exists(save_path):  # if image is not exist
            # print('...', path_list[1] + '/' + path_list[2], '\t', i + 1, '/', len_data)
            image = dataset_test.load_image(image_id)
            # image = imread('F:\datasets\PWML_T1_data\PWML_T1_cut_to_skull_8b\images/8/77.png')
            # image = np.stack([image, image, image], axis=-1)
            image, window, scale, padding, crop = utils.resize_image(image, min_dim=config.IMAGE_MIN_DIM,
                                                                     max_dim=config.IMAGE_MAX_DIM,
                                                                     chn_dim=config.IMAGE_CHANNEL_COUNT,
                                                                     padding=config.IMAGE_PADDING)
            images.append(image)
            whole_path.append(save_path)
            if len(images) == config.BATCH_SIZE:
                # Detect objects
                results = model.detect(images, verbose=0)
                # Detect, plot, save
                for n, result in enumerate(results):
                    # Processing pred_enlarge_list
                    if 'rois' in result:
                        roi_length = len(result['rois'])
                        #         print("rois:", roi_length)
                        if roi_length == 0:
                            pred = np.zeros([config.IMAGE_MAX_DIM, config.IMAGE_MAX_DIM],
                                            dtype=bool)
                        else:
                            pred = result['masks'][:, :, 0]
                            for i in range(roi_length - 1):
                                pred = np.maximum(pred, result['masks'][:, :, i + 1])  # instance to semantic
                    else:
                        pred = result['masks']
                    pred = np.squeeze(pred)
                    imsave(whole_path[n], pred)
                images = []
                whole_path = []
        else:
            print(save_path, ' already exist!')
