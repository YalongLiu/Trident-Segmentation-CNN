# coding: utf-8
import numpy as np
from frames.config import Config
from imgaug import augmenters as iaa

from keras.optimizers import Adam, SGD
from frames.metrics import mean_iou, dice_coef_metrics, self_balancing_focal_loss, focal_loss


############################################################
#  Configurations
############################################################
class ModelConfig(Config):
    # Modify to your own path
    DATA_DIR = './test_data'  # choose input data
    # DATA_DIR = 'F:/datasets/PWML/cut2skull/8b'  # choose input data
    WEIGHTS_PATH = ''  # Weights to load
    # WEIGHTS_PATH = "F:/datasets/weights/tridentsegnet_pwml_cut2skull_8b_98765_20190922T1712_0050_pred_0.7.h5"

    ###################################################################################################################
    #                         Modifications to the following section are NOT recommended!                             #
    ###################################################################################################################
    # Choose model and data
    MODLE_NAME = 'TridentSegNet'  # choose model
    LOAD_DATA_NAME = 'PWML_3slice_random'  # choose the load_data_*.py to load data
    TASK_TYPE = 'Segmentation'

    RANDOM_SEED = 98765  # Random seed
    LAYERS = 'all'
    LEARNING_RATE = 0.0002  # lr
    EPOCHES = 50  # Epochs
    GPU_COUNT = 1  # NUMBER OF GPUs to use. When using only a CPU, this needs to be set to 1.
    IMAGES_PER_GPU = 8  # Adjust depending on your GPU memory. Batch size is 4 (GPUs * images/GPU).
    DATA_RATIO = [0.8, 0.2]  # 5-fold
    NUM_CLASSES = 2
    CARDINALITY = 64
    MEAN_PIXEL = np.array([56.49, 56.49, 56.49])
    PATCH_SIZE = [128, 128]

    LOGDIR = './logs'
    # Inference directors
    INPUT_DIR = DATA_DIR + "/input"
    GT_DIR = DATA_DIR + "/gt"
    PRED_DIR = DATA_DIR + '_pred'
    GEN_DIR = DATA_DIR + '_gen'

    OPTIMIZER = Adam
    # OPTIMIZER = SGD
    LOSS_FUNCTION = [self_balancing_focal_loss]
    # LOSS_FUNCTION = [focal_loss]
    METRICS = [mean_iou, dice_coef_metrics]

    SAVE_BEST = False
    SAVE_EPOCHES = 1
    PATIENCE = 1

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 50

    # Input image resizing
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256
    IMAGE_PADDING = True
    IMAGE_CHANNEL_COUNT = 3

    augmentation = iaa.Sequential(children=(iaa.Crop(percent=(0, [0, 0.05], 0, [0, 0.05]), keep_size=True), iaa.Fliplr(
        0.5), iaa.Affine(rotate=(-10, 10)), iaa.Multiply((0.8, 1.2))))

    # The name of model to save
    NAME = MODLE_NAME + '_' + LOAD_DATA_NAME + '_' + str(RANDOM_SEED) + '_'
