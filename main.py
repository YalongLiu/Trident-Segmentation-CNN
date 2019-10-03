# coding: utf-8
import os
import random
import warnings
import logging
from numpy.random import seed
from tensorflow import set_random_seed

# import configs data and model
from configs.pwml import config_TridentSegNet as ModelConfig
import load_data.load_data_PWML_3slice_random as load_data
import models.model_TridentSegNet as modellib

# Choose a mode(inference or training)
mode = 'inference'
# mode = 'training'

if __name__ == "__main__":
    config = ModelConfig.ModelConfig()  # input config in different models

    # Set random seed
    random.seed(config.RANDOM_SEED)
    seed(config.RANDOM_SEED)
    set_random_seed(config.RANDOM_SEED)

    # config.display()
    print('Name:', config.NAME)

    # Load dataset
    dataset_train = load_data.TargetDataset(config)
    dataset_train.load_samples('train', config)
    dataset_train.prepare()
    dataset_test = load_data.TargetDataset(config)
    dataset_test.load_samples('test', config)
    dataset_test.prepare()

    # Import libs
    from frames import self_utils

    # Train mode
    if mode == 'training':
        # Create model in training mode
        model = modellib.DefineModel(mode=mode, config=config, model_dir=config.LOGDIR)
        # Load weights
        if config.WEIGHTS_PATH != '':
            if not os.path.exists(config.WEIGHTS_PATH):
                logging.warning(config.WEIGHTS_PATH + " cannot be found!")
            model.load_weights(config.WEIGHTS_PATH, by_name=True, exclude=[])
        print("Training " + config.LAYERS + "...")
        config.NAME = config.NAME + '_'
        model.train(dataset_train, dataset_test, augmentation=config.augmentation)

    # Inference mode
    if mode == 'inference':
        # Weight name
        weight_name = config.WEIGHTS_PATH
        weight_name = weight_name.split('/')
        tmp_name = weight_name[-1].split('.')[0]
        tmp_name = tmp_name.split('_')[-1]
        pred_name = weight_name[-2] + '_' + tmp_name + '_pred'
        print(pred_name)

        config.IMAGES_PER_GPU = 1
        config.BATCH_SIZE = 1
        # Create model in training mode
        model = modellib.DefineModel(mode=mode, config=config, model_dir=os.path.join(os.getcwd(), "logs"))
        # Load weights
        model.load_weights(config.WEIGHTS_PATH, by_name=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self_utils.segmentation_post_process(dataset_test, model, config, pred_name)
