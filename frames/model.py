import datetime
import keras
import logging
import multiprocessing
import numpy as np
import random
import re
import os
import imgaug

from frames.self_utils import log, time_show, generate_patches, merge_patches
import frames.utils as utils


class BatchNorm(keras.layers.BatchNormalization):
    """Extends the Keras BatchNormalization class to allow a central place
    to make changes if needed.

    Batch normalization has a negative effect on training if batches are small
    so this layer is often frozen (via setting in Config class) and functions
    as linear layer.
    """

    def call(self, inputs, training=None):
        """
        Note about training values:
            None: Train BN layers. This is the normal mode
            False: Freeze BN layers. Good when batch size is small
            True: (don't use). Set layer in training mode even when making inferences
        """
        return super(self.__class__, self).call(inputs, training=training)


class BaseModel:
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        # assert mode in ['training', 'inference', 'generate', 'test']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.set_log_dir()
        self.keras_model = self.build()
        # Pre-defined layer regular expressions
        layer_regex = {
            # all layers but the backbone
            "heads": r"(outputs)",
            # All layers
            "all": ".*",
        }
        if self.config.LAYERS in layer_regex.keys():
            self.config.LAYERS = layer_regex[self.config.LAYERS]

    def build(self):
        pass

    def load_image_gt(self, dataset, image_id, augmentation=None):
        """Load and return ground truth data for an image (image, mask, bounding boxes).

        augment: (deprecated. Use augmentation instead). If true, apply random
            image augmentation. Currently, only horizontal flipping is offered.
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
            For example, passing imgaug.augmenters.Fliplr(0.5) flips images
            right/left 50% of the time.

        Returns:
        image: [height, width, 3]
        mask: [height, width, instance_count]. The height and width are those
            of the image unless use_mini_mask is True, in which case they are
            defined in MINI_MASK_SHAPE.
        """
        image = dataset.load_image(image_id)
        mask, class_ids = dataset.load_mask(image_id)

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape

        # Augmentation
        if augmentation:
            assert np.max(image) >= 1, 'The max of image should be >1'
            if np.max(image) == 1:
                image = np.multiply(image, 255)
                image = image.astype(np.uint8)
            if np.max(mask) == 1:
                mask = np.multiply(mask, 255)
                mask = mask.astype(np.uint8)

            # Augmenters that are safe to apply to masks
            # Some, such as Affine, have settings that make them unsafe, so always
            # test your augmentation on masks
            MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes", "Fliplr", "Flipud", "CropAndPad",
                               "Affine", "PiecewiseAffine"]

            def hook(images, augmenter, parents, default):
                """Determines which augmenters to apply to masks."""
                return augmenter.__class__.__name__ in MASK_AUGMENTERS

            # Make augmenters deterministic to apply similarly to images and masks
            det = augmentation.to_deterministic()
            image = det.augment_image(image)
            # Change mask to np.uint8 because imgaug doesn't support np.bool
            mask = det.augment_image(mask, hooks=imgaug.HooksImages(activator=hook))
            # Verify that shapes didn't change
            assert image.shape == image_shape, "Augmentation shouldn't change image size"
            assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
            # Change mask back to bool
            # mask = mask.astype(np.bool)

        image, window, scale, pad_list, crop = utils.resize_image(image, min_dim=self.config.IMAGE_MIN_DIM,
                                                                  max_dim=self.config.IMAGE_MAX_DIM,
                                                                  chn_dim=self.config.IMAGE_CHANNEL_COUNT,
                                                                  padding=self.config.IMAGE_PADDING)
        image = image - self.config.MEAN_PIXEL / 255
        if self.config.NUM_CLASSES == 2:
            mask = utils.rescale_mask(mask, scale, pad_list)  # interpolation
        else:
            mask = utils.resize_mask(mask, scale, pad_list)  # no interpolation

        return image.astype(np.float32), mask.astype(np.float32)

    def data_generator(self, dataset, shuffle=True, random_rois=0, detection_targets=False, augmentation=None,
                       no_augmentation_sources=None):
        b = 0  # batch item index
        image_index = -1
        image_id = 0
        image_ids = np.copy(dataset.image_ids)
        error_count = 0
        no_augmentation_sources = no_augmentation_sources or []

        # Keras requires a generator to run indefinately.
        while True:
            try:
                image_index = (image_index + 1) % len(image_ids)
                if shuffle and image_index == 0:
                    np.random.shuffle(image_ids)

                image_id = image_ids[image_index]

                # If the image source is not to be augmented pass None as augmentation
                if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                    image, gt = self.load_image_gt(dataset, image_id, augmentation=None)
                else:
                    image, gt = self.load_image_gt(dataset, image_id, augmentation=augmentation)

                if np.sum(gt) == 0:
                    logging.warning('GT is all Zeros!')
                    continue
                # Pre-progress
                image, gt = generate_patches(image, gt[:, :, 0], self.config.PATCH_SIZE)
                choice_num = random.randint(0, len(image) - 1)
                image = image[choice_num]
                gt = gt[choice_num]
                gt = np.expand_dims(gt, axis=-1)

                image = np.swapaxes(image, 0, 2)
                image = np.swapaxes(image, 1, 2)
                image = np.expand_dims(image, axis=-1)

                # Init batch arrays
                if b == 0:
                    batch_images = np.zeros((self.config.BATCH_SIZE,) + image.shape, dtype=np.float32)
                    batch_gts = np.zeros((self.config.BATCH_SIZE,) + gt.shape, dtype=np.float32)
                # Add to batch
                batch_images[b] = image
                batch_gts[b] = gt

                b += 1
                # Batch full?
                if b >= self.config.BATCH_SIZE:
                    inputs = [batch_images]
                    outputs = [batch_gts]
                    yield inputs, outputs
                    b = 0

            except (GeneratorExit, KeyboardInterrupt):
                raise
            except:
                # Log it and skip the image
                logging.exception("Error processing image {}".format(dataset.image_info[image_id]))
                error_count += 1
                if error_count > 5:
                    raise

    def set_log_dir(self, model_path=None):
        """Sets the model log directory and epoch counter.

        model_path: If None, or a format different from what this code uses then set a new log directory and start
        epochs from 0. Otherwise, extract the log directory and the epoch counter from the file name.
        """
        # Set date and epoch counter as if starting a new model
        self.epoch = 0
        now = datetime.datetime.now()

        # If we have a model path with date and epochs use them
        if model_path:
            # Continue from we left of. Get epoch and date from the file name
            # A sample model path might look like:
            # \path\to\logs\coco20171029T2315\mask_rcnn_coco_0001.h5 (Windows)
            # /path/to/logs/coco20171029T2315/mask_rcnn_coco_0001.h5 (Linux)
            # regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\]mask\_rcnn\_[\w-]+(\d{4})\.h5"
            regex = r".*[/\\][\w-]+(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})[/\\][\w-]+(\d{4})\.h5"
            m = re.match(regex, model_path)
            if m:
                now = datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
                                        int(m.group(5)))
                # Epoch number in file is 1-based, and in Keras code it's 0-based.
                # So, adjust for that then increment by one to start from the next epoch
                self.epoch = int(m.group(6)) - 1 + 1
                print('Re-starting from epoch %d' % self.epoch)

        # Directory for training logs
        self.log_dir = os.path.join(self.model_dir, "{}{:%Y%m%dT%H%M}".format(self.config.NAME.lower(), now))

        # Path to save after each epoch. Include placeholders that get filled by Keras.
        # self.checkpoint_path = os.path.join(self.log_dir, "{}*epoch*.h5".format(self.config.NAME.lower()))
        # self.checkpoint_path = self.checkpoint_path.replace("*epoch*", "{epoch:04d}")
        self.checkpoint_path = os.path.join(self.log_dir, "{epoch:04d}.h5")

    def ancestor(self, tensor, name, checked=None):
        """Finds the ancestor of a TF tensor in the computation graph.
        tensor: TensorFlow symbolic tensor.
        name: Name of ancestor tensor to find
        checked: For internal use. A list of tensors that were already
                 searched to avoid loops in traversing the graph.
        """
        checked = checked if checked is not None else []
        # Put a limit on how deep we go to avoid very long loops
        if len(checked) > 500:
            return None
        # Convert name to a regex and allow matching a number prefix
        # because Keras adds them automatically
        if isinstance(name, str):
            name = re.compile(name.replace("/", r"(\_\d+)*/"))

        parents = tensor.op.inputs
        for p in parents:
            if p in checked:
                continue
            if bool(re.fullmatch(name, p.name)):
                return p
            checked.append(p)
            a = self.ancestor(p, name, checked)
            if a is not None:
                return a
        return None

    def model_compile(self):
        self.keras_model.compile(optimizer=self.config.OPTIMIZER(lr=self.config.LEARNING_RATE),
                                 loss=self.config.LOSS_FUNCTION, metrics=self.config.METRICS)

    def set_trainable(self, layer_regex, keras_model=None, indent=0, verbose=0):
        """Sets model layers as trainable if their names match
        the given regular expression.
        """
        # Print message on the first call (but not on recursive calls)
        if verbose > 0 and keras_model is None:
            log("Selecting layers to train")

        keras_model = keras_model or self.keras_model

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") \
            else keras_model.layers

        for layer in layers:
            # Is the layer a model?
            if layer.__class__.__name__ == 'Model':
                print("In model: ", layer.name)
                self.set_trainable(
                    layer_regex, keras_model=layer, indent=indent + 4)
                continue

            if not layer.weights:
                continue
            # Is it trainable?
            trainable = bool(re.fullmatch(layer_regex, layer.name))
            # Update layer. If layer is a container, update inner layer.
            if layer.__class__.__name__ == 'TimeDistributed':
                layer.layer.trainable = trainable
            else:
                layer.trainable = trainable
            # Print trainable layer names
            if trainable and verbose > 0:
                log("{}{:20}   ({})".format(" " * indent, layer.name,
                                            layer.__class__.__name__))

    def find_trainable_layer(self, layer):
        """If a layer is encapsulated by another layer, this function
        digs through the encapsulation and returns the layer that holds
        the weights.
        """
        if layer.__class__.__name__ == 'TimeDistributed':
            return self.find_trainable_layer(layer.layer)
        return layer

    def get_trainable_layers(self):
        """Returns a list of layers that have weights."""
        layers = []
        # Loop through all layers
        for l in self.keras_model.layers:
            # If layer is a wrapper, find inner trainable layer
            l = self.find_trainable_layer(l)
            # Include layer if it has weights
            if l.get_weights():
                layers.append(l)
        return layers

    def load_weights(self, filepath, by_name=False, exclude=None):
        """Modified version of the corresponding Keras function with
        the addition of multi-GPU support and the ability to exclude
        some layers from loading.
        exclude: list of layer names to exclude
        """
        import h5py
        # Conditional import to support versions of Keras before 2.2
        # TODO: remove in about 6 months (end of 2018)
        try:
            from keras.engine import saving
        except ImportError:
            # Keras before 2.2 used the 'topology' namespace.
            from keras.engine import topology as saving

        if exclude:
            by_name = True

        if h5py is None:
            raise ImportError('`load_weights` requires h5py.')
        f = h5py.File(filepath, mode='r')
        if 'layer_names' not in f.attrs and 'model_weights' in f:
            f = f['model_weights']

        # In multi-GPU training, we wrap the model. Get layers
        # of the inner model because they have the weights.
        keras_model = self.keras_model
        layers = keras_model.inner_model.layers if hasattr(keras_model, "inner_model") else keras_model.layers

        # Exclude some layers
        if exclude:
            layers = filter(lambda l: l.name not in exclude, layers)

        if by_name:
            saving.load_weights_from_hdf5_group_by_name(f, layers)
        else:
            saving.load_weights_from_hdf5_group(f, layers)
        if hasattr(f, 'close'):
            f.close()

        # Update the log directory
        self.set_log_dir(filepath)

    def train(self, train_dataset, val_dataset, custom_callbacks=None, augmentation=None, no_augmentation_sources=None):
        """Train the model.
        train_dataset, val_dataset: Training and validation Dataset objects.
        learning_rate: The learning rate to train with
        epochs: Number of training epochs. Note that previous training epochs
                are considered to be done alreay, so this actually determines
                the epochs to train in total rather than in this particaular
                call.
        layers: Allows selecting wich layers to train. It can be:
            - A regular expression to match layer names to train
            - One of these predefined values:
              heads: The RPN, classifier and mask heads of the network
              all: All the layers
              3+: Train CNN stage 3 and up
              4+: Train CNN stage 4 and up
              5+: Train CNN stage 5 and up
        augmentation: Optional. An imgaug (https://github.com/aleju/imgaug)
            augmentation. For example, passing imgaug.augmenters.Fliplr(0.5)
            flips images right/left 50% of the time. You can pass complex
            augmentations as well. This augmentation applies 50% of the
            time, and when it does it flips images right/left half the time
            and adds a Gaussian blur with a random sigma in range 0 to 5.

                augmentation = imgaug.augmenters.Sometimes(0.5, [
                    imgaug.augmenters.Fliplr(0.5),
                    imgaug.augmenters.GaussianBlur(sigma=(0.0, 5.0))
                ])
        custom_callbacks: Optional. Add custom callbacks to be called
            with the keras fit_generator method. Must be list of type keras.callbacks.
        no_augmentation_sources: Optional. List of sources to exclude for
            augmentation. A source is string that identifies a dataset and is
            defined in the Dataset class.
        """
        assert self.mode == "training", "Create model in training mode."

        # Data generators
        train_generator = self.data_generator(train_dataset, shuffle=True, augmentation=augmentation,
                                              no_augmentation_sources=no_augmentation_sources)
        val_generator = self.data_generator(val_dataset, shuffle=True)

        # Create log_dir if it does not exist
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        # Callbacks
        callbacks = [
            keras.callbacks.TensorBoard(log_dir=self.log_dir, histogram_freq=0, write_graph=True, write_images=False),
            keras.callbacks.ModelCheckpoint(self.checkpoint_path, monitor='val_loss', verbose=0, save_weights_only=True,
                                            save_best_only=self.config.SAVE_BEST, all_epochs=self.config.EPOCHES),
            keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=self.config.PATIENCE, min_lr=0)
        ]

        # Add custom callbacks to the list
        if custom_callbacks:
            callbacks += custom_callbacks

        # Train
        log("\nStarting at epoch {}. LR={}\n".format(self.epoch, self.config.LEARNING_RATE))
        log("Checkpoint Path: {}".format(self.checkpoint_path))
        self.set_trainable(self.config.LAYERS)
        self.model_compile()

        if os.name is 'nt':
            workers = 0
        else:
            workers = multiprocessing.cpu_count()

        self.keras_model.fit_generator(
            train_generator,
            initial_epoch=self.epoch,
            epochs=self.config.EPOCHES,
            steps_per_epoch=self.config.STEPS_PER_EPOCH,
            callbacks=callbacks,
            validation_data=next(val_generator),
            validation_steps=self.config.VALIDATION_STEPS,
            max_queue_size=100,
            workers=workers,
            use_multiprocessing=True
        )
        self.epoch = max(self.epoch, self.config.EPOCHES)
        time_show([], [])

    def detect(self, images, verbose=0):
        """Runs the detection pipeline.
        images: List of images
        Returns a list of dicts, one dict per image. The dict contains:
        masks: [H, W, N] instance binary masks
        """
        assert len(images) == 1, "len(images) should == 1"

        if verbose:
            log("Processing {} images".format(len(images)))
            for image in images:
                log("image", image)

        # Pre-progress
        image = images[0]
        image = image - self.config.MEAN_PIXEL / 255

        # Process detections
        results = []
        mask_patches = []
        image_patches, bboxs = generate_patches(image_ori=image, patch_size=self.config.PATCH_SIZE)
        for image_patch in image_patches:
            image_patch = np.swapaxes(image_patch, 0, 2)
            image_patch = np.swapaxes(image_patch, 1, 2)
            image_patch = np.expand_dims(image_patch, axis=-1)
            mask_patches.append(np.squeeze(self.keras_model.predict([[image_patch]])))

        mask = merge_patches(image, mask_patches, bboxs)
        mask = mask > 0.7
        mask = np.uint8(mask)
        results.append({"masks": mask})
        return results
