
import os
import random
import tensorflow as tf
import numpy as np

class DatasetSequenceImagenet(tf.keras.utils.Sequence):
    def __init__(self, dataset_path, batch_size, split, image_size = (256, 256), shuffle=True,
                 train_samples=None, test_samples=None, val_samples=None):
        """
        :param dataset_path: path to the dataset, which contains the train and test folders.
        Inside each folder there the images with no labels, since we only need the images to train the model.
        :param batch_size:
        :param split: 'train', 'valid' or 'test'
        :param image_size: the size of the images to be loaded
        :param shuffle: whether to shuffle the dataset
        :param train_samples: Maximum number of samples to use for training. Default is None, which means all the samples
        :param test_samples: Maximum number of samples to use for testing. Default is None, which means all the samples
        :param val_samples: Maximum number of samples to use for validation. Default is None, which means all the samples
        """
        self.train_path = os.path.join(dataset_path, 'train')
        self.test_path = os.path.join(dataset_path, 'test')
        self.val_path = os.path.join(dataset_path, 'val')
        self.batch_size = batch_size
        self.image_size = image_size
        self.split = split
        self.train = []   # list of training image names
        self.test = []    # list of testing image names
        self.val = []     # list of validation image names
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.val_samples = val_samples

        # populate the lists with all the images names in the directory
        if self.split == 'train':
            # The file names are stored in "train.txt"
            with open(os.path.join(dataset_path, 'train.txt'), 'r') as f:
                for i, line in enumerate(f):
                    self.train.append(os.path.join(self.train_path, line.strip()))

        elif self.split == 'test':
            for i in os.listdir(self.test_path):
                self.test.append(os.path.join(self.test_path, i))

        elif self.split == 'val':
            for i in os.listdir(self.val_path):
                self.val.append(os.path.join(self.val_path, i))

        else:
            raise ValueError("The split must be \'train\', \'test\' or \'val\'")

        # shuffle the data
        if shuffle:
            if self.split == 'train':
                random.shuffle(self.train)
            elif self.split == 'test':
                random.shuffle(self.test)
            elif self.split == 'val':
                random.shuffle(self.val)

        # Limit the number of samples
        if self.split == 'train':
            if self.train_samples is not None:
                self.train = self.train[:self.train_samples]
        elif self.split == 'test':
            if self.test_samples is not None:
                self.test = self.test[:self.test_samples]
        elif self.split == 'val':
            if self.val_samples is not None:
                self.val = self.val[:self.val_samples]


    def __len__(self):
        if self.split == 'train':
            return len(self.train) // self.batch_size
        elif self.split == 'test':
            return len(self.test) // self.batch_size
        elif self.split == 'val':
            return len(self.val) // self.batch_size


    def __getitem__(self, idx):
        if self.split == 'train':
            batch = self.train[idx * self.batch_size:(idx + 1) * self.batch_size]
            return self.preprocessing(np.array([self.__load_image(file) for file in batch]))
        elif self.split == 'test':
            batch = self.test[idx * self.batch_size:(idx + 1) * self.batch_size]
            return self.preprocessing(np.array([self.__load_image(file) for file in batch]))
        elif self.split == 'val':
            batch = self.val[idx * self.batch_size:(idx + 1) * self.batch_size]
            return self.preprocessing(np.array([self.__load_image(file) for file in batch]))

        else:
            raise ValueError("The split must be \'train\', \'test\' or \'val\'")


    def preprocessing(self, batch_images):
        """
        Returns a tuple of (input, output) to feed the network
        The first element of the tuple is the input: the V (value) channel of the HSV image
        The second element is the output: The image in HSV color space
        :param batch_images: a batch of images in RGB color space
        :return: a tuple of (input, output) to feed the network
        """

        # Convert the images to float32
        batch_images= tf.cast(batch_images, tf.float32)
        # Normalize the RGB images
        batch_images = batch_images / 255.0

        # Convert the images to HSV color space
        hsv_images = tf.image.rgb_to_hsv(batch_images)

        # Extract the V channel
        v_channels = hsv_images[:, :, :, 2]

        # Reshape the V channel to have the same number of dimensions as the input
        v_channels = tf.reshape(v_channels, (v_channels.shape[0], v_channels.shape[1], v_channels.shape[2], 1))

        # Return a tuple of (input, output) to feed the network
        return v_channels, hsv_images

    def __load_image(self, file_path):
        # Load the image with 3 RGB channels
        # We resize it to the specified size without distorting the image

        try:
            img = tf.keras.utils.load_img(file_path, target_size=self.image_size, color_mode='rgb', keep_aspect_ratio=True)
        except:
            # If an error occurs, we load a pre-defined image to prevent the training from crashing
            # This image is just the first image in the dataset (shark)
            img = tf.keras.utils.load_img(".\\imagenet_2010\\train\\n01484850\\n01484850_17.JPEG", target_size=self.image_size, color_mode='rgb', keep_aspect_ratio=True)

        if img is None:
            img = tf.keras.utils.load_img(".\\imagenet_2010\\train\\n01484850\\n01484850_17.JPEG", target_size=self.image_size, color_mode='rgb', keep_aspect_ratio=True)

        # Convert the image to a numpy array
        img = tf.keras.preprocessing.image.img_to_array(img)
        return img
