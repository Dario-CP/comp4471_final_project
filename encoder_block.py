"""
Encoder block for the U-Net in tensorflow
Returns a tuple (output, skip_connection):
    "output" is the block output
    "skip_connection" is the output before the max pooling
"""

import tensorflow as tf

class EncoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), max_pool=True, padding='same', activation='relu'):
        super(EncoderBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            activation=activation)

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            activation=activation)

        if max_pool:
            self.pool = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
        else:
            self.pool = tf.keras.layers.Lambda(lambda x: x) # Identity function

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        skip_connection = x
        x = self.pool(x)
        return x, skip_connection

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'conv2': self.conv2,
            'pool': self.pool
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # The output shape is the input shape divided by 2 in the spatial dimensions because of the max pooling 2x2
        # Note that we also output the skip connection
        return tf.TensorShape([input_shape[0], input_shape[1] // 2, input_shape[2] // 2, self.conv2.filters]),\
               tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], self.conv2.filters])
