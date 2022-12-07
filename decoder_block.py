"""
Decoder block for the U-Net in tensorflow
Gets a tuple (output, skip_connection) as input:
    "output" is the output of the previous block
    "skip_connection" is the output of the symmetrical encoder block before the max pooling
Returns the block output

We will use Conv2DTranspose
"""

import tensorflow as tf

class DecoderBlock(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'):
        super(DecoderBlock, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            activation=activation)

        self.conv2 = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                            strides=strides, padding=padding,
                                            activation=activation)

        self.up = tf.keras.layers.Conv2DTranspose(filters=filters, kernel_size=(2, 2), strides=(2, 2), padding='valid')

    def call(self, inputs):
        x, skip_connection = inputs
        x = self.up(x)
        x = tf.concat([x, skip_connection], axis=-1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv1': self.conv1,
            'conv2': self.conv2,
            'up': self.up
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        # The output shape is the input shape multiplied by 2 in the spatial dimensions
        return tf.TensorShape([input_shape[0][0], input_shape[0][1] * 2, input_shape[0][2] * 2, self.conv2.filters])
