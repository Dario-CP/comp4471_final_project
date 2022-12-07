"""
U-net model for HSV image colorization
"""

import tensorflow as tf
from encoder_block import EncoderBlock
from decoder_block import DecoderBlock

class UNet(tf.keras.Model):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder1 = EncoderBlock(filters=64)
        self.encoder2 = EncoderBlock(filters=128)
        self.encoder3 = EncoderBlock(filters=256)
        self.encoder4 = EncoderBlock(filters=512)

        # The last encoder block has no max pooling
        self.encoder5 = EncoderBlock(filters=1024, max_pool=False)

        self.decoder1 = DecoderBlock(filters=512)
        self.decoder2 = DecoderBlock(filters=256)
        self.decoder3 = DecoderBlock(filters=128)
        self.decoder4 = DecoderBlock(filters=64)

        self.conv = tf.keras.layers.Conv2D(filters=2, kernel_size=(1, 1), strides=(1, 1), padding='same')

    def call(self, inputs):
        x1, skip1 = self.encoder1(inputs)
        x2, skip2 = self.encoder2(x1)
        x3, skip3 = self.encoder3(x2)
        x4, skip4 = self.encoder4(x3)

        x5, skip5 = self.encoder5(x4)

        x = self.decoder1((x5, skip4))
        x = self.decoder2((x, skip3))
        x = self.decoder3((x, skip2))
        x = self.decoder4((x, skip1))

        x = self.conv(x)

        # Concatenate the input (channel V) to the output (channels H and S)
        x = tf.concat([x, inputs], axis=-1)

        return x

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'encoder1': self.encoder1,
            'encoder2': self.encoder2,
            'encoder3': self.encoder3,
            'encoder4': self.encoder4,
            'encoder5': self.encoder5,
            'decoder1': self.decoder1,
            'decoder2': self.decoder2,
            'decoder3': self.decoder3,
            'decoder4': self.decoder4,
            'conv': self.conv
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0], input_shape[1], input_shape[2], 3])

    # We need this method to be able to plot the summary of the model
    def build_graph(self):
        x = tf.keras.Input(shape=(256, 256, 1)) # Change 256 to None if you want to use any size
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
