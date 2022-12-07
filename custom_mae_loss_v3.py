import tensorflow as tf

class CustomMAELossV3(tf.keras.losses.Loss):
    """
    Custom loss that calculates the MAE loss for each channel of the predicted image,
    and gives more weight to the saturation channel by multiplying the MAE loss
    of the saturation channel by a hyperparameter alpha.
    """
    def __init__(self, name='custom_mae_loss', alpha=1.0):
        super().__init__(name=name)
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """
        :param y_true: ground truth image in HSV color space
        :param y_pred: predicted image in HSV color space
        :return:
        """

        # compute the MAE loss separately for each channel of the predicted image
        mae_loss_hue = tf.reduce_mean(tf.abs(y_true[:, :, :, 0] - y_pred[:, :, :, 0]))
        mae_loss_sat = tf.reduce_mean(tf.abs(y_true[:, :, :, 1] - y_pred[:, :, :, 1]))
        mae_loss_val = tf.reduce_mean(tf.abs(y_true[:, :, :, 2] - y_pred[:, :, :, 2]))

        # Return the average of the MAE losses, with more weight to the saturation channel
        return (mae_loss_hue + mae_loss_sat * self.alpha + mae_loss_val) / 3
