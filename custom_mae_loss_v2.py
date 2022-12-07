import tensorflow as tf

class CustomMAELossV2(tf.keras.losses.Loss):
    """
    """
    def __init__(self, name='custom_mae_loss_v2'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        """
        :param y_true: ground truth image in HSV color space
        :param y_pred: predicted image in HSV color space
        :return:
        """

        # get the saturation channel of the predicted image
        y_pred_sat = y_pred[:, :, :, 1]
        # compute the mean difference between 1 and the saturation channel
        K = tf.reduce_mean(tf.abs(1 - y_pred_sat))
        # compute the MSE loss
        mae_loss = tf.reduce_mean(tf.abs(y_true - y_pred))
        # add the term K to the MSE loss
        return mae_loss * K
