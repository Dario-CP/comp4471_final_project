import tensorflow as tf

class CustomMSELoss(tf.keras.losses.Loss):
    """
    Custom loss that adds a term K to the MSE loss,
    where K is the mean difference between 1 and the saturation channel of the predicted image.
    K is multiplied by a hyperparameter alpha that controls the importance of the term K.
    """
    def __init__(self, name='custom_mse_loss', alpha=1.0):
        super().__init__(name=name)
        self.alpha = alpha

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
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        # add the term K to the MSE loss
        return mse_loss + K * self.alpha
