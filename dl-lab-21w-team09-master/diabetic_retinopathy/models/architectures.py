import gin
import tensorflow as tf
import ssl
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

ssl._create_default_https_context = ssl._create_unverified_context
IMG_SIZE = 256
@gin.configurable
def ResNet_50(input_shape, n_classes):
#def UNet(input_shape, n_classes, base_filters, n_blocks, dense_units, dropout_rate):
    """Defines an architecture combined with pretrained nmodel ResNet50
    Parameters:
        input_shape (tuple: 3): input shape of the neural network
        n_classes (int): number of classes
    Returns:
        (keras.Model): keras model object
    """
    #assert n_blocks > 0, 'Number of blocks has to be at least 1.'
    inputs = tf.keras.Input(input_shape)
    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs,input_shape = (256,256,3))
    base_model.trainable = False
    out = base_model.output
    out = GlobalAveragePooling2D()(out)
    out = Dense(1024, activation ='relu')(out)
    outputs = Dense(n_classes,activation='softmax')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='ResNet50')