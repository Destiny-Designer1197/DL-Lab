import os
import matplotlib.pyplot as plt
import gin
import tensorflow as tf
from tensorflow.python.ops.gradients_util import _Inputs

@gin.configurable
def LSTM_for_HO(input_shape,n_classes,dense_units):
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')

def LSTM(input_shape,n_classes):
    """Defines a simple LSTM architecture.
    Parameters:
        input_shape (tuple: 2): (window_size,n_features)
        n_classes (int): number of classes, corresponding to the number of output neurons

    Returns:
        (keras.Model): keras model object
    """
    inputs = tf.keras.Input(input_shape)
    out = tf.keras.layers.LSTM(256, return_sequences=True)(inputs)
    out = tf.keras.layers.Dropout(0.2)(out)
    out = tf.keras.layers.LSTM(128, return_sequences=True)(out)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(out)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name='LSTM')
    
def CNN_LSTM(input_shape,n_classes):
    
    """Defines a CNN-LSTM architecture.
    Parameters:
        input_shape (tuple: 2): (window_size,n_features)
        n_classes (int): number of classes, corresponding to the number of output neurons
        
    Returns:
        (keras.Model): keras model object
    """
    inputs = tf.keras.Input(input_shape)
    conv1 = tf.keras.layers.Conv1D(filters=64,kernel_size=3,activation='relu',padding='same')(inputs)
    conv2 = tf.keras.layers.Conv1D(filters=64,kernel_size=3,activation='relu',padding='same')(conv1)
    drop1 = tf.keras.layers.Dropout(0.2)(conv2)
    pool1 = tf.keras.layers.MaxPooling1D(pool_size=1)(drop1)
    lstm1 = tf.keras.layers.LSTM(128, return_sequences=True)(pool1)
    drop2 = tf.keras.layers.Dropout(0.2)(lstm1)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(drop2)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='CNN_LSTM')

def gated_ru(input_shape,n_classes):
    
    """Defines a Gated Recurrent Unit architecture.
    Parameters:
        input_shape (tuple: 2): (window_size,n_features)
        n_classes (int): number of classes, corresponding to the number of output neurons
        
    Returns:
        (keras.Model): keras model object
    """
    
    inputs = tf.keras.Input(input_shape)
    gru_1 = tf.keras.layers.GRU(256, activation='tanh', return_sequences=True)(inputs)
    gru_2 = tf.keras.layers.GRU(128, activation='tanh', return_sequences=True)(gru_1)
    outputs = tf.keras.layers.Dense(n_classes, activation='softmax')(gru_2)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name='gated_ru')
