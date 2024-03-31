import gin
import logging
import tensorflow as tf
from tensorflow._api.v2 import data
import tensorflow_datasets as tfds
from input_pipeline.preprocessing import read_TFRecord

data_dir = 'human_activity_recognition/tfrecords'

ds_train_tfrecord= data_dir +'/train.tfrecord'
ds_test_tfrecord = data_dir +'/test.tfrecord'
ds_val_tfrecord = data_dir +'/val.tfrecord'

balanced_ds_train_tfrecord = data_dir + '/balanced_train.tfrecord'
balanced_ds_test_tfrecord= data_dir + '/balanced_test.tfrecord'
balanced_ds_val_tfrecord = data_dir + '/balanced_val.tfrecord'

@gin.configurable
def load(name,balanced):
        """ Load dataset
         name - name of dataset
         balanced - if dataset is balanced
        """
        if name == "hapt":
                logging.info(f"Preparing dataset {name}...")
                if balanced:
                        ds_train= read_TFRecord(balanced_ds_train_tfrecord)
                        ds_test = read_TFRecord(balanced_ds_test_tfrecord)
                        ds_val = read_TFRecord(balanced_ds_val_tfrecord)
                else: 
                        ds_train= read_TFRecord(ds_train_tfrecord)
                        ds_test = read_TFRecord(ds_test_tfrecord)
                        ds_val = read_TFRecord(ds_val_tfrecord)
                
                #return ds_train, ds_test, ds_val
                return prepare(ds_train,ds_val,ds_test)

        else:
                raise ValueError

@gin.configurable
def prepare(ds_train,ds_val,ds_test,batch_size,shuffle_size,caching):
        """ Prepare dataset

        Parameter:  
        ds_train, ds_val, ds_test: train dataset, validation dataset, test dataset
        batch_size: batch size of the dataset
        shuffle_size: buffer size of shuffling
        caching: if the dataste is cached

        Return:
        ds_train, ds_val, ds_test
        """
        # Prepare training dataset
        ds_train = ds_train.shuffle(shuffle_size)
        ds_train = ds_train.batch(batch_size)
        ds_train = ds_train.repeat(-1)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

        # Prepare validation dataset
        ds_val = ds_val.batch(batch_size)
        if caching:
                ds_val = ds_val.cache()
        ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

        # Prepare test dataset
        ds_test = ds_test.batch(batch_size)
        if caching:
                ds_test = ds_test.cache()
        ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

        return ds_train,ds_test,ds_val
