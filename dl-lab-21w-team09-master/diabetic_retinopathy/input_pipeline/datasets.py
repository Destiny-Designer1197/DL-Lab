from cgi import test
from random import shuffle
import gin
import os
import logging
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd 
from input_pipeline.preprocessing import read_TFRecord,augment_images,make_dataset,load_train_ds_in_df,load_test_ds_in_df
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 256

@gin.configurable
def load(name,image_train_dir,label_train_dir,image_test_dir,label_test_dir,batch_size,shuffle_size,caching):
        """ Load dataset

        Parameter:  
        name: name of the dataset
        image_train_dir: relative path to the train image dataset
        label_train_dir: relative path to the label.csv
        image_test_dir: relative path to the test image dataset
        label_test_dir: relative path to the label.csv
        batch_size: batch size of the dataset
        shuffle_size: buffer size of shuffling
        caching: if the dataste is cached

        Return:
        ds_train, ds_val, ds_test
        """
        if name == "idrid":
                logging.info(f"Preparing dataset {name}...")

                train_img_path_list = []
                test_img_path_list =[]
                val_img_path_list= []

                train_label_list = []
                test_label_list = []
                val_label_list = []

                def _list_ds_path(df,img_path_list,label_list):
                        for index,row in df.iterrows():
                                img_path_list.append(str(row['path']))
                                label_list.append(int(row['category']))
                        return img_path_list,label_list

                # load, split and balance data in train_df, val_df, test_df
                train_df,val_df = load_train_ds_in_df(image_train_dir,label_train_dir,train_val_ratio=0.2,n_samples=150,method='split_balanced')
                test_df = load_test_ds_in_df(image_test_dir,label_test_dir)
                #print('test_df', test_df)

                # create path lists for ds_train, ds_val, ds_test
                train_image_path_list,train_label_list = _list_ds_path(train_df,train_img_path_list,train_label_list)
                val_image_path_list,val_label_list = _list_ds_path(val_df, val_img_path_list,val_label_list)
                test_image_path_list,test_label_list = _list_ds_path(test_df,test_img_path_list,test_label_list)

                # make data set
                ds_train = make_dataset(train_image_path_list,train_label_list)
                ds_val = make_dataset(val_image_path_list,val_label_list)
                ds_test = make_dataset(test_image_path_list,test_label_list)

                return prepare(ds_train,ds_val,ds_test,batch_size,shuffle_size,caching)
        else:
                raise ValueError

@gin.configurable
def prepare(ds_train, ds_val, ds_test, batch_size, shuffle_size, caching):
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
        ds_train = ds_train.map(augment_images,num_parallel_calls=tf.data.experimental.AUTOTUNE)
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

        return ds_train,ds_val,ds_test