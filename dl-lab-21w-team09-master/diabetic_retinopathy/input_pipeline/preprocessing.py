import gin
import tensorflow as tf
import pandas as pd
import numpy as np
import os
import cv2
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from math import ceil
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.gen_parsing_ops import parse_example
import tensorflow_addons as tfa
import math

IMG_SIZE = 256

@gin.configurable
def make_dataset(img_paths_list,labels_list,target_width,target_height):
    """ Make dataset from dataframe

        Parameter:  
        img_paths_list - paths to images in train dataset
        label_list -l abels in train dataset
        target_width - target width of resized images
        shuffle_size - target height of resized images

        Return:
        dataset
    """
    def parse_image(filename):
        image = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(image,channels=3)
        image = tf.image.central_crop(image,central_fraction = 0.95)
        image = tf.image.crop_to_bounding_box(image, offset_height=0, offset_width=0, target_height = 2700, target_width = 3580)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image,[target_width, target_height],antialias=True) # bilinear interpolation returns value between [0,1]
        return image

    filenames_ds = tf.data.Dataset.from_tensor_slices(img_paths_list)
    images_ds = filenames_ds.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.one_hot(labels_list, depth = 2))
    ds = tf.data.Dataset.zip((images_ds, labels_ds))

    return ds

def load_data_in_df(image_dir,label_dir):
    """ Load original data in dataframe

        Parameter:  
        img_dir - path to the images
        label_dir - path to the label.csv file

        Return:
        dataframe
    """
    retina_df = pd.read_csv(label_dir) 
    retina_df['Image_ID'] = retina_df['Image name'].map(lambda x: x.split('_')[1])

    retina_df['path'] = retina_df['Image name'].map(lambda x: os.path.join(image_dir,'{}.jpg'.format(x)))
    retina_df['Retinopathy_grade'] = retina_df['Retinopathy grade']
    retina_df['Risk_of_macular_edema'] = retina_df['Risk of macular edema ']

    retina_df['category'] = retina_df['Retinopathy_grade'].map(lambda x: 1 if (x>1) else 0) 
    retina_df.dropna(axis = 'columns',inplace = True)
    retina_df = retina_df[['Image_ID', 'Retinopathy_grade','Risk_of_macular_edema','category','path']].drop_duplicates()

    def _visualize_original_data(retina_df):
        count = retina_df.groupby(['Retinopathy_grade']).nunique()
        print(count)
        grade = [0,1,2,3,4]
        num_image = count['Image_ID']
        plt.bar(grade,num_image)
        plt.xlabel('Retinopathy grade ')
        plt.ylabel('Number of Images')
        plt.title('Distribution of sevirity on train data and test data')
        plt.legend(['train dataset','test dataset'])
        #plt.show()
        plt.savefig('diabetic_retinopathy/images/label_distribution.jpg')
    
    #_visualize_original_data(retina_df)
    def _visualize_balanced_data(retina_df):
        count = retina_df.groupby(['category']).nunique()
        print(count)
        grade = [0,1]
        num_image = count['Image_ID']
        plt.bar(grade,num_image)
        plt.xlabel('Retinopathy category ')
        plt.ylabel('Number of Images')
        plt.title('Distribution of category on train data and test data')
        plt.legend(['train dataset','test dataset'])
        #plt.show()
        plt.savefig('diabetic_retinopathy/images/label_distribution_after_resampling.jpg')
    
    #_visualize_original_data(retina_df)
    #_visualize_balanced_data(retina_df)
    return retina_df

def load_train_ds_in_df(image_dir,label_dir,train_val_ratio,n_samples,method):
    """ Load training data in dataframe

        Parameter:  
        img_dir - path to the images
        label_dir - path to the label.csv file
        train_val_ratio - train:val ratio 
        n_samples: number of samples for resampling
        method: method deployed to the original dataset

        Return:
        train_df,val_df or original_df
    """
    def _split_trains_ds(df):
        train_ids, valid_ids = train_test_split(df['Image_ID'], test_size = train_val_ratio, random_state = 123, stratify = df['category'])
        train_df = df[df['Image_ID'].isin(train_ids)] # return train dataset
        val_df = df[df['Image_ID'].isin(valid_ids)] # return val dataset
        return train_df,val_df
    
    def _balance_ds(train_df):#balance dataset
        balanced_train_df = train_df.groupby(['category']).apply(lambda x: x.sample(n_samples, random_state = 42,replace = True)).reset_index(drop = True) 
        return balanced_train_df

    original_loaded_df = load_data_in_df(image_dir,label_dir) #load original dataset
    train_df, val_df = _split_trains_ds(original_loaded_df) #split dataset
    balanced_df = _balance_ds(original_loaded_df) # balance dataset

    if method == 'balanced':
        return balanced_df
    elif method == 'split':
        return train_df,val_df
    elif method == 'split_balanced':
        return _balance_ds(train_df),val_df
    else:
        return original_loaded_df

def load_test_ds_in_df(image_dir,label_dir):
    """ Load testing data in dataframe

        Parameter:  
        img_dir - path to the images
        label_dir - path to the label.csv file

        Return:
        test_df
    """
    test_df = load_data_in_df(image_dir,label_dir)
    pd.set_option('display.max_rows',None)
    return test_df

def preprocess_images(path,method_key):
    """ Preprocess images with opencv library

        Parameter:  
        path - path to the images
        method_key - method deployed to the images

        Return:
        image
    
        Code reference:
        https://www.kaggle.com/ratthachat/aptos-eye-preprocessing-in-diabetic-retinopathy
    """
    def _crop_image_from_gray(img,tolerance=7):
        """
        Crop out black borders
        https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
        """  
        if img.ndim == 2:
            mask = img>tolerance
            return img[np.ix_(mask.any(1),mask.any(0))]
        elif img.ndim == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            mask = gray_img>tolerance
            
            check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
            if (check_shape == 0): # image is too dark so that we crop out everything,
                return img # return original image
            else:
                img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
                img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
                img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        #         print(img1.shape,img2.shape,img3.shape)
                img = np.stack([img1,img2,img3],axis=-1)
        #         print(img.shape)

            return img

    def _color_crop(path, sigmaX=10):
        img = cv2.imread(path)
        #print(img.shape)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = _crop_image_from_gray(img)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.addWeighted ( img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
        #print(img.shape)
        
        return img

    def _circle_crop(path,sigmaX=30):
        """
        Create circular crop around image centre   
        https://www.kaggle.com/taindow/pre-processing-train-and-test-images 
        """  
        img = cv2.imread(path)
        img = _crop_image_from_gray(img)

        height, width, depth = img.shape
        largest_side = np.max((height, width))
        img = cv2.resize(img, (largest_side, largest_side))

        height, width, depth = img.shape

        x = int(width / 2)
        y = int(height / 2)
        r = np.amin((x, y))

        circle_img = np.zeros((height, width), np.uint8)
        cv2.circle(circle_img, (x, y), int(r), 1, thickness=-1)
        img = cv2.bitwise_and(img, img, mask=circle_img)
        img = _crop_image_from_gray(img)
        img = cv2.addWeighted (img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)

        return img
    
    if method_key == "color_crop": 
        return _color_crop(path)
    else:
        return _circle_crop(path)

def augment_images(image,label):
    """ Augment images with random transformations

        Parameter:  
        image - original image
        label - label of the image

        Return:
        image, label
    """
    def _random_crop(image):
        shape = tf.shape(image)
        dims_factor = tf.random.uniform([], 0.5, 1.0, dtype=tf.float32)
        height_dim  = tf.multiply(dims_factor, tf.cast(shape[0], tf.float32))
        width_dim   = tf.multiply(dims_factor, tf.cast(shape[1], tf.float32))
        image = tf.image.random_crop(image, [height_dim, width_dim, 3])
        return image
    
    def _random_rotate90(image):
        rotate_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if rotate_prob > .75:
            image = tf.image.rot90(image, k=3) # rotate 270º
        elif rotate_prob > .5:
            image = tf.image.rot90(image, k=2) # rotate 180º
        elif rotate_prob > .25:
            image = tf.image.rot90(image, k=1) # rotate 90º
        return image
    
    def _random_rotate(image, angle=60):
        rotate_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        max_angle   = angle*math.pi/180
        rotation    = tf.random.uniform(shape=[], minval=-max_angle, maxval=max_angle, dtype=tf.float32)
        
        image = tfa.image.rotate(image, rotation, interpolation = "BILINEAR")
        image = tf.image.central_crop(image, central_fraction=0.8)
        return image
    
    def _random_flip(image):
        flip_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
        if flip_prob > 0.75 :
            image = tf.image.transpose(image)
        elif flip_prob > 0.4:
            image = tf.image.random_flip_up_down(image)
        elif flip_prob > 0.3:
            image = tf.image.random_flip_left_right(image)
        return image
    
    def _random_lighting(image):
        image = tf.image.random_brightness(image, 0.2,)#apply random brightness
        image = tf.image.random_contrast(image, 0.6, 1.4,) #apply random contrast
        image = tf.image.random_hue(image, 0.0) #apply random hue
        image = tf.image.random_saturation(image, 0.5, 1.5,)#apply random saturation
        return image

    light_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    crop_prob  = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    flip_prob  = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    rot_prob   = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    rot90_prob = tf.random.uniform([], 0, 1.0, dtype=tf.float32)
    # apply random lightning
    if light_prob >= 0.5:
        image = _random_lighting(image)
        
    # apply random crops
    if crop_prob >= 0.5:
        image = _random_crop(image)      
    
    # apply random flips
    if flip_prob >= 0.5:
        image = _random_flip(image)
    
    # apply random rotations
    if rot_prob >= 0.5:
        image = _random_rotate(image)
    
    # apply random90 degree rotations
    if rot90_prob >= 0.5:
        image = _random_rotate90(image)
    
    image = tf.image.resize(image,[IMG_SIZE,IMG_SIZE])

    return image, label

def create_TFRecord(key,image_dir,df):
    """ 
    Write TFRecord
    """
    def serialize(image_string,label,image_id):
        def _bytes_feature(value): #Returns a bytes_list from a string / byte.
            if isinstance(value, type(tf.constant(0))):
                value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

        def _int64_feature(value):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=[value])) #Returns an int64_list from a bool / enum / int / uint.

        image_shape = tf.io.decode_jpeg(image_string).shape
        #print(image_shape)
        
        feature = {
            'height': _int64_feature(image_shape[0]),
            'width': _int64_feature(image_shape[1]),
            'channel': _int64_feature(image_shape[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_string),
            'image_ID':_int64_feature(image_id)
        }
        # Create a feature message using tf.train.Example
        return tf.train.Example(features = tf.train.Features(feature = feature))

    # Write the raw image files to `images.tfrecords`.
    data_dir = 'diabetic_retinopathy/tfrecords'
    if key == 'train':
        record_file = data_dir + '/train_images.tfrecords'
    elif key == 'test':
        record_file = data_dir + '/test_images.tfrecords'
    else: 
        record_file = data_dir + '/val_images.tfrecords'

    df['path'] = df['Image_ID'].map(lambda x: os.path.join(image_dir,'IDRiD_{}.jpg'.format(x)))

    #print(df)
    with tf.io.TFRecordWriter(record_file) as writer:
        for index,row in df.iterrows():
            filename = row['path']
            label = row['Retinopathy_grade']
            image_id = int(row['Image_ID'])
            image_string = open(filename,'rb').read() 
            tf_example = serialize(image_string,label,image_id) # First, process the two images into `tf.train.Example` messages.
            writer.write(tf_example.SerializeToString()) # Then, write to a `.tfrecords` file.

def read_TFRecord(tfrecord_dir):
    """ 
    Read TFRecord
    """
    #Create a dict describing the features
    image_feature_description = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'channel': tf.io.FixedLenFeature([], tf.int64),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'image_ID': tf.io.FixedLenFeature([], tf.int64),
    }

    def _parse_image_function(example_proto):
        # Parse the input tf.train.Example proto using the dictionary above.
        feature_dict = tf.io.parse_single_example(example_proto, image_feature_description)
        #feature_dict['image_raw'] = tf.io.decode_jpeg(feature_dict['image_raw'])
        feature_dict['image_raw'] = tf.cast(tf.io.decode_jpeg(feature_dict['image_raw']),tf.float32)   
        feature_dict['image_raw'] = tf.image.resize_with_pad(feature_dict['image_raw'],IMG_SIZE, IMG_SIZE) / 255.0 #resize and rescale the images
        return feature_dict
        #return feature_dict['image_raw'], feature_dict['label']

    raw_image_ds = tf.data.TFRecordDataset(tfrecord_dir)
    parsed_image_dataset = raw_image_ds.map(_parse_image_function)

    return parsed_image_dataset