import gin
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

OVERLAP_PERCENT = 0.5 #overlap percentage
SAMPLING_FREQ = 50 # 50Hz 
SAMPLIN_TIME_WINDOW = 5 # 5s
WINDOW_SIZE = SAMPLIN_TIME_WINDOW * SAMPLING_FREQ # n_samples
N_CLASSES = 12

keylist_train = ['exp01_user01','exp02_user01', 'exp03_user02', 'exp04_user02', 'exp05_user03', 'exp06_user03'
                        ,'exp07_user04', 'exp08_user04', 'exp09_user05', 'exp10_user05', 'exp11_user06', 'exp12_user06'
                        ,'exp13_user07', 'exp14_user07', 'exp15_user08', 'exp16_user08', 'exp17_user09', 'exp18_user09'
                        ,'exp19_user10', 'exp20_user10', 'exp21_user10', 'exp22_user11', 'exp23_user11', 'exp24_user12'
                        ,'exp25_user12', 'exp26_user13', 'exp27_user13', 'exp28_user14', 'exp29_user14', 'exp30_user15'
                        ,'exp31_user15', 'exp32_user16', 'exp33_user16', 'exp34_user17', 'exp35_user17', 'exp36_user18'
                        ,'exp37_user18', 'exp38_user19', 'exp39_user19', 'exp40_user20', 'exp41_user20', 'exp42_user21', 'exp43_user21']

keylist_test = ['exp44_user22', 'exp45_user22', 'exp46_user23', 'exp47_user23', 'exp48_user24', 'exp49_user24'
                        ,'exp50_user25', 'exp51_user25', 'exp52_user26', 'exp53_user26', 'exp54_user27', 'exp55_user27']

keylist_val = ['exp56_user28', 'exp57_user28', 'exp58_user29', 'exp59_user29', 'exp60_user30', 'exp61_user30']

def load_data_in_df(data_dir):
        """ Load data in dataframe

        Parameters:
        data_dir: path to raw data

        Return:
        dataframe
        """
        # all data paths listed according to the name
        sorted_raw_data_paths = sorted(glob.glob(data_dir+'*.txt'))

        def _load_data(data_dir,columns):
                processedList = []
                with open(data_dir,'r') as file:
                        for i, line in enumerate(file):
                                try:
                                        line = line.split() #delete whitespaces in-between
                                        last = line[2].strip() #delete all tailing whitespaces 
                                        if last == '':
                                                break
                                except:
                                        print('Error at line number: ', i)
                                temp = [float(line[0]), float(line[1]), float(last)] #store acc_X, acc_Y,acc_Z from each row
                                processedList.append(temp)

                processedArray = np.array(processedList) #transfer list to array for data processing
                temp_df= pd.DataFrame(data = processedArray, columns = columns)
                return temp_df
        
        def _load_label(data_dir,columns):
                processedList = []
                with open(data_dir,'r') as file:
                        for i, line in enumerate(file):
                                try:
                                        line = line.split() #delete whitespaces in-between
                                        start_time_point = line[3]
                                        end_time_point = line[4].strip() #delete all tailing whitespaces 
                                except:
                                        print('Error at line number: ', i)
                                temp = [int(line[0]), int(line[1]),int(line[2]),int(start_time_point),int(end_time_point)]
                                processedList.append(temp)
                
                processedArray = np.array(processedList)
                temp_df= pd.DataFrame(data = processedArray, columns = columns)
                return temp_df

        raw_data_dict = {}
        
        acc_columns = ['acc_X','acc_Y','acc_Z']
        gyro_columns = ['gyro_X','gyro_Y','gyro_Z']
        label_columns = ['exp_id','user_id','activity_id','start_time_point','end_time_point']
        
        for path_index in range(0,61):
                #key: expxx_userxx, e.g. exp01_user01, extracted from the string of the filename
                key = sorted_raw_data_paths[path_index][-16:-4]
                #print(key)
                raw_acc_df = _load_data(sorted_raw_data_paths[path_index],acc_columns)
                raw_gyro_df = _load_data(sorted_raw_data_paths[path_index+61],gyro_columns)
                raw_df = pd.concat([raw_acc_df, raw_gyro_df], axis=1)# concatenate dataframes 
                raw_data_dict[key] = raw_df  
        
        label_df = _load_label(sorted_raw_data_paths[122],label_columns)

        return label_df,raw_data_dict

def delete_unlabeled_data(label_df,raw_data_dict):
        """Delete all unlabelled data and combine two data frames
        keyword arguments:
        label_df -- dataframe of all label data
        raw_data_dic -- dictionary of all 6-channel data (in total: 61 items -> 61 experiments)
        """
        new_data_dict = {}

        exp_id = label_df['exp_id'].astype(str).str.zfill(2)
        user_id = label_df['user_id'].astype(str).str.zfill(2)
        activity_id = label_df['activity_id']
        start_time_point = label_df['start_time_point']
        end_time_point = label_df['end_time_point']
        rows_per_df = [len(raw_data_dict[key]) for key in sorted(raw_data_dict.keys())] # how many rows in each experiment

        # build the keys for searching in the dict
        keylist = []
        for i in range(len(exp_id)):
                key = "exp"+exp_id[i]+"_"+"user"+user_id[i] # make the string keys in form of expxx_userxx
                keylist.append(key) # make a key list, 1214 items in this keylist -> since 1214 rows in labels.txt

        cut_length = 0 # deleted rows according to the label
        cut_start = 0  # from which row will the data deleted
        j = 0 #index j for rows_per_df list
        
        for i in range(len(keylist)):
                # e.g. in labels.txt in hapt data set, first row: 1 1 5 250 1232
                # temp_key = keylist[0] = exp01_user01, temp_df stores all 6-channel data from exp01_user01
                # tmep_cut_start = 0, temp_cut_end = 250, temp_cut_length = 250-0 = 250
                # temp_act_start = start_time_point[0] = 250, temp_act_end = end_time_point = 1232 
                # activity_id = 5 
                temp_key = keylist[i]
                temp_df = raw_data_dict.get(temp_key) # get the value with the key, the value is the dataframe for each experiment

                temp_cut_start = cut_start # start point of the cut -> from this point no following action  
                temp_cut_end = start_time_point[i] - cut_length # end point of the cut -> from this point user has action (with considertion of the amount of the deleted rows)
                temp_cut_length = temp_cut_end - temp_cut_start 

                temp_df.drop(temp_df.index[temp_cut_start:temp_cut_end],inplace = True) # delete unlabelled rows in exp01_user01 dataframe

                temp_df.loc[start_time_point[i]:end_time_point[i],'label'] = activity_id[i]-1

                cut_length += temp_cut_length
                cut_start = end_time_point[i] + 1 - cut_length

                if(i != len(keylist)-1):
                        if(temp_key!= keylist[i+1]): # finish reading the same keys from the keylist
                        # e.g. the 22th row: 1 1 2 17298 17970 (for exp01_user01). the 23th row:  2 1 5 251 1226 (for exp02_user01)
                                temp_df.drop(temp_df.index[cut_start:(rows_per_df[j]-cut_length)],inplace = True)
                                new_data_dict[temp_key] = temp_df # store the changed dataframe into a new dict with the key expxx_userxx
                                cut_start = 0 # reste cut start
                                cut_length = 0
                                j += 1 
                        else:
                                continue
                else: # finish reading the keylist and store the changed 61th dataframe into the new dictionary
                                temp_df.drop(temp_df.index[cut_start:(rows_per_df[j]-cut_length)],inplace = True)
                                new_data_dict[temp_key] = temp_df 

        return new_data_dict


def balance_ds(data_dict):
        """
        Balance dataset
        """
        balance_data_dict = {}

        for key in data_dict.keys():
                temp_balanced_df = pd.DataFrame()
                temp_df = data_dict[key]
                temp_counts = temp_df['label'].value_counts()
                # print('.....',key,'.....')
                # print(temp_counts)

                min_count = temp_counts.min()

                a1 = temp_df[temp_df['label']==1].head(min_count).copy()
                a2 = temp_df[temp_df['label']==2].head(min_count).copy()
                a3 = temp_df[temp_df['label']==3].head(min_count).copy()
                a4 = temp_df[temp_df['label']==4].head(min_count).copy()
                a5 = temp_df[temp_df['label']==5].head(min_count).copy()
                a6 = temp_df[temp_df['label']==6].head(min_count).copy()
                a7 = temp_df[temp_df['label']==7].head(min_count).copy()
                a8 = temp_df[temp_df['label']==8].head(min_count).copy()
                a9 = temp_df[temp_df['label']==9].head(min_count).copy()
                a10 = temp_df[temp_df['label']==10].head(min_count).copy()
                a11 = temp_df[temp_df['label']==11].head(min_count).copy()
                a12 = temp_df[temp_df['label']==12].head(min_count).copy()

                temp_balanced_df = temp_balanced_df.append([a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12])
                pd.set_option('display.max_rows',None)
                #print(key,min_count)
                #print(temp_balanced_df,temp_balanced_df.shape)
                balance_data_dict[key] = temp_balanced_df

        return balance_data_dict
 
def normalization(data_dict):
        """Normalize input channels
        keyword arguments:
        data_dic -- the preprocessed data after the function preprocessing_ds()
        """
        def _zscore_normlization(column_values):
            new_column_values = (column_values-np.mean(column_values))/np.std(column_values)
            return new_column_values

        normalized_new_data_dic = {}

        for key in data_dict.keys():
                temp_df = data_dict[key]
                cols = list(temp_df.columns)
                cols.remove('label')
                for col in cols:
                        temp_df[col] = _zscore_normlization(temp_df[col]) 

                normalized_new_data_dic[key] = temp_df

                #pd.set_option('display.max_rows',None)
                #print(key)
                #print(temp_df)

        return normalized_new_data_dic

def make_sliding_window(data_dict):
        """Create 6+1 feature dicts with sliding window
        keyword arguments:
        data_dic -- the preprocessed data after the function normalization()
        """
        acc_x_windows = {}
        acc_y_windows = {}
        acc_z_windows = {}
        gyro_x_windows = {}
        gyro_y_windows = {}
        gyro_z_windows = {}
        label_windows = {}

        def _sliding_window(dataset,window_size,overlap_percentage):
                if overlap_percentage != 0: 
                        dataset = dataset.window(window_size,int(window_size*overlap_percentage),drop_remainder = True) #Returns a list consisting of windows.
                else:
                        dataset = dataset.window(window_size,drop_remainder = True)

                # maps map func across this dataset and flatterns the result
                dataset = dataset.flat_map(lambda window: window.batch(window_size))
                # retun the list of windows
                window_list = list(dataset.as_numpy_iterator())
                return window_list
                
        def _get_windows_size(feature_dict):
                for key in feature_dict.keys():
                    print(key,len(feature_dict[key]))
        
        def _make_window_dict(str,temp_df,istrain):
                feature_df = temp_df[str] 
                if str == 'label':
                        temp_df_tensor = tf.constant(feature_df,tf.int64) # convert each column to a tensor, this step is not necessary
                else:
                        temp_df_tensor = tf.constant(feature_df,tf.float64)
                temp_df_tensor_to_ds= tf.data.Dataset.from_tensor_slices((temp_df_tensor)) # create a Dataset for each column
                if istrain == True:
                        temp_feature_windows_ds = _sliding_window(temp_df_tensor_to_ds,WINDOW_SIZE,OVERLAP_PERCENT) # create a Dataset of "windows" for each column
                else:
                        temp_feature_windows_ds = _sliding_window(temp_df_tensor_to_ds,WINDOW_SIZE,0)
                return temp_feature_windows_ds 
        
        def _make_feature_dict_for_ds(data_dict,is_train_ids):
                for key in data_dict.keys():
                        temp_df = data_dict[key]
                        # save the dataset of "windows" into dictionaries according to the key exp01_user01
                        # e.g. in dictionary acc_x_windows, 
                        # key = 'exp01_user01', values -> a list consisting of 110 windows, each window is in window_size(250)
                        acc_x_windows[key] = _make_window_dict('acc_X',temp_df,is_train_ids)
                        acc_y_windows[key] = _make_window_dict('acc_Y',temp_df,is_train_ids)
                        acc_z_windows[key] = _make_window_dict('acc_Z',temp_df,is_train_ids)
                        gyro_x_windows[key] = _make_window_dict('gyro_X',temp_df,is_train_ids)
                        gyro_y_windows[key] = _make_window_dict('gyro_Y',temp_df,is_train_ids)
                        gyro_z_windows[key] = _make_window_dict('gyro_Z',temp_df,is_train_ids)
                        label_windows[key] = _make_window_dict('label',temp_df,is_train_ids)

        train_dict = {key:data_dict[key] for key in keylist_train}
        test_dict = {key:data_dict[key] for key in keylist_test}
        val_dict = {key:data_dict[key] for key in keylist_val}

        _make_feature_dict_for_ds(train_dict,True)
        _make_feature_dict_for_ds(test_dict,False)
        _make_feature_dict_for_ds(val_dict,False)

        #_get_windows_size(acc_x_windows)

        return acc_x_windows,acc_y_windows,acc_z_windows,gyro_x_windows,gyro_y_windows,gyro_z_windows,label_windows

def create_TFRecord(acc_x_dict,acc_y_dict,acc_z_dict,gyro_x_dict,gyro_y_dict,gyro_z_dict,lable_dict):
        """Create tfrecords
        Parameters:
        the feature dictionary storing sliding windows for each features according to the key expxx_userxx
        Return: 
        none, tfrecords created
        """
        data_dir = 'human_activity_recognition/tfrecords'
        # ds_train_tfrecord = data_dir + '/balanced_train.tfrecord'
        # ds_test_tfrecord= data_dir + '/balanced_test.tfrecord'
        # ds_val_tfrecord = data_dir + '/balanced_val.tfrecord'

        ds_train_tfrecord = data_dir + '/train.tfrecord'
        ds_test_tfrecord= data_dir + '/test.tfrecord'
        ds_val_tfrecord = data_dir + '/val.tfrecord'

        def _write_tfrecord(writer,key):
            for i in range(len(acc_x_dict[key])):

                f1 = acc_x_dict[key][i]
                f2 = acc_y_dict[key][i]
                f3 = acc_z_dict[key][i]
                f4 = gyro_x_dict[key][i]
                f5 = gyro_y_dict[key][i]
                f6 = gyro_z_dict[key][i]
                label = lable_dict[key][i]

                example = serialize(f1,f2,f3,f4,f5,f6,label)
                writer.write(example)
        
        # write ds_train in tfrecord
        with tf.io.TFRecordWriter(ds_train_tfrecord) as writer:
                for key in keylist_train:
                    _write_tfrecord(writer,key)
        
        #write ds_test in tfrecord
        with tf.io.TFRecordWriter(ds_test_tfrecord) as writer:
                for key in keylist_test:
                    _write_tfrecord(writer,key)

        #write ds_val in tfrecord
        with tf.io.TFRecordWriter(ds_val_tfrecord) as writer:
                for key in keylist_val:
                    _write_tfrecord(writer,key)

def serialize(f1,f2,f3,f4,f5,f6,label):
    """Creates a tf.train.Example message ready to be written to a file.

    Keyword arguments:
    numeric_feature_names: acc_X,acc_Y,acc_Z
    label: 
    """
    def _float_array_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value)) #Returns a float_list from a float / double.

    def _int64_array_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value)) #Returns an int64_list from a bool / enum / int / uint.

    feature = {
        'acc_X':_float_array_feature(f1),
        'acc_Y':_float_array_feature(f2),
        'acc_Z':_float_array_feature(f3),
        'gyro_X':_float_array_feature(f4),
        'gyro_Y':_float_array_feature(f5),
        'gyro_Z':_float_array_feature(f6),
        'label': _int64_array_feature(label)
    }
    # Create a feature message using tf.train.Example
    example = tf.train.Example(features = tf.train.Features(feature = feature))
    return example.SerializeToString()

def read_TFRecord(tfrecord_dir):
        # define feature structure and decode features
        def _parse_func(example):
                feature_description = {
                        'acc_X': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.float32),
                        'acc_Y': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.float32),
                        'acc_Z': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.float32),
                        'gyro_X': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.float32),
                        'gyro_Y': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.float32),
                        'gyro_Z': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.float32),
                        'label': tf.io.FixedLenFeature([WINDOW_SIZE,],tf.int64)
                }
                features = tf.io.parse_single_example(example,feature_description)
  
                feature = tf.transpose([features['acc_X'],features['acc_Y'],features['acc_Z'],features['gyro_X'],features['gyro_Y'],features['gyro_Z']])
                #label = tf.transpose([features['label']])
                label =  features['label']
                one_hot_encoded_label = tf.one_hot(label,N_CLASSES)
                #feature = tf.reshape(feature, [250,6,1])
                #one_hot_encoded_label= tf.reshape((features['label']),[250,1])
                #pd.set_option('display.max_rows',None)
                #print(one_hot_encoded_label)
                #return(feature,label)
                return (feature,one_hot_encoded_label)
        
        dataset = tf.data.TFRecordDataset(tfrecord_dir)
        dataset = dataset.map(_parse_func)

        return dataset