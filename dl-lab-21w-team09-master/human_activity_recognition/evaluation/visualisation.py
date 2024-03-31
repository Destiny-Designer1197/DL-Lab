from cProfile import label
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from input_pipeline.preprocessing import load_data_in_df,delete_unlabeled_data
import tensorflow as tf

def visualize_triaxial_signals(predictions,user_id,exp_id,plot_width,plot_height):# pass one dict key, which belongs to the test dataset
    """ Visualize a sequence from the test dataste

    Parameters: 
    predictions - predicted results from model
    user_id - user nr.
    exp_id - exp nr.
    plot_width, plot_height: size of plot
    """
    def _map_bar_color(columns):
        colors = {
            1: 'bisque',
            2: 'lavender',
            3: 'burlywood',
            4: 'tan',
            5: 'moccasin',
            6: 'thistle',
            7: 'gold',
            8: 'khaki',
            9: 'darkkhaki',
            10: 'yellowgreen',
            11: 'palegreen',
            12: 'lightskyblue'
        }
        #vis_df['label'] = vis_df['label'].map(lambda x:x+1)
        return columns.map(colors)

    raw_data_paths = 'human_activity_recognition/HAPT Data Set/RawData/*'
    label_df,data_dict = load_data_in_df(raw_data_paths)
    labelled_data_dict = delete_unlabeled_data(label_df,data_dict)

    key = "exp"+str(exp_id).zfill(2)+"_"+"user"+str(user_id).zfill(2) # make key for the search in the dict
    vis_df = labelled_data_dict[key]
    vis_df['label'] = vis_df['label'].map(lambda x:x+1)

    # x-axis for all plots, nr. of the samples
    nr_samples = [j for j in range(len(vis_df))] 
    # Define the figure and setting dimensions width and height
    fig_acc = plt.figure(figsize=(plot_width,plot_height))

#-------------------------------------------first plot: accelorometer data of (exp44_user22)------------------------------------------------#
    # acceleration columns are the first 3 columns acc_X, acc_Y and acc_Z
    ax_X = vis_df['acc_X'] # copy acc_X
    ax_Y = vis_df['acc_Y'] # copy acc_Y
    ax_Z = vis_df['acc_Z'] # copy acc_Z
        
    figure_Ylabel_acc ='Acceleration in 1g' # the figure y axis info
    title_acc = "acceleration signals for all activities performed by user "+ str(user_id) +' in experience '+str(exp_id) # set the title in each case

    plt.subplot(4,1,1)
    plt.bar(nr_samples,height=6,width =1,bottom = -2,color = _map_bar_color(vis_df['label']))
    # ploting each signal component
    _ =plt.plot(nr_samples,ax_X,color='r',label='acc_X')
    _ =plt.plot(nr_samples,ax_Y,color='b',label='acc_Y')
    _ =plt.plot(nr_samples,ax_Z,color='g',label='acc_Z')

    # Set the figure info defined earlier
    _ = plt.ylabel(figure_Ylabel_acc) # set Y axis info 
    _ = plt.xlabel('Index of the labelled samples') # Set X axis info (same label in all cases)
    _ = plt.title(title_acc) # Set the title of the figure

    # localise the figure's legends
    _ = plt.legend(loc="upper left")# upper left corner

#-------------------------------------------second plot: gyroscope data (exp44_user22)----------------------------------------------------#
    gy_X = vis_df['gyro_X']
    gy_Y = vis_df['gyro_Y']
    gy_Z = vis_df['gyro_Z']

    figure_Ylabel_gyro = 'Angular Velocity in radian per second [rad/s]' # the figure y axis info
    title_gyro = "gyroscope signals for all activities performed by user "+ str(user_id) +' in experience '+str(exp_id) # set the title in each case

    plt.subplot(4,1,2)
    plt.bar(nr_samples,height=8,width =1,bottom = -4,color = _map_bar_color(vis_df['label']))
    # ploting each signal component
    __ = plt.plot(nr_samples,gy_X,color='r',label='gyro_X')
    __ = plt.plot(nr_samples,gy_Y,color='b',label='gyro_Y')
    __ = plt.plot(nr_samples,gy_Z,color='g',label='gyro_Z')

    # Set the figure info defined earlier
    __ = plt.ylabel(figure_Ylabel_gyro) # set Y axis info 
    __ = plt.xlabel('Index of the labelled samples') # Set X axis info (same label in all cases)
    __ = plt.title(title_gyro) # Set the title of the figure

    # localise the figure's legends
    __ = plt.legend(loc="upper left")# upper left corner


#-----------------third plot: predictions of a selected sequence (exp44_user22)from ds_test ---------------------------#
    plt.subplot(4,1,3)
    predictions = tf.argmax(predictions,2)
    reshaped_predictions = tf.reshape(predictions,[-1])
    pred_arr = reshaped_predictions.numpy() + 1 #restore labels
    sliced_pred_arr = pred_arr[0:len(vis_df)] 
    label_data = {'label': sliced_pred_arr}
    loaded_df = pd.DataFrame(label_data)
    plt.bar(nr_samples,height=8,width =1,color = _map_bar_color(loaded_df['label']))
    plt.title('Prediction for the sequence exp44_user22 in ds_test')
    plt.xlabel('Index of the labelled samples')

#-----------------------------------------------fourth plot: colorbar----------------------------------------------------#
    plt.subplot(4,1,4)
    data = {'label_name': [1,2,3,4,5,6,7,8,9,10,11,12]}
    df = pd.DataFrame(data)
    plt.xticks(df['label_name'])
    plt.bar(df['label_name'],height=1,width = 1,color = _map_bar_color(df['label_name']))
    plt.title('Color Bar')

#-----------------------------------------------SUMMARY----------------------------------------------------#
    plt.tight_layout()
    plt.savefig('human_activity_recognition/images/visualisation_of_single_activity.png')