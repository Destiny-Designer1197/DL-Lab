from matplotlib import figure
import tensorflow as tf
import io
import glob
import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
import seaborn as sns
import matplotlib.pyplot as plt
import shutil
from tensorflow.python.ops.numpy_ops import np_config

class ConfusionMatrix(tf.keras.metrics.Metric):

    def __init__(self,n_classes, name="confusion_matrix_metric" , **kwargs):
        super(ConfusionMatrix, self).__init__(name=name, **kwargs)
        print("Confusion Matrix: INIT")
        self.n_classes = n_classes
        self.total_cm = self.add_weight("total", shape=(n_classes,n_classes), initializer="zeros")

    def reset_states(self):
        print("Confusion Matrix: RESET")
        for s in self.variables:
            s.assign(tf.zeros(shape=s.shape))

    def update_state(self, labels, predictions, sample_weight = None):
        """
        Update the confusion matrix
        """
        print("Confusion Matrix: UPDATE ")
        predictions=tf.argmax(predictions,2)
        labels = tf.argmax(labels,2)
        
        reshaped_predictions = tf.reshape(predictions,[-1])
        reshaped_labels = tf.reshape(labels,[-1])
        self.total_cm.assign_add(self.make_confusion_matrix(reshaped_labels,reshaped_predictions))
        return self.total_cm
    
    def result(self):
        print("Confusion Matrix: RESULT")
        return self.total_cm.numpy().astype(np.int32)

    def make_confusion_matrix(self,labels, predictions):
        """
        Make a confusion matrix
        """
        cm=tf.math.confusion_matrix(labels,predictions, dtype=tf.float32, num_classes=self.n_classes)
        return cm
    
    def plot_conf_mat(self,issaved):
        """
        Plot confusion matrix
        """

        def _plot_to_image(figure):
            buffer = io.BytesIO()
            plt.savefig(buffer,format ="png")
            plt.close(figure)
            buffer.seek(0)
            image = tf.image.decode_png(buffer.getvalue(),channels=3)
            image = tf.expand_dims(image,0)
            return image

        matrix = self.total_cm.numpy().astype(np.int32)
        df_cm = pd.DataFrame(
            matrix,
            index = np.arange(start=1, stop=self.n_classes+1),
            columns= np.arange(1,self.n_classes+1)
        )
        plt.figure(figsize=(20,20))
        ticklabels = ['1-Walking','2-Walking_Upstairs','3-Walking_Downstairs','4-SITTING','5-STANDING','6-LAYING','7-STAND_TO_SIT','8-SIT_TO_STAND','9-SIT_TO_LIE','10-LIE_TO_SIT','11-STAND_TO_LIE','12-LIE_TO_STAND']
        sn_plot= sns.heatmap(df_cm,annot = True, fmt= "d",cbar=True,xticklabels=ticklabels,yticklabels=ticklabels)
        #sn_plot= sns.heatmap(df_cm,annot = True, fmt= "d",cbar=True)
        plt.title("Confusion Matrix")
        plt.xlabel("Predictions")
        plt.ylabel("labels")
        image = sn_plot.get_figure()
        if issaved == False:
            image= _plot_to_image(image)
        else:
            image.savefig('human_activity_recognition/images/evaluation_confusion_matrix.png')

        return image
    
 
