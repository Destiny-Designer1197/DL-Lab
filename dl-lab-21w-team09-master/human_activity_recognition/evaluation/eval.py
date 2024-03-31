import logging
from numpy import argmax
from tensorflow.python.keras.utils.generic_utils import Progbar
from tqdm import tqdm
import os
import tensorflow as tf
import numpy as np
from evaluation.metrics import ConfusionMatrix
from evaluation.visualisation import visualize_triaxial_signals
import matplotlib.pyplot as plt
import scipy.misc
import PIL 
from datetime import datetime

def evaluate(model, checkpoint, ds_test, run_paths):
    """ Evaluate the trained model

    Parameters:
        model: model 
        checkpoints: checkpoints stored the exact values obtained from the training
        ds_test: test dataset
        run_paths: path stored checkpoints

    Note:
    1. Visualisation of a sequence from test dataset.
    2. Best test accuracy and average test accuracy are computed.

    """
    # load checkpoints
    loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')
    manager = tf.train.CheckpointManager(checkpoint=checkpoint, max_to_keep=3, directory=run_paths["path_ckpts_train"])
    checkpoint.restore(tf.train.latest_checkpoint(run_paths['path_ckpts_train'])).expect_partial()
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    
    # tesnsorboard summary writers
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    test_log_dir = 'human_activity_recognition/logs/gradient_tape/' + current_time + '/test'
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)
    
    # initiate a confusion matrix
    confusion_matrix = ConfusionMatrix(n_classes=12)

    def _test_step(images, labels):
        """
        evaluate a single step for given images and labels
        """
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)
        return predictions

    with tqdm(total=len(list(ds_test)), desc= 'Evaluation') as pbar:
        best_test_accuracy = 0
        average_test_accuracy = 0
        for idx, (images, labels) in enumerate(ds_test):
            step = idx + 1
            predictions = _test_step(images, labels)

            if step == 1: #visualisation for the sequnce in ds_test: exp44_user22
                visualize_triaxial_signals(predictions,user_id=22,exp_id=44,plot_width=20,plot_height=20)

            confusion_matrix.update_state(labels,predictions) 

            with test_summary_writer.as_default():
                    tf.summary.scalar("Loss", test_loss.result(), step=step)
                    tf.summary.scalar("Accuracy", test_accuracy.result(), step=step)
                    tf.summary.image(
                        "Confusion matrix",
                        confusion_matrix.plot_conf_mat(False),
                        step=step
                    )
                        
            if test_accuracy.result() > best_test_accuracy:
                best_test_accuracy = test_accuracy.result()
            average_test_accuracy = (average_test_accuracy * idx + test_accuracy.result())/step
    
            template = "Test_Loss: {}, Test_Accuracy: {}"
            logging.info(template.format(test_loss.result(),test_accuracy.result() * 100))

            pbar.update(1)

    template = "Summary: Best_Test_Accuracy: {}, Average_Test_Accuracy: {}"
    logging.info(template.format(best_test_accuracy * 100,average_test_accuracy * 100)) 

    confusion_matrix.plot_conf_mat(True)


    

