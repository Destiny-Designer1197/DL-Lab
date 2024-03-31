from asyncio import run
from posixpath import relpath
import re
import gin
import os
import logging
from absl import app, flags
import tensorflow as tf
from tensorflow.python.eager.execute import convert_to_mixed_eager_tensors
from tensorflow.python.ops.gen_experimental_dataset_ops import load_dataset
from train import Trainer
#from evaluation.eval import Tester
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import LSTM,CNN_LSTM
from input_pipeline.preprocessing import load_data_in_df,delete_unlabeled_data,balance_ds,normalization,make_sliding_window,create_TFRecord
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define constants, can be modified
WINDOW_SIZE = 250
N_FEATURES = 6
N_CLASSES = 12

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train',True, 'Specify whether to train or evaluate a model.')

def main(argv):
    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['human_activity_recognition/configs/config_test.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_test, ds_val= load()
    
    # model
    #model = LSTM(input_shape = (WINDOW_SIZE,N_FEATURES) ,n_classes = N_CLASSES)
    model = CNN_LSTM(input_shape = (WINDOW_SIZE,N_FEATURES) ,n_classes = N_CLASSES)
    #model.summary()
 
    if FLAGS.train:
        logging.info(f"Start Training...")
        trainer = Trainer(model, ds_train, ds_val, run_paths)
        for _ in trainer.train():
            continue
    else:
        logging.info(f"Start Evaluation...")

        checkpoint = tf.train.Checkpoint(step=tf.Variable(1),myOptimizer=tf.keras.optimizers.Adam(learning_rate=0.001), myModel=model)
        
        evaluate(model,checkpoint,ds_test,run_paths)

if __name__ == '__main__':
    app.run(main)
