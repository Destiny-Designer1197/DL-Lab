from posixpath import split
import gin
import logging
from absl import app, flags
from evaluation.deep_visualisation import visualisation_GradCam
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline.datasets import load
from utils import utils_params, utils_misc
from models.architectures import ResNet_50
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

IMG_SIZE = 256

FLAGS = flags.FLAGS
flags.DEFINE_boolean('train',True, 'Specify whether to train or evaluate a model.')

def main(argv):
    #generate folder structures
    run_paths = utils_params.gen_run_folder()

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings(['diabetic_retinopathy/configs/config_test.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    ds_train, ds_val, ds_test = load()

    #model
    model = ResNet_50(input_shape=(IMG_SIZE,IMG_SIZE,3), n_classes=2)
    #model.summary()

    if FLAGS.train:
        logging.info(f"Start Training...")
        trainer = Trainer(model, ds_train, ds_val,run_paths)
        for _ in trainer.train():
            continue
    else:
        logging.info(f"Start Evaluation...")
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1),myOptimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), myModel=model)
        evaluate(model,
                checkpoint,
                ds_test,
                run_paths
                )
        visualisation_GradCam(model)
        
if __name__ == "__main__":
    app.run(main)