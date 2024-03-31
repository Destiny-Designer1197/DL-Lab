import gin
import tensorflow as tf
import logging
from evaluation.metrics import ConfusionMatrix
from datetime import datetime
import os
import shutil

@gin.configurable
class Trainer(object):
    def __init__(self, model, ds_train, ds_val, run_paths, total_steps, log_interval, ckpt_interval,n_classes):
        # Summary Writer
        log_dir = 'diabetic_retinopathy/logs/gradient_tape/'
        
        if os.path.exists(log_dir):
                shutil.rmtree(log_dir)
        
        current_time = datetime.now().strftime("%Y%m%d-%H%M%S")

        train_log_dir = log_dir + current_time + '/train'
        val_log_dir = log_dir + current_time + '/val'
  
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        # Loss objective
        #self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.loss_object = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        #self.loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        #self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        #self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        self.test_accuracy = tf.keras.metrics.CategoricalAccuracy(name='test_accuracy')

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        #self.ds_info = ds_info
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval
        self.n_classes = n_classes
        self.train_confusion_matrix_metrics = ConfusionMatrix(n_classes=self.n_classes, name="train_confusion_matrix_metrics")
        self.test_confusion_matrix_metrics = ConfusionMatrix(n_classes=self.n_classes, name="test_confusion_matrix_metrics")
        self.epoch_counter = tf.Variable(initial_value=0, trainable=False)

        # Checkpoint Manager
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1),myOptimizer=self.optimizer, myModel=self.model)
       
        # Checkpoint Manager  
        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint=checkpoint, max_to_keep=3, directory=self.run_paths["path_ckpts_train"])

        if self.checkpoint_manager.latest_checkpoint:
            print("Checkpoint: Restored from {}".format(self.checkpoint_manager.latest_checkpoint))
        else:
            print("Checkpoint: Initializing from scratch.")

    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=False)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)
        self.train_confusion_matrix_metrics.update_state(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)
        self.test_confusion_matrix_metrics.update_state(labels, predictions)

    def train(self):
        with tf.profiler.experimental.Profile('diabetic_retinopathy/logs/gradient_tape/train/'):
            for idx, (images, labels) in enumerate(self.ds_train):
                    step = idx + 1
                    self.train_step(images, labels)

                    if step % self.log_interval == 0:

                        # Reset test metrics
                        self.test_loss.reset_states()
                        self.test_accuracy.reset_states()

                        for test_images, test_labels in self.ds_val:
                                self.test_step(test_images, test_labels)

                        template = 'Step {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
                        logging.info(template.format(step,
                                                    self.train_loss.result(),
                                                    self.train_accuracy.result() * 100,
                                                    self.test_loss.result(),
                                                    self.test_accuracy.result() * 100))

                        # Write summary to tensorboard
                        with self.train_summary_writer.as_default():
                            tf.summary.scalar('Loss',self.train_loss.result() , step = step)
                            tf.summary.scalar('Accuracy',self.train_accuracy.result() * 100, step = step)
                            tf.summary.image('Train ConfusionMatrix',self.train_confusion_matrix_metrics.plot_conf_mat(issaved=False),step=step)

                        with self.val_summary_writer.as_default():
                            tf.summary.scalar('Loss',self.test_loss.result() , step = step)
                            tf.summary.scalar('Accuracy',self.test_accuracy.result() * 100, step = step)
                            tf.summary.image('Val ConfusionMatrix',self.test_confusion_matrix_metrics.plot_conf_mat(issaved=False),step=step)
                                        
                        # Reset train metrics
                        self.train_loss.reset_states()
                        self.train_accuracy.reset_states()

                        yield self.test_accuracy.result().numpy()

                    if step % self.ckpt_interval == 0:
                        logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                        # Save checkpoint
                        try:
                            checkpoint_path = self.checkpoint_manager.save()
                            logging.info(f"Saving checkpoint to: '{checkpoint_path}'")
                        except:
                            logging.error(f"Failed to save checkpoint at checkpoint interval")

                    if step % self.total_steps == 0:
                        logging.info(f'Finished training after {step} steps.')
                        # Save final checkpoint
                        try:
                            # Save the final checkpoint.
                            last_checkpoint = self.checkpoint_manager.save()
                            logging.info(f"Last checkpoint: '{last_checkpoint}'")
                        except:
                            last_checkpoint = None
                            logging.error(f"Failed to save last checkpoint")
                        return self.test_accuracy.result().numpy()