#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 8/22/2023 2:50 PM
# @Author  : xiaomanl
# @File    : fcn.py
# @Software: PyCharm
# FCN fcn
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D,BatchNormalization,Activation
from tensorflow.keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, RootMeanSquaredError,MeanSquaredError,LogCoshError
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback
# from tensorflow.keras.utils import multi_gpu_model
import time
import tensorflow.keras.backend as K
#import pandas_profilling
import matplotlib.pyplot as plt
from model.base_class import BaseModel
from utils import Train_predict_compare, calculate_running_time, CustomDataGenerator
from tensorflow.keras.utils import custom_object_scope, Sequence
import keras
import logging
import pandas as pd
import os
import numpy as np
import joblib
import glob


# tf.config.optimizer.set_jit(True)
# tf.test.is_built_with_cuda()
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)


class LogPrintCallback(tf.keras.callbacks.Callback):

    def __init__(self, interval=50):
        super(LogPrintCallback, self).__init__()
        self.interval = interval
        self.seen = 0

    def on_epoch_begin(self, epoch, logs=None):
        if (epoch + 1) % self.interval == 0:
            self.epoch_start_time = time.time()
        else:
            self.epoch_start_time = float("-inf")


    def on_epoch_end(self, epoch, logs=None):
        self.seen += logs.get('size', 0)
        if (epoch + 1) % self.interval == 0:
            print("Epoch {}/{}".format(epoch+1, self.params['epochs']))
            metrics_log = ''
            for k in logs:
                val = logs[k]
                if abs(val) > 1e-3:
                    metrics_log += ' - %s: %.4f' % (k, val)
                else:
                    metrics_log += ' - %s: %.4e' % (k, val)
            Cost_time = (time.time() - self.epoch_start_time) * 1000 / self.interval
            print('{}/{} .... - Epoch time: {:.2f} ms {}'.format(epoch+1, self.params['epochs'],
                                                                               Cost_time, metrics_log))


class FCN(BaseModel):
    def __init__(self, configs, processed_features, processed_labels):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("FCN")
        self.mode_config = self.configs["FCN-Model_config"]
        self.build = self.mode_config["build"]

        self.verbose = self.mode_config["verbose"]
        self.batch_size = self.mode_config["batch_size"]
        self.nb_epochs = self.mode_config["nb_epochs"]

        self.model_save_label = "test"
        self.isplot = self.mode_config["ifplot"]
        #,MeanSquaredError, RootMeanSquaredError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, LogCoshError,MeanAbsoluteError
        self.metrics = [MeanAbsoluteError(), MeanSquaredError(), RootMeanSquaredError(), MeanAbsolutePercentageError(name = "my mpe"), MeanSquaredLogarithmicError(), LogCoshError()]

        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col":self.true_col})
        self.label_name_list = self.configs["label_name_list"]
        self.model_name = self.configs["select_model"]
        self.use_pre_trained_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]
        self.select_model = self.configs["select_model"]
        self.workload_name = self.configs["workload_names"][0]
        self.workload_scale = configs["label_scale"][self.workload_name]
        self.config_save_path = self.configs["config_save_path"]
        self.output_path = configs["output_path"]
        self.model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs["model_history_path"]


    def NN_init(self):
        tf.keras.initializers.he_normal(seed=None)

    def conv1d(self, x, w, b, stride=1, padding=0):
        """
        Applies a 1D convolution operation on a given input x with a given filter w and bias b.
        """
        # Get input dimensions
        batch_size, in_channels, in_length = x.shape
        out_channels, in_channels, kernel_size = w.shape

        # Calculate output dimensions
        out_length = (in_length + 2 * padding - kernel_size) // stride + 1

        # Add padding to the input if necessary
        if padding > 0:
            x = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode='constant')

        # Initialize the output tensor
        out = np.zeros((batch_size, out_channels, out_length))

        # Perform the convolution operation
        for i in range(out_length):
            x_slice = x[:, :, i * stride:i * stride + kernel_size]
            out[:, :, i] = np.sum(x_slice * w, axis=(1, 2)) + b

        return out

    def batch_normalization(self, x, gamma, beta, epsilon=1e-8):
        """
        Applies batch normalization to the input x using the provided gamma and beta parameters.
        """
        mean = np.mean(x, axis=0)
        variance = np.var(x, axis=0)
        x_hat = (x - mean) / np.sqrt(variance + epsilon)
        out = gamma * x_hat + beta
        return out

    def activation_function(self, x, activation):
        """
        Applies the specified activation function to the input x.
        """
        if activation == "relu":
            return np.maximum(0, x)
        elif activation == "sigmoid":
            return 1 / (1 + np.exp(-x))
        elif activation == "tanh":
            return np.tanh(x)
        elif activation == "softmax":
            exps = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exps / np.sum(exps, axis=-1, keepdims=True)
        else:
            raise ValueError("Unsupported activation function: {}".format(activation))



    def build_model(self):

        file = glob.glob(self.model_path + '/*.hdf5')[0]
        with custom_object_scope({'customed_mse': self.customed_mse}):
            model = tf.keras.models.load_model(file)
        self.logger.info("Load fcn from: {}".format(file))

        return model

    def data_check(self, x, y):
        tf.debugging.assert_scalar(x)


    def predict(self):
        model = self.build_model()
            # fcn = joblib.load(save_name)
        y_pred = model.predict(self.x_test)
        # convert the predicted from binary to integer
        #y_pred = np.argmax(y_pred, axis=1)
        # keras.backend.clear_session()
        return y_pred


    @calculate_running_time
    def infer(self, train_with_all_data=False, result=None):
        self.logger.debug("Begin training the fcn")

        x_test = self.processed_features
        y_test = self.processed_labels


        self.x_test = self.data_reshape(x_test)
        self.y_test = self.data_reshape(y_test)

        self.input_shape = self.x_test.shape[1:]


        y_predict = self.predict()
        result = self.evaluate(self.y_test, y_predict)
        if self.isplot:
            save_path = self.output_path
            Train_predict_compare(self.configs, y_predict, y_test, save_path)

        return result

    def data_reshape(self, data):
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, 1, y))
        data_reshape = np.asarray(data_reshape).astype("float32")
        # columns = data.columns
        return data_reshape

    def evaluate(self, y_test, y_predict):

        results = pd.DataFrame()
        # Calculate the Absolute Error (AE) for dataframe
        cols = self.label_name_list
        true_name = self.true_col
        predict_name = self.predict_col

        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[2]))
        y_predict = np.reshape(y_predict, (y_predict.shape[0], y_predict.shape[2]))

        ae = np.abs(y_test - y_predict)
        se = (y_test - y_predict) ** 2
        ape = np.abs(y_test - y_predict) / y_test * 100

        for i, col in enumerate(cols):
            y_test_col = f'{true_name[i]}'
            y_predict_col = f'{predict_name[i]}'
            ae_col = "AE_" + col
            se_col = "SE_" + col
            ape_col = "APE(%)_" + col
            result = pd.DataFrame({y_test_col: y_test[:, i], y_predict_col: y_predict[:, i], ae_col: ae[:, i],se_col: se[:, i], ape_col: ape[:, i]})
            results = pd.concat([results, result], axis=1)

            mae = np.mean(ae[:, i])
            mse = np.mean(se[:, i])
            mape = np.mean(ape[:, i])
            p50_ape = round(np.quantile(ape[:, i], 0.5), 4)
            p90_ape = round(np.quantile(ape[:, i], 0.9), 4)
            p95_ape = round(np.quantile(ape[:, i], 0.95), 4)
            p99_ape = round(np.quantile(ape[:, i], 0.99), 4)
            max_ape = round(np.max(ape[:, i]), 4)

            count_3 = np.sum(ape[:, i] < 3)
            proportion_3 = round(count_3 / len(ape) * 100, 5)

            count_5 = np.sum(ape[:, i] < 5)
            proportion_5 = round(count_5 / len(ape) * 100, 4)

            count_10 = np.sum(ape[:, i] < 10)
            proportion_10 = round(count_10 / len(ape) * 100, 4)
            self.logger.info(f"--------------------------------{col}workload overall metric----------------------------------------")
            self.logger.info(f"MAE: {mae:.4f}")
            self.logger.info(f"MSE: {mse:.4f}")
            self.logger.info(f"MAPE(%): {mape:.4f}%")
            self.logger.info(f"Accuracy (APE < 3%) for {col}: {proportion_3}%")
            self.logger.info(f"Accuracy (APE < 5%) for {col}: {proportion_5}%")
            self.logger.info(f"Accuracy (APE < 10%) for {col}: {proportion_10}%")
            self.logger.info(f"P90 APE(%) for {col}: {p90_ape}%")
            self.logger.info(f"P95 APE(%) for {col}: {p95_ape}%")
            self.logger.info(f"P99 APE(%) for {col}: {p99_ape}%")
            self.logger.info(f"MAX APE(%) for {col}: {max_ape}%")

        return results

    def customed_mse(self, y_true, y_pred):
        weights = K.square(y_pred - y_true)
        squared_difference = K.square(y_pred - y_true)
        weighted_squared_difference = squared_difference * weights
        return K.mean(weighted_squared_difference, axis=-1)

    def plot_epochs_metric(self, hist, file_name, metric="loss"):
        history_dict = hist.history
        loss_values = history_dict[metric]
        val_loss_values = history_dict['val_'+ metric]
        epochs = range(1, self.nb_epochs + 1)

        # plt.figure()
        plt.plot(epochs, loss_values, 'bo', label='Training '+metric)
        plt.plot(epochs, val_loss_values, 'b', label='Validation '+metric)
        plt.title('Training and validation ' + metric)
        plt.ylabel(metric, fontsize='large')
        plt.xlabel('Epoch', fontsize='large')
        plt.legend(['train', 'val'], loc='upper right')
        plt.savefig(file_name, bbox_inches='tight')
        plt.close()


def save_logs(output_directory, hist, y_pred, y_true, duration, lr=True, y_true_val=None, y_pred_val=None):
    hist_df = pd.DataFrame(hist.history)
    hist_df.to_csv(output_directory + 'history.csv', index=False)


    index_best_model = hist_df['loss'].idxmin()
    row_best_model = hist_df.loc[index_best_model]

    df_best_model = pd.DataFrame(data=np.zeros((1, 6), dtype=np.float), index=[0],
                                 columns=['best_model_train_loss', 'best_model_val_loss', 'best_model_train_acc',
                                          'best_model_val_acc', 'best_model_learning_rate', 'best_model_nb_epoch'])

    df_best_model['best_model_train_loss'] = row_best_model['loss']
    df_best_model['best_model_val_loss'] = row_best_model['val_loss']
    df_best_model['best_model_train_acc'] = row_best_model['accuracy']
    df_best_model['best_model_val_acc'] = row_best_model['val_accuracy']
    if lr == True:
        df_best_model['best_model_learning_rate'] = row_best_model['lr']
    df_best_model['best_model_nb_epoch'] = index_best_model

    df_best_model.to_csv(output_directory + 'df_best_model.csv', index=False)

    # for FCN there is no hyperparameters fine tuning - everything is static in code

    # plot losses
def weighted_loss(weights):

    def categorical_crossentropy_masked(y_true, y_pred):
        """
        y_true (sample_number, max_timestep, onehot)
        """

        label_weight = tf.convert_to_tensor(weights)
        label_weight = tf.cast(label_weight, tf.float32)
        y_true_weight = label_weight * y_true


        mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
        mask = tf.cast(mask, tf.bool)

        y_true_weight = tf.boolean_mask(y_true_weight, mask)
        y_pred = tf.boolean_mask(y_pred, mask)

        crossentroy = K.mean(-tf.reduce_sum(y_true_weight * tf.log(y_pred),axis = -1))
        # loss = K.mean(K.categorical_crossentropy(y_true, y_pred))

        return crossentroy

    return categorical_crossentropy_masked

def masked_accuracy(y_true, y_pred):

    mask = tf.reduce_sum(y_true, axis=-1)  # (sample_num, max_timestep)
    mask = tf.cast(mask, tf.bool)

    y_true = tf.boolean_mask(y_true, mask) # (allsamples_left_timesteps, feature_num)
    y_pred = tf.boolean_mask(y_pred, mask)

    y_true = K.argmax(y_true, axis=-1)
    y_pred = K.argmax(y_pred, axis=-1)

    correct_bool = tf.equal(y_true, y_pred)
    mask_accuracy = K.sum(tf.cast(correct_bool, tf.int32)) / tf.shape(correct_bool)[0]
    mask_accuracy = tf.cast(mask_accuracy, tf.float32)

    return mask_accuracy



class NBatchLogger(Callback):
    def __init__(self, display):
        self.seen = 0
        self.display = display

    def on_batch_end(self, batch, logs):
        self.seen += logs.get('size', 0)
        if self.seen % self.display == 0:
            # you can access loss, accuracy in self.params['metrics']
            print('\n{}/{} - loss ....\n'.format(self.seen, self.params['nb_sample']))
