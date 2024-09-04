

from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, RootMeanSquaredError,MeanSquaredError,LogCoshError
import tensorflow as tf
# import pydevd_pycharm
# pydevd_pycharm.settrace('192.168.3.53', port=2245, stdoutToServer=True, stderrToServer=True)


import matplotlib.pyplot as plt
import numpy as np
from keras.utils import custom_object_scope
from model.base_class import BaseModel
from utils import calculate_running_time, LogPrintCallback, CustomDataGenerator, Train_predict_compare, Self_Attention, MultiHeadAtten
import os
import logging
import pandas as pd
import yaml
import sys
from keras.utils import pad_sequences
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
import glob



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from einops import rearrange, repeat
from tqdm import tqdm


# tf.config.optimizer.set_jit(True)
# tf.test.is_built_with_cuda()
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)



class Mamba(BaseModel):
    def __init__(self, configs, processed_features, processed_labels):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("Mamba")
        self.mode_config = self.configs["Mamba_config"]
        self.build = self.mode_config["build"]
        self.verbose = self.mode_config["verbose"]
        self.batch_size = self.mode_config["batch_size"]
        self.nb_epochs = self.mode_config["nb_epochs"]
        self.mem_processed_numa_features = processed_features[0][0]
        self.mem_processed_char_feature = processed_features[0][1]
        self.cpu_processed_numa_features = processed_features[1][0]
        self.cpu_processed_char_feature = processed_features[1][1]
        self.system_processed_numa_features = processed_features[2][0]
        self.system_processed_char_feature = processed_features[2][1]
        self.workload_processed_numa_features = processed_features[3][0]
        self.workload_processed_char_feature = processed_features[3][1]
        self.model_save_label = "test"
        self.isplot = self.mode_config["ifplot"]
        self.label_name_list = self.configs["label_name_list"]
        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col": self.true_col})
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
        self.patience = self.mode_config["lr_patience"]
        self.factor = float(self.mode_config["lr_factor"])
        self.cooldown = float(self.mode_config["lr_cooldown"])
        self.min_lr = float(self.mode_config["min_lr"])
        self.opm_init_lr = float(self.mode_config["opm_init_lr"])
        self.decay_rate = float(self.mode_config["decay_rate"])
        self.decay_steps = self.mode_config["decay_steps"]
        self.n_feature_maps = self.mode_config["n_feature_maps"]
        self.freeze_train_model = self.configs["freeze_train_model"]
        self.device = torch.device('cuda')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs["model_history_path"]





    def build_model(self, single_model_path):
        self.logger.info("read model from {}".format(single_model_path))
        file = glob.glob(single_model_path)[0]
        # model_save_path = os.path.join(self.output_path, "model").replace("\\", "/")
        # model_save_name = os.path.join(model_save_path, f"{self.select_model}.pth").replace("\\", "/")
        # # register customed_mse to custom_object_scope
        # data_save_name = os.path.join(model_save_path, f"{self.select_model}_data.pth").replace("\\", "/")

        self.logger.info("Load MODEL from: {}".format(file))
        checkpoint = torch.load(file)
        model = checkpoint['model']



        return model

    def predict(self, single_model_path):
        features = np.concatenate([
            self.mem_numer_x_test,
            self.cpu_numer_x_test,
            self.system_numer_x_test
        ], axis=1)


        char_features = np.concatenate([
            self.workload_char_x_test,
            self.system_char_x_test
        ], axis=1)

        labels = self.y_test
        model = self.build_model(single_model_path)


        model.eval()
        dataset = CustomDataset(features, char_features, labels)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        predictions = []
        labels_list = []
        with torch.no_grad():
            for inputs, char, labels in dataloader:
                inputs = inputs.to(device)
                char = char.to(device)
                outputs = model(inputs, char)
                predictions.append(outputs.cpu())
                if (outputs > 100000).any():
                    print("error")
                labels_list.append(labels)


        predictions_tensor = torch.cat(predictions)
        labels_tensor = torch.cat(labels_list)


        return predictions_tensor, labels_tensor


    @calculate_running_time
    def infer(self, single_model_path, train_with_all_data=False, result=None):
        self.logger.debug("Begin testing the model")

        self.mem_char_x_test = self.data_reshape(self.mem_processed_char_feature)
        self.mem_numer_x_test = self.data_reshape(self.mem_processed_numa_features)
        self.cpu_char_x_test = self.data_reshape(self.cpu_processed_char_feature)
        self.cpu_numer_x_test = self.data_reshape(self.cpu_processed_numa_features)
        self.system_char_x_test = self.data_reshape(self.system_processed_char_feature)
        self.system_numer_x_test = self.data_reshape(self.system_processed_numa_features)
        self.workload_char_x_test = self.data_reshape(self.workload_processed_char_feature)
        self.workload_numer_x_test = self.data_reshape(self.workload_processed_numa_features)


        self.y_test = self.data_reshape(self.processed_labels)

        self.mem_char_x_train_shape = self.mem_char_x_test.shape[1:]
        self.cpu_char_x_train_shape = self.cpu_char_x_test.shape[1:]
        self.system_char_x_train_shape = self.system_char_x_test.shape[1:]
        self.workload_char_x_train_shape = self.workload_char_x_test.shape[1:]
        self.mem_numer_x_train_shape = self.mem_numer_x_test.shape[1:]
        self.cpu_numer_x_train_shape = self.cpu_numer_x_test.shape[1:]
        self.system_numer_x_train_shape = self.system_numer_x_test.shape[1:]
        self.workload_numer_x_train_shape = self.workload_numer_x_test.shape[1:]



        self.y_predict_tensor, self.y_test_tensor = self.predict(single_model_path)
        result, metric_overall = self.evaluate(self.y_predict_tensor, self.y_test_tensor)
        if self.isplot:
            save_path = self.output_path
            Train_predict_compare(self.configs, self.y_predict_tensor, self.y_test_tensor, save_path)

        return result, metric_overall

    def data_reshape(self, data):
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns
        return data_reshape

    def evaluate(self, y_predict, y_test):

        criterion = torch.nn.MSELoss()
        loss = criterion(y_predict.unsqueeze(-1), y_test)

        results = pd.DataFrame()


        cols = self.label_name_list
        true_name = self.true_col
        predict_name = self.predict_col

        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[1]))
        if len(y_predict.shape) == 3:
            y_predict = np.reshape(y_predict, (y_predict.shape[0], y_predict.shape[1]))

        ae = np.abs(y_test - y_predict)
        se = (y_test - y_predict) ** 2
        ape = np.abs(y_test - y_predict) / y_test * 100

        ae = ae.cpu().numpy()
        se = se.cpu().numpy()
        ape = ape.cpu().numpy()

        for i, col in enumerate(cols):
            y_test_col = f'{true_name[i]}'
            y_predict_col = f'{predict_name[i]}'
            ae_col = "AE_" + col
            se_col = "SE_" + col
            ape_col = "APE(%)_" + col
            result = pd.DataFrame(
                {y_test_col: y_test[:, i], y_predict_col: y_predict[:, i], ae_col: ae[:, i], se_col: se[:, i],
                 ape_col: ape[:, i]})
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

            self.logger.info(f"------------------------------overall metrics snippet of {col}----------------------------------")
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

        overall_mae = np.mean(ae)
        overall_mse = np.mean(se)
        overall_mape = np.mean(ape)
        overall_p50_ape = round(np.quantile(ape, 0.5), 4)
        overall_p90_ape = round(np.quantile(ape, 0.9), 4)
        overall_p95_ape = round(np.quantile(ape, 0.95), 4)
        overall_p99_ape = round(np.quantile(ape, 0.99), 4)
        overall_max_ape = round(np.max(ape), 4)
        overall_count_3 = np.sum(ape < 3)
        overall_proportion_3 = round(overall_count_3 / ape.size * 100, 5)
        overall_count_5 = np.sum(ape < 5)
        overall_proportion_5 = round(overall_count_5 / ape.size * 100, 4)
        overall_count_10 = np.sum(ape < 10)
        overall_proportion_10 = round(overall_count_10 / ape.size * 100, 4)
        metric_overall = {"Overall MAE": overall_mae, "Overall MSE": overall_mse, "Overall MAPE(%)": overall_mape,
             "Overall Accuracy (APE < 3%)": overall_proportion_3, "Overall Accuracy (APE < 5%)": overall_proportion_5,
             "Overall Accuracy (APE < 10%)": overall_proportion_10, "Overall P50 APE(%)": overall_p50_ape,
             "Overall P90 APE(%)": overall_p90_ape, "Overall P95 APE(%)": overall_p95_ape,
             "Overall P99 APE(%)": overall_p99_ape, "Overall MAX APE(%)": overall_max_ape}

        # 输出总的指标
        self.logger.info(f"------------------------------overall metrics----------------------------------")
        self.logger.info(f"Overall MAE: {overall_mae:.4f}")
        self.logger.info(f"Overall MSE: {overall_mse:.4f}")
        self.logger.info(f"Overall MAPE(%): {overall_mape:.4f}%")
        self.logger.info(f"Overall Accuracy (APE < 3%): {overall_proportion_3}%")
        self.logger.info(f"Overall Accuracy (APE < 5%): {overall_proportion_5}%")
        self.logger.info(f"Overall Accuracy (APE < 10%): {overall_proportion_10}%")
        self.logger.info(f"Overall P50 APE(%): {overall_p50_ape}%")
        self.logger.info(f"Overall P90 APE(%): {overall_p90_ape}%")
        self.logger.info(f"Overall P95 APE(%): {overall_p95_ape}%")
        self.logger.info(f"Overall P99 APE(%): {overall_p99_ape}%")
        self.logger.info(f"Overall MAX APE(%): {overall_max_ape}%")

        return results, metric_overall



    def plot_epochs_metric(self, hist, file_name, metric="loss"):
        try:
            history_dict = hist.history
            epochs = range(1, self.nb_epochs + 1)
            loss_values = history_dict[metric]
            plt.plot(epochs, loss_values, 'bo', label='Training ' + metric)
            if 'val_'+ metric in history_dict:
                val_loss_values = history_dict['val_'+ metric]
                plt.plot(epochs, val_loss_values, 'b', label='Validation '+metric)
            plt.title('Training and validation ' + metric)
            plt.ylabel(metric, fontsize='large')
            plt.xlabel('Epoch', fontsize='large')
            plt.legend(['train', 'val'], loc='upper right')
            plt.savefig(file_name, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self.logger.warning(f"Error plot metric: {e}")



class CustomDataset(Dataset):
    def __init__(self, features, char_features, labels):
        self.features = features
        self.labels = labels
        self.char_features = char_features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):

        features = self.features[idx].astype('float32')

        label = self.labels[idx].astype('float32')


        features = torch.tensor(features)

        label = torch.tensor(label)
        if self.char_features is not None:
            char_features = self.char_features[idx].astype('float32')
            char_features = torch.tensor(char_features)

            return features, char_features, label
        else:
            return features, label
