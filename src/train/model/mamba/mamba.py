

import keras
from keras.models import Model
from keras.layers import Dense, Conv1D,BatchNormalization,Activation, Input, Flatten, concatenate, ZeroPadding1D, ZeroPadding2D, ZeroPadding3D
from keras.metrics import MeanAbsoluteError, MeanAbsolutePercentageError, MeanSquaredLogarithmicError, RootMeanSquaredError,MeanSquaredError,LogCoshError
import tensorflow as tf
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
# from tensorflow.keras.utils import multi_gpu_model
import keras.backend as K
from keras.backend import reshape
from keras.layers import Embedding
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import custom_object_scope
from model.base_class import BaseModel
from train_utils import  calculate_running_time, LogPrintCallback, CustomDataGenerator, Train_predict_compare, Self_Attention, MultiHeadAtten
import os
import logging
import pandas as pd
import yaml
import sys
from keras.utils import pad_sequences
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import math
#from .mamba_mini import Mamba1, ModelArgs
# from .mamba_formal import Mamba1, ModelArgs
from .mamba1_formal import Mamba1, ModelArgs
from .mamba2_formal import Mamba2, Model2Args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from einops import rearrange, repeat
from tqdm import tqdm
from torchsummary import summary
from dataclasses import dataclass, asdict
from typing import Union, Optional

# try:
#     from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
# except ImportError:
#     causal_conv1d_fn, causal_conv1d_update = None, None
#
# try:
#     from causal_conv1d.causal_conv1d_varlen import causal_conv1d_varlen_states
# except ImportError:
#     causal_conv1d_varlen_states = None
#
# try:
#     from mamba_ssm.ops.triton.selective_state_update import selective_state_update
# except ImportError:
#     selective_state_update = None
#
# from torch.utils.tensorboard import SummaryWriter
#
# from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
#
# from mamba_ssm.distributed.tensor_parallel import ColumnParallelLinear, RowParallelLinear
# from mamba_ssm.distributed.distributed_utils import all_reduce, reduce_scatter
#
# from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
# from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined

# tf.config.optimizer.set_jit(True)
# tf.test.is_built_with_cuda()
# mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
# policy = tf.keras.mixed_precision.Policy("mixed_float16")
# tf.keras.mixed_precision.set_global_policy(policy)



class Mamba(BaseModel):
    def __init__(self, configs, processed_features, processed_labels, train_indices, test_indices, k_fold_save_path):
        super().__init__( configs, processed_features, processed_labels, train_indices, test_indices)
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
        self.save_path = k_fold_save_path
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
        self.loss_dict = {}
        self.current_epoch = 0
        self.char_token_order = self.configs["char_token_order"]





    def build_model(self):
        # 获取输入特征的维度
        input_dim = 36  # 特征维度
        output_dim = self.y_train.shape[1]  # 输出维度
        # args = ModelArgs(
        #     d_model=128,  # 模型的维度
        #     n_layer=7,  # 层数
        #     vocab_size=100,  # 词汇表大小
        #     d_state=8,  # 默认值，可以省略
        #     expand=2,  # 默认值，可以省略
        #     dt_rank='auto',  # 默认值，可以省略
        #     d_conv=5,  # 默认值，可以省略
        #     pad_vocab_size_multiple=8,  # 默认值，可以省略
        #     conv_bias=True,  # 默认值，可以省略
        #     bias=True,
        #     output_dim=output_dim,
        #     input_dim=input_dim,
        #     embedding_dim=4
        # )
        # args = ModelArgs(
        #     d_model=512,
        #     d_state = 8,
        #     d_conv = 4,
        #     conv_init =1,
        #     expand = 2,
        #     headdim = 16,
        #     d_ssm = 16,
        #     # If not None, we only apply SSM on this many dimensions, the rest uses gated MLP
        #     ngroups = 1,
        #     A_init_range = (1, 16),
        #     D_has_hdim = False,
        #     rmsnorm = False,
        #     norm_before_gate = False,
        #     dt_min = 1e-5,
        #     dt_max = 0.1,
        #     dt_init_floor = 1e-4,
        #     dt_limit = (0.0, float("inf")),
        #     bias = False,
        #     conv_bias = True,
        #     # Fused kernel and sharding options
        #     chunk_size = 256,
        #     use_mem_eff_path = True,
        #     layer_idx = None,  # Absorb kwarg for general module
        #     process_group = None,
        #     sequence_parallel = True,
        #     device = self.device,
        #     dtype = None,
        #     output_dim=output_dim,
        #     input_dim=input_dim,
        #     embedding_dim=4
        # )
        self.model_args = ModelArgs(
            d_model=64,
            d_state=8,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            output_dim=output_dim,
            input_dim=input_dim,
            n_layer=2,
            embedding_dim=4,
            shapes=self.shape,
            char_token_order=self.char_token_order
            )
        # self.model_args = Model2Args(
        #     d_model=128,
        #     d_state=128,
        #     d_conv=4,
        #     conv_init=1,
        #     expand=4,
        #     headdim=64,
        #     ngroups=1,
        #     A_init_range=(1, 16),
        #     dt_min=0.001,
        #     dt_max=0.1,
        #     dt_init_floor=1e-4,
        #     dt_limit=(0.0, float("inf")),
        #     learnable_init_states=False,
        #     activation="swish",
        #     bias=False,
        #     conv_bias=True,  # Fused kernel and sharding options
        #     chunk_size=256,
        #     use_mem_eff_path=True,
        #     layer_idx=None,  # Absorb kwarg for general module
        #     device=None,
        #     dtype=None,
        #     output_dim=output_dim,
        #     input_dim=input_dim,
        #     n_layer=2,
        #     embedding_dim=4,
        #     shapes=self.shape,
        #     char_token_order=self.char_token_order)


        net = Mamba1(self.model_args).to(self.device)

        self.criterion1 = nn.MSELoss()
        self.criterion2 = nn.L1Loss()
        self.criterion3 = nn.SmoothL1Loss(10)

        self.optimizer = optim.Adam(net.parameters(), lr=0.001)
        # self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.9)

        # self.scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=10,
                                                         threshold=0.1, threshold_mode='abs', cooldown=5,
                                                         min_lr=1e-9, eps=1e-8)
        if self.use_pre_trained_model:
            self.logger.warning("Use pre-trained model")
            checkpoint = torch.load(self.pre_trained_model_path)
            net.load_state_dict(checkpoint['model_state_dict'])
            self.criterion1.load_state_dict(checkpoint['criterion1_state_dict'])
            self.criterion2.load_state_dict(checkpoint['criterion2_state_dict'])
            self.criterion3.load_state_dict(checkpoint['criterion3_state_dict'])
            # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.current_epoch = checkpoint['current_epoch']
            self.num_epochs = checkpoint['epoch']
            self.loss_dict = checkpoint['loss']
            self.model_args = Model2Args(**checkpoint['model_args'])
            self.logger.info(f"Load pre-trained model from {self.pre_trained_model_path}")

        self.logger.info("Model built successfully")
        # summary(net, input_size=(1,input_dim, 1))


        return net


    def train(self):
        features = np.concatenate([
            self.mem_numer_x_train,
            self.cpu_numer_x_train,
            self.system_numer_x_train
        ], axis=1)
        # todo do not change the order of char_features
        char_features = np.concatenate([
            self.workload_char_x_train,
            self.system_char_x_train
        ], axis=1)

        print("workload_char_x_train shape: ", self.workload_char_x_train.shape)
        print("system_char_x_train shape: ", self.system_char_x_train.shape)

        labels = self.y_train.values

        self.model = self.build_model()

        if (self.verbose == True):
            self.model.summary()

        # feature_columns = self.cpu_numer_x_train.columns
        label_columns = self.y_train.columns

        train_data = CustomDataset(features, char_features, labels)

        trainloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)


        self.logger.info(f"Using device: {self.device}")

        self.num_epochs = 300 + self.current_epoch

        for epoch in range(self.current_epoch, self.num_epochs):
            running_loss_dict = {'mse': 0.0, 'mae': 0.0, "smooth_l1": 0.0}
            self.model.train()
            # self.logger.info(
            #     f"Epoch {epoch + 1}/{self.num_epochs} - Start - Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=True)
            # Training loop
            for i, data in enumerate(progress_bar, 0):
                inputs, char, label = data
                inputs, char, label = inputs.to(self.device), char.to(self.device), label.to(self.device)  # 将数据移动到GPU

                self.optimizer.zero_grad()
                # self.logger.info(f"inputs shape: {inputs.shape}, char shape: {char.shape}, label shape: {label.shape}")

                outputs = self.model(inputs, char)
                loss1 = self.criterion1(outputs, label.unsqueeze(1))
                loss2 = self.criterion2(outputs, label.unsqueeze(1))
                loss3 = self.criterion3(outputs, label.unsqueeze(1))
                total_loss = loss3
                total_loss.backward()
                self.optimizer.step()

                running_loss_dict['mse'] += loss1.item()
                running_loss_dict['mae'] += loss2.item()
                running_loss_dict['smooth_l1'] += loss3.item()

                progress_bar.set_postfix({
                    'mse': running_loss_dict['mse'] / (i + 1),
                    'mae': running_loss_dict['mae'] / (i + 1),
                    'smooth_l1': running_loss_dict['smooth_l1'] / (i + 1)}
                )
            ave_loss1 = running_loss_dict['mse'] / len(trainloader)
            ave_loss2 = running_loss_dict['mae'] / len(trainloader)
            ave_loss3 = running_loss_dict['smooth_l1'] / len(trainloader)
            self.loss_dict[epoch] = {"mse": ave_loss1, "mae": ave_loss2, "smooth_l1": ave_loss3}
            self.scheduler.step(ave_loss3)  # 更新学习率


            # torch.save(self.model.state_dict(), 'model.ckpt')
            if (epoch + 1) % 20 == 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    current_lr = param_group['lr']
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.num_epochs} -Learning rate for param group {i}: {current_lr}")
                # torch.save(self.model.state_dict(), 'model.ckpt')
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} finished with loss1: {ave_loss1:.3f}, loss2: {ave_loss2:.3f}, loss3: {ave_loss3:.3f}")
                self.save_model(
                    label=f"epoch_{epoch + 1}_loss1_{ave_loss1:.3f}_loss2_{ave_loss2:.3f}_loss3_{ave_loss3:.3f}")
                self.logger.info(f"Model saved to path: {self.output_path}")
                self.validate(epoch)
            self.current_epoch = epoch
        self.logger.info("Finished Training")

        # log = pd.DataFrame(hist.history)
        # log.to_csv(self.output_path + "/trainlog")
        self.logger.info("Finished training the model")
        # self.save_coef()
        if self.is_save_model:
            self.save_model()

        else:
            self.logger.info("train without saving model")

        return self.model

    def train_without_validate(self):

        features = np.concatenate([
            self.mem_numer_x_train,
            self.cpu_numer_x_train,
            self.system_numer_x_train
        ], axis=1)

# todo do not change the order of char_features
        char_features = np.concatenate([
            self.workload_char_x_train,
            self.system_char_x_train
        ], axis=1)
        print("workload_char_x_train shape: ", self.workload_char_x_train.shape)
        print("system_char_x_train shape: ", self.system_char_x_train.shape)

        labels = self.y_train.values

        self.model = self.build_model()

        if (self.verbose == True):
            self.model.summary()

        # feature_columns = self.cpu_numer_x_train.columns
        label_columns = self.y_train.columns
        train_data = CustomDataset(features, char_features,  labels)
        trainloader = DataLoader(train_data, batch_size=1, shuffle=True, num_workers=8)

        self.logger.info(f"Using device: {self.device}")



        self.num_epochs = 300 + self.current_epoch

        for epoch in range(self.current_epoch, self.num_epochs):
            running_loss_dict = {'mse': 0.0, 'mae': 0.0, "smooth_l1": 0.0}
            self.model.train()
            # self.logger.info(
            #     f"Epoch {epoch + 1}/{self.num_epochs} - Start - Current Learning Rate: {self.optimizer.param_groups[0]['lr']}")
            progress_bar = tqdm(trainloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}", leave=True)
            for i, data in enumerate(progress_bar, 0):
                inputs, char, label = data
                inputs, char, label = inputs.to(self.device), char.to(self.device), label.to(self.device)  # 将数据移动到GPU

                self.optimizer.zero_grad()
                # self.logger.info(f"inputs shape: {inputs.shape}, char shape: {char.shape}, label shape: {label.shape}")

                outputs = self.model(inputs, char)
                loss1 = self.criterion1(outputs, label.unsqueeze(1))
                loss2 = self.criterion2(outputs, label.unsqueeze(1))
                loss3 = self.criterion3(outputs, label.unsqueeze(1))
                total_loss = loss3
                total_loss.backward()
                self.optimizer.step()

                running_loss_dict['mse'] += loss1.item()
                running_loss_dict['mae'] += loss2.item()
                running_loss_dict['smooth_l1'] += loss3.item()

                progress_bar.set_postfix({
                    'mse': running_loss_dict['mse'] / (i + 1),
                'mae': running_loss_dict['mae'] / (i + 1),
                    'smooth_l1': running_loss_dict['smooth_l1'] / (i + 1)}
                )
            ave_loss1 = running_loss_dict['mse'] / len(trainloader)
            ave_loss2 = running_loss_dict['mae'] / len(trainloader)
            ave_loss3 = running_loss_dict['smooth_l1'] / len(trainloader)
            self.loss_dict[epoch] = {"mse": ave_loss1, "mae": ave_loss2, "smooth_l1": ave_loss3}
            self.scheduler.step(ave_loss3)  # 更新学习率

            # torch.save(self.model.state_dict(), 'model.ckpt')
            if (epoch + 1) % 20 == 0:
                for i, param_group in enumerate(self.optimizer.param_groups):
                    current_lr = param_group['lr']
                    self.logger.info(
                        f"Epoch {epoch + 1}/{self.num_epochs} -Learning rate for param group {i}: {current_lr}")
                # torch.save(self.model.state_dict(), 'model.ckpt')
                self.logger.info(
                    f"Epoch {epoch + 1}/{self.num_epochs} finished with loss1: {ave_loss1:.3f}, loss2: {ave_loss2:.3f}, loss3: {ave_loss3:.3f}")
                self.save_model(label=f"epoch_{epoch + 1}_loss1_{ave_loss1:.3f}_loss2_{ave_loss2:.3f}_loss3_{ave_loss3:.3f}")
                self.logger.info(f"Model saved to path: {self.output_path}")
            self.current_epoch = epoch
        self.logger.info("Finished Training")


        # log = pd.DataFrame(hist.history)
        # log.to_csv(self.output_path + "/trainlog")
        self.logger.info("Finished training the model")
        # self.save_coef()
        if self.is_save_model:
            self.save_model()

        else:
            self.logger.info("train without saving model")

        return self.model
    def save_model(self, label=""):
        save_path = os.path.join(self.output_path, "model").replace("\\", "/")
        # self.plot_epochs_metric(hist, save_path + "/loss.png")
        # log.to_csv(save_path + "/trainloss.csv")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        data_save_name = os.path.join(save_path, f"{self.select_model}_data.pth").replace("\\", "/")

        model_save_name = os.path.join(save_path, f"{self.select_model}_{label}.pth").replace("\\", "/")
        weight_save_name = os.path.join(save_path, f"{self.select_model}_{label}_weight.pth").replace("\\", "/")
        # x_train_save_name = os.path.join(save_path, "processed_x_train.csv").replace("\\", "/")
        # x_train = pd.concat([char_x_train, numer_x_train], axis=1)
        # x_train.to_csv(x_train_save_name, index=False)
        with open(os.path.join(self.config_save_path, "config.yaml").replace("\\", "/"), 'w') as f:
            yaml.dump(self.configs, f)
        # torch.save({
        #     "features": features,
        #     "labels": labels
        # }, data_save_name)
        checkpoint ={
            'model': self.model,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': self.num_epochs,  # 可选：保存当前的epoch数
            "loss": self.loss_dict,
            'criterion1_state_dict': self.criterion1.state_dict(),
            'criterion2_state_dict': self.criterion2.state_dict(),
            'criterion3_state_dict': self.criterion3.state_dict(),
            'current_epoch': self.current_epoch + 1,
            'model_args': asdict(self.model_args)
        }
        torch.save(checkpoint, model_save_name)
        torch.save(self.model.state_dict(), weight_save_name)
        self.logger.warning(f"saving model to: {save_path}")

    def validate(self, epoch=None):
        best_val_loss = float('inf')
        feature_test = np.concatenate([
            self.mem_numer_x_test,
            self.cpu_numer_x_test,
            self.system_numer_x_test
        ], axis=1)

        char_test_features = np.concatenate([
            self.workload_char_x_test,
            self.system_char_x_test
        ], axis=1)

        labels_test = self.y_test.values

        label_test_columns = self.y_test.columns
        test_data = CustomDataset(feature_test, char_test_features, labels_test)
        testloader = DataLoader(test_data, batch_size=1, shuffle=True, num_workers=8)
        # Validation loop
        self.model.eval()
        val_running_loss_dict = {'mse': 0.0, 'mae': 0.0, "smooth_l1": 0.0}
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, char, label = data
                inputs, char, label = inputs.to(self.device), char.to(self.device), label.to(self.device)
                outputs = self.model(inputs, char)
                loss1 = self.criterion1(outputs, label.unsqueeze(1))
                loss2 = self.criterion2(outputs, label.unsqueeze(1))
                loss3 = self.criterion3(outputs, label.unsqueeze(1))
                val_running_loss_dict['mse'] += loss1.item()
                val_running_loss_dict['mae'] += loss2.item()
                val_running_loss_dict['smooth_l1'] += loss3.item()

        ave_val_loss1 = val_running_loss_dict['mse'] / len(testloader)
        ave_val_loss2 = val_running_loss_dict['mae'] / len(testloader)
        ave_val_loss3 = val_running_loss_dict['smooth_l1'] / len(testloader)
        ave_val_loss = ave_val_loss3  # 假设 smooth_l1 是主要的验证指标

        self.logger.info(
            f"Validation - Epoch {epoch + 1}/{self.num_epochs} - mse: {ave_val_loss1:.3f}, mae: {ave_val_loss2:.3f}, smooth_l1: {ave_val_loss3:.3f}")

        # Save the best model
        if ave_val_loss < best_val_loss:
            best_val_loss = ave_val_loss
            self.save_model(label=f"best_model_{epoch + 1}_loss1_{ave_val_loss1:.3f}_loss2_{ave_val_loss2:.3f}_loss3_{ave_val_loss3:.3f}_validate")
            self.logger.info(f"Best model saved with smooth_l1 loss: {best_val_loss:.3f}")


        return


    @calculate_running_time
    def run(self, train_with_all_data=False, result=None):
        self.logger.debug("Begin training the model")
        if not train_with_all_data:
            self.mem_char_x_train = self.get_train_data(self.mem_processed_char_feature)
            self.mem_numer_x_train = self.get_train_data(self.mem_processed_numa_features)
            self.cpu_char_x_train = self.get_train_data(self.cpu_processed_char_feature)
            self.cpu_numer_x_train = self.get_train_data(self.cpu_processed_numa_features)
            self.system_char_x_train = self.get_train_data(self.system_processed_char_feature)
            self.system_numer_x_train = self.get_train_data(self.system_processed_numa_features)
            self.workload_char_x_train = self.get_train_data(self.workload_processed_char_feature)
            self.workload_numer_x_train = self.get_train_data(self.workload_processed_numa_features)

            self.mem_char_x_test = self.get_test_data(self.mem_processed_char_feature)
            self.mem_numer_x_test = self.get_test_data(self.mem_processed_numa_features)
            self.cpu_char_x_test = self.get_test_data(self.cpu_processed_char_feature)
            self.cpu_numer_x_test = self.get_test_data(self.cpu_processed_numa_features)
            self.system_char_x_test = self.get_test_data(self.system_processed_char_feature)
            self.system_numer_x_test = self.get_test_data(self.system_processed_numa_features)
            self.workload_char_x_test = self.get_test_data(self.workload_processed_char_feature)
            self.workload_numer_x_test = self.get_test_data(self.workload_processed_numa_features)

            self.y_train = self.processed_labels.iloc[self.train_indices, :]
            self.y_test = self.processed_labels.iloc[self.test_indices, :]

            self.mem_char_x_train_shape = self.mem_char_x_train.shape[1:]
            self.cpu_char_x_train_shape = self.cpu_char_x_train.shape[1:]
            self.system_char_x_train_shape = self.system_char_x_train.shape[1:]
            self.workload_char_x_train_shape = self.workload_char_x_train.shape[1:]
            self.mem_numer_x_train_shape = self.mem_numer_x_train.shape[1:]
            self.cpu_numer_x_train_shape = self.cpu_numer_x_train.shape[1:]
            self.system_numer_x_train_shape = self.system_numer_x_train.shape[1:]
            self.workload_numer_x_train_shape = self.workload_numer_x_train.shape[1:]

            self.shape = [self.mem_char_x_train_shape, self.mem_numer_x_train_shape, self.cpu_char_x_train_shape,
                          self.cpu_numer_x_train_shape,
                          self.system_char_x_train_shape, self.system_numer_x_train_shape,
                          self.workload_char_x_train_shape, self.workload_numer_x_train_shape]

            model = self.train()
            result = self.evaluate(self.y_test, self.y_predict)
            if self.isplot:
                save_path = self.output_path
                Train_predict_compare(self.configs, self.y_predict, self.y_test, save_path)
        else:
            self.logger.warning("Train model with all the data, without validation!")
            self.mem_char_x_train = self.data_reshape(self.mem_processed_char_feature)
            self.mem_numer_x_train = self.data_reshape(self.mem_processed_numa_features)
            self.cpu_char_x_train = self.data_reshape(self.cpu_processed_char_feature)
            self.cpu_numer_x_train = self.data_reshape(self.cpu_processed_numa_features)
            self.system_char_x_train = self.data_reshape(self.system_processed_char_feature)
            self.system_numer_x_train = self.data_reshape(self.system_processed_numa_features)
            self.workload_char_x_train = self.data_reshape(self.workload_processed_char_feature)
            self.workload_numer_x_train = self.data_reshape(self.workload_processed_numa_features)
            self.y_train = self.processed_labels

            self.mem_char_x_train_shape = self.mem_char_x_train.shape[1:]
            self.cpu_char_x_train_shape = self.cpu_char_x_train.shape[1:]
            self.system_char_x_train_shape = self.system_char_x_train.shape[1:]
            self.workload_char_x_train_shape = self.workload_char_x_train.shape[1:]
            self.mem_numer_x_train_shape = self.mem_numer_x_train.shape[1:]
            self.cpu_numer_x_train_shape = self.cpu_numer_x_train.shape[1:]
            self.system_numer_x_train_shape = self.system_numer_x_train.shape[1:]
            self.workload_numer_x_train_shape = self.workload_numer_x_train.shape[1:]
            self.shape=[self.mem_char_x_train_shape, self.mem_numer_x_train_shape, self.cpu_char_x_train_shape, self.cpu_numer_x_train_shape,
                   self.system_char_x_train_shape, self.system_numer_x_train_shape, self.workload_char_x_train_shape, self.workload_numer_x_train_shape]
            self.logger.info(f"system_processed_char_feature: {self.system_processed_char_feature.shape}")




            model = self.train_without_validate()
            self.copy_model()

        return result, "Noone"


    def get_train_data(self, data):
        print("data type: ", type(data))
        if data.empty:
            return data
        data = data.iloc[self.train_indices, :]
        data.reset_index(drop=True)
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns

        return data_reshape

    def data_reshape(self, data):
        x, y = data.shape
        # todo: x 1 y
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns
        # data_reshape = torch.tensor(data_reshape, dtype=torch.float32).to(self.device)
        return data_reshape

    def get_test_data(self, data):
        if data.empty:
            return data
        data = data.iloc[self.test_indices, :]
        data.reset_index(drop=True)
        x, y = data.shape
        data_reshape = np.reshape(data.values, (x, y, 1))
        columns = data.columns
        return data_reshape

    def evaluate(self, y_test, y_predict):

        results = pd.DataFrame()
        # Calculate the Absolute Error (AE) for dataframe
        cols = self.label_name_list
        true_name = self.true_col
        predict_name = self.predict_col

        y_test = np.reshape(y_test, (y_test.shape[0], y_test.shape[2]))
        if len(y_predict.shape) == 3:
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

        return results

    def customed_mse(self, y_true, y_pred):
        weights = K.square(y_pred - y_true)
        squared_difference = K.square(y_pred - y_true)
        weighted_squared_difference = squared_difference * weights
        return K.mean(weighted_squared_difference, axis=-1)

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