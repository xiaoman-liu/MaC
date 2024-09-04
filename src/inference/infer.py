#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/2022 4:49 PM
# @Author  : xiaomanl
# @File    : test.py
# @Software: PyCharm

import os
import sys

import pandas as pd
from pathlib import Path
import logging
import joblib

from utils import calculate_running_time, mkdir, save_data_encoder, read_config, set_logger
from generate_data import DataLoader
from data_preprocess import FeatureEncoder, GroupFeatureEmbedding
from data_postprocess import DataPostprocessor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils.additional import merge_K_fold_results
from utils import param_search, DatabaseManager
from model import LinearModel, XgboostModel, FCN, FCN_Vec, ResNet, AttenResNet, MultiAttenResNet, RandomForest, SVMModel, LSTMModel, Ridge, GroupMultiAttenResNet, Mamba
import glob
from tabulate import tabulate
os.environ["CUDA_VISIBLE_DEVICES"]="0"
# import pydevd_pycharm
# pydevd_pycharm.settrace('192.168.3.48', port=2245, stdoutToServer=True, stderrToServer=True)

class Predictor:
    def __init__(self, output_path="../../infer_results", config_path="./"):
        self.root_dir = Path(__file__).resolve().parent
        self.configs = read_config(self.root_dir, config_path, output_path)
        self.logger = self._set_logger()
        self.select_model = self.configs["select_model"]
        self.output_path = self.configs["output_path"]
        self.if_label_scale = self.configs["if_label_scale"]

        self.label_scaler = StandardScaler()
        self.workload_name = self.configs["workload_names"][0]
        self.label_scale = self.configs["label_scale"].get(self.workload_name, 1)
        self.encoder_path = self.configs["encoder_path"]

        self.model_dict = self.configs["model_dict"]
        _, self.model_name = self.model_dict[self.select_model].rsplit('.', 1)
        self.model = globals()[self.model_name]
        self.infer_upload_sql = self.configs["infer_upload_sql"]


    def get_model(self):
        if self.select_model == "Ridge" or self.select_model == "Lasoo" or self.select_model == "ElasticNet":
            model =  globals()["LinearModel"]
        else:
            model = globals()[self.model_name]
        return model

    def _set_logger(self):
        logger = set_logger(self.configs)
        return logging.getLogger("InferModule")


    def load_data(self):
        self.data_loader = DataLoader(self.configs)
        self.filter_data, self.label,self.all_train_data = self.data_loader.run()

    def preprocess_data(self):
        self.configs["filtered_columns"] = list(self.filter_data.columns)
        if self.select_model == "GroupMultiAttenResNet" or self.select_model == "Mamba":
            self.data_processor = GroupFeatureEmbedding(self.configs)
        else:
            self.data_processor = FeatureEncoder(self.configs)
        self.processed_feature = self.data_processor.run(self.filter_data)
        if self.if_label_scale:
            label_temp = self.label.values
            label_temp = self.label_scaler.fit_transform(label_temp)
            self.label = pd.DataFrame(label_temp, columns=self.label.columns, index=self.label.index)
        else:
            self.label /= self.label_scale

    def only_predict(self, save_path, single_model_path):
        self.param_search = self.select_model == "XgboostModel" and self.configs["XgboostModel_config"]["param_search"]
        mkdir(save_path)

        self.model_module = self.model(self.configs, self.processed_feature, self.label)
        self.predicted_results, metric_overall = self.model_module.infer(single_model_path)
        return self.predicted_results, metric_overall

    def postprecess_data(self, save_path=None):
        # if self.if_label_scale:
        #     label_temp = self.label.values
        #     label_temp = self.label_scaler.inverse_transform(label_temp)
        #     self.label = pd.DataFrame(label_temp, columns=self.label.columns, index=self.label.index)
        # else:
        #     self.label *= self.label_scale
        self.data_postprocessor = DataPostprocessor(save_path, self.configs, self.filter_data, self.label, self.predicted_results)
        self.all_validate_dataset = self.data_postprocessor.run(label_scaler=self.label_scaler)
        return self.all_validate_dataset

    def upload_sql(self, variation_data="", test_results=""):
        uploader = DatabaseManager(self.configs)
        uploader.run("inference_result", self.all_validate_dataset)



    @calculate_running_time
    def run(self):
        model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs[
            "model_history_path"]
        file = glob.glob(model_path + '/*.pth')
        metrics = []
        self.load_data()
        self.preprocess_data()
        for f in file:
            try:
                if "weight" in f:
                    continue
                self.logger.info("Begin to load model from {}".format(f))
                model_path = f
                self.logger.info("Begin to infer")
                results, metric_overall = self.only_predict(self.output_path, model_path)
                upload_data = self.postprecess_data()
                upload_data.to_csv(self.configs["output_path"] + "/infer_results.csv", index=False)
                metric_overall["model_path"] = model_path
                metrics.append(metric_overall)

                if self.infer_upload_sql:
                    self.upload_sql()
            except Exception as e:
                self.logger.error("Failed to infer model from {}".format(f))
                self.logger.error(str(e))
                continue
        df = pd.DataFrame(metrics)
        df_sorted = df.sort_values(by="Overall MAE")
        table_string = tabulate(df_sorted, headers='keys', tablefmt='grid', showindex=False)
        self.logger.info("\n" + table_string)
        # self.logger.info(df_sorted.to_string(index=False))


        return self.configs



if __name__ == "__main__":
    Predictor = Predictor()
    configs = Predictor.run()



