#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/10/2023 10:49 PM
# @Author  : xiaomanl
# @File    : xgb.py
# @Software: PyCharm

import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
from joblib import dump
import joblib
from utils import Train_predict_compare, calculate_running_time, DatabaseManager
from model.base_class import BaseModel
import sys
import glob
import yaml
from sklearn.multioutput import MultiOutputRegressor
from tabulate import tabulate
import numpy as np


#import pydotplus
#import dtreeviz

class RandomForest(BaseModel):
    def __init__(self, configs=None, processed_features=None, processed_labels=None):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("RandomForest")
        self.mode_config = self.configs["RandomForest_config"]
        self.train_param = self.mode_config["train_param"]
        self.isplot = self.mode_config["ifplot"]
        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col":self.true_col})
        self.label_name_list = self.configs["label_name_list"]
        self.model_name = self.configs["select_model"]
        self.use_train_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]

        self.max_depth=self.train_param["max_depth"]
        self.n_estimators = self.train_param["n_estimators"]
        self.verbosity = self.train_param["verbosity"]
        self.n_jobs = self.train_param["n_jobs"]
        self.oob_score = True if self.train_param["oob_score"] else False
        self.random_state = self.train_param["random_state"]
        # self.eta = self.train_param["eta"]

        self.use_pre_trained_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]
        self.select_model = self.configs["select_model"]
        self.if_feature_ranking = self.mode_config["feature_ranking"]
        self.config_save_path = self.configs["config_save_path"]
        self.use_multiple_label = self.configs["use_multiple_label"]
        self.coff2sql = self.mode_config["coff2sql"] if "coff2sql" in self.mode_config else 0
        self.train_with_all_data = self.configs["train_with_all_data"]
        self.workload_name = self.configs["workload_names"][0]
        self.model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs["model_history_path"]



    def build_model(self):
        file = glob.glob(self.model_path + '/*.joblib')[0]
        model = joblib.load(file)
        self.logger.info("Load model from: {}".format(file))
        return model

    def predict(self, model, x_test, y_test):
        """
        load saved fcn and predict_orig
        :return:
        """
        # if self.configs["use_train_model"]:
        #
        #     #fcn = joblib.load(open(self.output_path+ "/linear_model.joblib", "rb"))
        #     fcn = joblib.load(open("./output" + "/xgb.joblib", "rb"))
        y_predict = model.predict(x_test)
        y_predict = pd.DataFrame(y_predict, columns=self.predict_col)
        y_predict = y_predict
        square_r = model.score(x_test, y_test)
        self.logger.warning("Train model square_r: {}".format(square_r))
        self.logger.info(
            f"The shape of y_predict is: \033[1;34;34m{y_predict.shape[0]}\033[0m rows and \033[1;34;34m{y_predict.shape[1]}\033[0m columns.")

        return y_predict


    def evaluate(self, y_test, y_predict):

        # Calculate the Absolute Error (AE) for dataframe

        results = pd.DataFrame()

        # Calculate the Absolute Error (AE) for dataframe
        for i in range(len(self.label_name_list)):
            label_name = self.label_name_list[i]
            true_name = self.true_col[i]
            predict_name = self.predict_col[i]

            y_test = y_test.reset_index(drop=True)
            ae = abs(y_test[label_name] - y_predict[predict_name])
            # Calculate the Mean Absolute Error (MAE)
            se = (y_test[label_name] - y_predict[predict_name]) ** 2

            # Calculate the Mean Absolute Percentage Error (MAPE)
            ape = abs(y_test[label_name] - y_predict[predict_name]) / y_test[label_name] * 100


            result = pd.concat([y_test[label_name], y_predict[predict_name], ae, se, ape], axis=1)
            result.columns = [f'{true_name}', f'{predict_name}', f'AE_{label_name}', f'SE_{label_name}', f'APE(%)_{label_name}']
            results = pd.concat([results, result], axis=1)
        return results



    @calculate_running_time
    def infer(self, train_with_all_data=False, result=None):
        """
        fcn train and predict_orig
        :return:
        """
        self.logger.info("Begin predict the model")


        x_test = self.processed_features
        y_test = self.processed_labels
        model = self.build_model()
        y_predict = self.predict(model,x_test, y_test)
        result = self.evaluate(y_test, y_predict)
        if self.isplot:
            save_path = self.output_path
            Train_predict_compare(self.configs, y_predict, y_test, save_path)


        return result


