#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/2022 4:06 PM
# @Author  : xiaomanl
# @File    : lr.py
# @Software: PyCharm
from sklearn.multioutput import MultiOutputRegressor
from statsmodels.stats.outliers_influence import variance_inflation_factor
from joblib import dump
import joblib
import itertools
from tabulate import tabulate
import logging
import pandas as pd
import os
import sys
import numpy as np
from utils import Train_predict_compare, calculate_running_time, mkdir, NorScaler, MinMaxScaler, OneHotEncoder
from model.base_class import BaseModel
import yaml
import joblib
import glob






class Ridge(BaseModel):
    def __init__(self, configs, processed_features, processed_labels):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("Ridge")
        self.mode_config = self.configs["Ridge_config"]
        self.isplot = self.mode_config["ifplot"]
        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col":self.true_col})
        self.label_name_list = self.configs["label_name_list"]
        self.model_name = self.configs["select_model"]
        self.use_pre_trained_model = self.configs["use_train_model"]
        self.select_model = self.configs["select_model"]
        self.test_names = self.configs["test_names"]
        self.model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs["model_history_path"]


    def build_model(self):
        file = glob.glob(self.model_path + '/*.joblib')[0]
        model = joblib.load(file)
        self.logger.info("Load fcn from: {}".format(file))
        return model


    def predict(self, model, x_test, y_test):
        """
        load saved fcn and predict_orig
        :return:
        """
        # if self.configs["use_train_model"]:
        #     #fcn = joblib.load(open(self.output_path+ "/lr.joblib", "rb"))
        #     fcn = joblib.load(open("./output" + "/lr.joblib", "rb"))
        y_predict = model.predict(x_test)
        y_predict = pd.DataFrame(y_predict, columns=self.predict_col)
        # x_test.loc[x_test.isna().any(axis=1), x_test.isna().any(axis=0)]

        square_r = model.score(x_test, y_test)
        self.logger.warning("Train fcn square_r: {}".format(square_r))
        self.logger.info(
            f"The shape of y_predict is: \033[1;34;34m{y_predict.shape[0]}\033[0m rows and \033[1;34;34m{y_predict.shape[1]}\033[0m columns.")

        return y_predict

    def evaluate(self, y_test, y_predict):
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
        self.logger.info("Begin predict the linear")


        x_test = self.processed_features
        y_test = self.processed_labels
        model = self.build_model()
        y_predict = self.predict(model, x_test, y_test)
        result = self.evaluate(y_test, y_predict)
        if self.isplot:
            save_path = self.output_path
            Train_predict_compare(self.configs, y_predict, y_test, save_path)


        return result



