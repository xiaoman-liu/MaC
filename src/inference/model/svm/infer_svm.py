#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 11/22/2022 4:06 PM
# @Author  : xiaomanl
# @File    : lr.py
# @Software: PyCharm
from sklearn import svm
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
from utils import Train_predict_compare, calculate_running_time, mkdir, NorScaler, MinMaxScaler, OneHotEncoder, DatabaseManager
from model.base_class import BaseModel
import yaml
import shutil
import glob






class SVMModel(BaseModel):
    def __init__(self, configs, processed_features, processed_labels):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("SVMModel")
        self.mode_config = self.configs["SVMModel_config"]
        self.isplot = self.mode_config["ifplot"]
        self.predict_col = ["Predict_" + item  for item in self.configs["label_name_list"]]
        self.true_col = ["True_" + item  for item in self.configs["label_name_list"]]
        self.configs.update({"predict_col": self.predict_col, "true_col":self.true_col})
        self.label_name_list = self.configs["label_name_list"]
        self.model_name = self.configs["select_model"]
        self.use_pre_trained_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]
        self.select_model = self.configs["select_model"]
        self.test_names = self.configs["test_names"]
        self.config_save_path = self.configs["config_save_path"]
        self.coff2sql = self.mode_config["coff2sql"] if "coff2sql" in self.mode_config else 0
        self.train_params = self.mode_config["train_param"]
        self.train_with_all_data = self.configs["train_with_all_data"]
        self.workload_name = self.configs["workload_names"][0]
        self.fr_tb = self.configs["feature_ranking_tb"].get(self.workload_name, None)

        self.C = self.train_params["C"]
        self.epsilon = self.train_params["epsilon"]
        self.kernel = self.train_params["kernel"]
        self.gamma = self.train_params["gamma"]
        self.degree = self.train_params["degree"]
        self.coef0 = self.train_params["coef0"]
        self.shrinking = self.train_params["shrinking"]
        self.tol = self.train_params["tol"]
        self.cache_size = self.train_params["cache_size"]
        self.class_weight = self.train_params["class_weight"]
        self.verbose = self.train_params["verbose"]
        self.max_iter = self.train_params["max_iter"]
        self.decision_function_shape = self.train_params["decision_function_shape"]
        self.break_ties = self.train_params["break_ties"]
        self.random_state = self.train_params["random_state"]

        self.model_path = self.configs["model_path"] if self.configs["model_path"] else self.configs[
            "model_history_path"]



    def build_model(self):
        file = glob.glob(self.model_path + '/*.joblib')[0]
        model = joblib.load(file)
        self.logger.info("Load model from: {}".format(file))
        return model

    def upload_feature_importance2sql(self, data):

        self.logger.info("<---------------------------begin to upload model coff")

        engine = DatabaseManager(self.configs)
        table_name = self.fr_tb
        if table_name is None:
            self.logger.warning("You don't have feature ranking table")
            return
        engine.insert_data(table_name, data)
        self.logger.info("successfully upload to sql -------------------------------->")



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
        self.logger.warning("Train model square_r: {}".format(square_r))
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
        self.logger.info("Begin predict the model")


        x_test = self.processed_features
        y_test = self.processed_labels
        model = self.build_model()
        y_predict = self.predict(model, x_test, y_test)
        result = self.evaluate(y_test, y_predict)
        if self.isplot:
            save_path = self.output_path
            Train_predict_compare(self.configs, y_predict, y_test, save_path)


        return result



