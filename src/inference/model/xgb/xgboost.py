#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/10/2023 10:49 PM
# @Author  : xiaomanl
# @File    : xgb.py
# @Software: PyCharm

import logging
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import os
import pandas as pd
from joblib import dump
import joblib
from utils import Train_predict_compare, calculate_running_time
from model.base_class import BaseModel
import sys
import glob
from sklearn.ensemble import RandomForestRegressor


#import pydotplus
#import dtreeviz

class XgboostModel(BaseModel):
    def __init__(self, configs=None, processed_features=None, processed_labels=None):
        super().__init__( configs, processed_features, processed_labels)
        self.logger = logging.getLogger("XgboostModel")
        self.mode_config = self.configs["XgboostModel_config"]
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
        self.learning_rate = self.train_param["learning_rate"]
        self.verbosity = self.train_param["verbosity"]
        self.booster = self.train_param["booster"]
        self.n_jobs = self.train_param["n_jobs"]
        self.random_state = self.train_param["random_state"]
        self.max_delta_step = self.train_param["max_delta_step"]
        self.colsample_bytree = self.train_param["colsample_bytree"]
        self.subsample = self.train_param["subsample"]
        self.multi_output = True if self.train_param["multi_output"] and self.configs["use_multiple_label"] else False
        # self.eta = self.train_param["eta"]

        self.use_pre_trained_model = self.configs["use_train_model"]
        self.pre_trained_model_path = self.configs["pre_trained_model_path"]
        self.select_model = self.configs["select_model"]
        self.if_feature_ranking = self.mode_config["feature_ranking"]
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
        self.logger.info("Begin predict the xgboost")
        y_predict = model.predict(x_test)
        y_predict = pd.DataFrame(y_predict, columns=self.predict_col)
        y_predict = y_predict
        square_r = model.score(x_test, y_test)
        self.logger.warning("Train model square_r: {}".format(square_r))
        self.logger.info(
            f"The shape of y_predict is: \033[1;34;34m{y_predict.shape[0]}\033[0m rows and \033[1;34;34m{y_predict.shape[1]}\033[0m columns.")

        return y_predict

    def plot_model(self):
        self.logger.info("#"*50)
        # plot_tree(self.fcn,fmap='',num_trees=0,rankdir='UT',ax=None)
        # plt.show()
        model = self.model
        # graph = xgb.to_graphviz(self.fcn.get_booster().trees_to_dataframe())
        graph = xgb.to_graphviz(self.model.get_booster(), num_trees=0)
        graph.format = 'png'
        curPath = os.path.abspath(os.path.dirname(__file__))
        graph.view(os.path.join(curPath,f"model_view_2"))

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

    def feature_ranking(self, model, x_train):

        result = pd.DataFrame(columns=[f"rank_{i}" for i in range(1, 51)])

        # Loop through each group of data
        feature_set = set()
        variation_summary = pd.DataFrame()
        feature_importance = model.get_booster().get_score(importance_type="gain")
        total = sum(feature_importance.values())
        feature_importance = {key: round(val / total * 100, 3) for key, val in feature_importance.items()}

        # Sort the feature importance in descending order
        sorted_feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        for i, (column, value) in enumerate(sorted_feature_importance):
            result.at[0, f"rank_{i+1}"] = f"{value}%:{column}"

        header = set([key for key, value in feature_importance.items() if value > 0.1])
        if not feature_set:
            feature_set.update(header)
        else:
            feature_set = feature_set.union(header)


        # Add the top 10 feature ranking to the result dataframe

        feature_list = pd.DataFrame({"feature_list": list(feature_set)})
        all_feature_set = set(list(x_train.columns))
        slience_header = pd.DataFrame(all_feature_set - feature_set)
        slience_header.to_csv(f'{self.output_path}/slience_header.csv', index=False, header=False)

        result.to_csv(f'{self.output_path}/feature_ranking.csv', index=False)

        counts = result[[f"rank_{i}" for i in range(1, 41)]].apply(pd.Series.value_counts)
        counts.to_csv(f'{self.output_path}/feature_ranking_count.csv')
        feature_list.to_csv(f'{self.output_path}/feature_list.csv', index=False)
        self.logger.info(f"Feature ranking results are saving to {self.output_path}/feature_ranking.csv")

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


