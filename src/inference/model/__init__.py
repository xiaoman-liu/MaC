#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 7/3/2023 6:03 PM
# @Author  : xiaomanl
# @File    : __init__.py
# @Software: PyCharm

from model import lr, fcn, fcn_vec, xgb,base_class
from model.lr import LinearModel
from model.xgb import  XgboostModel
from model.fcn import FCN
from model.fcn_vec import FCN_Vec
from model.base_class import BaseModel
from model.resnet import ResNet
from model.atten_resnet import AttenResNet
from model.res_trans_net import MultiAttenResNet
from model.rf import RandomForest
from model.svm import SVMModel
from model.lstm import LSTMModel
from model.ridge.ridge import Ridge
from model.group_multi_atten_resnet import GroupMultiAttenResNet
from model.mamba.mamba import Mamba
# from model.mamba.mamba2 import Mamba2
