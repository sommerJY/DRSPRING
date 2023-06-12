


# 되는지 확인해야함 
salloc --partition=1gpu -N 1 -n 1 --tasks-per-node=1 --comment="test"

conda activate DS_1


import numpy as np
import pandas as pd
import pickle 
import gzip

import os, sys

import json

import matplotlib.pyplot as plt

import keras 
import tensorflow
import keras as K
import tensorflow as tf
from keras import backend
from tensorflow.keras.backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout
import random


TOOL_PATH = '/home01/k040a01/03.DeepSynergy/01.Data/'
JY_DATA_PATH = '/home01/k040a01/02.M3V5/M3V5_W32_349_DATA/'


hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model

