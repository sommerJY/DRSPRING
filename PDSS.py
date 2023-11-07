
import sys
import random
import shutil
import math
import os
import pandas as pd
import numpy as np
import json
import networkx as nx
import copy
import pickle
import joblib
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import sklearn
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = sklearn.preprocessing.OneHotEncoder
from sklearn import metrics
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
import sklearn.model_selection

import datetime
from datetime import *

import torch_geometric.nn as pyg_nn
import torch_geometric.transforms as T
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import torch_geometric.nn.conv

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.QED import qed

from scipy.sparse import coo_matrix
from scipy import sparse
from scipy import stats

now_path = os.getcwd()
sys.path.append(now_path+'/utils')

from M2_Model import *
from M2_prep_input import *


parser = argparse.ArgumentParser()
parser.add_argument("save_file", type=str, help='path to save')    # extra value
parser.add_argument("--mode", type=str, default='train', help='select mode')  # extra value
parser.add_argument("--saved_model", type=str, default=None, help='saved model location')  # extra value
parser.add_argument("--early_stopping", type=str, default=None, help='whether to perform early stopping')  # extra value
parser.add_argument("--DrugAsmiles", type=str, default=None, help='put drug A smiles file here')   # extra value
parser.add_argument("--DrugBsmiles", type=str, default=None, help='put drug B smiles file here')   # extra value
parser.add_argument("--M1_DrugA", type=str, default=None, help='put M1 derived drug A Result name here')   # extra value
parser.add_argument("--M1_DrugB", type=str, default=None, help='put M1 derived drug B Result name here')   # extra value
parser.add_argument("--Basal_Cell", type=str, default=None, help='put new cell line basal expression')   # extra value

args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)



if args.mode=='train':
	print('Preparing training data...')
	T_train, T_val, T_test = make_merged_data()
elif args.mode=='new_data':
	print('Processing Your Input...')
else :
	print('Error. Please check the mode')


start_time = datetime.now()


config = {
	'max_epoch': int(1000),
	'G_chem_layer' : int(3), 
	'G_chem_hdim' : int(16), 
	'G_exp_layer' : int(4), 
	'G_exp_hdim' : int(32),
	'dsn_layer' : '128-64-32', 
	'snp_layer' : '32-16-8', 
	'dropout_1' : float(0.1),
	'dropout_2' : float(0.1),
	'batch_size' : int(16),
	'learning_rate' : float(0.001),
	'EarlyStop' : args.early_stopping
}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


if args.mode=='train':
	training_model(args.save_file, T_train, T_val, T_test, args.early_stopping, device, config)
	print("Finished training and saved the model")
elif args.mode=='new_data':
	if args.Basal_Cell == None :
		pred_cell_synergy(args.save_file, config, args.saved_model, args.DrugAsmiles, args.DrugBsmiles, args.M1_DrugA, args.M1_DrugB, None)
	else :
		pred_cell_synergy(args.save_file, config, args.saved_model, args.DrugAsmiles, args.DrugBsmiles, args.M1_DrugA, args.M1_DrugB, args.Basal_Cell)
	print("Finished testing!")

end_time = datetime.now()
print('Total Time Spent : {}'.format(end_time - start_time))



