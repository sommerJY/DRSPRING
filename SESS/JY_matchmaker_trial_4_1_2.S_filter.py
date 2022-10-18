
< failed exp >


import rdkit
import os
import os.path as osp
from math import ceil
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch.utils.data import Dataset

from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool #DiffPool
from torch_geometric.nn import SAGEConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pickle
import joblib
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import sklearn
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = sklearn.preprocessing.OneHotEncoder
import datetime
from datetime import *
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
												
# Graph convolution model
import torch_geometric.nn as pyg_nn
# Graph utility function
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import torch_geometric.nn.conv
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem

import networkx as nx
import copy
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy import stats
import sklearn.model_selection

import sys
import random
import shutil
import math

import ray
from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd





WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_4.1/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/' #
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET/'
cid_PATH = '/home01/k006a01/01.DATA/PC_ATC_LIST/'



# our sider 
JY_SIDER = pd.read_csv("/st06/jiyeonH/13.DD_SESS/SIDER/sider_all.csv", sep = ',')
JY_SIDER['CID'] = [int(a.split('CID1')[1]) for a in list(JY_SIDER.cid)]
JY_SIDER_CID = list(set(JY_SIDER['CID']))


# 응 그러하다. 내 의도대로 진행하고 싶었지만 문제는 176 중에서 SIDER 랑 맞는게 49개 밖에 없는거 실화냐 
ori_in_JYSIDER = [a for a in ori_176 if a in JY_SIDER_CID]


# OFFSIDES
import json
JY_OFF = pd.read_csv("/st06/jiyeonH/13.DD_SESS/OFFSIDES/OFFSIDES.csv", sep = ',', low_memory = False)
JY_OFF = JY_OFF.sort_values('drug_rxnorn_id')
JY_OFF = JY_OFF.reset_index(drop = True)
JY_OFF = JY_OFF.loc[0:3206556,]
JY_OFF['RXN_ID'] = [int(a) for a in list(JY_OFF['drug_rxnorn_id'])]

OFF_RX = list(set(JY_OFF['drug_rxnorn_id']))

with open('/st06/jiyeonH/13.DD_SESS/PC_RxNORM/PubChemAnnotations_NLM_RxNorm_Terminology_heading_RXCUI.json', 'r') as j_file:
    RXCUI = json.load(j_file)


# tot list 
RXCUI['Annotations']['Annotation']
# 챙길거 
RXCUI['Annotations']['Annotation'][0]['SourceID']
# CID
RXCUI['Annotations']['Annotation'][0]['LinkedRecords']['CID']



RXCUI_DF = pd.DataFrame(columns=['SourceID', 'CID'])
for a in RXCUI['Annotations']['Annotation'] :
    SourceID = [int(b) for b in [a['SourceID']]]
    try :
        CID = a['LinkedRecords']['CID']
    except :
        CID = ['' for a in range(len(SourceID))]
    mini_df = pd.DataFrame({'SourceID' :SourceID, 'CID':CID })
    RXCUI_DF = pd.concat([RXCUI_DF, mini_df])

RXCUI_DF_OFF = pd.merge(JY_OFF, RXCUI_DF, left_on = 'RXN_ID', right_on = 'SourceID', how = 'left')
RXCUI_DF_OFF_check = list(set(RXCUI_DF_OFF['CID'])) # 765

ori_in_JYOFF = [a for a in ori_176 if a in RXCUI_DF_OFF_check] # 18... 오지게 적네 


# 58 (49+18)

JY_comp_cids = list(set(ori_in_JYSIDER + ori_in_JYOFF))
JY_comp_cids


tmp1 = A_B_C_S_SET[A_B_C_S_SET.drug_row_cid.isin(JY_comp_cids)]
tmp2 = tmp1[tmp1.drug_col_cid.isin(JY_comp_cids)]


## similarity matrix by side effect 

DSGAT_dir = '/st06/jiyeonH/11.TOX/DSGAT/DSGAT-master/original_data/'

#  drug - SE term - Freq 
sup_1 = pd.read_csv(DSGAT_dir+'Supplementary_Data_1.txt', sep = '\t')
#  drug - SE term - Source (근데 1에 있는 내용이 2에 다 있지는 않음)
sup_2 = pd.read_csv(DSGAT_dir+'Supplementary_Data_2.txt', sep = '\t')
# SE term - TopMedDRA Term
sup_3 = pd.read_csv(DSGAT_dir+'Supplementary_Data_3.txt', sep = '\t')
# SE term - SecondLevelMedDRATerm Term
sup_4 = pd.read_csv(DSGAT_dir+'Supplementary_Data_4.txt', sep = '\t')




import scipy 
import scipy.io as sio
import csv

MAT_dir = '/st06/jiyeonH/11.TOX/DSGAT/DSGAT-master/data_WS/'

node_label_file = scipy.io.loadmat(MAT_dir+'side_effect_label_750.mat')
node_label = node_label_file['node_label'] # (994, 243)


raw_freq_file = sio.loadmat(MAT_dir+'raw_frequency_750.mat')

# >>> raw_freq_file.keys()
# dict_keys(['__header__', '__version__', '__globals__', 'R', 'drugs', 'sideeffects'])
raw_freq_file['R'].shape -> (750, 994)
# 왜 근데 750개 한정이지. SIDER 에서는 일단 더 많지 않았나 -> 얘네도 다른 논문에서 따온거 
raw = raw_freq_file['R']

index_pair = np.where(raw != 0)
index_arr = np.arange(0, index_pair[0].shape[0], 1)






def load_drug_smile(file):
    reader = csv.reader(open(file))
    drug_dict = {}
    drug_smile = []
    for item in reader:
        name = item[0]
        smile = item[1]
        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)
    return drug_dict, drug_smile

drug_dict, drug_smile = load_drug_smile(MAT_dir+'drug_SMILES_750.csv')




freq mat 이 누군지가 중요 
그걸 10CV 로 보았음 

k=10
index_pair = np.where(raw != 0)
index_arr = np.arange(0, index_pair[0].shape[0], 1) # (37071,)
np.random.shuffle(index_arr)
x = [] # len 10 -> index 를 10개 리스트로 만든거라서. 
n = math.ceil(index_pair[0].shape[0] / k) # 3708


for i in range(k): # K(10) 개로 자르는 과정임 
    if i == k - 1:
        x.append(index_arr[0:].tolist())
    else:
        x.append(index_arr[0:n].tolist())
        index_arr = index_arr[n:]


dic = {}
for i in range(k): # 그래서 해당 CV 의 애들은 0 matrix 로 만들어버리기 
    mask = np.ones(raw.shape)
    mask[index_pair[0][x[i]], index_pair[1][x[i]]] = 0
    dic['mask' + str(i)] = mask

# dic['mask9'].shape
# (750, 994) # 994 sim 은 어디서 나왔을까 
array([[1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       ...,
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 1., ..., 1., 1., 1.],
       [1., 1., 0., ..., 1., 1., 1.]])


mask mat 로 사용될 애인것 같음 

test_mask = dic['mask9']

frequencyMat = raw * test_mask
이렇게 되면 원개 raw 는 
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 4., ..., 0., 0., 0.],
       [4., 0., 0., ..., 0., 0., 0.],
       [0., 0., 5., ..., 0., 0., 0.]])

이랬는데, 

array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.],
       ...,
       [0., 0., 4., ..., 0., 0., 0.],
       [4., 0., 0., ..., 0., 0., 0.],
       [0., 0., 0., ..., 0., 0., 0.]]) # 여기서 5 하나 빠진거 볼수 있음 





frequencyMat[i]




from sklearn.neighbors import kneighbors_graph
from sklearn.decomposition import PCA
# neighbors.kneighbors_graph(X, n_neighbors, *)
# Compute the (weighted) graph of k-Neighbors for points in X.
# sklearn.neighbors.kneighbors_graph(X, n_neighbors, *, mode='connectivity', metric='minkowski', p=2, metric_params=None, include_self=False, n_jobs=None)

frequencyMat = frequencyMat.T # 약물 * label 
metric = 'cosine' # '0: cosine, 1: jaccard, 2: euclidean' 
knn = 10 # 이웃 기준

if pca:
    pca_ = PCA(n_components=256)
    similarity_pca = pca_.fit_transform(frequencyMat) # 994 * 256 그래서 SE sim 만듬 
    print('PCA 信息保留比例： ')
    print(sum(pca_.explained_variance_ratio_))
    A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
else:
    A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)

G = nx.from_numpy_matrix(A.todense())
edges = []

for (u, v) in G.edges():
    edges.append([u, v])
    edges.append([v, u])

edges = np.array(edges).T
edges = torch.tensor(edges, dtype=torch.long)

# load  side_effect_label mat ，用node_label做点信息 994*243
node_label = scipy.io.loadmat(side_effect_label)['node_label']
feat = torch.tensor(node_label, dtype=torch.float)
sideEffectsGraph = Data(x=feat, edge_index=edges)

raw_frequency = scipy.io.loadmat(raw_file)
raw = raw_frequency['R']

# make data_WS Pytorch mini-batch processing ready
train_data = myDataset(root='data_WS', dataset='drug_sideEffect_data' + str(id - 1))
train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
test_loader = DataLoader(train_data, batch_size=1, shuffle=False)
