# pyg 로 explain 하기 



# 모델 하나를 데려와야함 
# 아예 진행된거 기반으로
# 그리고 그거에 맞게 데이터를 좀 가공해야할것 같은 느낌  




explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

node_index = 10 # which node index to explain
explanation = explainer(data.x, data.edge_index, index=node_index) # 이게 끝인가봄? 




예시 데이터 관련 
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
dataset = ExplainerDataset(
    graph_generator=BAGraph(num_nodes=300, num_edges=5),
    motif_generator='house',
    num_motifs=80,
)

# Generate Barabási-Albert base graph
graph_generator = BAGraph(num_nodes=300, num_edges=500)
# Create the InfectionDataset to the generated base graph
dataset = InfectionDataset(
    graph_generator=graph_generator,
    num_infected_nodes=50,
    max_path_length=3
)





explanation.visualize_feature_importance(feature_importance.png, top_k=10)

explanation.visualize_graph('subgraph.png', backend="networkx")






####################3
final 예시 코드 
import os.path as osp

import torch
import torch.nn.functional as F

from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.nn import GCNConv

import pandas as pd 
import numpy as np
import os 


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
from rdkit import DataStructs

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


dataset = 'Cora'
path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, dataset)
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)
    # 
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

#러닝은 러닝대로 
# explainer 는 따로 싸줘야하는듯 

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)
node_index = 10
explanation = explainer(data.x, data.edge_index, index=node_index)
print(f'Generated explanations in {explanation.available_explanations}')

path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")



################################################## 
################################################## 
################################################## 




class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, cell_dim ,
	out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.cell_dim = cell_dim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_chem, self.G_hiddim_chem)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_Common_dim)])
		self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		##
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_Common_dim)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		##
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_Common_dim+self.G_Common_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		##
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1]+self.cell_dim , self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		for conv in self.G_convs_1_chem :
			conv.reset_parameters()
		for bns in self.G_bns_1_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def calc_batch_label (self, syn, feat) :
		batchnum = syn.shape[0]
		nodenum = feat.shape[0]/batchnum
		Num = [a for a in range(batchnum)]
		Rep = np.repeat(Num, nodenum)
		batch_labels = torch.Tensor(Rep).long()
		if torch.cuda.is_available():
			batch_labels = batch_labels.cuda()
		return batch_labels
	#
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, cell, syn ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
		Exp_batch_label = self.calc_batch_label(syn, EXP1)
		#
		for G_1_C in range(len(self.G_convs_1_chem)):
			if G_1_C == len(self.G_convs_1_chem)-1 :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = F.dropout(Drug1_F, p=self.inDrop, training=self.training)
				Drug1_F = self.pool(Drug1_F, Drug_batch_label )
				Drug1_F = self.tanh(Drug1_F)
				G_1_C_out = Drug1_F
			else :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = self.G_bns_1_chem[G_1_C](Drug1_F)
				Drug1_F = F.elu(Drug1_F)
		#
		for G_2_C in range(len(self.G_convs_1_chem)):
			if G_2_C == len(self.G_convs_1_chem)-1 :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = F.dropout(Drug2_F, p=self.inDrop, training=self.training)
				Drug2_F = self.pool(Drug2_F, Drug_batch_label )
				Drug2_F = self.tanh(Drug2_F)
				G_2_C_out = Drug2_F
			else :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = self.G_bns_1_chem[G_2_C](Drug2_F)
				Drug2_F = F.elu(Drug2_F)
		#
		for G_1_E in range(len(self.G_convs_1_exp)):
			if G_1_E == len(self.G_convs_1_exp)-1 :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = F.dropout(EXP1, p=self.inDrop, training=self.training)
				EXP1 = self.pool(EXP1, Exp_batch_label )
				EXP1 = self.tanh(EXP1)
				G_1_E_out = EXP1
			else :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = self.G_bns_1_exp[G_1_E](EXP1)
				EXP1 = F.elu(EXP1)
		#
		for G_2_E in range(len(self.G_convs_1_exp)):
			if G_2_E == len(self.G_convs_1_exp)-1 :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, Exp_batch_label )
				EXP2 = self.tanh(EXP2)
				G_2_E_out = EXP2
			else :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = self.G_bns_1_exp[G_2_E](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.relu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.relu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2, cell ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else :
				X = self.SNPs[L3](X)
		return X


OLD_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W322_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V5_W322_349_MIS2')))

my_config = ANA_DF_CSV[ANA_DF_CSV.trial_id=='30ec9826'] # 349 


best_model = MY_expGCN_parallel_model(
        3, 64 , 8,
        4, 3, 16,
        [256,128,64], [256,128,64], [128,32], 
        91, 1,
        0.2, 0.2
        )

	
state_dict = torch.load(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/test', 'model.pth'), map_location=torch.device('cpu'))
best_model.load_state_dict(state_dict)



explainer = Explainer(
    model=best_model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)


문제는... 
데이터 형태가..
원래는 pyg 에서 제공하는 데이터 모양새는 

dataset = Planetoid(root='/tmp/Cora', name='Cora')

dataset = TUDataset(root='/tmp/Cora', name='PROTEINS')


dataset.x
torch.Size([43471, 3])
dataset.y
torch.Size([2708])
len(dataset)
1113
>>> dataset[0]
Data(edge_index=[2, 162], x=[42, 3], y=[1])
>>> dataset[1]
Data(edge_index=[2, 92], x=[27, 3], y=[1])





from torch_geometric.data import Data as pyg_Data
from torch_geometric.loader import DataLoader as pyg_DataLoader


train_data_0
val_data_0
test_data_0


data_list = [
    pyg_Data(train_data_0['drug1_feat']),
    pyg_Data(train_data_0['drug2_feat']),
    pyg_Data(train_data_0['drug1_adj']),
    pyg_Data(train_data_0['drug2_adj']),
    pyg_Data(train_data_0['GENE_A']),
    pyg_Data(train_data_0['GENE_B']),
    pyg_Data(train_data_0['TARGET_A']),
    pyg_Data(train_data_0['TARGET_B']),
    pyg_Data(train_data_0['cell_BASAL']),
    pyg_Data(train_data_0['cell']),
    pyg_Data(train_data_0['y']),
    ]



gene_feature_1 = torch.concat([train_data_0['GENE_A'],train_data_0['TARGET_A'].view(-1,349,1), train_data_0['cell_BASAL'].view(-1,349,1)], dim=2)
gene_index_1 = JY_ADJ_IDX.view(1,2,-1).repeat(gene_feature.shape[0],1,1)
chem_feature_1 = train_data_0['drug1_feat']
chem_index_1 = train_data_0['drug1_adj']

gene_feature_1 = torch.concat([train_data_0['GENE_A'],train_data_0['TARGET_A'].view(-1,349,1), train_data_0['cell_BASAL'].view(-1,349,1)], dim=2)
gene_index_1 = JY_ADJ_IDX.view(1,2,-1).repeat(gene_feature.shape[0],1,1)
chem_feature_1 = train_data_0['drug1_feat']
chem_index_1 = train_data_0['drug1_adj']



edge_list_1 = []
for a in train_data_0['drug1_adj'] :
    tmp = a.long().to_sparse().indices()
    edge_list_1.append(tmp)

chem_1_list = [
    pyg_Data(x = gene_feature_1[0], edge_index = gene_index_1[0]),
    pyg_Data(x = gene_feature_1[1], edge_index = gene_index_1[1]),
    ]

chem_1_loader = pyg_DataLoader(chem_1_list, batch_size=1)

for a in enumerate(chem_1_loader) :
    a

with torch.no_grad():
    target = model(x, edge_index, batch, edge_label_index)



model_config = ModelConfig(
        mode='regression',
        task_level='graph',
        return_type='raw',
    )

explainer = Explainer(
        model=best_model,
        algorithm=GNNExplainer(epochs=2),
        explanation_type='phenomenon',
        node_mask_type='object', # node 하나하나의 중요성을 보겠다는 의미 
        edge_mask_type=None,
        model_config=model_config,
    )


explainer = Explainer(
        model=best_model,
        algorithm=GNNExplainer(epochs=2),
        explanation_type='model',
        node_mask_type='object', # node 하나하나의 중요성을 보겠다는 의미 
        edge_mask_type=None,
        model_config=model_config,
    )







# 얘는 edge 만 보는 애라고 함 
explainer = Explainer(
    model=model,
    algorithm=PGExplainer(epochs=30, lr=0.003),
    explanation_type='phenomenon',
    edge_mask_type='object',
    model_config=dict(
        mode='regression',
        task_level='graph',
        return_type='raw',
    ),
    # Include only the top 10 most important edges:
    threshold_config=dict(threshold_type='topk', value=10),
)

# PGExplainer needs to be trained separately since it is a parametric
# explainer i.e it uses a neural network to generate explanations:
for epoch in range(30):
    for batch in chem_1_loader:
        loss = explainer.algorithm.train(
            epoch, model, batch.x, batch.edge_index, target=batch.target)

# Generate the explanation for a particular graph:
explanation = explainer(dataset[0].x, dataset[0].edge_index)
print(explanation.edge_mask)



그래서 지금 데이터 모양새까지는 이해가 됐는데,
문제는 지금 모델은 input 이 두개임(대가리가 두개라)
근데 explainer 에서 인식하는 node 모양새는 그렇지가 않은거. 
그럼 애초부터 묶어서 넣어줘야해? 
그건 좀 에반디
그럼 차라리 
모델을 자르는게 나을듯 
아 근데 이게 phenomenon 대상으로 하는거면 얘기가 또 다름...
-> 왜냐면 target 에 대한 loss 를 기반으로 node 점수를 계산하는거라
explainer 에 넣어주긴 해야해.. 
그럼 그냥 explainer 코드를 바꾸는건? 



explanation = explainer(data.x, data.edge_index)
print(f'Generated explanations in {explanation.available_explanations}')


model_config = ModelConfig(
        mode='regression',
        task_level='graph',
        return_type='raw',
    )


explainer = Explainer(
        model=cut_test,
        algorithm=GNNExplainer(epochs=2),
        explanation_type='model',
        node_mask_type='object', # node 하나하나의 중요성을 보겠다는 의미 
        edge_mask_type=None,
        model_config=model_config,
    )

explanation = explainer(chem_1_list[0].x, chem_1_list[0].edge_index)



path = 'feature_importance.png'
explanation.visualize_feature_importance(path, top_k=10)
print(f"Feature importance plot has been saved to '{path}'")

path = 'subgraph.pdf'
explanation.visualize_graph(path)
print(f"Subgraph visualization plot has been saved to '{path}'")






class cut_gene_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, cell_dim ,
	out_dim, inDrop, drop):
		super(cut_gene_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.cell_dim = cell_dim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_Common_dim)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		##
	#
	def forward(self, EXP1, EXP_ADJ ):
		EXP_ADJ_WGT = [1 for a in range(EXP_ADJ.shape[1])]
		Exp_batch_label = [1 for a in range(EXP1.shape[0])]
		#
		for G_1_E in range(len(self.G_convs_1_exp)):
			if G_1_E == len(self.G_convs_1_exp)-1 :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = F.dropout(EXP1, p=self.inDrop, training=self.training)
				#EXP1 = self.pool(EXP1, Exp_batch_label )
				EXP1 = self.tanh(EXP1)
				G_1_E_out = EXP1
			else :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = self.G_bns_1_exp[G_1_E](EXP1)
				EXP1 = F.elu(EXP1)
		#
		return G_1_E_out


cut_test = cut_gene_model(3, 64 , 8,
        4, 3, 16,
        [256,128,64], [256,128,64], [128,32], 
        91, 1,
        0.2, 0.2)


exp_keys = ['G_convs_1_exp.0.bias', 'G_convs_1_exp.0.lin.weight', 'G_convs_1_exp.1.bias', 'G_convs_1_exp.1.lin.weight', 'G_convs_1_exp.2.bias', 'G_convs_1_exp.2.lin.weight', 'G_convs_1_exp.3.bias', 'G_convs_1_exp.3.lin.weight', 'G_bns_1_exp.0.weight', 'G_bns_1_exp.0.bias', 'G_bns_1_exp.0.running_mean', 'G_bns_1_exp.0.running_var', 'G_bns_1_exp.0.num_batches_tracked', 'G_bns_1_exp.1.weight', 'G_bns_1_exp.1.bias', 'G_bns_1_exp.1.running_mean', 'G_bns_1_exp.1.running_var', 'G_bns_1_exp.1.num_batches_tracked', 'G_bns_1_exp.2.weight', 'G_bns_1_exp.2.bias', 'G_bns_1_exp.2.running_mean', 'G_bns_1_exp.2.running_var', 'G_bns_1_exp.2.num_batches_tracked']

pretrained_dict = {k: v for k, v in state_dict.items() if k in exp_keys}

cut_test.load_state_dict(pretrained_dict)


cut_test()



nx.draw(G, pos, node_size=1500, node_color='yellow', font_size=8, font_weight='bold')


왜 안되는건지 모르겠지만 model 을 target 으로 하는걸로 그림 그리기 가능해지면 그냥 그거 예시 교수님 보여드리기 
집가면 일단 내용부터 쓰기
민지랑 같이 쓰는거에 데이터 정리 마무리 하기 
















