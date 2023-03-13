# 내가 만든 모델에 원래 데이터 넣기 


import rdkit
import os 
import os.path as osp
from math import ceil
import pandas as pd 
import numpy as np
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

import tensorflow as tf
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



class performance_metrics():
    def __init__(self, y, pred):
        super(performance_metrics, self).__init__()
    def pearson(y, pred):
        pear = stats.pearsonr(y, pred)
        pear_value = pear[0]
        pear_p_val = pear[1]
        print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
        return pear_value
    #
    def spearman(y, pred):
        spear = stats.spearmanr(y, pred)
        spear_value = spear[0]
        spear_p_val = spear[1]
        print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
        return spear_value
    #
    def mse(y, pred):
        err = mean_squared_error(y, pred)
        print("Mean squared error is {}".format(err))
        return err
    #
    def squared_error(y,pred):
        errs = []
        for i in range(y.shape[0]):
            err = (y[i]-pred[i]) * (y[i]-pred[i])
            errs.append(err)
        return np.asarray(errs)



MM_DATA = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/DrugCombinationData.tsv', sep = '\t')
CEL_DATA =pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/E-MTAB-3610.sdrf.txt', sep = '\t')


def data_loader(drug1_chemicals, drug2_chemicals, cell_line_gex, comb_data_name):
	print("File reading ...")
	comb_data = pd.read_csv(comb_data_name, sep="\t")
	cell_line = pd.read_csv(cell_line_gex,header=None)
	chem1 = pd.read_csv(drug1_chemicals,header=None)
	chem2 = pd.read_csv(drug2_chemicals,header=None)
	synergies = np.array(comb_data["synergy_loewe"])
	#
	cell_line = np.array(cell_line.values)
	chem1 = np.array(chem1.values)
	chem2 = np.array(chem2.values)
	return chem1, chem2, cell_line, synergies


trial_MM_DATA = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/DrugCombinationData.tsv'
trial_drug1 = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/drug1_chem.csv'
trial_drug2 = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/drug2_chem.csv'
trial_CEL_gex = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/cell_line_gex.csv'

trial_chem1, trial_chem2, trial_cell_line, trial_synergies = data_loader(trial_drug1, trial_drug2, trial_CEL_gex, trial_MM_DATA )

train_indx = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/train_inds.txt'
val_indx = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/val_inds.txt'
test_indx = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/test_inds.txt'


def prepare_data(chem1, chem2, cell_line, synergies, norm, 
	train_ind_fname, val_ind_fname, test_ind_fname):
	print("Data normalization and preparation of train/validation/test data")
	test_ind = list(np.loadtxt(test_ind_fname,dtype=np.int))
	val_ind = list(np.loadtxt(val_ind_fname,dtype=np.int))
	train_ind = list(np.loadtxt(train_ind_fname,dtype=np.int))
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	# chem 1 이랑 2 를 붙인다고? 
	train1 = np.concatenate((chem1[train_ind,:],chem2[train_ind,:]),axis=0) # 엥 왜 
	train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
	val_data['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem1[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train2 = np.concatenate((chem2[train_ind,:],chem1[train_ind,:]),axis=0)
	train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
	val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem2[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
	train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
	val_cell_line, mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(cell_line[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(cell_line[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train_data['drug1'] = np.concatenate((train_data['drug1'],train_cell_line),axis=1)
	train_data['drug2'] = np.concatenate((train_data['drug2'],train_cell_line),axis=1)
	#
	val_data['drug1'] = np.concatenate((val_data['drug1'],val_cell_line),axis=1)
	val_data['drug2'] = np.concatenate((val_data['drug2'],val_cell_line),axis=1)
	#
	test_data['drug1'] = np.concatenate((test_data['drug1'],test_cell_line),axis=1)
	test_data['drug2'] = np.concatenate((test_data['drug2'],test_cell_line),axis=1)
	#
	train_data['y'] = np.concatenate((synergies[train_ind],synergies[train_ind]),axis=0)
	val_data['y'] = synergies[val_ind]
	test_data['y'] = synergies[test_ind]
	print(test_data['drug1'].shape)
	print(test_data['drug2'].shape)
	return train_data, val_data, test_data


norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data(trial_chem1, trial_chem2, trial_cell_line, trial_synergies, norm,
											train_indx, val_indx, test_indx)

min_s = np.amin(train_data['y']) # array mean 구해주기 
loss_weight = np.log(train_data['y'] - min_s + np.e) # log 취해서 다 고만고만한 값이 됨 


layers = {}
layers['DSN_1'] = '2048-4096-2048' # architecture['DSN_1'][0] # layers of Drug Synergy Network 1
layers['DSN_2'] = '2048-4096-2048' #architecture['DSN_2'][0] # layers of Drug Synergy Network 2
layers['SPN'] = '2048-1024' #architecture['SPN'][0] # layers of Synergy Prediction Network
modelName = 'matchmaker.w' # args.saved_model_name # name of the model to save the weights


# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100

dsn1_layers = layers["DSN_1"].split("-")
dsn2_layers = layers["DSN_2"].split("-")
snp_layers = layers["SPN"].split("-")

beta_1=0.9
beta_2=0.999
# default 



class MM_DATASET(Dataset): 
	def __init__(self, drug1, drug2, y ):
			self.drug1 = torch.Tensor(drug1) # input 1
			self.drug2 = torch.Tensor(drug2) # input 2
			self.y = y
		#
	def __len__(self): 
			return len(self.drug1)
		#
	def __getitem__(self, index): 
			return self.drug1[index], self.drug2[index], self.y[index]



T_train = MM_DATASET(train_data['drug1'],train_data['drug2'],train_data['y'])
T_val = MM_DATASET(val_data['drug1'],val_data['drug2'],val_data['y'])
T_test = MM_DATASET(test_data['drug1'],test_data['drug2'],test_data['y'])
#T_loss_weight = torch.Tensor(loss_weight)

loaders = {
	'train' : torch.utils.data.DataLoader(T_train, batch_size = batch_size),
	'eval' : torch.utils.data.DataLoader(T_val, batch_size = batch_size),
	'test' : torch.utils.data.DataLoader(T_test, batch_size = batch_size),
}


def weighted_mse_loss(input, target, weight):
		return (weight * (input - target) ** 2).mean()



class MY_parallel_model(torch.nn.Module):
	def __init__(self, drug1_indim, drug2_indim, layers_1, layers_2, layers_3, out_dim, inDrop, drop):
		super(MY_parallel_model, self).__init__()
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.input_dim1 = drug1_indim
		self.input_dim2 = drug2_indim
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		#
		self.Convs_1 = nn.ModuleList([torch.nn.Linear(self.input_dim1, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		#
		self.Convs_2 = nn.ModuleList([torch.nn.Linear(self.input_dim2, self.layers_2[0] )])
		self.Convs_2.extend([torch.nn.Linear(self.layers_2[a], self.layers_2[a+1]) for a in range(len(self.layers_2)-1)])
		#
		self.SNPs = nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self): 
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.Convs_2:
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def forward(self, input_drug1, input_drug2 ):
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.relu(input_drug1)
			else : 
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_2)):
			if L2 != len(self.Convs_2)-1 :
				input_drug2 = self.Convs_2[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.relu(input_drug2)
			else : 
				input_drug2 = self.Convs_2[L2](input_drug2)
		#
		X = torch.cat((input_drug1,input_drug2),1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.relu(X)
			else : 
				X = self.SNPs[L3](X)
		return X
	






# start_epochs = 1
n_epochs = 1000

use_cuda = False
#
loss_weight2 = loss_weight.tolist()
# batch_cut_weight = torch.Tensor([loss_weight2[i:i+batch_size] for i in range(0,len(loss_weight2), batch_size)])
tt = [loss_weight2[i:i+batch_size] for i in range(0,len(loss_weight2), batch_size)]

batch_cut_weight = [torch.Tensor(t) for t in tt]

#
loaders = {
	'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"]),
	'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"]),
	'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"]),
}
#
#
#
#


def save_ckp(state, is_best, checkpoint_path, best_model_path):
	"""
	state: checkpoint we want to save
	is_best: is this the best checkpoint; min validation loss
	checkpoint_path: path to save checkpoint
	best_model_path: path to save best model
	"""
	f_path = checkpoint_path # save checkpoint data to the path given, checkpoint_path
	torch.save(state, f_path)
	if is_best: # if it is a best model, min validation loss
		best_fpath = best_model_path+'{}.best_model.pt'.format(state['epoch']) # copy that checkpoint file to best path given, best_model_path
		shutil.copyfile(f_path, best_fpath)



def load_ckp(checkpoint_fpath, model, optimizer):
	"""
	checkpoint_path: path to save checkpoint
	model: model that we want to load checkpoint parameters into       
	optimizer: optimizer we defined in previous training
	"""
	checkpoint = torch.load(checkpoint_fpath) # load check point
	model.load_state_dict(checkpoint['state_dict']) # initialize state_dict from checkpoint to model
	optimizer.load_state_dict(checkpoint['optimizer']) # initialize optimizer from checkpoint to optimizer
	valid_loss_min = checkpoint['valid_loss_min'] # initialize valid_loss_min from checkpoint to valid_loss_min
	return model, optimizer, checkpoint['epoch'], valid_loss_min.item() # return model, optimizer, epoch value, min validation loss 



 

def original_to_torch():
n_epochs=  1000
valid_loss_min = np.Inf
use_cuda = False
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100
dsn1_layers = [2048, 512 , 2048 ]
dsn2_layers = [2048, 512 , 2048 ] 
snp_layers = [1024 , 1024]
train_loss_all = []
valid_loss_all = []
checkpoint_path= '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/checkpoint/current_checkpoint.pt'
best_model_path='/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/best_model/'
save_path = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1'
valid_loss_min_input = np.Inf
criterion = weighted_mse_loss
#
model = MY_parallel_model(
	1396, 1396, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, drop
	)	
#
optimizer = torch.optim.Adam(model.parameters(), lr = l_rate)
#
for epoch in range(1, n_epochs+1):
	now = datetime.now()
	train_loss = 0.0
	valid_loss = 0.0
	#
	###################
	# train the model #
	###################
	model.train()
	for batch_idx_t, (drug1, drug2, y) in enumerate(loaders['train']):
		# move to GPU
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		## find the loss and update the model parameters accordingly
		# clear the gradients of all optimized variables
		optimizer.zero_grad()
		output = model(drug1, drug2)
		y = y.view(-1,1)
		wc = batch_cut_weight[batch_idx_t].view(-1,1)
		loss = criterion(output, y, wc ) # weight 더해주기 
		loss.backward()
		optimizer.step()
		## record the average training loss, using something like
		## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
		train_loss = train_loss + loss.item()
	#
	######################    
	# validate the model #
	######################
	model.eval()
	for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['eval']):
		# move to GPU
		if use_cuda:
			data, target = data.cuda(), target.cuda()
		## update the average validation loss
		output = model(drug1, drug2)
		y = y.view(-1,1)
		MSE = nn.MSELoss()
		loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
		# update average validation loss 
		valid_loss = valid_loss + loss.item()
	#
	# calculate average losses
	TRAIN_LOSS = train_loss/(batch_idx_t+1)
	train_loss_all.append(TRAIN_LOSS)
	VAL_LOSS = valid_loss/(batch_idx_v+1)
	valid_loss_all.append(VAL_LOSS)
	#
	# print training/validation statistics 
	done = datetime.now()
	time_spent = done-now
	print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:2f}sec'.format(
		epoch, 
		TRAIN_LOSS,
		VAL_LOSS,
		time_spent.total_seconds()
		))
	#
	checkpoint = {
		'epoch': epoch,
		'valid_loss_min': VAL_LOSS,
		'state_dict': model.state_dict(),
		'optimizer': optimizer.state_dict(),
	}
	#
	save_ckp(checkpoint, False, checkpoint_path, best_model_path)
	#
	if VAL_LOSS <= valid_loss_min :
		print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, VAL_LOSS))
		# save checkpoint as best model
		save_ckp(checkpoint, True, checkpoint_path, best_model_path)
		valid_loss_min = VAL_LOSS
		if epoch >= 200 :
			print('Early stopping!\nStart to test process.')
			#return model
#

너무 초반에 내려가고 그 다음에 내려간 애들은 의미가 거의 없어서 
일단 마지막 애랑
초반의 애랑 비교해야할듯 



plot_loss(train_loss_all, valid_loss_all, save_path, 'ORIGINAL')
loss_df = pd.DataFrame({'TrainLoss' :train_loss_all, 'ValidLoss':valid_loss_all })
loss_df.to_csv(save_path+'/ORIGINAL_LOSS_DF.csv', sep = '\t')
# return trained model
	return model


original_to_torch()











# 마지막 끝난대로 하는 경우 
all_pred = []

model.eval()
for batch_idx_T, (drug1, drug2, y) in enumerate(loaders['test']):
	## update the average validation loss
	output = model(drug1, drug2)
	y = y.view(-1,1)
	MSE = nn.MSELoss()
	loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
	# update average validation loss 
	test_loss = test_loss + loss.item()
	outputs = output.squeeze().tolist()
	all_pred = all_pred + outputs

mse_value = performance_metrics.mse(test_data['y'], all_pred)
spearman_value = performance_metrics.spearman(test_data['y'], all_pred)
pearson_value = performance_metrics.pearson(test_data['y'], all_pred)





# val loss 최소로 내려간 경우 
min_checkpoint = torch.load('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/best_model/44.best_model.pt') # load check point
min_model = MY_parallel_model(
	1396, 1396, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, drop
	)	
min_model.load_state_dict(min_checkpoint['state_dict'])

all_pred = []

min_model.eval()
for batch_idx_T, (drug1, drug2, y) in enumerate(loaders['test']):
	## update the average validation loss
	output = min_model(drug1, drug2)
	y = y.view(-1,1)
	MSE = nn.MSELoss()
	loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
	# update average validation loss 
	test_loss = test_loss + loss.item()
	outputs = output.squeeze().tolist()
	all_pred = all_pred + outputs

mse_value = performance_metrics.mse(test_data['y'], all_pred)
spearman_value = performance_metrics.spearman(test_data['y'], all_pred)
pearson_value = performance_metrics.pearson(test_data['y'], all_pred)




# Why diff result ? 
# check MSE result 


y_true = np.array([[1,2,3],[6,7,8]])
y_pred = np.array([[0,0,0],[1,1,1]])
weight = np.array([5,10])
weight_r = weight.reshape(1,2)
weight_t= weight.reshape(2,1)
weight_p = np.array([5/15,10/15])
weight_pr = weight_p.reshape(1,2)
weight_pt= weight_p.reshape(2,1)

0. TF no weight 
tf_m = tf.keras.metrics.MeanSquaredError()
tf_result = tf_m(y_true, y_pred) # 20.666668
tf_result.numpy()

loss = tf.reduce_mean(tf.square(y_true - y_pred)) # 20 # 그냥 6 으로 나눈거 


1. TF weight 추가 
tf_m = tf.keras.metrics.MeanSquaredError()
tf_result = tf_m(y_true, y_pred, sample_weight=weight)
tf_result.numpy() # 26.000002


2. 예상 공식
loss = tf.reduce_mean(weight_t*tf.square(y_true - y_pred)) # 195 
왜 26 이 나오려면.... 어떻게... 하지 
나는 그냥 곱해준건데 
와 시발 찾은듯 
tmp_loss = np.mean(tf.square(y_true - y_pred), axis =1)*weight
sum(tmp_loss)/sum(weight) # 25.3 근데 말이 안돼... 그리고 맞지도 않아 시벌 


(weight_t * (y_pred - y_true) ** 2).sum() / weight_t.sum()
한번만 더 해보자




아 아닌가 
tf_sd = tf.math.squared_difference(y_pred, y_true)
tf_sd = tf.cast(tf_sd, dtype=tf.float64)
tf_dot = weight_t * tf_sd
tf.math.reduce_mean(tf_dot) # 13.0... 이건 아닌데 근데 내가 보기에도 한번 노나줘야하는데 
tf.math.reduce_mean(np.sum(tf_dot/sum(weight), axis = 1)) # 39.00000000000001

tf_dot = weight_pt * tf_sd
np.mean(np.sum(tf_dot,axis=1)) # 내 머릿속에서 나오는 애들은 자꾸 왜 이런값이지 39.0




3. torch just mse
T_y_true = torch.Tensor(np.array([[1,2,3],[6,7,8]]))
T_y_pred = torch.Tensor(np.array([[0,0,0],[1,1,1]]))
T_weight = torch.Tensor(np.array([5,10]))
T_weight2 = T_weight.view(-1,2)

tch_m = torch.nn.MSELoss()
tch_result = tch_m(T_y_true, T_y_pred) # 20.6667

4. Torch 
tch_sd = (T_y_pred - T_y_true) ** 2
tch_res = (T_weight * (y_pred - y_true) ** 2).mean()


5. sklearn mse





# tensorflow 

np.mean(TF_sd * np.array([[5],[10]])) # 195


TF_FF = np.mean(weight*TF_sd)
TF_res = tf.keras.metrics.mean_squared_error(y_true, y_pred, sample_weight=weight).numpy() #190 

sum(weight*square(ans-pred)) 

또 누군가의 의견 
tf.math.reduce_sum(weight * tf.math.square(y_true - y_pred)) / tf.math.reduce_sum(weight)



이건 또 다른 방법 (focal loss)
-> 분류 성능이 높은 클래스에 대해서는 down weighting 
-> 분류가 힘든 데이터에 대한 트레이닝을 강조 
def focal_loss(y_true, y_pred):
    gamma = 2.0, alpha = 0.25
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))

출처: https://3months.tistory.com/414 [Deep Play:티스토리]




# pytorch 

T_y_true = torch.Tensor(np.array([[1,2,3],[6,7,8]]))
T_y_pred = torch.Tensor(np.array([[0,0,0],[1,1,1]]))
T_weight = torch.Tensor(np.array([5,10]))
T_weight2 = T_weight.view(-1,2)

MS_sd = (y_pred - y_true) ** 2
MY = (weight * (y_pred - y_true) ** 2).mean()
# 26 
T_diff = torch.square(T_y_true - T_y_pred)
torch.sum(T_weight2 * T_diff) / tf.math.reduce_sum(weight) # 78


 # MSEloss
 
def mse(t1, t2):
	diff = t1 - t2
	return torch.sum(diff * diff) / diff.numel()
https://pythonguides.com/pytorch-mseloss/#:~:text=PyTorch%20MSELoss%20code%20is%20defined,actual%20value%20and%20predicted%20value.

pyt_diff = torch.Tensor(y_true-y_pred)

torch.sum(pyt_diff * pyt_diff) / pyt_diff.numel() 