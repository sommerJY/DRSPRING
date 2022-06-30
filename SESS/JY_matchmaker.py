
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



seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)




# graph 마다 확인 
ID_GENE_ORDER_mini = ID_G.nodes()
IAS_PPI = nx.adjacency_matrix(ID_G)

JY_GRAPH = ID_G
JY_GRAPH_ORDER = ID_G.nodes()
JY_ADJ = nx.adjacency_matrix(ID_G)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices() 
JY_IDX_WEIGHT = ID_WEIGHT_SCORE


# DC set 확인 
A_B_C_S = CELLO_DC_BETA.reset_index()
A_B_C_S_SET = A_B_C_S[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index()


# LINCS 확인 
BETA_ORDER_pre =[list(L_matching_list.PPI_name).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = L_matching_list.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ORDER = list(BETA_ORDER_DF.entrez)


def get_LINCS_data(DrugA_SIG, DrugB_SIG):
	DrugA_EXP = BETA_BIND[['id',DrugA_SIG]]
	DrugB_EXP = BETA_BIND[['id',DrugB_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.entrez] 
	DrugA_EXP_ORD = DrugA_EXP.iloc[BIND_ORDER]
	DrugB_EXP_ORD = DrugB_EXP.iloc[BIND_ORDER]
	#
	ARR = np.array([list(DrugA_EXP_ORD[DrugA_SIG]), list(DrugB_EXP_ORD[DrugB_SIG])])
	SUM = np.sum(ARR, axis = 0)
	return DrugA_EXP_ORD, DrugB_EXP_ORD, ARR.T, SUM


def get_morgan(smiles):
	result = []
	try:
		tmp = Chem.MolFromSmiles(smiles)
		result.append(tmp)
	except ValueError as e :
		tmp.append("NA")
	return result[0]


def get_CHEMI_data(Drug_SIG, bitsize):
	A_SM = BETA_SELEC_SIG[BETA_SELEC_SIG.sig_id == Drug_SIG]['canonical_smiles']
	A_SM = A_SM.values[0]
	#
	A_morgan = get_morgan(A_SM)
	bi = {}
	A_FP = rdkit.Chem.AllChem.GetMorganFingerprintAsBitVect(A_morgan, radius=2, nBits = bitsize , bitInfo=bi)
	if len(A_FP)==bitsize :
		A_FP_LIST = list(A_FP.ToBitString())
	else : 
		A_FP_LIST = ['0']*bitsize
	#
	return A_FP_LIST


def get_synergy_data(DrugA_SIG, DrugB_SIG, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.BETA_sig_id_x == DrugA_SIG]
	ABCS2 = ABCS1[ABCS1.BETA_sig_id_y == DrugB_SIG]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe)
	return synergy_score



MY_chem_A = torch.empty(size=(A_B_C_S_SET.shape[0], 256))
MY_chem_B= torch.empty(size=(A_B_C_S_SET.shape[0], 256))
MY_exp_A = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_exp_B = torch.empty(size=(A_B_C_S_SET.shape[0], 978))
MY_exp_AB = torch.empty(size=(A_B_C_S_SET.shape[0], 978, 2))
MY_syn =  torch.empty(size=(A_B_C_S_SET.shape[0],1))



for IND in range(A_B_C_S_SET.shape[0]):
	DrugA_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_x']
	DrugB_SIG = A_B_C_S_SET.iloc[IND,]['BETA_sig_id_y']
	Cell = A_B_C_S_SET.iloc[IND,]['DrugCombCello']
	#
	bitsize = 256
	DrugA_FP = [int(a) for a in get_CHEMI_data(DrugA_SIG, bitsize)]
	DrugB_FP = [int(a) for a in get_CHEMI_data(DrugB_SIG, bitsize)]
	#
	EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
	#
	AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
	#
	MY_chem_A[IND] = torch.Tensor(DrugA_FP)
	MY_chem_B[IND] = torch.Tensor(DrugB_FP)
	MY_exp_A[IND] = torch.Tensor(EXP_A.iloc[:,1])
	MY_exp_B[IND] = torch.Tensor(EXP_B.iloc[:,1])
	MY_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
	MY_syn[IND] = torch.Tensor([AB_SYN])


MY_chem_A_tch = torch.tensor(np.array(MY_chem_A))
MY_chem_B_tch = torch.tensor(np.array(MY_chem_B))
MY_exp_A_tch = torch.tensor(np.array(MY_exp_A))
MY_exp_B_tch = torch.tensor(np.array(MY_exp_B))
MY_exp_AB_tch = torch.tensor(np.array(MY_exp_AB))
MY_syn_tch = torch.tensor(np.array(MY_syn))


def normalize(X, means1=None, std1=None, means2=None, std2=None, 
	feat_filt=None, norm='tanh_norm'):
	if std1 is None:
		std1 = np.nanstd(X, axis=0) # nan 무시하고 표준편차 구하기 
	if feat_filt is None:
		feat_filt = std1!=0
	X = X[:,feat_filt]
	X = np.ascontiguousarray(X)
	if means1 is None:
		means1 = np.mean(X, axis=0)
	X = (X-means1)/std1[feat_filt]
	if norm == 'norm':
		return(X, means1, std1, feat_filt)
	elif norm == 'tanh':
		return(np.tanh(X), means1, std1, feat_filt)
	elif norm == 'tanh_norm':
		X = np.tanh(X)
		if means2 is None:
			means2 = np.mean(X, axis=0)
		if std2 is None:
			std2 = np.std(X, axis=0)
		X = (X-means2)/std2
		X[:,std2==0]=0
		return(X, means1, std1, means2, std2, feat_filt)


def prepare_data(MY_chem1_tch, MY_chem2_tch, MY_exp_tch, MY_syn_tch, norm ) :
	chem1_train, chem1_tv, chem2_train, chem2_tv, exp_train, exp_tv, syn_train, syn_tv  = sklearn.model_selection.train_test_split(
		MY_chem1_tch, MY_chem2_tch, MY_exp_tch, MY_syn_tch, 
		test_size=3230, random_state=42 )
	chem1_val, chem1_test, chem2_val, chem2_test, exp_val, exp_test, syn_val, syn_test  = sklearn.model_selection.train_test_split(
		chem1_tv, chem2_tv, exp_tv, syn_tv, 
		test_size=1230, random_state=42 )
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	train1 = np.concatenate((chem1_train, chem2_train),axis=0) 
	train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
	val_data['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem1_val,mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1_test,mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train2 = np.concatenate((chem2_train, chem1_train),axis=0)
	train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
	val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem2_val,mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2_test,mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train3 = np.concatenate((exp_train, exp_train), axis=0)
	train_exp, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
	val_exp, mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(exp_val, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	test_exp, mean1, std1, mean2, std2, feat_filt = normalize(exp_test, mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train_data['drug1'] = np.concatenate((train_data['drug1'],train_exp),axis=1)
	train_data['drug2'] = np.concatenate((train_data['drug2'],train_exp),axis=1)
	#
	val_data['drug1'] = np.concatenate((val_data['drug1'],exp_val),axis=1)
	val_data['drug2'] = np.concatenate((val_data['drug2'],exp_val),axis=1)
	#
	test_data['drug1'] = np.concatenate((test_data['drug1'],test_exp),axis=1)
	test_data['drug2'] = np.concatenate((test_data['drug2'],test_exp),axis=1)
	#		
	train_data['y'] = np.concatenate((syn_train,syn_train),axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	print(test_data['drug1'].shape)
	print(test_data['drug2'].shape)
	return train_data, val_data, test_data


norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data(MY_chem1_tch, MY_chem2_tch, MY_exp_tch, MY_syn_tch, norm)
# 6000 vs 2000 vs 1230



# prepare layers of the model and the model name
layers = {}
layers['DSN_1'] = '2048-4096-2048' # architecture['DSN_1'][0] # layers of Drug Synergy Network 1
layers['DSN_2'] = '2048-4096-2048' #architecture['DSN_2'][0] # layers of Drug Synergy Network 2
layers['SPN'] = '2048-1024' #architecture['SPN'][0] # layers of Synergy Prediction Network
modelName = 'matchmaker.w' # args.saved_model_name # name of the model to save the weights




class MY_parallel_model(torch.nn.Module):
	def __init__(self, train_data, layers_1, layers_2, layers_3, out_dim, inDrop, drop):
		super(MY_parallel_model, self).__init__()
		self.train_data = train_data
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.input_dim1 = train_data['drug1'].shape[1]
		self.input_dim2 = train_data['drug2'].shape[1]
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
	



# performance_metrics

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



# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 100
earlyStop_patience = 20

dsn1_layers = layers["DSN_1"].split("-")
dsn2_layers = layers["DSN_2"].split("-")
snp_layers = layers["SPN"].split("-")




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



# calculate weights for weighted MSE loss only for train data 
min_s = np.amin(train_data['y'])
loss_weight = np.log(train_data['y'] - min_s + np.e) # log 취해서 다 고만고만한 값이 됨 
loss_weight_cut = [loss_weight[i:i+100] for i in range(0,len(loss_weight), 100)]
loss_weight_cut_tch = torch.Tensor(loss_weight_cut)
아 이렇게 하면 안됨.
나는 너무

MM_MODEL = MY_parallel_model(train_data, dsn1_layers, dsn2_layers, snp_layers, 1, 0.2, 0.5)	

# 잠깐 테스트로 시간 줄이기 
T_train = MM_DATASET(train_data['drug1'],train_data['drug2'],train_data['y'])
T_val = MM_DATASET(val_data['drug1'],val_data['drug2'],val_data['y'])
T_test = MM_DATASET(test_data['drug1'],test_data['drug2'],test_data['y'])
T_loss_weight = torch.Tensor(loss_weight)

loaders = {
	'train' : torch.utils.data.DataLoader(T_train, batch_size = batch_size),
	'eval' : torch.utils.data.DataLoader(T_val, batch_size = batch_size),
	'test' : torch.utils.data.DataLoader(T_test, batch_size = batch_size),
}


def MY_train(start_epochs, n_epochs, valid_loss_min_input, loaders, model, optimizer, criterion, loss_weight_cut_tch, use_cuda, checkpoint_path, best_model_path):
	"""
	Keyword arguments:
	start_epochs -- the real part (default 0.0)
	n_epochs -- the imaginary part (default 0.0)
	valid_loss_min_input
	loaders
	model
	optimizer
	criterion
	use_cuda
	checkpoint_path
	best_model_path
	returns trained model
	"""
	valid_loss_min = valid_loss_min_input # initialize tracker for minimum validation loss
	train_loss_all = []
	valid_loss_all = []
	#
	for epoch in range(start_epochs, n_epochs+1):
		now = datetime.now()
		train_loss = 0.0
		valid_loss = 0.0
		#
		###################
		# train the model #
		###################
		model.train()
		for batch_idx, (drug1, drug2, y) in enumerate(loaders['train']):
			# move to GPU
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = model(drug1, drug2)
			y = y.view(-1,1)
			wc = loss_weight_cut_tch[batch_idx].view(-1,1)
			loss = criterion(output, y, wc ) # weight 더해주기 
			loss.backward()
			optimizer.step()
			train_loss_all.append(loss.item())
			## record the average training loss, using something like
			## train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
			train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
		#
		######################    
		# validate the model #
		######################
		model.eval()
		for batch_idx, (drug1, drug2, y) in enumerate(loaders['eval']):
			# move to GPU
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			## update the average validation loss
			output = model(drug1, drug2)
			y = y.view(-1,1)
			MSE = nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			valid_loss_all.append(loss.item())
			# update average validation loss 
			valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))
		#
		# calculate average losses
		train_loss = train_loss/len(loaders['train'].dataset)
		valid_loss = valid_loss/len(loaders['eval'].dataset)
		#
		# print training/validation statistics 
		done = datetime.now()
		time_spent = done-now
		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:2f}sec'.format(
			epoch, 
			train_loss,
			valid_loss,
			time_spent.total_seconds()
			))
		#
		# create checkpoint variable and add important data
		checkpoint = {
			'epoch': epoch,
			'valid_loss_min': valid_loss,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		#
		# save checkpoint
		save_ckp(checkpoint, False, checkpoint_path, best_model_path)
		#
		## TODO: save the model if validation loss has decreased
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
			# save checkpoint as best model
			save_ckp(checkpoint, True, checkpoint_path, best_model_path)
			valid_loss_min = valid_loss
	#
	plot_loss(train_loss_all ,valid_loss_all, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MY1' )
	loss_df = pd.DataFrame({'TrainLoss' :train_loss_all, 'ValidLoss':valid_loss_all })
	loss_df.to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/LOSS_DF.csv', sep = '\t')
	# return trained model
	return model


def weighted_mse_loss(input, target, weight):
		return (weight * (input - target) ** 2).mean()


optimizer = torch.optim.Adam(MM_MODEL.parameters(), lr = l_rate)
Check_PATH = "/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/checkpoint/current_checkpoint.pt"
Best_PATH = "/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/best_model/"
criterion = weighted_mse_loss
use_cuda = False


trained_model = MY_train(1, 200, np.Inf, loaders, MM_MODEL, optimizer, criterion, loss_weight_cut_tch, use_cuda,Check_PATH , Best_PATH)


def plot_loss(train_loss, valid_loss, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
	plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
	#minposs = valid_loss.index(min(valid_loss))+1
	#plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
	plt.xlim(0, len(train_loss)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.loss_plot.png'.format(path, plotname), bbox_inches = 'tight')


def plot_acc(train_acc, valid_acc, path, plotname):
	fig = plt.figure(figsize=(10,8))
	plt.plot(range(1,len(train_acc)+1),train_acc, label='Training Loss')
	plt.plot(range(1,len(valid_acc)+1),valid_acc,label='Validation Loss')
	#minposs = valid_acc.index(min(valid_acc))+1
	#plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')
	plt.xlabel('epochs')
	plt.ylabel('loss')
	plt.ylim(0, math.ceil(max(train_acc))) # 일정한 scale
	plt.xlim(0, len(train_acc)+1) # 일정한 scale
	plt.grid(True)
	plt.legend()
	plt.tight_layout()
	fig.savefig('{}/{}.acc_plot.png'.format(path,plotname), bbox_inches = 'tight')








#################################################
#################################################
#################################################



# load to eval 
RE_model = MY_parallel_model(train_data, dsn1_layers, dsn2_layers, snp_layers, 1, 0.2, 0.5)	
optimizer = optim.Adam(RE_model.parameters(), lr=0.001)
ckp_path = Check_PATH
best_path = Best_PATH+'67.best_model.pt'

RE_MODEL, optimizer, start_epoch, valid_loss_min = load_ckp(best_path, RE_model, optimizer)

RE_MODEL.eval()
test_MSE = 0.0
for batch_idx, (drug1, drug2, y) in enumerate(loaders['test']):
	with torch.no_grad():
		output = RE_MODEL(drug1, drug2)
		y = y.view(-1,1)
		MSE = nn.MSELoss()
		loss = MSE(output, y) 
		test_MSE += loss.float()

print('MSE on test: {}'.format(round(test_MSE.item()/len(loaders['test'].dataset), 2)))


valid_loss = valid_loss/len(loaders['eval'].dataset)











################################################
wanna use ray ?
wanna use ray ?
wanna use ray ?
wanna use ray ?


################################################
wanna use ray ?
wanna use ray ?
wanna use ray ?
wanna use ray ?


################################################
wanna use ray ?
wanna use ray ?
wanna use ray ?
wanna use ray ?




Check_PATH = "/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/checkpoint/current_checkpoint.pt"
Best_PATH = "/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1/best_model/"
criterion = weighted_mse_loss
use_cuda = False

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



CONFIG={
	"batch_size" : tune.choice([50,100]),
	"feat_size_1" : tune.choice([4096, 2048, 1024, 512]),
	"feat_size_2" : tune.choice([4096, 2048, 1024, 512]),
	"feat_size_3" : tune.choice([4096, 2048, 1024, 512]),
	"feat_size_4" : tune.choice([4096, 2048, 1024, 512]),
	"dropout_1" : tune.choice([0.01, 0.2, 0.5, 0.8]),
	"dropout_2" : tune.choice([0.01, 0.2, 0.5, 0.8]),
	"lr" : tune.choice([0.00001, 0.0001, 0.001]),
}


# 이건 넣어놓고 자자 인간적으로
# 그렇게 인간이 아니게 되었다고 한다 

T_train = MM_DATASET(train_data['drug1'],train_data['drug2'],train_data['y'])
T_val = MM_DATASET(val_data['drug1'],val_data['drug2'],val_data['y'])
T_test = MM_DATASET(test_data['drug1'],test_data['drug2'],test_data['y'])
T_loss_weight = torch.Tensor(loss_weight)

RAY_train = ray.put(T_train)
RAY_val = ray.put(T_val)
RAY_test = ray.put(T_test)
RAY_loss_weight = ray.put(loss_weight)



def RAY_MY_train(config, checkpoint_dir=None):
	# start_epochs = 1
	n_epochs = 200
	# valid_loss_min_input = np.Inf
	criterion = weighted_mse_loss
	use_cuda = False
	# checkpoint_path = Check_PATH
	# best_model_path = Best_PATH
	# batch_size = 100
	#
	#
	#
	T_train = ray.get(RAY_train)
	T_val = ray.get(RAY_val)
	T_test = ray.get(RAY_test)
	T_loss_weight = ray.get(RAY_loss_weight)
	batch_cut_weight = torch.Tensor([T_loss_weight[i:i+config["batch_size"]] for i in range(0,len(T_loss_weight), config["batch_size"])])
	# batch cut 수정해야함 딱 안떨어지는 batch 있을수도 있음 
	#
	loaders = {
		'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"]),
		'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"]),
		'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"]),
	}
	#
	#
	dsn1_layers = [2048, config["feat_size_1"] , config["feat_size_2"] ]
	dsn2_layers = [2048, config["feat_size_1"] , config["feat_size_2"] ] 
	snp_layers = [config["feat_size_3"] ,config["feat_size_4"] ,]
	inDrop = config["dropout_1"] 
	Drop = config["dropout_2"] 
	#
	#
	MM_MODEL = MY_parallel_model(
		1234,1234, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
		)	
	#
	optimizer = torch.optim.Adam(MM_MODEL.parameters(), lr = config["lr"] )
	if checkpoint_dir :
		checkpoint = os.path.join(checkpoint_dir, "checkpoint")
		model_state, optimizer_state = torch.load(checkpoint)
		model.load_state_dict(model_state)
		optimizer.load_state_dict(optimizer_state)
	#
	#
	# valid_loss_min = valid_loss_min_input # initialize tracker for minimum validation loss
	train_loss_all = []
	valid_loss_all = []
	#
	#
	for epoch in range(n_epochs):
		now = datetime.now()
		train_loss = 0.0
		valid_loss = 0.0
		#
		###################
		# train the model #
		###################
		MM_MODEL.train()
		for batch_idx_t, (drug1, drug2, y) in enumerate(loaders['train']):
			# move to GPU
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			## find the loss and update the model parameters accordingly
			# clear the gradients of all optimized variables
			optimizer.zero_grad()
			output = MM_MODEL(drug1, drug2)
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
		MM_MODEL.eval()
		for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['eval']):
			# move to GPU
			if use_cuda:
				data, target = data.cuda(), target.cuda()
			## update the average validation loss
			output = MM_MODEL(drug1, drug2)
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
		#
		with tune.checkpoint_dir(step = epoch) as checkpoint_dir:
			path = os.path.join(checkpoint_dir, "checkpoint")
			torch.save((MM_MODEL.state_dict(), optimizer.state_dict()), path)
			torch.save(MM_MODEL.state_dict(), './model.pth')
		tune.report(TrainLoss= TRAIN_LOSS,  ValLoss=VAL_LOSS )
	#
	# plot_loss(train_loss_all ,valid_loss_all, '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MY1' )
	# return trained model
	# return model


reporter = CLIReporter(
		metric_columns=["TrainLoss", "ValLoss",  "training_iteration"])

optuna_search = OptunaSearch(metric="ValLoss", mode="min")

ASHA_scheduler = tune.schedulers.ASHAScheduler(
	time_attr='training_iteration', metric="ValLoss", mode="min", max_t= 200, grace_period = 20)


ANALYSIS = tune.run( # 끝내지 않음 
	tune.with_parameters(RAY_MY_train),  
	name = '22.05.24.MM_trial_1',
	num_samples=100, 
	config=CONFIG, 
	resources_per_trial={'cpu': 5},
	progress_reporter = reporter,
	search_alg = optuna_search,
	scheduler = ASHA_scheduler
	)

,
	resume = True





from ray.tune import ExperimentAnalysis
#anal_df = ExperimentAnalysis("/home/jiyeonH/ray_results/22.05.24.MM_trial_1")
ANA_DF = ANALYSIS.dataframe()
ANA_ALL_DF = ANALYSIS.trial_dataframes


DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
# /home/jiyeonH/ray_results/22.05.24.MM_trial_1/RAY_MY_train_51c5f85e_1_batch_size=100,dropout_1=0.8,dropout_2=0.8,feat_size_1=512,feat_size_2=2048,feat_size_3=1024,feat_size_4=1_2022-05-25_11-56-48

# 190

mini_df = ANA_ALL_DF[DF_KEY]
plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MM_trial_1.no1' )


checkpoint = '/checkpoint_000199'
TOPVAL_PATH = DF_KEY + checkpoint

dsn1_layers = [2048, 512 , 2048 ]
dsn2_layers = [2048, 512 , 2048 ] 
snp_layers = [1024 , 1024]
inDrop = 0.8
Drop = 0.8
#
#
best_model = MY_parallel_model(
	1234,1234, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
	)	

state_dict = torch.load(os.path.join(TOPVAL_PATH, "checkpoint"))
best_model.load_state_dict(state_dict[0])


test_loss = 0
pred = []
ans = []

best_model.eval()

for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['test']):
	with torch.no_grad() :
		output = best_model(drug1, drug2)
		y = y.view(-1,1)
		MSE = nn.MSELoss()
		loss = MSE(output, y) 
		OUT = output.squeeze().tolist()
		Y = y.squeeze().tolist()
		pred = pred+OUT
		ans = ans + Y
		test_loss = test_loss + loss.item()

test_loss/(batch_idx_v+1)
pearson(ans, pred)
spearman(ans, pred)








# 최저를 찾으려면 

TOT_min = np.Inf
TOT_key = ""
for key in ANA_ALL_DF.keys():
	trial_min = min(ANA_ALL_DF[key]['ValLoss'])
	if trial_min < TOT_min :
		TOT_min = trial_min
		TOT_key = key

# /home/jiyeonH/ray_results/22.05.24.MM_trial_1/RAY_MY_train_52efc2d2_3_batch_size=100,dropout_1=0.01,dropout_2=0.2,feat_size_1=4096,feat_size_2=4096,feat_size_3=512,feat_size_4=_2022-05-25_11-56-50
# 170

mini_df = ANA_ALL_DF[TOT_key]

plot_loss(list(mini_df.TrainLoss), list(mini_df.ValLoss), '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/my_trial_1', 'MM_trial_1.no2' )



checkpoint = "/checkpoint_"+str(mini_df[mini_df.ValLoss == TOT_min].index.item()).zfill(6)
TOPVAL_PATH = TOT_key + checkpoint

dsn1_layers = [2048, 4096 , 4096 ]
dsn2_layers = [2048, 4096 , 4096 ] 
snp_layers = [512 , 4096]
inDrop = 0.01
Drop = 0.2
#
#
best_model = MY_parallel_model(
	1234,1234, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, Drop
	)	

state_dict = torch.load(os.path.join(TOPVAL_PATH, "checkpoint"))
best_model.load_state_dict(state_dict[0])


test_loss = 0
pred = []
ans = []

best_model.eval()

for batch_idx_v, (drug1, drug2, y) in enumerate(loaders['test']):
	with torch.no_grad() :
		output = best_model(drug1, drug2)
		y = y.view(-1,1)
		MSE = nn.MSELoss()
		loss = MSE(output, y) 
		OUT = output.squeeze().tolist()
		Y = y.squeeze().tolist()
		pred = pred+OUT
		ans = ans + Y
		test_loss = test_loss + loss.item()

test_loss/(batch_idx_v+1)
pearson(ans, pred)
spearman(ans, pred)









def draw_stack(NAME):
	correct = 0
	total = len(data_loader_test.dataset)
	pred_list= []
	ans_list = []
	with torch.no_grad():
		for step, data in enumerate(data_loader_test):
			# get the inputs; data is a list of [inputs, labels]
			X = data[0]
			ADJ = data[1]
			ANS = data[2].flatten().long()
			t = [a for a in range(data_loader_test.batch_size)]
			rr = np.repeat(t, data_loader_test.dataset.list_feature.shape[1])
			batch_labels = torch.Tensor(rr).long()
			#
			pred = best_model(X, ADJ, batch_labels).max(dim=1)[1]
			#pred = best_model(data[0:2])[0].max(dim=1)[1]
			pred_list += pred.tolist()
			ans_list += ANS.tolist()
			correct += pred.eq(ANS.view(-1)).sum().item()
	#	
	accuracy = correct/total
	fpr, tpr, thresholds = metrics.roc_curve(ans_list, pred_list)
	roc_auc = metrics.auc(fpr, tpr)
	metrics.confusion_matrix(ans_list, pred_list)
	sns.heatmap(metrics.confusion_matrix(ans_list, pred_list), annot=True)
	print("Accuracy : {}".format(accuracy))
	print("ROC_AUC : {}".format(roc_auc))
	#
	plt.savefig("/st06/jiyeonH/11.TOX/MY_TRIAL_4/{}.png".format(NAME))
	plt.close()




####### 내가 만든 checkpoint 인데, ray 에서는 별로 필요 ㄴ 




		# create checkpoint variable and add important data
		checkpoint = {
			'epoch': epoch,
			'valid_loss_min': valid_loss,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
		}
		#
		save_ckp(checkpoint, False, checkpoint_path, best_model_path)
		#
		if valid_loss <= valid_loss_min:
			print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,valid_loss))
			# save checkpoint as best model
			save_ckp(checkpoint, True, checkpoint_path, best_model_path)
			valid_loss_min = valid_loss

		print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} in {:2f}sec'.format(
			epoch, 
			TRAIN_LOSS,
			VALID_LOSS,
			time_spent.total_seconds()
			))
		#















