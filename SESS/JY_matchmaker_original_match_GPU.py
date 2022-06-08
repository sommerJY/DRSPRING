
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

import sys
import random
import shutil
import math


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

MM_PATH = '/home01/k006a01/PRJ.01/'

MM_DATA = pd.read_csv(MM_PATH+'DrugCombinationData.tsv', sep = '\t')
CEL_DATA =pd.read_csv(MM_PATH+'E-MTAB-3610.sdrf.txt', sep = '\t')


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


trial_MM_DATA = MM_PATH+'DrugCombinationData.tsv'
trial_drug1 = MM_PATH+'data/drug1_chem.csv'
trial_drug2 = MM_PATH+'data/drug2_chem.csv'
trial_CEL_gex = MM_PATH+'data/cell_line_gex.csv'

trial_chem1, trial_chem2, trial_cell_line, trial_synergies = data_loader(trial_drug1, trial_drug2, trial_CEL_gex, trial_MM_DATA )

train_indx = MM_PATH+'data/train_inds.txt'
val_indx = MM_PATH+'data/val_inds.txt'
test_indx = MM_PATH+'data/test_inds.txt'


def prepare_data(chem1, chem2, cell_line, synergies, norm, 
	train_ind_fname, val_ind_fname, test_ind_fname):
	print("Data normalization and preparation of train/validation/test data")
	test_ind = list(np.loadtxt(test_ind_fname,dtype=np.int64))
	val_ind = list(np.loadtxt(val_ind_fname,dtype=np.int64))
	train_ind = list(np.loadtxt(train_ind_fname,dtype=np.int64))
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
	


loss_weight2 = loss_weight.tolist()
# batch_cut_weight = torch.Tensor([loss_weight2[i:i+batch_size] for i in range(0,len(loss_weight2), batch_size)])
# tt = [loss_weight2[i:i+batch_size] for i in range(0,len(loss_weight2), batch_size)]



#loaders = {
#	'train' : torch.utils.data.DataLoader(T_train, batch_size = config["batch_size"]),
#	'eval' : torch.utils.data.DataLoader(T_val, batch_size = config["batch_size"]),
#	'test' : torch.utils.data.DataLoader(T_test, batch_size = config["batch_size"]),
#}
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







def original_to_torch(use_cuda=False):  
    n_epochs=  1000
    valid_loss_min = np.Inf
    l_rate = 0.0001
    inDrop = 0.2
    drop = 0.5
    batch_size = 128
    earlyStop_patience = 100
    dsn1_layers = [2048, 512 , 2048 ]
    dsn2_layers = [2048, 512 , 2048 ] 
    snp_layers = [1024 , 1024]
    train_loss_all = []
    valid_loss_all = []
    checkpoint_path= MM_PATH+'TRIAL_1/checkpoint/current_checkpoint.pt'
    best_model_path=MM_PATH+'TRIAL_1/best_model/'
    save_path = MM_PATH+'TRIAL_1/'
    valid_loss_min_input = np.Inf
    criterion = weighted_mse_loss
    # 
    model = MY_parallel_model(
        1396, 1396, dsn1_layers, dsn2_layers, snp_layers, 1, inDrop, drop
        )	
    # 
    if use_cuda:
        model = nn.DataParallel(model)
        model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr = l_rate)
    patience = 100
    trigger_times=0
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
                drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
            ## find the loss and update the model parameters accordingly
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            output = model(drug1, drug2)
            y = y.view(-1,1)
            wc = torch.Tensor(loss_weight2[batch_size*(batch_idx_t):batch_size*(batch_idx_t+1)]).view(-1,1)
            # batch_cut_weight[batch_idx_t].view(-1,1)
            if use_cuda:
                wc = wc.cuda()
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
                drug1, drug2, y = drug1.cuda(), drug2.cuda(), y.cuda()
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
            trigger_times = 0
        else : 
            trigger_times += 1
            print("trigger times : {}".format(str(trigger_times)))
            if trigger_times > patience :
                print('Early stopping!\nStart to test process.')
                return model

    #



batch 따라서 weight 변하는거랑 OK 
early stopping 한번만 확인해서 OK
KISTI 서버에 넣어보기 OK 
그러면 내일은 내내 DB 계획이랑 지금 브리핑 할 사항들 정리해서 알려드릴 수 있을것 같음 









