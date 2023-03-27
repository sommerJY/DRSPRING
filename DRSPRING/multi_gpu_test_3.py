


# 이건 pytorch github 에서 보고 따라하는거 
# 이거 보고했는데도 틀리면 
# ip 쓰는 방식이 잘못된것 같음 

# 아직 진행중


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

#import ray
#from ray import tune
#from functools import partial
#from ray.tune.schedulers import ASHAScheduler
#from ray.tune import CLIReporter
#from ray.tune.suggest.optuna import OptunaSearch
#from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group, destroy_process_group

import torch.distributed as dist
from argparse import ArgumentParser
import torch.multiprocessing as mp

from utils import *
from get_input import *
from layers import *


#os.environ['MASTER_ADDR'] = ip_address
#print('ip_address is :' + ip_address, flush = True)
#os.environ['MASTER_PORT'] = '8888' 
#os.environ['WORLD_SIZE'] = str(world_size)
#dist.init_process_group(
#		backend='nccl',
#		init_method='tcp://localhost:8888',
#		world_size=world_size,
#		rank=rank


def ddp_setup(rank, world_size) : 
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "12355"
	dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


class Trainer :
	def __init__(self, model: torch.nn.Module, train_data: DataLoader, optimizer: torch.optim.Optimizer, gpu_id: int, save_every: int,) -> None:
		self.gpu_id = gpu_id
		self.model = model.to(gpu_id)
		self.train_data = train_data
		self.optimizer = optimizer
		self.criterion = torch.nn.MSELoss()
		self.save_every = save_every
		self.model = DDP(model, device_ids=[gpu_id])
		#
	#
	def _run_batch(self, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y):
		self.optimizer.zero_grad()
		output = self.model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
		loss = self.criterion(output, targets)
		loss.backward()
		self.optimizer.step()
	#
	def _run_epoch(self, epoch):
		b_sz = len(next(iter(self.train_data))[0])
		print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
		self.train_data.sampler.set_epoch(epoch)
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(self.train_data):
			print("gpu : {}, epoch : {}/{}, batchid = {}".format(self.gpu_id, epoch+1, 100, batch_idx_t), flush = True)
			expA = expA.view(-1,3)
			expB = expB.view(-1,3)
			adj_w = adj_w.squeeze()
			#
			drug1_f = drug1_f.to(self.gpu_id)
			drug2_f = drug2_f.to(self.gpu_id)
			drug1_a = drug1_a.to(self.gpu_id)
			expA = expA.to(self.gpu_id)
			expB = expB.to(self.gpu_id)
			adj = adj.to(self.gpu_id)
			cell = cell.to(self.gpu_id)
			adj_w = adj_w.to(self.gpu_id)			
			y = y.to(self.gpu_id)
			#
			this_loss = self._run_batch(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			print("gpu : {}, epoch : {}/{}, loss = {}".format(self.gpu_id, epoch+1, 100, loss), flush = True)
	#
	def _save_checkpoint(self, epoch):
		ckp = self.model.module.state_dict()
		PATH = "/home01/k020a01/TEST/checkpoint.pt"
		torch.save(ckp, PATH)
		print(f"Epoch {epoch} | Training checkpoint saved at {PATH}")
	#
	def train(self, max_epochs: int):
		for epoch in range(max_epochs):
			self._run_epoch(epoch)
			if self.gpu_id == 0 and epoch % self.save_every == 0:
				self._save_checkpoint(epoch)




def load_train_objs():
	NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
	LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
	DC_PATH = '/home01/k020a01/01.Data/DrugComb/'
	PRJ_PATH ='/home01/k020a01/TEST/'
	#
	A_B_C_S_SET_SM, T_train_0, T_val_0, T_test_0, batch_cut_weight = prepare_input(
			MJ_NAME = 'M3V4', WORK_DATE = '23.03.16', MISS_NAME = 'MIS2', file_name = 'M3V4ccle_MISS2_FULL', 
			WORK_NAME = 'WORK_20', CELL_CUT = 200)
	#
	train_set = T_train_0  # load your dataset
	#
	dsn1_layers = [128, 128, 128]
	dsn2_layers = [64,64,64]
	snp_layers = [32,32]
	inDrop = 0.5
	Drop = 0.2
	#
	model = MY_expGCN_parallel_model(
		3, T_train_0.gcn_drug1_F.shape[-1] , 4,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		3, 3 , 10,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
	)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01 )
	return train_set, model, optimizer



def prepare_dataloader(dataset: Dataset, batch_size: int):
	return DataLoader(
		dataset,
		batch_size=batch_size,
		collate_fn = graph_collate_fn,
		pin_memory=True,
		shuffle=False,
		num_workers=4,
		sampler=DistributedSampler(dataset)
	)



def main(rank: int, world_size: int, save_every: int, total_epochs: int, batch_size: int):
	ddp_setup(rank, world_size)
	dataset, model, optimizer = load_train_objs()
	train_data = prepare_dataloader(dataset, batch_size)
	trainer = Trainer(model, train_data, optimizer, rank, save_every)
	trainer.train(total_epochs)
	destroy_process_group()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='simple distributed training job')
	parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
	parser.add_argument('save_every', type=int, help='How often to save a snapshot')
	parser.add_argument('--batch_size', default=16, type=int, help='Input batch size on each device (default: 32)')
	args = parser.parse_args()
	#
	world_size = torch.cuda.device_count()
	mp.spawn(main, args=(world_size, args.save_every, args.total_epochs, args.batch_size), nprocs=world_size)
	#mp.spawn(main, args=(world_size, 1, 100, 32), nprocs=world_size)





