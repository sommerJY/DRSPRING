
# 0316
#하던거 : __main__ 을 쓰는 형식으로 사용해야 argparse 가 제대로 작동할것 같음 
#그래서 spawn 관련된거 다시 확인해봐야함 
#근데 ip 를 나눠서 던지는거까지는 되는것 같은데
#모델 하나에 대해서 실험 성공만 되면 ray 에 연결시키는것도 가능할듯? 
#근데 굳이 그래야하나 싶기도 하고
#hyperparameter 쓰지 않고 전체 데이터를 8gpu 에 learning 하는게 얼마나 걸리는지 확인을 좀 해봐야할것 같음 
#data distribution 정도는 쓸줄 알아야 pytorch 에 GPU 쓸줄 안다고 볼 수 있을듯 #
#근데 이러면 데이터 가져오는 코드를 어떻게 확인하지 

# 0317
# 그나마 nvidia-smi 에서 잡히는 코드 
# 문제가 뭘까 왜 다 제대로 설정해주고 nvidia 에서도 먹는데 
# train 내에서 아무것도 print 를 못할까 






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

import torch.multiprocessing as mp
import torch.distributed as dist

from torch.utils.data.distributed import DistributedSampler


import torch.distributed as dist
from argparse import ArgumentParser
import torch.multiprocessing as mp

from utils import *
from get_input import *
from layers import *










def JY_train( gpu, args ) :
	NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
	LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
	DC_PATH = '/home01/k020a01/01.Data/DrugComb/'
	PRJ_PATH ='/home01/k020a01/TEST'
	#
	args.gpu = gpu
	print('gpu:',gpu)
	#rank = args.local_ranks * args.ngpus+gpu
	#rank = int(os.environ.get("SLURM_NODEID")) * args.ngpus + gpu
	rank = int(os.environ['SLURM_PROCID'])
	print('rank:', rank)
	#
	dist.init_process_group(
		backend='nccl',
		init_method='tcp://localhost:8888',
		world_size=args.world_size,
		rank=rank
	)
	#
	torch.manual_seed(0) # node 들 사이에서의 randomness control
	torch.cuda.set_device(args.gpu)
	args.main = (args.rank == 0)
	#
	A_B_C_S_SET_SM, T_train_0, T_val_0, T_test_0, batch_cut_weight = prepare_input(
		MJ_NAME = 'M3V4', WORK_DATE = '23.03.16', MISS_NAME = 'MIS2', file_name = 'M3V4ccle_MISS2_FULL', 
		WORK_NAME = 'WORK_20', CELL_CUT = 200)
	#
	train_sampler = torch.utils.data.distributed.DistributedSampler(
		T_train_0, num_replicas = args.world_size, rank = rank, seed=  24
	)
	train_loader = torch.utils.data.DataLoader(
		T_train_0, batch_size=int(128/args.ngpus), collate_fn = graph_collate_fn, shuffle =False, num_workers=4,
		pin_memory = True, sampler = train_sampler
	)
	#
	#
	dsn1_layers = [128, 128, 128]
	dsn2_layers = [64,64,64]
	snp_layers = [32,32]
	inDrop = 0.5
	Drop = 0.2
	#
	#
	model = MY_expGCN_parallel_model(
		3, T_train_0.gcn_drug1_F.shape[-1] , 4,      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		3, 3 , 10,      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
	)
	model.cuda(args.gpu)
	model = torch.nn.parallel.DistributedDataParallel(
		model, device_ids = [args.gpu], find_unused_parameters = True
	)
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01 )
	criterion = torch.nn.MSELoss()
	for epoch in range(args.epochs) :
		print("epoch : " + epoch, flush = True)
		train_loader.sampler.set_epoch(epoch)
		running_loss = 0
		last_loss = 0 
		#
		ans_list = []
		pred_list = []
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(train_loader) :
			print('gpu:' + gpu + ", batch id :"+ batch_idx_t, flush = True)
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
			adj_w = adj_w.squeeze()
			drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(args.gpus), drug2_f.cuda(args.gpus), drug1_a.cuda(args.gpus), drug2_a.cuda(args.gpus), expA.cuda(args.gpus), expB.cuda(args.gpus), adj.cuda(args.gpus), adj_w.cuda(args.gpus), y.cuda(args.gpus), cell.cuda(args.gpus) 
			optimizer.zero_grad()
			output = model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			loss = criterion(output, y)
			loss.backward()
			optimizer.step()
			#
			running_loss = running_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
		#
		last_loss = running_loss / (batch_idx_t+1)
		train_sc, _ = stats.spearmanr(pred_list, ans_list)
		train_pc, _ = stats.pearsonr(pred_list, ans_list)
		print("epoch : {}/{}, loss = {}".format(epoch+1, args.epochs, loss), flush = True)
		if rank == 0 :
			dict_model = {
				'state_dict' : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'epoch' : args.epochs,
			}
			torch.save(dict_model, PRJ_PATH+'/model.pth')


if __name__ == "__main__":
	#
	parser = argparse.ArgumentParser()
	parser.add_argument('--nodes', default=1, type=int)
	parser.add_argument('--local_ranks', default=0, type=int,help="Node's order number in [0, num_of_nodes-1]")
	parser.add_argument('--ip_address', type=str, required=True,help='ip address of the host node')
	parser.add_argument('--ngpus', default=1, type=int, help='number of gpus per node')
	parser.add_argument('--epochs', default=100, type=int, metavar='N',help='number of total epochs to run')
	args = parser.parse_args()
	#
	args.world_size = args.ngpus * args.nodes
	os.environ['MASTER_ADDR'] = args.ip_address
	print('ip_address is :' + args.ip_address, flush = True)
	os.environ['MASTER_PORT'] = '8888'
	os.environ['WORLD_SIZE'] = str(args.world_size)
	#
	mp.spawn(JY_train, nprocs=args.ngpus, args=(args,))

