
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
#import datetime
#from datetime import *
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
import time
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

from torch.utils.data.distributed import DistributedSampler


import torch.distributed as dist
from argparse import ArgumentParser
import torch.multiprocessing as mp

from utils import *
from get_input import *
from layers import *

import builtins
import torch.backends.cudnn as cudnn




parser = argparse.ArgumentParser(description='test')
parser.add_argument('--net_type', default='pyramidnet', type=str,
					help='networktype: resnet, and pyamidnet')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
					help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
					help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
					help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch_size', default=128, type=int,
					metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
					metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
					help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
					metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
					metavar='N', help='print frequency (default: 10)')
parser.add_argument('--depth', default=32, type=int,
					help='depth of the network (default: 32)')
parser.add_argument('--no-bottleneck', dest='bottleneck', action='store_false',
					help='to use basicblock for CIFAR datasets (default: bottleneck)')
parser.add_argument('--dataset', dest='dataset', default='imagenet', type=str,
					help='dataset (options: cifar10, cifar100, and imagenet)')
parser.add_argument('--no-verbose', dest='verbose', action='store_false',
					help='to print the status at every iteration')
parser.add_argument('--alpha', default=300, type=float,
					help='number of new channel increases per depth (default: 300)')
parser.add_argument('--expname', default='TEST', type=str,
					help='name of experiment')
parser.add_argument('--beta', default=0, type=float,
					help='hyperparameter beta')
parser.add_argument('--cutmix_prob', default=0, type=float,
					help='cutmix probability')

parser.add_argument('--resume', default='', type=str, metavar='PATH',
					help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=-1, type=int,
					help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
					help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
					help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
					help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
					help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
					help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
					help='Use multi-processing distributed training to launch '
						 'N processes per node, which has N GPUs. This is the '
						 'fastest way to use PyTorch for either single node or '
						 'multi node data parallel training')







class AverageMeter(object):
	"""Computes and stores the average and current value"""
	#
	def __init__(self):
		self.reset()
	#
	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	#
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def main() :
	args = parser.parse_args()
	args.distributed = args.world_size > 1 or args.multiprocessing_distributed
	ngpus_per_node = torch.cuda.device_count()
	if args.multiprocessing_distributed:
		args.world_size=ngpus_per_node*args.world_size
		mp.spawn(main_worker,nprocs=ngpus_per_node,args=(ngpus_per_node,args))
	else : 
		main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
	# 내용1 :gpu 설정
	print(gpu,ngpus_per_node)
	args.gpu = gpu
	#
	global best_err1, best_err5
	# 내용1-1: gpu!=0이면 print pass -> 아냐 안해
	#
	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))
	#
	if args.distributed:
		if args.dist_url=='env://' and args.rank==-1:
			args.rank=int(os.environ["RANK"])
		if args.multiprocessing_distributed:
			# gpu = 0,1,2,...,ngpus_per_node-1
			print("gpu는",gpu)
			args.rank=args.rank*ngpus_per_node + gpu
		# 내용1-2: init_process_group 선언
		torch.distributed.init_process_group(backend=args.dist_backend,init_method=args.dist_url,
											world_size=args.world_size,rank=args.rank)
	#
	# 내용5: 데이터 로딩
	NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
	LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
	DC_PATH = '/home01/k020a01/01.Data/DrugComb/'
	PRJ_PATH ='/home01/k020a01/TEST'
	#
	A_B_C_S_SET_SM, T_train_0, T_val_0, T_test_0, batch_cut_weight = prepare_input(
			MJ_NAME = 'M3V4', WORK_DATE = '23.03.16', MISS_NAME = 'MIS2', file_name = 'M3V4ccle_MISS2_FULL', 
			WORK_NAME = 'WORK_20', CELL_CUT = 200)
	#
	if args.distributed:
		train_sampler = torch.utils.data.distributed.DistributedSampler(T_train_0)
	else:
		train_sampler = None
	#
	train_loader = torch.utils.data.DataLoader( # collate 가 문제가 되는것 같음.... 흠... 
		T_train_0,
		batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,sampler=train_sampler)
	val_loader = torch.utils.data.DataLoader(
		T_val_0,
		batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
	#
	# # 내용2: model 정의
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
	# 내용3: multiprocess 설정
	if args.distributed:
		if args.gpu is not None:
			torch.cuda.set_device(args.gpu)
			model.cuda(args.gpu)
			# when using a single GPU per process and per DDP, we need to divide tha batch size
			# ourselves based on the total number of GPUs we have
			args.batch_size = int(args.batch_size / ngpus_per_node)
			args.workers = int((args.workers+ngpus_per_node-1)/ngpus_per_node)
			# 내용3-1: model ddp설정
			model = torch.nn.parallel.DistributedDataParallel(model,device_ids=[args.gpu])# args.gpu가 무슨 값인지 알고 싶다.
		else:
			model.cuda()
			# DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
			# 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
			model = torch.nn.parallel.DistributedDataParallel(model)
	elif args.gpu is not None:
		torch.cuda.set_device(args.gpu)
		model=model.cuda(args.gpu)
		raise NotImplementedError("Only DistributedDataParallel is supported.")
	else:
		raise NotImplementedError("Only DistributedDataparallel is supported.")
	# 내용4: criterion / optimizer 정의
	criterion = nn.CrossEntropyLoss().cuda(args.gpu)
	#
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay, nesterov=True)
	#
	# 내용 6: for문을 통한 training
	#
	cudnn.benchmark = True
	#stime = time.time()
	for epoch in range(100) :
		print("epoch : " + str(epoch), flush = True)
		train_loader.sampler.set_epoch(epoch)
		running_loss = 0
		last_loss = 0 
		#
		ans_list = []
		pred_list = []
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(train_loader) :
			print('gpu:' + str(args.gpu) + ", batch id :"+ str(batch_idx_t), flush = True)
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
		print("epoch : {}/{}, loss = {}".format(epoch+1, 100, loss), flush = True)
		if args.rank == 0 :
			dict_model = {
				'state_dict' : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'epoch' : epoch,
			}
			torch.save(dict_model, PRJ_PATH+'/model.pth')



if __name__ == "__main__":
	#
	main()


