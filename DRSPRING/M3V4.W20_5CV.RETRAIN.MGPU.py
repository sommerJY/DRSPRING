
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.nn import DataParallel
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from datetime import datetime
import pandas as pd 

from utils import *
from get_input import *
from layers import *


def ddp_setup():
	init_process_group(backend="nccl")

class Trainer:
	def __init__(
		self,
		model: torch.nn.Module,
		train_data: DataLoader,
		val_data: DataLoader,
		optimizer: torch.optim.Optimizer,
		save_every: int,
		snapshot_path: str,
	) -> None:
		self.gpu_id = int(os.environ["LOCAL_RANK"])
		self.train_data = train_data
		self.val_data = val_data
		self.optimizer = optimizer
		self.save_every = save_every
		self.epochs_run = 0
		self.snapshot_path = snapshot_path
		#if os.path.exists(snapshot_path):
		#	print("Loading snapshot")
		#	self._load_snapshot(snapshot_path)
		# original ver
		self.model = model.to(self.gpu_id)
		self.model = DDP(self.model, device_ids=[self.gpu_id])
		self.total_loss = []


	#def _load_snapshot(self, snapshot_path):
	#	loc = f"cuda:{self.gpu_id}"
	#	snapshot = torch.load(snapshot_path, map_location=loc)
	#	self.model.load_state_dict(snapshot["MODEL_STATE"])
	#	self.epochs_run = snapshot["EPOCHS_RUN"]
	#	print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

	def calc_batch_label (self, syn, feat) :
		batchnum = syn.shape[0]
		nodenum = feat.shape[0]/batchnum
		Num = [a for a in range(batchnum)]
		Rep = np.repeat(Num, nodenum)
		batch_labels = torch.Tensor(Rep).long()
		return batch_labels
	#
	def _run_batch(self, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y):
		#print('run batch')
		Drug_batch_label = self.calc_batch_label(y, drug1_f)
		Exp_batch_label = self.calc_batch_label(y, expA)
		Drug_batch_label.to(self.gpu_id)
		Exp_batch_label.to(self.gpu_id)
		#
		if self.model.training == True : 
			self.optimizer.zero_grad()	
			#
			output = self.model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y, Drug_batch_label, Exp_batch_label)
			#print('GPU{} output : {}'.format(self.gpu_id, output))
			criterion = torch.nn.MSELoss()
			loss = criterion(output, y)
			#print('GPU{} batch loss : {}'.format(self.gpu_id ,loss))
			#criterion = weighted_mse_loss
			#loss = criterion(output, targets, weight)
			loss.backward()
			self.optimizer.step()
		else :
			with torch.no_grad():
				output = self.model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y, Drug_batch_label, Exp_batch_label)
				# print('GPU{} output : {}'.format(self.gpu_id, output))
				criterion = torch.nn.MSELoss()
				loss = criterion(output, y)
				# print('GPU{} batch loss : {}'.format(self.gpu_id ,loss))
		pred_list = output.squeeze().tolist()
		ans_list = y.squeeze().tolist()
		return loss, pred_list, ans_list




	def _run_epoch(self, epoch):
		self.model.train()
		b_sz = len(next(iter(self.train_data))[0])
		# print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
		self.train_data.sampler.set_epoch(epoch)
		train_loss = 0
		all_pred_list = []
		all_ans_list = []
		for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(self.train_data):
			# print("TRAIN gpu : {}, epoch : {}/{}, batchid = {}".format(self.gpu_id, epoch+1, 1000, batch_idx_t), flush = True)
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
			this_loss, pred_list, ans_list = self._run_batch(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			train_loss += this_loss.item()
			all_pred_list = all_pred_list + pred_list
			all_ans_list = all_ans_list + ans_list
		train_sc, _ = stats.spearmanr(all_pred_list, all_ans_list)
		train_pc, _ = stats.pearsonr(all_pred_list, all_ans_list) 
		final_train_loss = train_loss/(batch_idx_t+1)
		print("TRAIN gpu : {}, epoch : {}/{}, loss = {}".format(self.gpu_id, epoch+1, 1000, final_train_loss), flush = True)
		return self.gpu_id, epoch, final_train_loss, train_sc, train_pc 


	def _validation(self, epoch):
		self.model.eval()
		b_sz = len(next(iter(self.val_data))[0])
		# print(f"[GPU{self.gpu_id}]| Batchsize: {b_sz} | Steps: {len(self.val_data)}")
		val_loss = 0
		all_pred_list = []
		all_ans_list = []
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(self.val_data):
			# print("VAL gpu : {},  batchid = {}".format(self.gpu_id,  batch_idx_v), flush = True)
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
			this_loss, pred_list, ans_list = self._run_batch(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			val_loss += this_loss.item()
			all_pred_list = all_pred_list + pred_list
			all_ans_list = all_ans_list + ans_list
		val_sc, _ = stats.spearmanr(all_pred_list, all_ans_list)
		val_pc, _ = stats.pearsonr(all_pred_list, all_ans_list) 
		final_val_loss = val_loss/(batch_idx_v+1)
		print("VAL gpu : {}, epoch : {}/{}, loss = {}".format(self.gpu_id,  epoch+1, 1000, final_val_loss), flush = True)
		return self.gpu_id, final_val_loss, val_sc, val_pc
			


	def _save_snapshot(self, epoch):
		snapshot = {
			"MODEL_STATE": self.model.module.state_dict(),
			"OPT_STATE" : self.optimizer.state_dict(),
			"EPOCHS_RUN": epoch,
		}
		checkpoint = str(epoch).zfill(4)
		os.makedirs( os.path.join(self.snapshot_path,'checkpoints'), exist_ok = True)
		epoch_snap_path = os.path.join(self.snapshot_path, 'checkpoints',"checkpoint{}.pt".format(checkpoint))
		torch.save(snapshot, epoch_snap_path)
		print(f"Epoch {epoch} | Training snapshot saved at {epoch_snap_path}")


	def train(self, max_epochs: int):
		for epoch in range(self.epochs_run, max_epochs):
			ep_st = datetime.now()
			train_gpu_id, epoch, train_loss, train_sc, train_pc = self._run_epoch(epoch)
			train_df = pd.DataFrame({
				"train_gpu_id" : [train_gpu_id],
				"epoch" : [epoch],
				"train_loss" : [train_loss],
				"train_sc" : [train_sc],
				"train_pc" : [train_pc]
			})
			train_df.to_csv(os.path.join(self.snapshot_path,'Train_DF.csv'), mode='a', index=False, header = False)
			if self.gpu_id == 0 : # and epoch % self.save_every == 0
				self._save_snapshot(epoch)
				val_gpu_id, val_loss, val_sc, val_pc = self._validation(epoch)
				val_df = pd.DataFrame({
				"val_gpu_id" : [val_gpu_id],
				"epoch" : [epoch],
				"val_loss" : [val_loss],
				"val_sc" : [val_sc],
				"val_pc" : [val_pc]
				})
				val_df.to_csv(os.path.join(self.snapshot_path,'Val_DF.csv'), mode='a', index=False, header = False)
				ep_end = datetime.now()
				print("iteration_time : {}s".format((ep_end - ep_st).seconds))
			# torch.distributed.barrier()	어디에 넣어야하는지 약간 헷갈....
			

		

		





def load_train_objs(args): # CSV 내용 추가해야함 
	NETWORK_PATH = '/home01/k020a01/01.Data/HumanNet/'
	LINCS_PATH = '/home01/k020a01/01.Data/LINCS/'
	DC_PATH = '/home01/k020a01/01.Data/DrugComb/'
	PRJ_PATH ='/home01/k020a01/TEST/'
	#
	A_B_C_S_SET_SM, T_train_0, T_val_0, T_test_0, batch_cut_weight = prepare_input(args.MJ_NAME, args.WORK_DATE, args.MISS_NAME, args.file_name, args.WORK_NAME, args.CELL_CUT)
	train_set = T_train_0  # load your dataset
	val_set = T_val_0
	#
	#
	# 아놔 test 비율 바꿨어야 하는데 시불
	from ray.tune import Analysis
	WORK_DATE = '23.04.10'
	PRJ_NAME = 'M3V5'
	MISS_NAME = 'MIS2'
	WORK_NAME = 'WORK_20'
	W_NAME = '349' # 349
	#
	anal_dir = "/home01/k020a01/ray_results/PRJ02.{}.{}.{}.{}/".format(WORK_DATE, PRJ_NAME, MISS_NAME, WORK_NAME)
	list_dir = os.listdir(anal_dir)
	exp_json = [a for a in list_dir if 'experiment_state' in a]
	exp_json
	# anal_df = ExperimentAnalysis(anal_dir+exp_json[2])
	anal_df = Analysis(anal_dir)
	#
	ANA_DF = anal_df.dataframe()
	ANA_ALL_DF = anal_df.trial_dataframes
	#
	# ## change by model##################################!!!!!!!!!!!!!!!!!!
	print('best_model : {}'.format('model 5'))
	max_cor = max(ANA_DF.sort_values('AV_V_SC')['AV_V_SC'])
	DF_KEY = ANA_DF[ANA_DF.AV_V_SC == max_cor]['logdir'].item()
	mini_df = ANA_ALL_DF[DF_KEY]
	my_config = ANA_DF[ANA_DF.logdir==DF_KEY]
	cck_num = mini_df[mini_df.AV_V_SC==max(mini_df.AV_V_SC)].index.item()
	checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
	#
	#
	dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
	dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
	snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#
	model = MY_expGCN_parallel_model(
		my_config['config/G_chem_layer'].item(), 
		T_train_0.gcn_drug1_F.shape[-1] , 
		my_config['config/G_chem_hdim'].item(),      # G_layer_chem, G_indim_chem, G_hiddim_chem, 
		my_config['config/G_exp_layer'].item(), 
		3 , 
		my_config['config/G_exp_hdim'].item(),      # G_layer_exp, G_indim_exp, G_hiddim_exp, 
		dsn1_layers, dsn2_layers, snp_layers,      # drug 1 layers, drug 2 layers, merged layer, 
		len(set(A_B_C_S_SET_SM.DrugCombCCLE)), 1,      # cell_dim ,out_dim,
		inDrop, Drop      # inDrop, drop
	) # load your model
	#
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01 )
	return train_set, val_set, model, optimizer


def prepare_trainloader(dataset: Dataset, batch_size: int, world_size: int):
	train_sampler = DistributedSampler(dataset, num_replicas = int(world_size), rank = int(os.environ["LOCAL_RANK"]))
	train_loader = DataLoader(
		dataset,
		batch_size=int(batch_size/world_size),
		collate_fn = graph_collate_fn,
		pin_memory=True,
		shuffle=False,
		sampler=train_sampler
	)
	return train_loader


def prepare_valloader(dataset: Dataset, batch_size: int, world_size: int):
	val_sampler = DistributedSampler(dataset, num_replicas = int(world_size), rank = int(os.environ["LOCAL_RANK"]))
	val_loader = DataLoader(
		dataset,
		batch_size=int(batch_size/world_size),
		collate_fn = graph_collate_fn,
		pin_memory=True,
		shuffle=False,
		sampler=val_sampler
		)
	return val_loader


def main(args, save_every: int, total_epochs: int, batch_size: int, world_size: int, snapshot_path: str ):
	ddp_setup()
	train_df = pd.DataFrame(columns =['train_gpu_id','epoch','train_loss','train_sc','train_pc'])
	val_df = pd.DataFrame(columns =['val_gpu_id','epoch','val_loss','val_sc','val_pc'])
	train_df.to_csv(os.path.join(snapshot_path,'Train_DF.csv'), index=False)
	val_df.to_csv(os.path.join(snapshot_path,'Val_DF.csv'), index=False)
	#
	train_dataset, val_dataset, model, optimizer = load_train_objs(args)
	train_data = prepare_trainloader(train_dataset, batch_size, world_size)
	val_data = prepare_valloader(val_dataset, batch_size, world_size)
	trainer = Trainer(model, train_data, val_data, optimizer, save_every, snapshot_path)
	trainer.train(total_epochs)
	destroy_process_group()


if __name__ == "__main__":
	import argparse
	parser = argparse.ArgumentParser(description='simple distributed training job')
	parser.add_argument('--MJ_NAME', default='M3V4',  type=str, help='mj model name')
	parser.add_argument('--WORK_DATE', default='23.03.16',  type=str, help='mj model name')
	parser.add_argument('--MISS_NAME', default='MIS2',  type=str, help='mj model name')
	parser.add_argument('--file_name', default='M3V4ccle_MISS2_FULL',  type=str, help='train data name')
	parser.add_argument('--WORK_NAME', default='WORK_20',  type=str, help='jy work name')
	parser.add_argument('--W_NAME', default='W20v1',  type=str, help='jy work name 2')
	parser.add_argument('--CELL_CUT', default='200',  type=int, help='cell line filter')
	#
	parser.add_argument('--total_epochs', default=1000,  type=int, help='Total epochs to train the model')
	parser.add_argument('--save_every', default=10, type=int, help='How often to save a snapshot')
	parser.add_argument('--snap_path', default='/home01/k020a01/TEST', type=str, help='where to save')
	parser.add_argument('--batch_size', default=32, type=int, help='Input batch size on each device (default: 32)')
	parser.add_argument('--world_size', default=4, type=int, help='world size')
	args = parser.parse_args()
	#
	main(args, args.save_every, args.total_epochs, args.batch_size, args.world_size, args.snap_path)



