
from torch_geometric.nn import GCNConv, Linear, global_mean_pool
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
import random
from prep_input import *


seed = 42
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)


class M2_Model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, 
	out_dim, inDrop, drop):
		super(M2_Model, self).__init__()
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
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
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
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, syn ):
		EXP_ADJ_WGT = [1 for a in range(EXP_ADJ.shape[1])]
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
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.elu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.elu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2 ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.elu(X)
			else :
				X = self.SNPs[L3](X)
		return X



class EarlyStopping:
    def __init__(self, patience=200):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience
    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
    def is_stop(self):
        return self.patience >= self.patience_limit


def weighted_mse_loss(input, target, weight):
	return (weight * (input - target) ** 2).mean()

# 이걸 한번더 고쳐야하나 고민 



def inner_train ( TRAIN_DATA, LOSS_WEIGHT, THIS_MODEL, THIS_OPTIMIZER , device) :
	THIS_MODEL.train()
    #
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	batch_cut_weight = LOSS_WEIGHT
    #
	for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y) in enumerate(TRAIN_DATA) :
		expA = expA.view(-1,3)#### 다른점 
		expB = expB.view(-1,3)#### 다른점 
		adj_w = adj_w.squeeze()
		# move to GPU
		if device.type != 'cpu':
			drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
		# 
		THIS_OPTIMIZER.zero_grad()
		output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y)
		wc = torch.Tensor(batch_cut_weight[batch_idx_t]).view(-1,1)
		if device.type != 'cpu':
			wc = wc.cuda()
		loss = weighted_mse_loss(output, y, wc ) # weight 더해주기 
		loss.backward()
		THIS_OPTIMIZER.step()
		#
		running_loss = running_loss + loss.item()
		pred_list = pred_list + output.squeeze().tolist()
		ans_list = ans_list + y.squeeze().tolist()
	#
	last_loss = running_loss / (batch_idx_t+1)
	train_sc, _ = stats.spearmanr(pred_list, ans_list)
	train_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, train_pc, train_sc, THIS_MODEL, THIS_OPTIMIZER     



def inner_val( VAL_DATA, THIS_MODEL, device) :
	THIS_MODEL.eval()
	#
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
    #
    MSE = torch.nn.MSELoss()
    #
	with torch.no_grad() :
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y) in enumerate(VAL_DATA) :
			expA = expA.view(-1,3)
			expB = expB.view(-1,3)
			adj_w = adj_w.squeeze()
			if device.type != 'cpu':
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda()
			output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y)
			loss = MSE(output, y) 
			# 
			running_loss = running_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
    #
	last_loss = running_loss / (batch_idx_v+1)
	val_sc, _ = stats.spearmanr(pred_list, ans_list)
	val_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, val_pc, val_sc, THIS_MODEL, ans_list, pred_list



def training_model(save_path, T_train, T_val, T_test, es, device, config_dict) :
    n_epochs = config_dict['max_epoch']
    # 
    dsn_layers = [int(a) for a in config_dict["dsn_layer"].split('-') ]
    snp_layers = [int(a) for a in config_dict["snp_layer"].split('-') ]
    # 
    train_loader = torch.utils.data.DataLoader(T_train, batch_size = config_dict["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=16)
    val_loader = torch.utils.data.DataLoader(T_val, batch_size = config_dict["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=16)
    test_loader = torch.utils.data.DataLoader(T_test, batch_size = config_dict["batch_size"], collate_fn = graph_collate_fn, shuffle =False, num_workers=16)
    loss_weight = get_loss_weight(T_train)
    batch_cut_weight = [loss_weight[i:i+config_dict["batch_size"]] for i in range(0,len(loss_weight), config_dict["batch_size"])]
    #
    early_stop = EarlyStopping(patience=200)
    #
    THIS_MODEL = M2_Model(
        config_dict["G_chem_layer"], T_train.gcn_drug1_F.shape[-1] , config_dict["G_chem_hdim"], 
        config_dict["G_exp_layer"], 3 , config_dict["G_exp_hdim"], 
        dsn_layers, dsn_layers, snp_layers,  1,  
        config_dict["dropout_1"], config_dict["dropout_2"] )
    #
	THIS_MODEL = THIS_MODEL.cuda()
    #
    THIS_optimizer = torch.optim.Adam(THIS_MODEL.parameters(), lr = config_dict["learning_rate"] )
    #
    train_loss_all = []
	valid_loss_all = []
	train_pearson_corr_all = []
	train_spearman_corr_all = []
	valid_pearson_corr_all = []
	valid_spearman_corr_all = []
    #
	for epoch in range(n_epochs):
		now = datetime.now()
		train_loss = 0.0
		valid_loss = 0.0
		#
		train_loss, train_pc, train_sc, THIS_MODEL, THIS_optimizer = inner_train(T_train, loss_weight, THIS_MODEL, THIS_optimizer, device)
		train_loss_all.append(train_loss)
		train_pearson_corr_all.append(train_pc)
		train_spearman_corr_all.append(train_sc)	
        #
		val_loss, val_pc, val_sc, THIS_MODEL, _, _ = inner_val(T_val, THIS_MODEL, device)
		valid_loss_all.append(val_loss)
		valid_pearson_corr_all.append(val_pc)
		valid_spearman_corr_all.append(val_sc) 
		#    
		done = datetime.now()
		time_spent = done-now
        #
        print('epoch : {}, TrainLoss : {}, ValLoss : {}'.format(epoch, train_loss, val_loss), flush=True)
        # 
        if config_dict['EarlyStop'] == 'es'
            early_stop.step(val_loss)
        #
        if epoch == n_epochs-1 : 
            test_loss, test_pc, test_sc, THIS_MODEL, ans_list, pred_list = inner_val(T_test, THIS_MODEL, device)
            print("Best Test Pearson is : {}".format(test_pc))
            test_dict = {'ANS' : ans_list, 'PRED' : pred_list}
            test_dict_DF = pd.DataFrame(test_dict)
            torch.save(THIS_MODEL.state_dict(), os.path.join(save_path, 'MODEL.pt'))
            test_dict_DF.to_csv(os.path.join(save_path, 'test_result_table.csv'))
        #
        if config_dict['EarlyStop'] == 'es' :
            if val_loss < np.min(valid_loss_all) :
                result_dict = {
                    'train_loss_all' : train_loss_all, 'valid_loss_all' : valid_loss_all, 
                    'train_pearson_corr_all' : train_pearson_corr_all, 'train_spearman_corr_all' : train_spearman_corr_all, 
                    'val_pearson_corr_all' : val_pearson_corr_all, 'val_spearman_corr_all' : val_spearman_corr_all, 
                    }
                result_dict_DF = pd.DataFrame(result_dict)
                print('Best Val Loss : {}'.format(val_loss))
                test_loss, test_pc, test_sc, THIS_MODEL, ans_list, pred_list = inner_val(T_test, THIS_MODEL, device)
                test_dict = {'ANS' : ans_list, 'PRED' : pred_list}
                test_dict_DF = pd.DataFrame(test_dict)
                torch.save(THIS_MODEL.state_dict(), os.path.join(save_path, 'MODEL.pt'))
            #
            if early_stop.is_stop():
                print('Early Stopping in epoch {}'.format(epoch))
                print("Best Test Pearson is : {}".format(test_pc))
                result_dict_DF.to_csv(os.path.join(save_path, 'train_result_table.csv'))
                test_dict_DF.to_csv(os.path.join(save_path, 'test_result_table.csv'))




def pred_simple_synergy (my_config, SM_A, SM_B, CELL):
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item()
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#      
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, 64 , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
				inDrop, Drop
				) 
	#
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		state_dict = torch.load(CKP_PATH) #### change ! 
	else:
		state_dict = torch.load(CKP_PATH, map_location=torch.device('cpu'))
	# 
	# 
	#print("state_dict_done", flush = True)
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)
	#
	#print("state_load_done", flush = True)
	#
	best_model.to(device)
	best_model.eval()
	#
	single = torch.Tensor([[0]])
	drug1_f, drug2_f, drug1_a, drug2_a, FEAT_A, FEAT_B, adj, adj_w = make_simple_input_data(SM_A, SM_B, CELL)
	output_1 = best_model(drug1_f, drug2_f, drug1_a, drug2_a, FEAT_A, FEAT_B, adj, adj_w, single) 
	output_2 = best_model(drug2_f, drug1_f, drug2_a, drug1_a, FEAT_B, FEAT_A, adj, adj_w, single) 
	result_value = np.round(np.mean([output_1.item(), output_2.item()]),4)
	print('Expected Loewe Score is : {}'.format(result_value))
	print('\n')






















