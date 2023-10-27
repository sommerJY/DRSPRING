from torch_geometric.nn import GCNConv, Linear, global_mean_pool
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from corr_test import rmse, correlation
from utils import EarlyStopping, after_process
import random



seed = 5214
np.random.seed(seed=seed)
random.seed(a=seed)
torch.manual_seed(seed)

num_gene = int(349)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, gene_hidden_channels, layer_count, glayer_count, hidden_layer_size):
        super().__init__()
        self.conv1 = GCNConv(64, hidden_channels)
        self.conv2 = nn.ModuleList([GCNConv(hidden_channels, hidden_channels) for i in range(layer_count)])
        self.geneconv1 = GCNConv(2, gene_hidden_channels)
        self.geneconv2 = nn.ModuleList([GCNConv(gene_hidden_channels, gene_hidden_channels) for i in range(glayer_count)])
        if hidden_channels >= gene_hidden_channels:
            dense_size = gene_hidden_channels
        else:
            dense_size = hidden_channels
        self.linear_x = Linear(hidden_channels, (dense_size), weight_initializer='glorot')
        self.linear_exp = Linear(gene_hidden_channels, (dense_size), weight_initializer='glorot')
        self.linear_dose = Linear(int(1), int(1), weight_initializer='glorot')
        self.linear_time = Linear(int(1), int(1), weight_initializer='glorot')
        self.linear1 = Linear(int((dense_size*2)+2), hidden_layer_size, weight_initializer='glorot')
        self.linear2 = Linear(hidden_layer_size, 349, weight_initializer='glorot')
    ##
    def forward(self, data, drop_pert): #, batch
        x, edge_index, gene_exp, gene_edge_index = data.x_s, data.edge_index_s, data.x_p, data.edge_index_p
        dose_vect, time_vect = data.dose, data.time 
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)
        for l in self.conv2:
            x = l(x, edge_index)
            x = F.elu(x)
            x = F.dropout(x, training=self.training)
        x = global_mean_pool(x, data.x_s_batch)
        x = self.linear_x(x)
        ########
        exp = self.geneconv1(gene_exp, gene_edge_index)
        exp = F.elu(exp)
        exp = F.dropout(exp, training=self.training)
        for gl in self.geneconv2:
            exp = gl(exp, gene_edge_index)
            exp = F.elu(exp)
            exp = F.dropout(exp, training=self.training)
        exp = global_mean_pool(exp, data.x_p_batch)
        #readout
        exp = self.linear_exp(exp)
        dose = self.linear_dose(dose_vect)
        time = self.linear_time(time_vect)
        out = torch.cat((x, exp, dose, time), 1)
        out = self.linear1(out)
        out = F.dropout(out, p = drop_pert, training=self.training)
        out = F.elu(out)
        out = self.linear2(out)# out = torch.tanh(out)
        return out
    


def training_model(data, save_file, device, es_value, max_epoch,learning_rate,drop_pert,hid_dim,ghid_dim,b_size,layer_count,glayer_count,hidden_layer_size):
    early_stop = EarlyStopping(patience=200)
    train_loader = DataLoader(data[0], batch_size=b_size, follow_batch=['x_s', 'x_p'], shuffle=True, num_workers=0)
    dev_loader = DataLoader(data[1], batch_size=b_size, follow_batch=['x_s', 'x_p'], shuffle=True, num_workers=0)
    test_loader = DataLoader(data[2], batch_size=b_size, follow_batch=['x_s', 'x_p'], shuffle=True, num_workers=0)
    #
    model = GCN(hid_dim, ghid_dim, layer_count, glayer_count, hidden_layer_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    train_loss_list, dev_loss_list, test_loss_list =[],[],[]
    pearson_list_train, pearson_list_dev, pearson_list_test = [],[],[]
    spearman_list_train, spearman_list_dev,spearman_list_test = [],[],[]
    rmse_list_train, rmse_list_dev, rmse_list_test = [],[],[]
    #
    best_test_pearson = float("-inf")
    #
    for epoch in range(max_epoch):
        print("Iteration %d:" % (epoch+1))
        model.train()
        epoch_loss = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        for i, data in enumerate(train_loader): 
            predict = model(data.to(device), drop_pert) 
            loss = criterion(predict, data.y)
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()
            epoch_loss += loss.item()
            predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
            lb_np = np.concatenate((lb_np, data.y.cpu().detach().numpy()), axis=0)
        print('Train loss:',epoch_loss/(i+1))
        train_loss_test =(epoch_loss / (i + 1))
        train_loss_list.append(train_loss_test)
        rmse_score = rmse(lb_np, predict_np)
        rmse_list_train.append(rmse_score)
        print('Train RMSE: %.4f' % rmse_score)
        pearson, _ = correlation(lb_np, predict_np, 'pearson')
        pearson_list_train.append(pearson)
        print('Train Pearson\'s correlation: %.4f' % pearson)
        spearman, _ = correlation(lb_np, predict_np, 'spearman')
        spearman_list_train.append(spearman)
        print('Train Spearman\'s correlation: %.4f' % spearman)
        #
        model.eval()
        epoch_loss_dev = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, data in enumerate(dev_loader): 
                predict = model(data.to(device), drop_pert) 
                loss = criterion(predict, data.y)
                epoch_loss_dev += loss.item()
                predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
                lb_np = np.concatenate((lb_np, data.y.cpu().detach().numpy()), axis=0)
            print('Dev loss:',epoch_loss_dev / (i + 1))
            dev_loss_list.append(epoch_loss_dev/(i+1))
            dev_loss_dev =(epoch_loss_dev / (i + 1))
            if es_value=='es':
                early_stop.step(dev_loss_dev)
            ###
            rmse_score = rmse(lb_np, predict_np)
            rmse_list_dev.append(rmse_score)
            print('Dev RMSE: %.4f' % rmse_score)
            pearson_dev, _ = correlation(lb_np, predict_np, 'pearson')
            pearson_list_dev.append(pearson_dev)
            print('Dev Pearson\'s correlation: %.4f' % pearson_dev)
            spearman, _ = correlation(lb_np, predict_np, 'spearman')
            spearman_list_dev.append(spearman)
            print('Dev Spearman\'s correlation: %.4f' % spearman)
            #
        model.eval()
        epoch_loss_test = 0
        lb_np = np.empty([0, num_gene])
        predict_np = np.empty([0, num_gene])
        with torch.no_grad():
            for i, data in enumerate(test_loader): 
                predict = model(data.to(device), drop_pert) 
                loss = criterion(predict, data.y)
                epoch_loss_test += loss.item()
                predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
                lb_np = np.concatenate((lb_np, data.y.cpu().detach().numpy()), axis=0)
            print('Test loss:',epoch_loss_test / (i + 1))
            test_loss_test =(epoch_loss_test / (i + 1))
            test_loss_list.append(test_loss_test)
            ###
            rmse_score = rmse(lb_np, predict_np)
            rmse_list_test.append(rmse_score)
            print('Test RMSE: %.4f' % rmse_score)
            pearson_test, _ = correlation(lb_np, predict_np, 'pearson')
            pearson_list_test.append(pearson_test)
            print('Test Pearson\'s correlation: %.4f' % pearson_test)
            spearman, _ = correlation(lb_np, predict_np, 'spearman')
            spearman_list_test.append(spearman)
            print('Test Spearman\'s correlation: %.4f' % spearman)
            #
            if best_test_pearson < pearson_test:
                best_test_pearson = pearson_test
            #
            path3 = save_file
            result_df = pd.DataFrame({"loss":train_loss_list,"dev_loss":dev_loss_list,"test_loss":test_loss_list,
                        "rmse":rmse_list_train,"dev_rmse":rmse_list_dev,"test_rmse":rmse_list_test,
                        "pearson":pearson_list_train,"dev_rmse":rmse_list_dev,"test_rmse":rmse_list_test,
                        "spearman":spearman_list_train,"dev_spearman":spearman_list_dev,"test_spearman":spearman_list_test})
            result_df.index = np.arange(1, len(result_df)+1)
            if epoch==999:
                print("Best Test Pearson is : ", best_test_pearson)
                torch.save(model.state_dict(), path3+"./trained_model.pt")
                result_df.to_csv(path3+"./result_df.csv")
            if es_value=='es' and early_stop.is_stop():
                print("Best Test Pearson is : ", best_test_pearson)
                torch.save(model.state_dict(), path3+"./trained_model.pt")
                result_df.to_csv(path3+"./result_df.csv")
                break
            #





def testing_model(data, save_file, saved_model,basal, device,learning_rate,drop_pert,hid_dim,ghid_dim,b_size,layer_count,glayer_count,hidden_layer_size):
    test_loader = DataLoader(data, batch_size=b_size, follow_batch=['x_s', 'x_p'], shuffle=True, num_workers=0)
    #
    model = GCN(hid_dim, ghid_dim, layer_count, glayer_count, hidden_layer_size).to(device)
    model.load_state_dict(torch.load(saved_model, map_location=device))
    model.to(device)
    model.eval()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    test_df= pd.read_csv("./final_data/raw/test_tf.csv")  
    gi_info_lm_list = test_df.iloc[:,5:354].columns.to_list()
    ccle_info_file = "./final_data/raw/ccle_lincs_convert.csv"
    ccle_info = pd.read_csv(ccle_info_file, low_memory=False)
    max_epoch = 1
    #
    for epoch in range(max_epoch):
        model.eval()
        predict_np = np.empty([0, num_gene])
        pert_list, cell_list, sm_list = [],[],[]
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                predict = model(data.to(device), drop_pert) 
                predict_np = np.concatenate((predict_np, predict.cpu().detach().numpy()), axis=0)
                pert_list = pert_list + data.pert_id
                cell_list = cell_list + data.cell_id
                sm_list = sm_list + data.smiles
    predict_df = pd.DataFrame(predict_np)
    predict_df.columns = gi_info_lm_list
    info_df = pd.DataFrame({'pert_id':pert_list}) 
    info_df['cell_id'] = cell_list
    info_df['smiles'] = sm_list
    if basal == 'lincs_wth_ccle_org_all.csv':
        info_dfcl = pd.merge(info_df,ccle_info,how='left',left_on='cell_id', right_on='cell_iname')
        info_dfpc = info_dfcl[['pert_id','ccle_name']]
    else:
        info_dfpc = info_df[['pert_id','cell_id']]
        info_dfpc.columns = ['pert_id','ccle_name']
    pred_np_processed = after_process(info_dfpc,predict_df)
    pred_np_processed.to_csv(save_file+'test_predicted_expression.csv')


