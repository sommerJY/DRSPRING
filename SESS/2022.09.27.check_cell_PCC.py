
# cell line check 

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# ATOM filter best model - M2 였음 

DF_KEY = list(ANA_DF.sort_values('ValLoss')['logdir'])[0]
mini_df = ANA_ALL_DF[DF_KEY]
my_config = ANA_DF[ANA_DF.logdir==DF_KEY]

cck_num =mini_df[mini_df.ValLoss==min(mini_df.ValLoss)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)

T_test = ray.get(RAY_test)

G_layer = my_config['config/G_layer'].item()
G_hiddim = my_config['config/G_hiddim'].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()


best_model = MY_expGCN_parallel_model(
				G_layer, T_test.gcn_drug1_F.shape[-1] , G_hiddim,
				G_layer, 2, G_hiddim,
				dsn1_layers, dsn2_layers, snp_layers, 17, 1,
				inDrop, Drop
				)

model_path = PRJ_PATH

device = "cuda:0" if torch.cuda.is_available() else "cpu"
state_dict = torch.load(os.path.join(model_path, 'M2_checkpoint'), map_location=torch.device('cpu'))

best_model.load_state_dict(state_dict[0])

best_model.eval()






# cell each cell data


def prepare_cell_data_GCN(CELL_chem_A_feat, CELL_chem_B_feat, CELL_chem_A_adj, CELL_chem_B_adj, CELL_exp_A, CELL_exp_B, CELL_tgt_A, CELL_tgt_B, CELL_syn, CELL_cell, norm ) :
    #
    the_data = {}
    the_data['drug1_feat'] = CELL_chem_A_feat
    the_data['drug2_feat'] = CELL_chem_B_feat
    the_data['drug1_adj'] = CELL_chem_A_adj
    the_data['drug2_adj'] = CELL_chem_B_adj
    the_data['EXP_A'] = CELL_exp_A
    the_data['EXP_B'] = CELL_exp_B
    the_data['TGT_A'] = CELL_tgt_A
    the_data['TGT_B'] = CELL_tgt_B
    the_data['y'] = CELL_syn
    the_data['cell'] = CELL_cell
    print(the_data['drug1_feat'].shape, flush=True)
    return the_data



def save_cell_result(cell_name):
    print(cell_name)
    CELL_table = A_B_C_S_SET[A_B_C_S_SET.DrugCombCello == cell_name]
    CELL_table = CELL_table.reset_index(drop=True)
    CELL_shape = CELL_table.shape[0]
    ##
    ##
    CELL_chem_A_feat = torch.empty(size=(CELL_shape, max_len, 64))
    CELL_chem_B_feat= torch.empty(size=(CELL_shape, max_len, 64))
    CELL_chem_A_adj = torch.empty(size=(CELL_shape, max_len, max_len))
    CELL_chem_B_adj= torch.empty(size=(CELL_shape, max_len, max_len))
    CELL_exp_A = torch.empty(size=(CELL_shape, 978))
    CELL_exp_B = torch.empty(size=(CELL_shape, 978))
    CELL_exp_AB = torch.empty(size=(CELL_shape, 978, 2))
    CELL_Cell = torch.empty(size=(CELL_shape, cell_one_hot.shape[1]))
    CELL_tgt_A = torch.empty(size=(CELL_shape, 978))
    CELL_tgt_B = torch.empty(size=(CELL_shape, 978))
    CELL_syn =  torch.empty(size=(CELL_shape,1))
    ##
    ##
    for IND in range(CELL_shape): #  
        DrugA_SIG = CELL_table.iloc[IND,]['BETA_sig_id_x']
        DrugB_SIG = CELL_table.iloc[IND,]['BETA_sig_id_y']
        Cell = CELL_table.iloc[IND,]['DrugCombCello']
        #
        k=1
        DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_SIG, k)
        DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_SIG, k)
        #
        EXP_A, EXP_B, EXP_AB, LINCS = get_LINCS_data(DrugA_SIG, DrugB_SIG)
        #
        Cell_Vec = get_cell(DrugA_SIG, DrugB_SIG, Cell)
        #
        TGT_A = get_targets(DrugA_SIG)
        TGT_B = get_targets(DrugB_SIG)
        #
        AB_SYN = get_synergy_data(DrugA_SIG, DrugB_SIG, Cell)
        #
        CELL_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
        CELL_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
        CELL_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
        CELL_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
        CELL_exp_A[IND] = torch.Tensor(EXP_A.iloc[:,1])
        CELL_exp_B[IND] = torch.Tensor(EXP_B.iloc[:,1])
        CELL_exp_AB[IND] = torch.Tensor(EXP_AB).unsqueeze(0)
        CELL_Cell[IND] = Cell_Vec
        CELL_tgt_A[IND] = torch.Tensor(TGT_A)
        CELL_tgt_B[IND] = torch.Tensor(TGT_B)
        CELL_syn[IND] = torch.Tensor([AB_SYN])
        #
        #
    cell_path = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
    torch.save(CELL_chem_A_feat, cell_path+'0927.{}.CELL_chem_A_feat.pt'.format(cell_name))
    torch.save(CELL_chem_B_feat, cell_path+'0927.{}.CELL_chem_B_feat.pt'.format(cell_name))
    torch.save(CELL_chem_A_adj, cell_path+'0927.{}.CELL_chem_A_adj.pt'.format(cell_name))
    torch.save(CELL_chem_B_adj, cell_path+'0927.{}.CELL_chem_B_adj.pt'.format(cell_name))
    torch.save(CELL_exp_A, cell_path+'0927.{}.CELL_exp_A.pt'.format(cell_name))
    torch.save(CELL_exp_B, cell_path+'0927.{}.CELL_exp_B.pt'.format(cell_name))
    torch.save(CELL_exp_AB, cell_path+'0927.{}.CELL_exp_AB.pt'.format(cell_name))
    torch.save(CELL_Cell, cell_path+'0927.{}.CELL_Cell.pt'.format(cell_name))
    torch.save(CELL_tgt_A, cell_path+'0927.{}.CELL_tgt_A.pt'.format(cell_name))
    torch.save(CELL_tgt_B, cell_path+'0927.{}.CELL_tgt_B.pt'.format(cell_name))
    torch.save(CELL_syn, cell_path+'0927.{}.CELL_syn.pt'.format(cell_name))



def get_cell_result(cell_name):
    print(cell_name)
    cell_path = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
    CELL_chem_A_feat = torch.load(cell_path+'0927.{}.CELL_chem_A_feat.pt'.format(cell_name))
    CELL_chem_B_feat = torch.load(cell_path+'0927.{}.CELL_chem_B_feat.pt'.format(cell_name))
    CELL_chem_A_adj = torch.load(cell_path+'0927.{}.CELL_chem_A_adj.pt'.format(cell_name))
    CELL_chem_B_adj = torch.load(cell_path+'0927.{}.CELL_chem_B_adj.pt'.format(cell_name))
    CELL_exp_A = torch.load(cell_path+'0927.{}.CELL_exp_A.pt'.format(cell_name))
    CELL_exp_B = torch.load(cell_path+'0927.{}.CELL_exp_B.pt'.format(cell_name))
    CELL_exp_AB = torch.load(cell_path+'0927.{}.CELL_exp_AB.pt'.format(cell_name))
    CELL_Cell = torch.load(cell_path+'0927.{}.CELL_Cell.pt'.format(cell_name))
    CELL_tgt_A = torch.load(cell_path+'0927.{}.CELL_tgt_A.pt'.format(cell_name))
    CELL_tgt_B = torch.load(cell_path+'0927.{}.CELL_tgt_B.pt'.format(cell_name))
    CELL_syn = torch.load(cell_path+'0927.{}.CELL_syn.pt'.format(cell_name))
    cell_dict = prepare_cell_data_GCN(CELL_chem_A_feat, CELL_chem_B_feat, CELL_chem_A_adj, CELL_chem_B_adj, CELL_exp_A, CELL_exp_B, CELL_tgt_A, CELL_tgt_B, CELL_syn, CELL_Cell, norm )
    #
    T_cell = DATASET_GCN_W_FT(
        torch.Tensor(cell_dict['drug1_feat']), torch.Tensor(cell_dict['drug2_feat']), 
        torch.Tensor(cell_dict['drug1_adj']), torch.Tensor(cell_dict['drug2_adj']),
        cell_dict['EXP_A'], cell_dict['EXP_B'], 
        cell_dict['TGT_A'], cell_dict['TGT_B'], 
        JY_ADJ_IDX, JY_IDX_WEIGHT_T, 
        torch.Tensor(cell_dict['y']),
        torch.Tensor(cell_dict['cell']))
    #
    #
    RAY_cell = ray.put(T_cell)
    T_cell = ray.get(RAY_cell)
    CELL_loader = torch.utils.data.DataLoader(T_cell, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
    #
    CELL_MSE, CELL_PR, CELL_SR = cell_corr(CELL_loader)
    tmp_df = pd.DataFrame({"cell_name" : [cell_name], 'MSE': [CELL_MSE], "PR": [CELL_PR], "SR": [CELL_SR]})
    return tmp_df




def cell_corr(CELL_loader) : 
    cell_loss = 0.0
    PRED_list = []
    Y_list = []
    with torch.no_grad():
        best_model.eval()
        for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) in enumerate(CELL_loader):
            expA = expA.view(-1,2)
            expB = expB.view(-1,2)
            adj_w = adj_w.squeeze()
            output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell) 
            MSE = torch.nn.MSELoss()
            loss = MSE(output, y)
            cell_loss = cell_loss + loss.item()
            Y_list = Y_list + y.view(-1).tolist()
            outputs = output.view(-1).tolist()
            PRED_list = PRED_list+outputs
    cell_loss = cell_loss/(batch_idx_t+1)
    if len(Y_list) >1 :
        pr,pp = stats.pearsonr(PRED_list, Y_list)
        sr,sp = stats.spearmanr(PRED_list, Y_list)
    else :
        pr = 0
        sr = 0
    return cell_loss, pr, sr














CELL_RESULT = pd.DataFrame(columns=["cell_name", 'MSE', "PR", "SR"])
cello_list = ['CVCL_0332', 'CVCL_0035', 'CVCL_0139', 'CVCL_0023', 'CVCL_0320', 'CVCL_0178', 'CVCL_0062', 'CVCL_0291', 'CVCL_0031', 'CVCL_0004', 'CVCL_0132', 'CVCL_0033', 'CVCL_2235', 'CVCL_0336', 'CVCL_0527', 'CVCL_A442', 'CVCL_0395']

for cello in cello_list :
    tmp_df = get_cell_result(cello)
    CELL_RESULT = pd.concat([CELL_RESULT, tmp_df])


CELL_RESULT.to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/RESULT1.csv', sep ='\t')

C_df = pd.merge(CELL_RESULT, DC_CELL_DF3[['DC_cellname','DrugCombCello','tissue']], left_on ='cell_name', right_on='DrugCombCello', how='left' )



rgb_values = sns.color_palette("Set2", 8)
color_map = dict(zip(list(set(C_df.tissue)), rgb_values))



# MSE

path = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
plotname = 'cell_MSE'
C_names = list(C_df.cell_name)
C_df_MSE = C_df.sort_values('MSE')
fig = plt.figure(figsize=(10,8))
plt.bar(C_df_MSE['cell_name'], C_df_MSE['MSE'])
plt.xticks(range(C_df_MSE.shape[0]), list(C_df_MSE['DC_cellname']), rotation=90)
for i in range(C_df_MSE.shape[0]):
	plt.annotate(str(list(C_df_MSE['MSE'])[i]), xy=(i,list(C_df_MSE['MSE'])[i]), ha='center', va='bottom', fontsize=7)

plt.tight_layout()
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')



# PR


path = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
plotname = 'cell_PR'
C_names = list(C_df.cell_name)
C_df_PR = C_df.sort_values('PR')
fig = plt.figure(figsize=(10,5))
plt.bar(C_df_PR['cell_name'], C_df_PR['PR'], color=C_df_PR['tissue'].map(color_map) ) # 
plt.xticks(range(C_df_PR.shape[0]), list(C_df_PR['DC_cellname']), rotation=90)
for i in range(C_df_PR.shape[0]):
	plt.annotate(str(round(list(C_df_PR['PR'])[i],2)), xy=(i,list(C_df_PR['PR'])[i]), ha='center', va='bottom', fontsize=9 ) # 

plt.tight_layout()
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')



# SR

path = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
plotname = 'cell_SR'
C_names = list(C_df.cell_name)
C_df_SR = C_df.sort_values('SR')
fig = plt.figure(figsize=(10,5))
plt.bar(C_df_SR['cell_name'], C_df_SR['SR'], color=C_df_SR['tissue'].map(color_map))
plt.xticks(range(C_df_SR.shape[0]), list(C_df_SR['DC_cellname']), rotation=90)
for i in range(C_df_SR.shape[0]):
	plt.annotate(str(round(list(C_df_SR['SR'])[i],2)), xy=(i,list(C_df_SR['SR'])[i]), ha='center', va='bottom', fontsize=9 ) # 

plt.tight_layout()
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')




#분포 확인 
plot_dir = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
import seaborn 

sns.set_context('talk', font_scale=1.1)
plt.figure(figsize=(10,7))
##### ax = sns.violinplot(y = 'PR', x= 'tissue', data =C_df, color = '#cde7ff', inner= 'quartile')
ax = sns.boxplot(y = 'PR', x= 'tissue', data =C_df, hue = 'tissue', palette = color_map, inner= 'quartile')
### ax = sns.violinplot(y = 'PR', x= 'tissue', data =C_df, hue = 'tissue', palette = color_map, inner= 'quartile')
#### sns.stripplot(y = 'MEAN', x= 'cat', data =means_df, color = 'black', edgecolor='gray')
#### ax = sns.stripplot(y = 'PR', x = 'tissue', data = C_df, edgecolor= 'gray', color = '#e83298') # palette ={'IN':'#444444', 'OUT':'#e83298' }hue = 'dot_col', 

ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=10)
plt.legend([],[],frameon=False)
plt.savefig(plot_dir+"B_PR_tissue.png", format = 'png', dpi=300, bbox_inches = 'tight')



# 빈도 다시 
path = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/2022.09.27.cellcheck/'
plotname = 'cell_freq'
C_list = list(A_B_C_S_SET_COH.DC_cellname)
C_names = list(set(C_list))
C_freq = [C_list.count(a) for a in C_names]
C_freq_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq})
C_freq_df = C_freq_df.sort_values('freq')
C_freq_df = pd.merge(C_freq_df, DC_CELL_DF3[['DC_cellname','tissue']], left_on ='cell', right_on='DC_cellname', how='left' )
fig = plt.figure(figsize=(10,5))
plt.bar(C_freq_df['cell'], C_freq_df['freq'], color=C_freq_df['tissue'].map(color_map))
plt.xticks(range(C_freq_df.shape[0]),list(C_freq_df['cell']), rotation=90)
for i in range(C_freq_df.shape[0]):
	plt.annotate(str(list(C_freq_df['freq'])[i]), xy=(i,list(C_freq_df['freq'])[i]), ha='center', va='bottom', fontsize=9 )

plt.tight_layout()
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')

