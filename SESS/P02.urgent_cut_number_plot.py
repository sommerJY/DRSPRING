
# 빈도 체크 

ABCS_train['used'] = 'train'
ABCS_val['used'] = 'val'
ABCS_test['used'] = 'test'

ABCS_used = pd.concat([ABCS_train, ABCS_val, ABCS_test])
ABCS_used = ABCS_used[['DrugCombCello','used']]

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCello.isin(A_B_C_S_SET.DrugCombCello)]
DC_CELL_info_filt['cell_onehot'] = [a for a in range(len(set(A_B_C_S_SET.DrugCombCello)))]
DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['DrugCombCCLE'])]
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'

DC_CELL_info_filt = DC_CELL_info_filt.reset_index(drop = True)


ABCS_train_COH =pd.merge(ABCS_train, DC_CELL_info_filt[['DrugCombCello','DC_cellname','tissue']], on = 'DrugCombCello', how = 'left'  )
ABCS_val_COH =pd.merge(ABCS_val, DC_CELL_info_filt[['DrugCombCello','DC_cellname','tissue']], on = 'DrugCombCello', how = 'left'  )
ABCS_test_COH =pd.merge(ABCS_test, DC_CELL_info_filt[['DrugCombCello','DC_cellname','tissue']], on = 'DrugCombCello', how = 'left'  )

ABCS_used_COH = pd.merge(ABCS_used, DC_CELL_info_filt[['DrugCombCello','DC_cellname','tissue']], on = 'DrugCombCello', how = 'left'  )

C_names = list(set(ABCS_used_COH.DC_cellname))

C_train_freq = [list(ABCS_train_COH.DC_cellname).count(a) for a in C_names]
C_val_freq = [list(ABCS_val_COH.DC_cellname).count(a) for a in C_names]
C_test_freq = [list(ABCS_test_COH.DC_cellname).count(a) for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'train_freq' : C_train_freq, 'val_freq' :C_val_freq, 'test_freq' :C_test_freq })
C_df['tot_freq'] = C_df['train_freq'] + C_df['val_freq'] + C_df['test_freq']
C_df = C_df.sort_values('tot_freq')

CELL_over_1000 = C_df[C_df.tot_freq>1000]
CELL_over_500 = C_df[C_df.tot_freq>500]
CELL_over_100 = C_df[C_df.tot_freq>100]



















PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_W0/'.format(MJ_NAME, MISS_NAME)

ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.csv'.format(MJ_NAME, MISS_NAME, WORK_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}.pickle'.format(MJ_NAME, MISS_NAME, WORK_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)

# M3 MIS2 W0 list 
trial_list = ["4cef1a8e","763ac58a","2df50b20","852ed9b4","370ff2d0","eaf0de8c","f258543a","55550dce","91f3b472","448ef8d4","2644c294","e290e766","adfb3c80","ae46ad40","ec9e5d20","ba3d661e","66db7a7a","c8b31198","ef1205d8","ee27fb32","c3ba2e4a","3ceec234","babd315c","b2db1ffe","167cb48e","74763da4","953a09ec","9e33ae9e","3c4a5d96","22023ddc","453110f4","e16fb7de","a8ac9014","a80a7866","d40964a6","ab79c5f8","5143303a","cab12408","b1060c54","5eefb3a0","6dc113de","94be52a0","26a6fbce","de9b4924","9c3d0d84","95fecb6c","792c79e4","726cc86c","e07d1aaa","711f007c","5faa2210","da67ed3c","2e163b38","becb936a","47f20492","1a1f3488","f11f43da","4028a41e","9de9f854","4d984f8e","96b97d54","bef7c6e6","f9bfc7ec","a2c9dcfe","30d2333e","5ba1d0f2","340967fc","daff9558","07b2ad90","e4c35ab8","b93f4e56","025128b2","cc0fd08c","cec36b1e","bef320dc","21e0ce14","ce3da6d2","293e3478","9af94750","c84e6b1e","8a52a9ee","873194b8","11ccb92a","bf8483ec","bedef0d0","bf1ec96c","beb4fc1c","b2920c54","be8871e2","bfd650be","bfabc52e","bf595640","bf451e46","bec86130","bfeabbc6","bfc0c096","bf979004","bf6f4e6e","bf317440","bf0b0b52","4d591058","4d837b72","4dcf12d0","4e9560b6","4e69f2be","4d28306e","4df9f496","4e1130ac","4e7f4f4c","4dba1ed4","4de52ed0","4d6dcc82","4e2b3fc4","4e3f34ac","4e53c5c0","4ead1512","4d9abb0c"]

ANA_DF = ANA_DF[ANA_DF.trial_id.isin(trial_list)]
all_keys = list(ANA_ALL_DF.keys())
all_keys_trials = [a.split('/')[5].split('_')[3] for a in all_keys]
all_keys_trials_ind = [all_keys_trials.index(a) for a in trial_list if a in list(ANA_DF.trial_id)]
new_keys = [all_keys[a] for a in all_keys_trials_ind]

TOPVAL_PATH = PRJ_PATH



# M3_MIS2_W0 모델 
import numpy as np
TOT_max = -np.Inf
TOT_key = ""
for key in new_keys:
	trial_max = max(ANA_ALL_DF[key]['SCOR'])
	if trial_max > TOT_max :
		TOT_max = trial_max
		TOT_key = key

print('best cor', flush=True)
print(TOT_key, flush=True)
mini_df = ANA_ALL_DF[TOT_key]
#TOPVAL_PATH = TOT_key
my_config = ANA_DF[ANA_DF.logdir==TOT_key]
cck_num =mini_df[mini_df.SCOR==max(mini_df.SCOR)].index.item()
checkpoint = "/checkpoint_"+str(cck_num).zfill(6)
#TOPVAL_PATH = TOT_key + checkpoint
print('best cor check', flush=True)
print(TOPVAL_PATH, flush=True)
R_6_V = max(mini_df.SCOR)
R_6_V
model_name = 'M4_checkpoint'





my_config = my_config
model_path = TOPVAL_PATH
model_name = "M4_checkpoint"
PRJ_PATH = PRJ_PATH
PRJ_NAME = MJ_NAME
MISS_NAME = MISS_NAME+'_'+WORK_NAME
number = 'M4'



use_cuda = False #  #   #  #  #  #
T_test = ray.get(RAY_test)
Test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config['config/batch_size'].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=my_config['config/n_workers'].item())
#
G_chem_layer = my_config['config/G_chem_layer'].item()
G_chem_hdim = my_config['config/G_chem_hdim'].item()
G_exp_layer = my_config['config/G_exp_layer'].item()
G_exp_hdim = my_config['config/G_exp_hdim'].item()
dsn1_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()]
dsn2_layers = [my_config['config/feat_size_0'].item(), my_config['config/feat_size_1'].item(), my_config['config/feat_size_2'].item()] 
snp_layers = [my_config['config/feat_size_3'].item() , my_config['config/feat_size_4'].item()]
inDrop = my_config['config/dropout_1'].item()
Drop = my_config['config/dropout_2'].item()
#       
best_model = MY_expGCN_parallel_model(
            G_chem_layer, T_test.gcn_drug1_F.shape[-1] , G_chem_hdim,
            G_exp_layer, 3, G_exp_hdim,
            dsn1_layers, dsn2_layers, snp_layers, 
            len(set(A_B_C_S_SET_SM.DrugCombCello)), 1,
            inDrop, Drop
            )
#
if torch.cuda.is_available():
    best_model = best_model.cuda()
    print('model to cuda', flush = True)
    if torch.cuda.device_count() > 1 :
        best_model = torch.nn.DataParallel(best_model)
        print('model to multi cuda', flush = True)
#
device = "cuda:0" if torch.cuda.is_available() else "cpu"
#
if torch.cuda.is_available():
    state_dict = torch.load(os.path.join(model_path, model_name))
else:
    state_dict = torch.load(os.path.join(model_path, model_name), map_location=torch.device('cpu'))
#
print("state_dict_done", flush = True)
if type(state_dict) == tuple:
    best_model.load_state_dict(state_dict[0])
else : 
    best_model.load_state_dict(state_dict)	
#
print("state_load_done", flush = True)
#
#
best_model.eval()
test_loss = 0.0
PRED_list = []
Y_list = T_test.syn_ans.squeeze().tolist()
with torch.no_grad():
    best_model.eval()
    for batch_idx_t, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(Test_loader):
        expA = expA.view(-1,3)
        expB = expB.view(-1,3)
        adj_w = adj_w.squeeze()
        if use_cuda:
            drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
        output= best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) 
        MSE = torch.nn.MSELoss()
        loss = MSE(output, y)
        test_loss = test_loss + loss.item()
        outputs = output.squeeze().tolist()
        PRED_list = PRED_list+outputs
#
TEST_LOSS = test_loss/(batch_idx_t+1)
R__T = TEST_LOSS
R__1 , R__2 = jy_corrplot(PRED_list, Y_list, PRJ_PATH,'P.{}.{}.{}_model'.format(PRJ_NAME, MISS_NAME, number) )






def jy_corrplot(PRED_list, Y_list, path, plotname ):
	jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg')
	pr,pp = stats.pearsonr(PRED_list, Y_list)
	print("Pearson correlation is {} and related p_value is {}".format(pr, pp), flush=True)
	sr,sp = stats.spearmanr(PRED_list, Y_list)
	print("Spearman correlation is {} and related p_value is {}".format(sr, sp), flush=True)
	jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
	jplot.ax_joint.scatter(PRED_list, Y_list)
	jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
	jplot.figure.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')
	return pr, sr









ABCS_test_result = ABCS_test[['DrugCombCello','type' ]]
ABCS_test_result['ANS'] = Y_list
ABCS_test_result['PRED'] = PRED_list

DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['DrugCombCCLE'])]
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'



ABCS_test_re = pd.merge(ABCS_test_result, DC_CELL_info_filt[['DrugCombCello','DC_cellname','cell_onehot', 'tissue']], on = 'DrugCombCello', how = 'left'  )




# cell line filter correlation 

CELL_over_1000
CELL_over_500
CELL_over_100

ABCS_test_re_1000 = ABCS_test_re[ABCS_test_re.DC_cellname.isin(CELL_over_1000.cell)] # 4723
ABCS_test_re_500 = ABCS_test_re[ABCS_test_re.DC_cellname.isin(CELL_over_500.cell)] # 4860
ABCS_test_re_100 = ABCS_test_re[ABCS_test_re.DC_cellname.isin(CELL_over_100.cell)] # 5151



def jy_corrplot(PRED_list, Y_list, path, plotname ):
	jplot = sns.jointplot(x=PRED_list, y=Y_list, ci=68, kind='reg')
	pr,pp = stats.pearsonr(PRED_list, Y_list)
	print("Pearson correlation is {} and related p_value is {}".format(pr, pp), flush=True)
	sr,sp = stats.spearmanr(PRED_list, Y_list)
	print("Spearman correlation is {} and related p_value is {}".format(sr, sp), flush=True)
	jplot.ax_joint.annotate(f'$pearson = {pr:.3f}, spearman = {sr:.3f}$',xy=(min(PRED_list)+ 0.01, max(Y_list)- 0.01 ), ha='left', va='center',)
	jplot.ax_joint.scatter(PRED_list, Y_list)
	jplot.set_axis_labels(xlabel='Predicted', ylabel='Answer', size=15)
	jplot.figure.savefig('{}/{}.corrplot.png'.format(path, plotname), bbox_inches = 'tight')
	return pr, sr

plot_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3_MIS2_W0/'

jy_corrplot(list(ABCS_test_re_1000.PRED), list(ABCS_test_re_1000.ANS),plot_path,'M3MI2W0_1000' )
jy_corrplot(list(ABCS_test_re_500.PRED), list(ABCS_test_re_500.ANS),plot_path,'M3MI2W0_500' )
jy_corrplot(list(ABCS_test_re_100.PRED), list(ABCS_test_re_100.ANS),plot_path,'M3MI2W0_100' )













test_cell_df = pd.DataFrame({'DC_cellname' : list(set(ABCS_test_re.DC_cellname))})

cell_P = []
cell_S = []
cell_num = []

for cell in list(test_cell_df.DC_cellname) :
    tmp_test_re = ABCS_test_re[ABCS_test_re.DC_cellname == cell]
    cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED)
    cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED)
    cell_nums = tmp_test_re.shape[0]
    cell_P.append(cell_P_corr)
    cell_S.append(cell_S_corr)
    cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num

test_cell_df = pd.merge(test_cell_df, DC_CELL_info_filt[['DC_cellname','tissue']], on = 'DC_cellname', how = 'left'  )
tissue_set = list(set(test_cell_df['tissue']))
color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"]
test_cell_df['tissue_oh'] = [color_set[tissue_set.index(a)] for a in list(test_cell_df['tissue'])]


# Spearman corr
test_cell_df = test_cell_df.sort_values('S_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['S_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df.shape[0]):
	plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['S_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_spearman'), bbox_inches = 'tight')
plt.close()


# Pearson corr
test_cell_df = test_cell_df.sort_values('P_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df.shape[0])]

plt.bar(x_pos, test_cell_df['P_COR'], color=test_cell_df['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df.shape[0]):
	plt.annotate(str(list(test_cell_df['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_pearson'), bbox_inches = 'tight')
plt.close()







# 1000 ver 

test_cell_df_1000 = test_cell_df[test_cell_df.DC_cellname.isin(CELL_over_1000.cell)]


# Spearman corr
test_cell_df_1000 = test_cell_df_1000.sort_values('S_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df_1000.shape[0])]

plt.bar(x_pos, test_cell_df_1000['S_COR'], color=test_cell_df_1000['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df_1000['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df_1000.shape[0]):
	plt.annotate(str(list(test_cell_df_1000['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df_1000['S_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_spearman_1000'), bbox_inches = 'tight')
plt.close()


# Pearson corr
test_cell_df_1000 = test_cell_df_1000.sort_values('P_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df_1000.shape[0])]

plt.bar(x_pos, test_cell_df_1000['P_COR'], color=test_cell_df_1000['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df_1000['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df_1000.shape[0]):
	plt.annotate(str(list(test_cell_df_1000['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df_1000['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_pearson_1000'), bbox_inches = 'tight')
plt.close()








# 500 ver 

test_cell_df_500 = test_cell_df[test_cell_df.DC_cellname.isin(CELL_over_500.cell)]


# 색 맘에 안듬 다시 

color_map = pd.DataFrame({
    'tissue':['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN'],
    'my_col':['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff']
    })
Telemagenta / Pastel Orange / Azure Blue / Traffic Green / light green / strawberry red / 옅은 파랑 /노랑 /보라

test_cell_df_500 = pd.merge(test_cell_df_500, color_map, on = 'tissue')


# Pearson corr
test_cell_df_500 = test_cell_df_500.sort_values('P_COR')

fig = plt.figure(figsize=(20,8))
x_pos = [a*2 for a in range(test_cell_df_500.shape[0])]

plt.bar(x_pos, test_cell_df_500['P_COR'], color=test_cell_df_500['my_col']) # 
plt.xticks(x_pos, list(test_cell_df_500['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 0.9, 0.1), fontsize= 18)
for i in range(test_cell_df_500.shape[0]):
	plt.annotate(str(list(test_cell_df_500['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df_500['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_pearson_500'), bbox_inches = 'tight')
plt.close()



# Spearman corr
test_cell_df_500 = test_cell_df_500.sort_values('S_COR')

fig = plt.figure(figsize=(20,8))
x_pos = [a*2 for a in range(test_cell_df_500.shape[0])]

plt.bar(x_pos, test_cell_df_500['S_COR'], color=test_cell_df_500['my_col']) # 
plt.xticks(x_pos,list(test_cell_df_500['DC_cellname']), rotation=90, fontsize= 18)
plt.yticks(np.arange(0, 0.9, 0.1), fontsize= 18)

for i in range(test_cell_df_500.shape[0]):
	plt.annotate(str(list(test_cell_df_500['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df_500['S_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_spearman_500'), bbox_inches = 'tight')
plt.close()




color_dict = {color_map['tissue'][a] : color_map['my_col'][a] for a in range(9)}

import seaborn
    
fig = plt.figure(figsize=(10,15))
seaborn.set(style = 'whitegrid') 
seaborn.violinplot(x = test_cell_df_500['tissue'], y =test_cell_df_500['P_COR'],  palette=color_dict)
ax = seaborn.stripplot(x = test_cell_df_500['tissue'], y =test_cell_df_500['P_COR'],
                   color="gray", edgecolor="white", s=15, linewidth=2.0)
ax.set_xticklabels(['BONE', 'BREAST', 'SKIN', 'BLOOD', 'LARGE_INTESTINE', 'PROSTATE', 'CENTRAL_NERVOUS', 'LUNG', 'OVARY' ], fontsize = 18, rotation=90)
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_tissue_violin'), bbox_inches = 'tight')
plt.close()

# range(len(test_cell_df_500['tissue'])), 




'BONE', 'BREAST', , 'SKIN''HAEMATOPOIETIC_AND_LYMPHOID_TISSUE',  'LARGE_INTESTINE' 'PROSTATE' 'CENTRAL_NERVOUS',  'LUNG', 'OVARY', 


# 100 ver 

test_cell_df_100 = test_cell_df[test_cell_df.DC_cellname.isin(CELL_over_100.cell)]


# Spearman corr
test_cell_df_100 = test_cell_df_100.sort_values('S_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df_100.shape[0])]

plt.bar(x_pos, test_cell_df_100['S_COR'], color=test_cell_df_100['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df_100['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df_100.shape[0]):
	plt.annotate(str(list(test_cell_df_100['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df_100['S_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_spearman_100'), bbox_inches = 'tight')
plt.close()


# Pearson corr
test_cell_df_100 = test_cell_df_100.sort_values('P_COR')

fig = plt.figure(figsize=(22,15))
x_pos = [a*2 for a in range(test_cell_df_100.shape[0])]

plt.bar(x_pos, test_cell_df_100['P_COR'], color=test_cell_df_100['tissue_oh']) # 
plt.xticks(x_pos,list(test_cell_df_100['DC_cellname']), rotation=90, fontsize= 18)
for i in range(test_cell_df_100.shape[0]):
	plt.annotate(str(list(test_cell_df_100['cell_num'])[i]), xy=(x_pos[i],list(test_cell_df_100['P_COR'])[i]), ha='center', va='bottom', fontsize= 18)

plt.legend()
plt.tight_layout()
fig.savefig('{}/{}.png'.format(PRJ_PATH, 'test_cell_pearson_100'), bbox_inches = 'tight')
plt.close()



