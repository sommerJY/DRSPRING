cell line 별 확인 

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




# 빈도 확인 
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3_MIS2_W0/'
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M1_MIS2_W0/'
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3_MIS0_W0/'

plotname = 'cell_freq'

C_names = list(set(ABCS_used_COH.DC_cellname))

C_train_freq = [list(ABCS_train_COH.DC_cellname).count(a) for a in C_names]
C_val_freq = [list(ABCS_val_COH.DC_cellname).count(a) for a in C_names]
C_test_freq = [list(ABCS_test_COH.DC_cellname).count(a) for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'train_freq' : C_train_freq, 'val_freq' :C_val_freq, 'test_freq' :C_test_freq })
C_df['tot_freq'] = C_df['train_freq'] + C_df['val_freq'] + C_df['test_freq']
C_df = C_df.sort_values('tot_freq')

fig, ax = plt.subplots(figsize=(30, 15))
## fig, ax = plt.subplots(figsize=(40, 15))

x_pos = [a*3 for a in range(C_df.shape[0])]
ax.bar(x_pos, list(C_df['train_freq']), label='train')
ax.bar(x_pos, list(C_df['val_freq']), bottom=list(C_df['train_freq']), label='Val')
ax.bar(x_pos, list(C_df['test_freq']), bottom=list(C_df['train_freq']+C_df['val_freq']), label='test')

plt.xticks(x_pos, list(C_df['cell']), rotation=90, fontsize=18)

for i in range(C_df.shape[0]):
	plt.annotate(str(int(list(C_df['tot_freq'])[i])), xy=(x_pos[i], list(C_df['tot_freq'])[i]), ha='center', va='bottom', fontsize=18)

ax.set_ylabel('cell nums')
ax.set_title('used cells')
plt.tight_layout()

plotname = 'total_cells'
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')
plt.close()




























my_config = my_config
model_path = TOPVAL_PATH
model_name = "M2_checkpoint"
PRJ_PATH = PRJ_PATH
PRJ_NAME = MJ_NAME
MISS_NAME = MISS_NAME+'_'+WORK_NAME
number = 'M2'


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



ABCS_test_result = ABCS_test[['DrugCombCello','type' ]]
ABCS_test_result['ANS'] = Y_list
ABCS_test_result['PRED'] = PRED_list

DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['DrugCombCCLE'])]
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'



ABCS_test_re = pd.merge(ABCS_test_result, DC_CELL_info_filt[['DrugCombCello','DC_cellname','cell_onehot', 'tissue']], on = 'DrugCombCello', how = 'left'  )



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























#######################################
#########################################
# M1 cell line 별 확인 

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




# 빈도 확인 
path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M1_MIS2_W0/'

plotname = 'cell_freq'

C_list = list(A_B_C_S_SET_COH.DC_cellname)
C_names = list(set(ABCS_used_COH.DC_cellname))

C_train_freq = [list(ABCS_train_COH.DC_cellname).count(a) for a in C_names]
C_val_freq = [list(ABCS_val_COH.DC_cellname).count(a) for a in C_names]
C_test_freq = [list(ABCS_test_COH.DC_cellname).count(a) for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'train_freq' : C_train_freq, 'val_freq' :C_val_freq, 'test_freq' :C_test_freq })
C_df['tot_freq'] = C_df['train_freq'] + C_df['val_freq'] + C_df['test_freq']
C_df = C_df.sort_values('tot_freq')

fig, ax = plt.subplots(figsize=(40, 15))
x_pos = [a*2 for a in range(C_df.shape[0])]
ax.bar(x_pos, list(C_df['train_freq']), label='train')
ax.bar(x_pos, list(C_df['val_freq']), bottom=list(C_df['train_freq']), label='Val')
ax.bar(x_pos, list(C_df['test_freq']), bottom=list(C_df['train_freq']+C_df['val_freq']), label='test')

plt.xticks(x_pos, list(C_df['cell']), rotation=90, fontsize=18)

for i in range(C_df.shape[0]):
	plt.annotate(str(int(list(C_df['tot_freq'])[i])), xy=(x_pos[i], list(C_df['tot_freq'])[i]), ha='center', va='bottom', fontsize=18)

ax.set_ylabel('cell nums')
ax.set_title('used cells')
plt.tight_layout()

plotname = 'total_cells'
fig.savefig('{}/{}.png'.format(path, plotname), bbox_inches = 'tight')
plt.close()





my_config = my_config
model_path = TOPVAL_PATH
model_name = "M2_checkpoint"
PRJ_PATH = PRJ_PATH
PRJ_NAME = MJ_NAME
MISS_NAME = MISS_NAME+'_'+WORK_NAME
number = 'M2'


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



ABCS_test_result = ABCS_test[['DrugCombCello','type' ]]
ABCS_test_result['ANS'] = Y_list
ABCS_test_result['PRED'] = PRED_list

DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['DrugCombCCLE'])]
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'



ABCS_test_re = pd.merge(ABCS_test_result, DC_CELL_info_filt[['DrugCombCello','DC_cellname','cell_onehot', 'tissue']], on = 'DrugCombCello', how = 'left'  )



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

