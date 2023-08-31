


NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'



NETWORK_PATH = '/home01/k040a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k040a01/01.Data/LINCS/'
DC_PATH = '/home01/k040a01/01.Data/DrugComb/'



# HS Drug pathway DB 활용 -> 349
print('NETWORK')
# HUMANNET 사용 

hunet_gsp = pd.read_csv(NETWORK_PATH+'HS-DB.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B','SC']

LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)
lm_entrezs = list(LINCS_978.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885
hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5]

len(set(list(hnet_L3['G_A']) + list(hnet_L3['G_B']))) # 611

ID_G = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')

# MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

#for nn in list(MSSNG):
#       ID_G.add_node(nn)

# edge 
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

# 원래는 edge score 있지만 일단은...
ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]




# 유전자 이름으로 붙이기 

new_node_names = []
for a in ID_G.nodes():
        tmp_name = LINCS_978[LINCS_978.gene_id == a ]['gene_symbol'].item() # 6118
        new_node_name = str(a) + '__' + tmp_name
        new_node_names = new_node_names + [new_node_name]

mapping = {list(ID_G.nodes())[a]:new_node_names[a] for a in range(len(new_node_names))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE





# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE



# LINCS exp order 따지기 
BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)




SAVE_PATH = '/home01/k040a01/02.M3V6/M3V6_349_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_349_FULL/'


file_name = 'M3V6_349_MISS2_ONEIL'
file_name = 'M3V8_349_MISS2_ONEIL'

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(file_name))
MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(file_name))
MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(file_name))
MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(file_name))
MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(file_name))
MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(file_name))
MY_Target_1_A = torch.load(SAVE_PATH+'{}.MY_Target_1_A.pt'.format(file_name))
MY_Target_1_B = torch.load(SAVE_PATH+'{}.MY_Target_1_B.pt'.format(file_name))
MY_CellBase = torch.load(SAVE_PATH+'{}.MY_CellBase.pt'.format(file_name))
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))



A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)

cid_a = list(A_B_C_S_SET_ADD2['CID_A'])
cid_b = list(A_B_C_S_SET_ADD2['CID_B'])
sm_a = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
sm_b = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
ccle = list(A_B_C_S_SET_ADD2['CELL'])



A_B_C_S_SET_ADD2['CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [sm_a[i] + '___' + sm_b[i]+ '___' + ccle[i] if sm_a[i] < sm_b[i] else sm_b[i] + '___' + sm_a[i]+ '___' + ccle[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]

A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)




MISS_filter = ['AOBO','AXBX','AXBO','AOBX'] # 

A_B_C_S_SET = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.ONEIL == 'O'] # 16422

# A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O'] # 11639

#A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] # 8086 -> 이걸 빼야하나 말아야하나 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]






# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/home01/k040a01/01.Data/CCLE/'
# CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'

ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']

ccle_cell_info_filt = ccle_cell_info[ccle_cell_info.DepMap_ID.isin(ccle_exp['Unnamed: 0'])]
ccle_names = [a for a in ccle_cell_info_filt.DrugCombCCLE if type(a) == str]


A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(ccle_names)]



data_ind = list(A_B_C_S_SET.index)

MY_chem_A_feat_RE = MY_chem_A_feat[data_ind]
MY_chem_B_feat_RE = MY_chem_B_feat[data_ind]
MY_chem_A_adj_RE = MY_chem_A_adj[data_ind]
MY_chem_B_adj_RE = MY_chem_B_adj[data_ind]
MY_g_EXP_A_RE = MY_g_EXP_A[data_ind]
MY_g_EXP_B_RE = MY_g_EXP_B[data_ind]
MY_Target_A = copy.deepcopy(MY_Target_1_A)[data_ind] ############## NEW TARGET !!!!!! #####
MY_Target_B = copy.deepcopy(MY_Target_1_B)[data_ind] ############## NEW TARGET !!!!!! #####
MY_CellBase_RE = MY_CellBase[data_ind]
MY_syn_RE = MY_syn[data_ind]


A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)





# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
        DC_CELL_DF2,
        pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left'  )



# drug binary 

DC_ALL_PATH = '/home01/k040a01/01.Data/DrugComb/'
# DC_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

all_chem_DF = pd.read_csv(DC_ALL_PATH+'DC_ALL_7555_ORDER.csv')
all_smiles = list(all_chem_DF.CAN_SMILES)

iMols = [Chem.MolFromSmiles(a.strip()) for a in all_smiles]

fps = [AllChem.GetMorganFingerprintAsBitVect(mol, radius = 3) for mol in iMols]

arr = np.empty((0,2048), int).astype(int)

for indd in range(len(fps)) :
    fpfp = fps[indd]
    array = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fpfp, array)
    arr = np.vstack((arr, array))



drug_A_arr = []

for AA in list(A_B_C_S_SET_COH.CID_A) :
    new_ind = all_chem_DF [ all_chem_DF.CID==AA ].index.item()
    arr_res = arr[new_ind]
    drug_A_arr.append(arr_res)

drug_A_arrS = torch.Tensor(np.vstack(drug_A_arr))


drug_B_arr = []

for BB in list(A_B_C_S_SET_COH.CID_B) :
    new_ind = all_chem_DF [ all_chem_DF.CID==BB ].index.item()
    arr_res = arr[new_ind]
    drug_B_arr.append(arr_res)

drug_B_arrS = torch.Tensor(np.vstack(drug_B_arr))




# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_names.sort()

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['CELL'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')


C_freq_filter = C_df



A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.CELL)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)





##### no target filter version 
#no_TF_CID = [104842, 208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 24748204, 3062316, 216239, 3385, 5288382, 5311, 59691338, 60750, 5329102, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
#no_TF_CELL = ['LOVO', 'A375', 'HT29', 'OVCAR3', 'SW-620', 'SK-OV-3', 'MDAMB436', 'NCIH23', 'RKO', 'UACC62', 'A2780', 'VCAP', 'A427', 'T-47D', 'ES2', 'PA1', 'RPMI7951', 'SKMES1', 'NCIH2122', 'HT144', 'NCIH1650', 'SW837', 'OV90', 'UWB1289', 'HCT116', 'A2058', 'NCIH520']

no_TF_CID = [104842, 208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 3062316, 24748204, 216239, 3385, 5288382, 5311, 59691338, 60750, 5329102, 11960529, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
no_TF_CELL = ['T-47D', 'RKO', 'ES2', 'RPMI7951', 'NCIH520', 'MSTO', 'NCIH2122', 'MDAMB436', 'OV90', 'KPL1', 'HT144', 'A375', 'PA1', 'CAOV3', 'OVCAR3', 'LOVO', 'NCIH1650', 'A427', 'VCAP', 'NCI-H460', 'SK-OV-3', 'DLD1', 'A2058', 'SW837', 'SKMES1', 'UWB1289', 'HCT116', 'A2780', 'ZR751', 'UACC62', 'SW-620', 'NCIH23', 'SKMEL30', 'HT29']



ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(no_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(no_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(no_TF_CELL)]



A_B_C_S_SET_COH = copy.deepcopy(ON_filt_3)



data_ind = list(A_B_C_S_SET_COH.index)

MY_chem_A_feat_RE2 = MY_chem_A_feat_RE[data_ind]
MY_chem_B_feat_RE2 = MY_chem_B_feat_RE[data_ind]
MY_chem_A_adj_RE2 = MY_chem_A_adj_RE[data_ind]
MY_chem_B_adj_RE2 = MY_chem_B_adj_RE[data_ind]
MY_g_EXP_A_RE2 = MY_g_EXP_A_RE[data_ind]
MY_g_EXP_B_RE2 = MY_g_EXP_B_RE[data_ind]
MY_Target_A2 = copy.deepcopy(MY_Target_A)[data_ind]
MY_Target_B2 = copy.deepcopy(MY_Target_B)[data_ind]
MY_CellBase_RE2 = MY_CellBase_RE[data_ind]
MY_syn_RE2 = MY_syn_RE[data_ind]

# merge 전 후로 index 달라지므로 뒤에 넣어줬음 
A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())



print('CIDs', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])
print(len(set(tmp2)) , flush = True)


print('CID_CID', flush = True)
print(len(set(A_B_C_S_SET_COH2.CID_CID)), flush = True)



print('CID_CID_CCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.cid_cid_cell)), flush = True)

print('DrugCombCCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.CELL)), flush = True)





# 일단 생 5CV


print("LEARNING")

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_no_dup_sm_sm = [setset.split('___')[0]+'___'+setset.split('___')[1] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({
        'setset' : data_no_dup.tolist(),
        'cell' : data_no_dup_cells,
        'SM_SM' : data_no_dup_sm_sm
         })

SM_SM_list = list(set(data_nodup_df.SM_SM))
SM_SM_list.sort()
sm_sm_list_1 = sklearn.utils.shuffle(SM_SM_list, random_state=42)

bins = [a for a in range(0, len(sm_sm_list_1), round(len(sm_sm_list_1)*0.2) )]
bins = bins[1:]
res = np.split(sm_sm_list_1, bins)

CV_1_smsm = list(res[0])
CV_2_smsm = list(res[1])
CV_3_smsm = list(res[2])
CV_4_smsm = list(res[3])
CV_5_smsm = list(res[4])
if len(res) > 5 :
        CV_5_smsm = list(res[4]) + list(res[5])

len(sm_sm_list_1)
len(CV_1_smsm) + len(CV_2_smsm) + len(CV_3_smsm) + len(CV_4_smsm) + len(CV_5_smsm)

CV_1_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_1_smsm)]['setset'])
CV_2_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_2_smsm)]['setset'])
CV_3_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_3_smsm)]['setset'])
CV_4_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_4_smsm)]['setset'])
CV_5_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_5_smsm)]['setset'])

CV_ND_INDS = {
        'CV0_train' : CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset,
        'CV0_test' : CV_5_setset,
        'CV1_train' : CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset,
        'CV1_test' : CV_1_setset,
        'CV2_train' : CV_3_setset + CV_4_setset + CV_5_setset + CV_1_setset,
        'CV2_test' : CV_2_setset,
        'CV3_train' : CV_4_setset + CV_5_setset + CV_1_setset + CV_2_setset,
        'CV3_test' : CV_3_setset,
        'CV4_train' : CV_5_setset + CV_1_setset + CV_2_setset + CV_3_setset,
        'CV4_test' : CV_4_setset
}

print(data_nodup_df.shape)
len( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset)
len(set( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset ))


MY_g_EXP_A_RE2 = MY_g_EXP_A_RE2.view(MY_g_EXP_A_RE2.shape[0],-1)
MY_g_EXP_B_RE2 = MY_g_EXP_B_RE2.view(MY_g_EXP_B_RE2.shape[0],-1)
















def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
MY_syn_RE2, norm ) :
        # 
        # CV_num = 0
        train_key = 'CV{}_train'.format(CV_num)
        test_key = 'CV{}_test'.format(CV_num)
        # 
        #
        ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[train_key])]
        ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]
        #
        tv_ind = list(ABCS_tv.index)
        random.shuffle(tv_ind)
        test_ind = list(ABCS_test.index)
        # 
        chem_feat_A_tv = drug_A_arrS[tv_ind]; chem_feat_A_test = drug_A_arrS[test_ind]
        chem_feat_B_tv = drug_B_arrS[tv_ind]; chem_feat_B_test = drug_B_arrS[test_ind]
        gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
        gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
        target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
        target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
        cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
        syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
        #
        tv_data = {}
        test_data = {}
        #
        tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
        test_data['drug1_feat'] = chem_feat_A_test
        #
        tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
        test_data['drug2_feat'] = chem_feat_B_test
        #
        tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
        test_data['GENE_A'] = gene_A_test
        #
        tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
        test_data['GENE_B'] = gene_B_test
        #
        tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
        test_data['TARGET_A'] = target_A_test
        #
        tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
        test_data['TARGET_B'] = target_B_test
        #
        tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
        test_data['cell_BASAL'] = cell_basal_test
        #
        tv_data['Merged_features'] = torch.concat(([tv_data['drug1_feat'], tv_data['drug2_feat'], tv_data['GENE_A'], tv_data['GENE_B'], tv_data['TARGET_A'], tv_data['TARGET_B'], tv_data['cell_BASAL'] ]), axis=1)
        test_data['Merged_features'] = torch.concat(([test_data['drug1_feat'], test_data['drug2_feat'], test_data['GENE_A'], test_data['GENE_B'], test_data['TARGET_A'], test_data['TARGET_B'], test_data['cell_BASAL'] ]), axis=1)
        #            
        tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
        test_data['y'] = syn_test
        #
        print(tv_data['drug1_feat'].shape, flush=True)
        print(test_data['drug1_feat'].shape, flush=True)
        return tv_data, test_data


def result_pearson(y, pred):
        pear = stats.pearsonr(y, pred)
        pear_value = pear[0]
        pear_p_val = pear[1]
        print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val), flush=True)


def result_spearman(y, pred):
        spear = stats.spearmanr(y, pred)
        spear_value = spear[0]
        spear_p_val = spear[1]
        print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val), flush=True)



def plot_loss(train_loss, valid_loss, path, plotname):
        fig = plt.figure(figsize=(10,8))
        plt.plot(range(1,len(train_loss)+1),train_loss, label='Training Loss')
        plt.plot(range(1,len(valid_loss)+1),valid_loss,label='Validation Loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.ylim(0, math.ceil(max(train_loss+valid_loss))) # 일정한 scale
        plt.xlim(0, len(train_loss)+1) # 일정한 scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        fig.savefig('{}/{}.loss_plot.png'.format(path, plotname), bbox_inches = 'tight')




seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)



# gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, gcn_exp_A, gcn_exp_B, gcn_tgt_A, gcn_tgt_B, gcn_adj, gcn_adj_weight, syn_ans, cell_info
norm = 'tanh_norm'


# leave combination 

# CV_0
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

# CV_1
train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

# CV_2
train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

# CV_3
train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

# CV_4
train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )



# leave cells out 
def prepare_data_GCN_cellout(cell_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
MY_syn_RE2, norm ) :
        # 
        cell_name = no_TF_CELL[cell_num]
        #train_key = 'CV{}_train'.format(CV_num)
        #test_key = 'CV{}_test'.format(CV_num)
        # 
        #
        ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!=cell_name]
        ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname==cell_name]
        #
        tv_ind = list(ABCS_tv.index)
        random.shuffle(tv_ind)
        test_ind = list(ABCS_test.index)
        # 
        chem_feat_A_tv = drug_A_arrS[tv_ind]; chem_feat_A_test = drug_A_arrS[test_ind]
        chem_feat_B_tv = drug_B_arrS[tv_ind]; chem_feat_B_test = drug_B_arrS[test_ind]
        gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
        gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
        target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
        target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
        cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
        syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
        #
        tv_data = {}
        test_data = {}
        #
        tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
        test_data['drug1_feat'] = chem_feat_A_test
        #
        tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
        test_data['drug2_feat'] = chem_feat_B_test
        #
        tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
        test_data['GENE_A'] = gene_A_test
        #
        tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
        test_data['GENE_B'] = gene_B_test
        #
        tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
        test_data['TARGET_A'] = target_A_test
        #
        tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
        test_data['TARGET_B'] = target_B_test
        #
        tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
        test_data['cell_BASAL'] = cell_basal_test
        #
        tv_data['Merged_features'] = torch.concat(([tv_data['drug1_feat'], tv_data['drug2_feat'], tv_data['GENE_A'], tv_data['GENE_B'], tv_data['TARGET_A'], tv_data['TARGET_B'], tv_data['cell_BASAL'] ]), axis=1)
        test_data['Merged_features'] = torch.concat(([test_data['drug1_feat'], test_data['drug2_feat'], test_data['GENE_A'], test_data['GENE_B'], test_data['TARGET_A'], test_data['TARGET_B'], test_data['cell_BASAL'] ]), axis=1)
        #            
        tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
        test_data['y'] = syn_test
        #
        print(tv_data['drug1_feat'].shape, flush=True)
        print(test_data['drug1_feat'].shape, flush=True)
        return tv_data, test_data



CID_0 = no_TF_CID[0:6]
CID_1 = no_TF_CID[6:12]
CID_2 = no_TF_CID[12:18]
CID_3 = no_TF_CID[18:24]
CID_4 = no_TF_CID[24:]


# leave cids out 
def prepare_data_GCN_cidout(CV_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
MY_syn_RE2, norm ) :
        # 
        cid_list = globals()['CID_{}'.format(CV_num)]
        #train_key = 'CV{}_train'.format(CV_num)
        #test_key = 'CV{}_test'.format(CV_num)
        # 
        #
        ABCS_tv = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(cid_list)==False) & (A_B_C_S_SET_SM.CID_B.isin(cid_list)==False) ];
        ABCS_test = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(cid_list)) | (A_B_C_S_SET_SM.CID_B.isin(cid_list))]
        #
        tv_ind = list(ABCS_tv.index)
        random.shuffle(tv_ind)
        test_ind = list(ABCS_test.index)
        # 
        chem_feat_A_tv = drug_A_arrS[tv_ind]; chem_feat_A_test = drug_A_arrS[test_ind]
        chem_feat_B_tv = drug_B_arrS[tv_ind]; chem_feat_B_test = drug_B_arrS[test_ind]
        gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
        gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
        target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
        target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
        cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
        syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
        #
        tv_data = {}
        test_data = {}
        #
        tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
        test_data['drug1_feat'] = chem_feat_A_test
        #
        tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
        test_data['drug2_feat'] = chem_feat_B_test
        #
        tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
        test_data['GENE_A'] = gene_A_test
        #
        tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
        test_data['GENE_B'] = gene_B_test
        #
        tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
        test_data['TARGET_A'] = target_A_test
        #
        tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
        test_data['TARGET_B'] = target_B_test
        #
        tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
        test_data['cell_BASAL'] = cell_basal_test
        #
        tv_data['Merged_features'] = torch.concat(([tv_data['drug1_feat'], tv_data['drug2_feat'], tv_data['GENE_A'], tv_data['GENE_B'], tv_data['TARGET_A'], tv_data['TARGET_B'], tv_data['cell_BASAL'] ]), axis=1)
        test_data['Merged_features'] = torch.concat(([test_data['drug1_feat'], test_data['drug2_feat'], test_data['GENE_A'], test_data['GENE_B'], test_data['TARGET_A'], test_data['TARGET_B'], test_data['cell_BASAL'] ]), axis=1)
        #            
        tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
        test_data['y'] = syn_test
        #
        print(tv_data['drug1_feat'].shape, flush=True)
        print(test_data['drug1_feat'].shape, flush=True)
        return tv_data, test_data



A_B_C_S_SET_SM['tissue'] = A_B_C_S_SET_SM['CELL'].apply(lambda x : '_'.join(x.split('_')[1:]))

# leave tissue out 
def prepare_data_GCN_tisout(tissue, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
MY_syn_RE2, norm ) :
        # 
        #
        ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue != tissue]
        ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue == tissue]
        #
        tv_ind = list(ABCS_tv.index)
        random.shuffle(tv_ind)
        test_ind = list(ABCS_test.index)
        # 
        chem_feat_A_tv = drug_A_arrS[tv_ind]; chem_feat_A_test = drug_A_arrS[test_ind]
        chem_feat_B_tv = drug_B_arrS[tv_ind]; chem_feat_B_test = drug_B_arrS[test_ind]
        gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
        gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
        target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
        target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
        cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
        syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
        #
        tv_data = {}
        test_data = {}
        #
        tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
        test_data['drug1_feat'] = chem_feat_A_test
        #
        tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
        test_data['drug2_feat'] = chem_feat_B_test
        #
        tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
        test_data['GENE_A'] = gene_A_test
        #
        tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
        test_data['GENE_B'] = gene_B_test
        #
        tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
        test_data['TARGET_A'] = target_A_test
        #
        tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
        test_data['TARGET_B'] = target_B_test
        #
        tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
        test_data['cell_BASAL'] = cell_basal_test
        #
        tv_data['Merged_features'] = torch.concat(([tv_data['drug1_feat'], tv_data['drug2_feat'], tv_data['GENE_A'], tv_data['GENE_B'], tv_data['TARGET_A'], tv_data['TARGET_B'], tv_data['cell_BASAL'] ]), axis=1)
        test_data['Merged_features'] = torch.concat(([test_data['drug1_feat'], test_data['drug2_feat'], test_data['GENE_A'], test_data['GENE_B'], test_data['TARGET_A'], test_data['TARGET_B'], test_data['cell_BASAL'] ]), axis=1)
        #            
        tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
        test_data['y'] = syn_test
        #
        print(tv_data['drug1_feat'].shape, flush=True)
        print(test_data['drug1_feat'].shape, flush=True)
        return tv_data, test_data









[ Random Forest ]
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
#X = data.drop("target", axis=1)
#y = data['target']
  
[ CV0 ]

X_train, X_test, y_train, y_test = np.array(train_data_0['Merged_features']), np.array(test_data_0['Merged_features']), np.array(train_data_0['y']), np.array(test_data_0['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'max_features': [round(np.sqrt(X_train.shape[1])), 128, 256], # 
}

grid_search = GridSearchCV(RandomForestRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search.fit(X_train, y_train)
print(grid_search.best_estimator_)

gs_df_0 = pd.DataFrame(columns = ['n_estimators', 'max_features', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_0['test_mean'] = list(grid_search.cv_results_['mean_test_score'])
gs_df_0['test_std'] = list(grid_search.cv_results_['std_test_score'])
gs_df_0['n_estimators'] = list(grid_search.cv_results_['param_n_estimators'])
gs_df_0['max_features'] = list(grid_search.cv_results_['param_max_features'])
gs_df_0['CV0'] = list(grid_search.cv_results_['split0_test_score'])
gs_df_0['CV1'] = list(grid_search.cv_results_['split1_test_score'])
gs_df_0['CV2'] = list(grid_search.cv_results_['split2_test_score'])
gs_df_0['CV3'] = list(grid_search.cv_results_['split3_test_score'])
gs_df_0['CV4'] = list(grid_search.cv_results_['split4_test_score'])

                    gs_df_0.at[0,'CV0'] = gs_df_00['CV0'][0] ; gs_df_0.at[1,'CV0'] = gs_df_00['CV0'][1] ; gs_df_0.at[2,'CV0'] = gs_df_00['CV0'][2] ; gs_df_0.at[3,'CV0'] = gs_df_00['CV0'][3]
                    gs_df_0.at[0,'CV1'] = gs_df_00['CV1'][0] ;gs_df_0.at[1,'CV1'] = gs_df_00['CV1'][1];gs_df_0.at[2,'CV1'] = gs_df_00['CV1'][2];gs_df_0.at[3,'CV1'] = gs_df_00['CV1'][3]
                    gs_df_0.at[0,'CV2'] = gs_df_00['CV2'][0] ;gs_df_0.at[1,'CV2'] = gs_df_00['CV2'][1];gs_df_0.at[2,'CV2'] = gs_df_00['CV2'][2];gs_df_0.at[3,'CV2'] = gs_df_00['CV2'][3]
                    gs_df_0.at[0,'CV3'] = gs_df_00['CV3'][0] ;gs_df_0.at[1,'CV3'] = gs_df_00['CV3'][1];gs_df_0.at[2,'CV3'] = gs_df_00['CV3'][2];gs_df_0.at[3,'CV3'] = gs_df_00['CV3'][3]
                    gs_df_0.at[0,'CV4'] = gs_df_00['CV4'][0] ;gs_df_0.at[1,'CV4'] = gs_df_00['CV4'][1];gs_df_0.at[2,'CV4'] = gs_df_00['CV4'][2];gs_df_0.at[3,'CV4'] = gs_df_00['CV4'][3]

                    gs_df_0.at[0,'test_mean'] = gs_df_00['test_mean'][0];gs_df_0.at[1,'test_mean'] = gs_df_00['test_mean'][1];gs_df_0.at[2,'test_mean'] = gs_df_00['test_mean'][2];gs_df_0.at[3,'test_mean'] = gs_df_00['test_mean'][3]

                    gs_df_0.at[0,'test_std'] = gs_df_00['test_std'][0];gs_df_0.at[1,'test_std'] = gs_df_00['test_std'][1];gs_df_0.at[2,'test_std'] = gs_df_00['test_std'][2];gs_df_0.at[3,'test_std'] = gs_df_00['test_std'][3]

ML_path = '/home01/k040a01/05.ML/'
gs_df_0.to_csv(ML_path+'RF_train_CV0.csv', index = False)



[ CV1 ]

X_train, X_test, y_train, y_test = np.array(train_data_1['Merged_features']), np.array(test_data_1['Merged_features']), np.array(train_data_1['y']), np.array(test_data_1['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'max_features': [round(np.sqrt(X_train.shape[1])), 128, 256], # 
}

grid_search_1 = GridSearchCV(RandomForestRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_1.fit(X_train, y_train)
print(grid_search_1.best_estimator_)

gs_df_1 = pd.DataFrame(columns = ['n_estimators', 'max_features', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_1['test_mean'] = list(grid_search.cv_results_['mean_test_score'])
gs_df_1['test_std'] = list(grid_search.cv_results_['std_test_score'])
gs_df_1['n_estimators'] = list(grid_search.cv_results_['param_n_estimators']) + list(grid_search.cv_results_['param_n_estimators'])
gs_df_1['max_features'] = list(grid_search.cv_results_['param_max_features'])
gs_df_1['CV0'] = list(grid_search.cv_results_['split0_test_score'])
gs_df_1['CV1'] = list(grid_search.cv_results_['split1_test_score'])
gs_df_1['CV2'] = list(grid_search.cv_results_['split2_test_score'])
gs_df_1['CV3'] = list(grid_search.cv_results_['split3_test_score'])
gs_df_1['CV4'] = list(grid_search.cv_results_['split4_test_score'])

ML_path = '/home01/k040a01/05.ML/'
gs_df_1.to_csv(ML_path+'RF_train_CV1.csv', index = False)


[ CV2 ]

X_train, X_test, y_train, y_test = np.array(train_data_2['Merged_features']), np.array(test_data_2['Merged_features']), np.array(train_data_2['y']), np.array(test_data_2['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'max_features': [round(np.sqrt(X_train.shape[1])), 128, 256], 
}

grid_search_2 = GridSearchCV(RandomForestRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_2.fit(X_train, y_train)
print(grid_search_2.best_estimator_)

gs_df_2 = pd.DataFrame(columns = ['n_estimators', 'max_features', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_2['test_mean'] = list(grid_search_2.cv_results_['mean_test_score'])
gs_df_2['test_std'] = list(grid_search_2.cv_results_['std_test_score'])
gs_df_2['n_estimators'] = list(grid_search.cv_results_['param_n_estimators'])
gs_df_2['max_features'] = list(grid_search_2.cv_results_['param_max_features'])
gs_df_2['CV0'] = list(grid_search_2.cv_results_['split0_test_score'])
gs_df_2['CV1'] = list(grid_search_2.cv_results_['split1_test_score'])
gs_df_2['CV2'] = list(grid_search_2.cv_results_['split2_test_score'])
gs_df_2['CV3'] = list(grid_search_2.cv_results_['split3_test_score'])
gs_df_2['CV4'] = list(grid_search_2.cv_results_['split4_test_score'])

ML_path = '/home01/k040a01/05.ML/'
gs_df_2.to_csv(ML_path+'RF_train_CV2.csv', index = False)




[ CV3 ]

X_train, X_test, y_train, y_test = np.array(train_data_3['Merged_features']), np.array(test_data_3['Merged_features']), np.array(train_data_3['y']), np.array(test_data_3['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'max_features': [round(np.sqrt(X_train.shape[1])), 128, 256], 
}

grid_search_3 = GridSearchCV(RandomForestRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_3.fit(X_train, y_train)
print(grid_search_3.best_estimator_)

gs_df_3 = pd.DataFrame(columns = ['n_estimators', 'max_features', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_3['test_mean'] = list(grid_search_3.cv_results_['mean_test_score'])
gs_df_3['test_std'] = list(grid_search_3.cv_results_['std_test_score'])
gs_df_3['n_estimators'] = list(grid_search_3.cv_results_['param_n_estimators'])
gs_df_3['max_features'] = list(grid_search_3.cv_results_['param_max_features'])
gs_df_3['CV0'] = list(grid_search_3.cv_results_['split0_test_score'])
gs_df_3['CV1'] = list(grid_search_3.cv_results_['split1_test_score'])
gs_df_3['CV2'] = list(grid_search_3.cv_results_['split2_test_score'])
gs_df_3['CV3'] = list(grid_search_3.cv_results_['split3_test_score'])
gs_df_3['CV4'] = list(grid_search_3.cv_results_['split4_test_score'])

ML_path = '/home01/k040a01/05.ML/'
gs_df_3.to_csv(ML_path+'RF_train_CV3.csv', index = False)



[ CV4 ]

X_train, X_test, y_train, y_test = np.array(train_data_4['Merged_features']), np.array(test_data_4['Merged_features']), np.array(train_data_4['y']), np.array(test_data_4['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'max_features': [round(np.sqrt(X_train.shape[1])), 128, 256], 
}

grid_search_4 = GridSearchCV(RandomForestRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_4.fit(X_train, y_train)
print(grid_search_4.best_estimator_)

gs_df_4 = pd.DataFrame(columns = ['n_estimators', 'max_features', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_4['test_mean'] = list(grid_search_4.cv_results_['mean_test_score'])
gs_df_4['test_std'] = list(grid_search_4.cv_results_['std_test_score'])
gs_df_4['n_estimators'] = list(grid_search_4.cv_results_['param_n_estimators'])
gs_df_4['max_features'] = list(grid_search_4.cv_results_['param_max_features'])
gs_df_4['CV0'] = list(grid_search_4.cv_results_['split0_test_score'])
gs_df_4['CV1'] = list(grid_search_4.cv_results_['split1_test_score'])
gs_df_4['CV2'] = list(grid_search_4.cv_results_['split2_test_score'])
gs_df_4['CV3'] = list(grid_search_4.cv_results_['split3_test_score'])
gs_df_4['CV4'] = list(grid_search_4.cv_results_['split4_test_score'])

ML_path = '/home01/k040a01/05.ML/'
gs_df_4.to_csv(ML_path+'RF_train_CV4.csv', index = False)







# 이제 이부분 다시 해야함 

def get_res(y_test_0, y_pred_0) : 
    Pcor_0, _ = stats.pearsonr(y_test_0, y_pred_0)
    Scor_0, _  = stats.spearmanr(y_test_0, y_pred_0)
    mse_0 = MSE(torch.Tensor(y_test_0), torch.Tensor(y_pred_0)).item()
    #
    return Pcor_0, Scor_0, mse_0

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'
gs_df_0 = pd.read_csv(ML_path+'RF_train_CV0.csv')
gs_df_1 = pd.read_csv(ML_path+'RF_train_CV1.csv')
gs_df_2 = pd.read_csv(ML_path+'RF_train_CV2.csv')
gs_df_3 = pd.read_csv(ML_path+'RF_train_CV3.csv')
gs_df_4 = pd.read_csv(ML_path+'RF_train_CV4.csv')

test_mean = np.mean([gs_df_0.test_mean, gs_df_2.test_mean, gs_df_3.test_mean, gs_df_4.test_mean], 0)
np.max(test_mean)

X_train_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train.npy'); X_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test.npy' ); y_train_0 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train.npy') ;y_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test.npy' ) 
X_train_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train.npy'); X_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test.npy' ); y_train_1 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train.npy') ;y_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test.npy' ) 
X_train_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train.npy'); X_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test.npy' ); y_train_2 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train.npy') ;y_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test.npy' ) 
X_train_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train.npy'); X_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test.npy' ); y_train_3 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train.npy') ;y_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test.npy' ) 
X_train_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train.npy'); X_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test.npy' ); y_train_4 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train.npy') ;y_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test.npy' ) 


y_train_0 = np.ravel(y_train_0) ; y_test_0 = np.ravel(y_test_0)
y_train_1 = np.ravel(y_train_1) ; y_test_1 = np.ravel(y_test_1)
y_train_2 = np.ravel(y_train_2) ; y_test_2 = np.ravel(y_test_2)
y_train_3 = np.ravel(y_train_3) ; y_test_3 = np.ravel(y_test_3)
y_train_4 = np.ravel(y_train_4) ; y_test_4 = np.ravel(y_test_4)


model_final_0 = RandomForestRegressor(n_estimators=1024, max_features=256)
model_final_1 = RandomForestRegressor(n_estimators=1024, max_features=256)
model_final_2 = RandomForestRegressor(n_estimators=1024, max_features=256)
model_final_3 = RandomForestRegressor(n_estimators=1024, max_features=256)
model_final_4 = RandomForestRegressor(n_estimators=1024, max_features=256)

model_final_0.fit(X_train_0, y_train_0)
model_final_1.fit(X_train_1, y_train_1)
model_final_2.fit(X_train_2, y_train_2)
model_final_3.fit(X_train_3, y_train_3)
model_final_4.fit(X_train_4, y_train_4)

y_pred_0 = model_final_0.predict(X_test_0)
y_pred_1 = model_final_1.predict(X_test_1)
y_pred_2 = model_final_2.predict(X_test_2)
y_pred_3 = model_final_3.predict(X_test_3)
y_pred_4 = model_final_4.predict(X_test_4)

pred_list = list(y_pred_0) + list(y_pred_1) + list(y_pred_2) + list(y_pred_3) + list(y_pred_4)
ans_list = list(y_test_0) + list(y_test_1) + list(y_test_2) + list(y_test_3) + list(y_test_4)

Pcor_0, Scor_0, mse_0 = get_res(y_test_0, y_pred_0)
Pcor_1, Scor_1, mse_1 = get_res(y_test_1, y_pred_1)
Pcor_2, Scor_2, mse_2 = get_res(y_test_2, y_pred_2)
Pcor_3, Scor_3, mse_3 = get_res(y_test_3, y_pred_3)
Pcor_4, Scor_4, mse_4 = get_res(y_test_4, y_pred_4)

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'
RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
RF_result['Pcor'] = [Pcor_0, Pcor_1, Pcor_2, Pcor_3, Pcor_4]
RF_result['Scor'] = [Scor_0, Scor_1, Scor_2, Scor_3, Scor_4]
RF_result['mse'] = [mse_0, mse_1, mse_2, mse_3, mse_4]

RF_result.to_csv(ML_path+'leaveCombi.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

mse = np.mean((np.array(ans_list) - np.array(pred_list)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_list))

CfI = st.t.interval(alpha=0.90, df=len(pred_list)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)

RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCombi_value.csv')








# leave cell out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse', 'CfI'])
pred_results = []
ans_list = []

for cell_num in range(0,34) :
    print(cell_num)
    train_data, test_data = prepare_data_GCN_cellout(cell_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = RandomForestRegressor(n_estimators=1024, max_features=256)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveCell.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCell_value.csv')






# leave drugs out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []

for cv_num in range(0,5) :
    print(cv_num)
    train_data, test_data = prepare_data_GCN_cidout(cv_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = RandomForestRegressor(n_estimators=1024, max_features=256)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveCID.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCID_value.csv')






# leave tissue out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []

tissue_list = ['PROSTATE', 'BREAST', 'LARGE_INTESTINE', 'LUNG', 'OVARY', 'SKIN', 'PLEURA']

for tissue in tissue_list :
    train_data, test_data = prepare_data_GCN_tisout(tissue, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = RandomForestRegressor(n_estimators=1024, max_features=256, n_jobs = 16)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveTis.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveTis_value.csv')
























[ XGBOOST]

import pandas as pd
import random
import os
import numpy as np

from sklearn.model_selection import GridSearchCV 
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
import sklearn.metrics as metrics

import xgboost as xgb

[ CV 0  ] 

X_train, X_test, y_train, y_test = np.array(train_data_0['Merged_features']), np.array(test_data_0['Merged_features']), np.array(train_data_0['y']), np.array(test_data_0['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'learning_rate': [0.001, 0.01, 0.1, 1], 
}

grid_search_0 = GridSearchCV(xgb.XGBRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_0.fit(X_train, y_train)
print(grid_search_0.best_estimator_)

gs_df_0 = pd.DataFrame(columns = ['n_estimators', 'learning_rate', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_0['test_mean'] = list(grid_search_0.cv_results_['mean_test_score'])
gs_df_0['test_std'] = list(grid_search_0.cv_results_['std_test_score'])
gs_df_0['n_estimators'] = list(grid_search_0.cv_results_['param_n_estimators'])
gs_df_0['learning_rate'] = list(grid_search_0.cv_results_['param_learning_rate'])
gs_df_0['CV0'] = list(grid_search_0.cv_results_['split0_test_score'])
gs_df_0['CV1'] = list(grid_search_0.cv_results_['split1_test_score'])
gs_df_0['CV2'] = list(grid_search_0.cv_results_['split2_test_score'])
gs_df_0['CV3'] = list(grid_search_0.cv_results_['split3_test_score'])
gs_df_0['CV4'] = list(grid_search_0.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'
gs_df_0.to_csv(ML_path+'XGB_train_CV0.csv', index = False)




[ CV 1  ] 

X_train, X_test, y_train, y_test = np.array(train_data_1['Merged_features']), np.array(test_data_1['Merged_features']), np.array(train_data_1['y']), np.array(test_data_1['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'learning_rate': [0.001, 0.01, 0.1, 1], 
}

grid_search_1 = GridSearchCV(xgb.XGBRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_1.fit(X_train, y_train)
print(grid_search_1.best_estimator_)

gs_df_1 = pd.DataFrame(columns = ['n_estimators', 'learning_rate', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_1['test_mean'] = list(grid_search_1.cv_results_['mean_test_score'])
gs_df_1['test_std'] = list(grid_search_1.cv_results_['std_test_score'])
gs_df_1['n_estimators'] = list(grid_search_1.cv_results_['param_n_estimators'])
gs_df_1['learning_rate'] = list(grid_search_1.cv_results_['param_learning_rate'])
gs_df_1['CV0'] = list(grid_search_1.cv_results_['split0_test_score'])
gs_df_1['CV1'] = list(grid_search_1.cv_results_['split1_test_score'])
gs_df_1['CV2'] = list(grid_search_1.cv_results_['split2_test_score'])
gs_df_1['CV3'] = list(grid_search_1.cv_results_['split3_test_score'])
gs_df_1['CV4'] = list(grid_search_1.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'
gs_df_1.to_csv(ML_path+'XGB_train_CV1.csv', index = False)


[ CV 2  ] 

X_train, X_test, y_train, y_test = np.array(train_data_2['Merged_features']), np.array(test_data_2['Merged_features']), np.array(train_data_2['y']), np.array(test_data_2['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'learning_rate': [0.001, 0.01, 0.1, 1], 
}

grid_search_2 = GridSearchCV(xgb.XGBRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_2.fit(X_train, y_train)
print(grid_search_2.best_estimator_)

gs_df_2 = pd.DataFrame(columns = ['n_estimators', 'learning_rate', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_2['test_mean'] = list(grid_search_2.cv_results_['mean_test_score'])
gs_df_2['test_std'] = list(grid_search_2.cv_results_['std_test_score'])
gs_df_2['n_estimators'] = list(grid_search_2.cv_results_['param_n_estimators'])
gs_df_2['learning_rate'] = list(grid_search_2.cv_results_['param_learning_rate'])
gs_df_2['CV0'] = list(grid_search_2.cv_results_['split0_test_score'])
gs_df_2['CV1'] = list(grid_search_2.cv_results_['split1_test_score'])
gs_df_2['CV2'] = list(grid_search_2.cv_results_['split2_test_score'])
gs_df_2['CV3'] = list(grid_search_2.cv_results_['split3_test_score'])
gs_df_2['CV4'] = list(grid_search_2.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'
gs_df_2.to_csv(ML_path+'XGB_train_CV2.csv', index = False)





[ CV 3  ] 

X_train, X_test, y_train, y_test = np.array(train_data_3['Merged_features']), np.array(test_data_3['Merged_features']), np.array(train_data_3['y']), np.array(test_data_3['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'learning_rate': [0.001, 0.01, 0.1, 1], 
}

grid_search_3 = GridSearchCV(xgb.XGBRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_3.fit(X_train, y_train)
print(grid_search_3.best_estimator_)

gs_df_3 = pd.DataFrame(columns = ['n_estimators', 'learning_rate', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_3['test_mean'] = list(grid_search_3.cv_results_['mean_test_score'])
gs_df_3['test_std'] = list(grid_search_3.cv_results_['std_test_score'])
gs_df_3['n_estimators'] = list(grid_search_3.cv_results_['param_n_estimators'])
gs_df_3['learning_rate'] = list(grid_search_3.cv_results_['param_learning_rate'])
gs_df_3['CV0'] = list(grid_search_3.cv_results_['split0_test_score'])
gs_df_3['CV1'] = list(grid_search_3.cv_results_['split1_test_score'])
gs_df_3['CV2'] = list(grid_search_3.cv_results_['split2_test_score'])
gs_df_3['CV3'] = list(grid_search_3.cv_results_['split3_test_score'])
gs_df_3['CV4'] = list(grid_search_3.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'
gs_df_3.to_csv(ML_path+'XGB_train_CV3.csv', index = False)




[ CV 4  ] 

X_train, X_test, y_train, y_test = np.array(train_data_4['Merged_features']), np.array(test_data_4['Merged_features']), np.array(train_data_4['y']), np.array(test_data_4['y'])

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'n_estimators': [128, 256, 512, 1024],
    'learning_rate': [0.001, 0.01, 0.1, 1], 
}

grid_search_4 = GridSearchCV(xgb.XGBRegressor(),
                           param_grid=param_grid, verbose = 2, error_score='raise')
grid_search_4.fit(X_train, y_train)
print(grid_search_4.best_estimator_)

gs_df_4 = pd.DataFrame(columns = ['n_estimators', 'learning_rate', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_4['test_mean'] = list(grid_search_4.cv_results_['mean_test_score'])
gs_df_4['test_std'] = list(grid_search_4.cv_results_['std_test_score'])
gs_df_4['n_estimators'] = list(grid_search_4.cv_results_['param_n_estimators'])
gs_df_4['learning_rate'] = list(grid_search_4.cv_results_['param_learning_rate'])
gs_df_4['CV0'] = list(grid_search_4.cv_results_['split0_test_score'])
gs_df_4['CV1'] = list(grid_search_4.cv_results_['split1_test_score'])
gs_df_4['CV2'] = list(grid_search_4.cv_results_['split2_test_score'])
gs_df_4['CV3'] = list(grid_search_4.cv_results_['split3_test_score'])
gs_df_4['CV4'] = list(grid_search_4.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'
gs_df_4.to_csv(ML_path+'XGB_train_CV4.csv', index = False)






# 이제 이부분 다시 해야함 

def get_res(y_test_0, y_pred_0) : 
    Pcor_0, _ = stats.pearsonr(y_test_0, y_pred_0)
    Scor_0, _  = stats.spearmanr(y_test_0, y_pred_0)
    mse_0 = MSE(torch.Tensor(y_test_0), torch.Tensor(y_pred_0)).item()
    #
    return Pcor_0, Scor_0, mse_0


ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'
gs_df_0 = pd.read_csv(ML_path+'XGB_train_CV0.csv')
gs_df_1 = pd.read_csv(ML_path+'XGB_train_CV1.csv')
gs_df_2 = pd.read_csv(ML_path+'XGB_train_CV2.csv')
gs_df_3 = pd.read_csv(ML_path+'XGB_train_CV3.csv')
gs_df_4 = pd.read_csv(ML_path+'XGB_train_CV4.csv')

test_mean = np.mean([gs_df_0.test_mean, gs_df_1.test_mean, gs_df_2.test_mean, gs_df_3.test_mean, gs_df_4.test_mean], 0)
list(test_mean).index(np.max(test_mean))

X_train_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train.npy'); X_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test.npy' ); y_train_0 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train.npy') ;y_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test.npy' ) 
X_train_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train.npy'); X_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test.npy' ); y_train_1 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train.npy') ;y_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test.npy' ) 
X_train_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train.npy'); X_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test.npy' ); y_train_2 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train.npy') ;y_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test.npy' ) 
X_train_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train.npy'); X_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test.npy' ); y_train_3 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train.npy') ;y_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test.npy' ) 
X_train_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train.npy'); X_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test.npy' ); y_train_4 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train.npy') ;y_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test.npy' ) 


y_train_0 = np.ravel(y_train_0) ; y_test_0 = np.ravel(y_test_0)
y_train_1 = np.ravel(y_train_1) ; y_test_1 = np.ravel(y_test_1)
y_train_2 = np.ravel(y_train_2) ; y_test_2 = np.ravel(y_test_2)
y_train_3 = np.ravel(y_train_3) ; y_test_3 = np.ravel(y_test_3)
y_train_4 = np.ravel(y_train_4) ; y_test_4 = np.ravel(y_test_4)


model_final_0 = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs= 16)
model_final_1 = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs= 16)
model_final_2 = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs= 16)
model_final_3 = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs= 16)
model_final_4 = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs= 16)


model_final_0.fit(X_train_0, y_train_0)
model_final_1.fit(X_train_1, y_train_1)
model_final_2.fit(X_train_2, y_train_2)
model_final_3.fit(X_train_3, y_train_3)
model_final_4.fit(X_train_4, y_train_4)

y_pred_0 = model_final_0.predict(X_test_0)
y_pred_1 = model_final_1.predict(X_test_1)
y_pred_2 = model_final_2.predict(X_test_2)
y_pred_3 = model_final_3.predict(X_test_3)
y_pred_4 = model_final_4.predict(X_test_4)

pred_list = list(y_pred_0) + list(y_pred_1) + list(y_pred_2) + list(y_pred_3) + list(y_pred_4)
ans_list = list(y_test_0) + list(y_test_1) + list(y_test_2) + list(y_test_3) + list(y_test_4)

Pcor_0, Scor_0, mse_0 = get_res(y_test_0, y_pred_0)
Pcor_1, Scor_1, mse_1 = get_res(y_test_1, y_pred_1)
Pcor_2, Scor_2, mse_2 = get_res(y_test_2, y_pred_2)
Pcor_3, Scor_3, mse_3 = get_res(y_test_3, y_pred_3)
Pcor_4, Scor_4, mse_4 = get_res(y_test_4, y_pred_4)

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'
RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
RF_result['Pcor'] = [Pcor_0, Pcor_1, Pcor_2, Pcor_3, Pcor_4]
RF_result['Scor'] = [Scor_0, Scor_1, Scor_2, Scor_3, Scor_4]
RF_result['mse'] = [mse_0, mse_1, mse_2, mse_3, mse_4]

RF_result.to_csv(ML_path+'leaveCombi.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

mse = np.mean((np.array(ans_list) - np.array(pred_list)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_list))

CfI = st.t.interval(alpha=0.90, df=len(pred_list)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)

RF_result2 = pd.DataFrame({'pred' : pred_list, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCombi_value.csv')




# leave cell out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []


#A_B_C_S_SET_SM.to_csv(ML_path + 'A_B_C_S_SET_SM.csv')
#torch.save(drug_A_arrS, ML_path + 'drug_A_arrS.pth')
#torch.save(drug_B_arrS, ML_path + 'drug_B_arrS.pth')
#torch.save(MY_g_EXP_A_RE2, ML_path + 'MY_g_EXP_A_RE2.pth' )
#torch.save(MY_g_EXP_B_RE2, ML_path + 'MY_g_EXP_B_RE2.pth' )
#torch.save(MY_Target_A2, ML_path + 'MY_Target_A2.pth' )
#torch.save(MY_Target_B2, ML_path + 'MY_Target_B2.pth' )
#torch.save(MY_CellBase_RE2, ML_path + 'MY_CellBase_RE2.pth' )
#torch.save(MY_syn_RE2, ML_path + 'MY_syn_RE2.pth' )

A_B_C_S_SET_SM = pd.read_csv(ML_path + 'A_B_C_S_SET_SM.csv')
drug_A_arrS = torch.load( ML_path + 'drug_A_arrS.pth')
drug_B_arrS = torch.load( ML_path + 'drug_B_arrS.pth')
MY_g_EXP_A_RE2 = torch.load( ML_path + 'MY_g_EXP_A_RE2.pth' )
MY_g_EXP_B_RE2 = torch.load( ML_path + 'MY_g_EXP_B_RE2.pth' )
MY_Target_A2 = torch.load( ML_path + 'MY_Target_A2.pth' )
MY_Target_B2 = torch.load( ML_path + 'MY_Target_B2.pth' )
MY_CellBase_RE2 = torch.load( ML_path + 'MY_CellBase_RE2.pth' )
MY_syn_RE2 = torch.load( ML_path + 'MY_syn_RE2.pth' )


for cell_num in range(0,34) :
    print(cell_num)
    train_data, test_data = prepare_data_GCN_cellout(cell_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs = 8)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)



RF_result.to_csv(ML_path+'leaveCell.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCell_value.csv')




# leave drugs out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []


for cv_num in range(5) :
    print(cv_num)
    train_data, test_data = prepare_data_GCN_cidout(cv_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs = 16)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)



RF_result.to_csv(ML_path+'leaveCID.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCID_value.csv')







# leave tissue out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []

tissue_list = ['PROSTATE', 'BREAST', 'LARGE_INTESTINE', 'LUNG', 'OVARY', 'SKIN', 'PLEURA']

for tissue in tissue_list :
    train_data, test_data = prepare_data_GCN_tisout(tissue, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = xgb.XGBRegressor(n_estimators=1024, learning_rate=0.1, n_jobs = 16)
    model_final.fit(X_train, y_train)
    y_pred = model_final.predict(X_test)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveTis.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveTis_value.csv')




















[ SVR ]
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVR
# import pickle5 as pickle
from joblib import parallel_backend
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV




ML_path = '/home01/k040a01/05.ML/'
A_B_C_S_SET_SM = pd.read_csv(ML_path + 'A_B_C_S_SET_SM.csv')
drug_A_arrS = torch.load( ML_path + 'drug_A_arrS.pth')
drug_B_arrS = torch.load( ML_path + 'drug_B_arrS.pth')
MY_g_EXP_A_RE2 = torch.load( ML_path + 'MY_g_EXP_A_RE2.pth' )
MY_g_EXP_B_RE2 = torch.load( ML_path + 'MY_g_EXP_B_RE2.pth' )
MY_Target_A2 = torch.load( ML_path + 'MY_Target_A2.pth' )
MY_Target_B2 = torch.load( ML_path + 'MY_Target_B2.pth' )
MY_CellBase_RE2 = torch.load( ML_path + 'MY_CellBase_RE2.pth' )
MY_syn_RE2 = torch.load( ML_path + 'MY_syn_RE2.pth' )





[ CV0 ]

X_train, X_test, y_train, y_test = np.array(train_data_0['Merged_features']), np.array(test_data_0['Merged_features']), np.array(train_data_0['y']), np.array(test_data_0['y'])

                # np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train' , X_train)
                # np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test' , X_test)
                # np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train' , y_train)
                # np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test' , y_test)

                X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train.npy')
                X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test.npy' )
                y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train.npy')
                y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test.npy' )



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid={
            'C': [0.001, 0.01, 1, 10],
            'gamma': [0.01, 0.05, 0.1, 0.5]
        }


grid_search_0 = GridSearchCV(SVR(kernel='rbf'),
                           param_grid=param_grid, verbose = 3, 
                           error_score='raise', n_jobs = 10)


grid_search_0.fit(X_train_scaled, y_train)

gs_df_0 = pd.DataFrame(columns = ['C', 'gamma', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_0['test_mean'] = list(grid_search_0.cv_results_['mean_test_score'])
gs_df_0['test_std'] = list(grid_search_0.cv_results_['std_test_score'])
gs_df_0['C'] = list(grid_search_0.cv_results_['param_C'])
gs_df_0['gamma'] = list(grid_search_0.cv_results_['param_gamma'])
gs_df_0['CV0'] = list(grid_search_0.cv_results_['split0_test_score'])
gs_df_0['CV1'] = list(grid_search_0.cv_results_['split1_test_score'])
gs_df_0['CV2'] = list(grid_search_0.cv_results_['split2_test_score'])
gs_df_0['CV3'] = list(grid_search_0.cv_results_['split3_test_score'])
gs_df_0['CV4'] = list(grid_search_0.cv_results_['split4_test_score'])

# ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'
ML_path = '/home01/k040a01/05.ML/'

gs_df_0.to_csv(ML_path+'SVR_train_CV0.csv', index = False)




[ CV1 ]

X_train, X_test, y_train, y_test = np.array(train_data_1['Merged_features']), np.array(test_data_1['Merged_features']), np.array(train_data_1['y']), np.array(test_data_1['y'])

                        np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train' , X_train)
                        np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test' , X_test)
                        np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train' , y_train)
                        np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test' , y_test)

                        X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train.npy')
                        X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test.npy' )
                        y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train.npy')
                        y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test.npy' )



from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid={
            'C': [0.001, 0.01, 1, 10],
            'gamma': [0.01, 0.05, 0.1, 0.5]
        }


grid_search_1 = GridSearchCV(SVR(kernel='rbf'),
                           param_grid=param_grid, verbose = 3, 
                           error_score='raise', n_jobs = 10)


grid_search_1.fit(X_train_scaled, y_train)

gs_df_1 = pd.DataFrame(columns = ['C', 'gamma', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_1['test_mean'] = list(grid_search_1.cv_results_['mean_test_score'])
gs_df_1['test_std'] = list(grid_search_1.cv_results_['std_test_score'])
gs_df_1['C'] = list(grid_search_1.cv_results_['param_C'])
gs_df_1['gamma'] = list(grid_search_1.cv_results_['param_gamma'])
gs_df_1['CV0'] = list(grid_search_1.cv_results_['split0_test_score'])
gs_df_1['CV1'] = list(grid_search_1.cv_results_['split1_test_score'])
gs_df_1['CV2'] = list(grid_search_1.cv_results_['split2_test_score'])
gs_df_1['CV3'] = list(grid_search_1.cv_results_['split3_test_score'])
gs_df_1['CV4'] = list(grid_search_1.cv_results_['split4_test_score'])

# ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'
# ML_path = '/home01/k040a01/05.ML/'

gs_df_1.to_csv(ML_path+'SVR_train_CV1.csv', index = False)




[ CV2 ]

X_train, X_test, y_train, y_test = np.array(train_data_2['Merged_features']), np.array(test_data_2['Merged_features']), np.array(train_data_2['y']), np.array(test_data_2['y'])

np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train' , X_train)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test' , X_test)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train' , y_train)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test' , y_test)

X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train.npy')
X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test.npy' )
y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train.npy')
y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test.npy' )



y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid={
            'C': [0.001, 0.01, 1, 10],
            'gamma': [0.01, 0.05, 0.1, 0.5]
        }


grid_search_2 = GridSearchCV(SVR(kernel='rbf'),
                           param_grid=param_grid, verbose = 2, 
                           error_score='raise', n_jobs = 16)


grid_search_2.fit(X_train, y_train)

gs_df_2 = pd.DataFrame(columns = ['C', 'gamma', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_2['test_mean'] = list(grid_search_2.cv_results_['mean_test_score'])
gs_df_2['test_std'] = list(grid_search_2.cv_results_['std_test_score'])
gs_df_2['C'] = list(grid_search_2.cv_results_['param_C'])
gs_df_2['gamma'] = list(grid_search_2.cv_results_['param_gamma'])
gs_df_2['CV0'] = list(grid_search_2.cv_results_['split0_test_score'])
gs_df_2['CV1'] = list(grid_search_2.cv_results_['split1_test_score'])
gs_df_2['CV2'] = list(grid_search_2.cv_results_['split2_test_score'])
gs_df_2['CV3'] = list(grid_search_2.cv_results_['split3_test_score'])
gs_df_2['CV4'] = list(grid_search_2.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'
# ML_path = '/home01/k040a01/05.ML/'

gs_df_2.to_csv(ML_path+'SVR_train_CV2.csv', index = False)



[ CV3 ]

X_train, X_test, y_train, y_test = np.array(train_data_3['Merged_features']), np.array(test_data_3['Merged_features']), np.array(train_data_3['y']), np.array(test_data_3['y'])

np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train' , X_train)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test' , X_test)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train' , y_train)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test' , y_test)

X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train.npy')
X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test.npy' )
y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train.npy')
y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test.npy' )



y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid={
            'C': [0.001, 0.01, 1, 10],
            'gamma': [0.01, 0.05, 0.1, 0.5]
        }


grid_search_3 = GridSearchCV(SVR(kernel='rbf'),
                           param_grid=param_grid, verbose = 2, 
                           error_score='raise', n_jobs = 10)


grid_search_3.fit(X_train, y_train)

gs_df_3 = pd.DataFrame(columns = ['C', 'gamma', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_3['test_mean'] = list(grid_search_3.cv_results_['mean_test_score'])
gs_df_3['test_std'] = list(grid_search_3.cv_results_['std_test_score'])
gs_df_3['C'] = list(grid_search_3.cv_results_['param_C'])
gs_df_3['gamma'] = list(grid_search_3.cv_results_['param_gamma'])
gs_df_3['CV0'] = list(grid_search_3.cv_results_['split0_test_score'])
gs_df_3['CV1'] = list(grid_search_3.cv_results_['split1_test_score'])
gs_df_3['CV2'] = list(grid_search_3.cv_results_['split2_test_score'])
gs_df_3['CV3'] = list(grid_search_3.cv_results_['split3_test_score'])
gs_df_3['CV4'] = list(grid_search_3.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'
# ML_path = '/home01/k040a01/05.ML/'

gs_df_3.to_csv(ML_path+'SVR_train_CV3.csv', index = False)




[ CV4 ]

X_train, X_test, y_train, y_test = np.array(train_data_4['Merged_features']), np.array(test_data_4['Merged_features']), np.array(train_data_4['y']), np.array(test_data_4['y'])

np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train' , X_train)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test' , X_test)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train' , y_train)
np.save('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test' , y_test)

X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train.npy')
X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test.npy' )
y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train.npy')
y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test.npy' )



y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid={
            'C': [0.001, 0.01, 1, 10],
            'gamma': [0.01, 0.05, 0.1, 0.5]
        }


grid_search_4 = GridSearchCV(SVR(kernel='rbf'),
                           param_grid=param_grid, verbose = 2, 
                           error_score='raise', n_jobs = 10)


grid_search_4.fit(X_train, y_train)

gs_df_4 = pd.DataFrame(columns = ['C', 'gamma', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_4['test_mean'] = list(grid_search_4.cv_results_['mean_test_score'])
gs_df_4['test_std'] = list(grid_search_4.cv_results_['std_test_score'])
gs_df_4['C'] = list(grid_search_4.cv_results_['param_C'])
gs_df_4['gamma'] = list(grid_search_4.cv_results_['param_gamma'])
gs_df_4['CV0'] = list(grid_search_4.cv_results_['split0_test_score'])
gs_df_4['CV1'] = list(grid_search_4.cv_results_['split1_test_score'])
gs_df_4['CV2'] = list(grid_search_4.cv_results_['split2_test_score'])
gs_df_4['CV3'] = list(grid_search_4.cv_results_['split3_test_score'])
gs_df_4['CV4'] = list(grid_search_4.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'
# ML_path = '/home01/k040a01/05.ML/'

gs_df_4.to_csv(ML_path+'SVR_train_CV4.csv', index = False)


norm = 'tanh_norm'






# 이제 이부분 다시 해야함 

def get_res(y_test_0, y_pred_0) : 
    Pcor_0, _ = stats.pearsonr(y_test_0, y_pred_0)
    Scor_0, _  = stats.spearmanr(y_test_0, y_pred_0)
    mse_0 = MSE(torch.Tensor(y_test_0), torch.Tensor(y_pred_0)).item()
    #
    return Pcor_0, Scor_0, mse_0

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'
gs_df_0 = pd.read_csv(ML_path+'SVR_train_CV0.csv')
gs_df_1 = pd.read_csv(ML_path+'SVR_train_CV1.csv')
gs_df_2 = pd.read_csv(ML_path+'SVR_train_CV2.csv')
gs_df_3 = pd.read_csv(ML_path+'SVR_train_CV3.csv')
gs_df_4 = pd.read_csv(ML_path+'SVR_train_CV4.csv')


test_mean = np.mean([gs_df_0.test_mean, gs_df_2.test_mean, gs_df_3.test_mean, gs_df_4.test_mean], 0)
list(test_mean).index(np.max(test_mean))


                X_train_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train.npy'); X_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test.npy' ); y_train_0 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train.npy') ;y_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test.npy' ) 
                X_train_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train.npy'); X_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test.npy' ); y_train_1 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train.npy') ;y_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test.npy' ) 
                X_train_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train.npy'); X_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test.npy' ); y_train_2 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train.npy') ;y_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test.npy' ) 
                X_train_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train.npy'); X_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test.npy' ); y_train_3 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train.npy') ;y_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test.npy' ) 
                X_train_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train.npy'); X_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test.npy' ); y_train_4 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train.npy') ;y_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test.npy' ) 


X_train_0, X_test_0, y_train_0, y_test_0 = np.array(train_data_0['Merged_features']), np.array(test_data_0['Merged_features']), np.array(train_data_0['y']), np.array(test_data_0['y'])
X_train_1, X_test_1, y_train_1, y_test_1 = np.array(train_data_1['Merged_features']), np.array(test_data_1['Merged_features']), np.array(train_data_1['y']), np.array(test_data_1['y'])
X_train_2, X_test_2, y_train_2, y_test_2 = np.array(train_data_2['Merged_features']), np.array(test_data_2['Merged_features']), np.array(train_data_2['y']), np.array(test_data_2['y'])
X_train_3, X_test_3, y_train_3, y_test_3 = np.array(train_data_3['Merged_features']), np.array(test_data_3['Merged_features']), np.array(train_data_3['y']), np.array(test_data_3['y'])
X_train_4, X_test_4, y_train_4, y_test_4 = np.array(train_data_4['Merged_features']), np.array(test_data_4['Merged_features']), np.array(train_data_4['y']), np.array(test_data_4['y'])


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled_0 = scaler.fit_transform(X_train_0) ; X_test_scaled_0 = scaler.transform(X_test_0)
scaler = StandardScaler()
X_train_scaled_1 = scaler.fit_transform(X_train_1) ; X_test_scaled_1 = scaler.transform(X_test_1)
scaler = StandardScaler()
X_train_scaled_2 = scaler.fit_transform(X_train_2) ; X_test_scaled_2 = scaler.transform(X_test_2)
scaler = StandardScaler()
X_train_scaled_3 = scaler.fit_transform(X_train_3) ; X_test_scaled_3 = scaler.transform(X_test_3)
scaler = StandardScaler()
X_train_scaled_4 = scaler.fit_transform(X_train_4) ; X_test_scaled_4 = scaler.transform(X_test_4)




y_train_0 = np.ravel(y_train_0) ; y_test_0 = np.ravel(y_test_0)
y_train_1 = np.ravel(y_train_1) ; y_test_1 = np.ravel(y_test_1)
y_train_2 = np.ravel(y_train_2) ; y_test_2 = np.ravel(y_test_2)
y_train_3 = np.ravel(y_train_3) ; y_test_3 = np.ravel(y_test_3)
y_train_4 = np.ravel(y_train_4) ; y_test_4 = np.ravel(y_test_4)


model_final_0 = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
model_final_1 = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
model_final_2 = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
model_final_3 = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
model_final_4 = SVR(kernel='rbf', C = 10.000, gamma = 0.01)


model_final_0.fit(X_train_scaled_0, y_train_0)
model_final_1.fit(X_train_scaled_1, y_train_1)
model_final_2.fit(X_train_scaled_2, y_train_2)
model_final_3.fit(X_train_scaled_3, y_train_3)
model_final_4.fit(X_train_scaled_4, y_train_4)


y_pred_0 = model_final_0.predict(X_test_scaled_0)
y_pred_1 = model_final_1.predict(X_test_scaled_1)
y_pred_2 = model_final_2.predict(X_test_scaled_2)
y_pred_3 = model_final_3.predict(X_test_scaled_3)
y_pred_4 = model_final_4.predict(X_test_scaled_4)

pred_list = list(y_pred_0) + list(y_pred_1) + list(y_pred_2) + list(y_pred_3) + list(y_pred_4)
ans_list = list(y_test_0) + list(y_test_1) + list(y_test_2) + list(y_test_3) + list(y_test_4)

Pcor_0, Scor_0, mse_0 = get_res(y_test_0, y_pred_0)
Pcor_1, Scor_1, mse_1 = get_res(y_test_1, y_pred_1)
Pcor_2, Scor_2, mse_2 = get_res(y_test_2, y_pred_2)
Pcor_3, Scor_3, mse_3 = get_res(y_test_3, y_pred_3)
Pcor_4, Scor_4, mse_4 = get_res(y_test_4, y_pred_4)

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
RF_result['Pcor'] = [Pcor_0, Pcor_1, Pcor_2, Pcor_3, Pcor_4]
RF_result['Scor'] = [Scor_0, Scor_1, Scor_2, Scor_3, Scor_4]
RF_result['mse'] = [mse_0, mse_1, mse_2, mse_3, mse_4]

RF_result.to_csv(ML_path+'leaveCombi.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

mse = np.mean((np.array(ans_list) - np.array(pred_list)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_list))

CfI = st.t.interval(alpha=0.90, df=len(pred_list)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)

RF_result2 = pd.DataFrame({'pred' : pred_list, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCombi_value.csv')







# leave cell out 


from sklearn.preprocessing import StandardScaler

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []

for cell_num in range(20,30) :
    print(cell_num)
    train_data, test_data = prepare_data_GCN_cellout(cell_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) ; X_test_scaled = scaler.transform(X_test)
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
    model_final.fit(X_train_scaled, y_train)
    y_pred = model_final.predict(X_test_scaled)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    print(tmp_df)
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveCell.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCell_value.csv')





# leave drugs out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []



for cv_num in range(5) :
    print(cv_num)
    train_data, test_data = prepare_data_GCN_cidout(cv_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) ; X_test_scaled = scaler.transform(X_test)
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
    model_final.fit(X_train_scaled, y_train)
    y_pred = model_final.predict(X_test_scaled)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveCID.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCID_value.csv')








# leave tissue out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []

tissue_list = ['PROSTATE', 'BREAST', 'LARGE_INTESTINE', 'LUNG', 'OVARY', 'SKIN', 'PLEURA']

for tissue in tissue_list :
    train_data, test_data = prepare_data_GCN_tisout(tissue, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) ; X_test_scaled = scaler.transform(X_test)
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = SVR(kernel='rbf', C = 10.000, gamma = 0.01)
    model_final.fit(X_train_scaled, y_train)
    y_pred = model_final.predict(X_test_scaled)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveTis.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveTis_value.csv')






















[ ELASTIC NET ]

from sklearn.linear_model import Lasso,ElasticNet,Ridge
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV

tot_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/XGB/'

A_B_C_S_SET_SM = pd.read_csv(tot_path + 'A_B_C_S_SET_SM.csv')
drug_A_arrS = torch.load( tot_path + 'drug_A_arrS.pth')
drug_B_arrS = torch.load( tot_path + 'drug_B_arrS.pth')
MY_g_EXP_A_RE2 = torch.load( tot_path + 'MY_g_EXP_A_RE2.pth' )
MY_g_EXP_B_RE2 = torch.load( tot_path + 'MY_g_EXP_B_RE2.pth' )
MY_Target_A2 = torch.load( tot_path + 'MY_Target_A2.pth' )
MY_Target_B2 = torch.load( tot_path + 'MY_Target_B2.pth' )
MY_CellBase_RE2 = torch.load( tot_path + 'MY_CellBase_RE2.pth' )
MY_syn_RE2 = torch.load( tot_path + 'MY_syn_RE2.pth' )








maybe we should do normalize in this data

norm = 'tanh_norm'

# leave combination 

train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )










[ CV 0 ] 

X_train, X_test, y_train, y_test = np.array(train_data_0['Merged_features']), np.array(test_data_0['Merged_features']), np.array(train_data_0['y']), np.array(test_data_0['y'])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

                    # 이름은 SVR 이지만 괜춘 
                    X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train.npy')
                    X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test.npy' )
                    y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train.npy')
                    y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test.npy' )


y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9], # 
}

grid_search_0 = GridSearchCV(ElasticNet(),
                           param_grid=param_grid, verbose = 2, error_score='raise', n_jobs = 10) # 

# grid_search_0.fit(X_train, y_train)
grid_search_0.fit(X_train_scaled, y_train)

print(grid_search_0.best_estimator_)

gs_df_0 = pd.DataFrame(columns = ['alpha', 'l1_ratio', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_0['test_mean'] = list(grid_search_0.cv_results_['mean_test_score'])
gs_df_0['test_std'] = list(grid_search_0.cv_results_['std_test_score'])
gs_df_0['alpha'] = list(grid_search_0.cv_results_['param_alpha'])
gs_df_0['l1_ratio'] = list(grid_search_0.cv_results_['param_l1_ratio'])
gs_df_0['CV0'] = list(grid_search_0.cv_results_['split0_test_score'])
gs_df_0['CV1'] = list(grid_search_0.cv_results_['split1_test_score'])
gs_df_0['CV2'] = list(grid_search_0.cv_results_['split2_test_score'])
gs_df_0['CV3'] = list(grid_search_0.cv_results_['split3_test_score'])
gs_df_0['CV4'] = list(grid_search_0.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
gs_df_0.to_csv(ML_path+'EN_train_norm_CV0.csv', index = False)



[ CV 1 ]
X_train, X_test, y_train, y_test = np.array(train_data_1['Merged_features']), np.array(test_data_1['Merged_features']), np.array(train_data_1['y']), np.array(test_data_1['y'])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



            # 이름은 SVR 이지만 괜춘 
            X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train.npy')
            X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test.npy' )
            y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train.npy')
            y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test.npy' )


y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9], # 
}

grid_search_1 = GridSearchCV(ElasticNet(),
                           param_grid=param_grid, verbose = 2, error_score='raise', n_jobs = 10)
grid_search_1.fit(X_train_scaled, y_train)
print(grid_search_1.best_estimator_)

gs_df_1 = pd.DataFrame(columns = ['alpha', 'l1_ratio', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_1['test_mean'] = list(grid_search_1.cv_results_['mean_test_score'])
gs_df_1['test_std'] = list(grid_search_1.cv_results_['std_test_score'])
gs_df_1['alpha'] = list(grid_search_1.cv_results_['param_alpha'])
gs_df_1['l1_ratio'] = list(grid_search_1.cv_results_['param_l1_ratio'])
gs_df_1['CV0'] = list(grid_search_1.cv_results_['split0_test_score'])
gs_df_1['CV1'] = list(grid_search_1.cv_results_['split1_test_score'])
gs_df_1['CV2'] = list(grid_search_1.cv_results_['split2_test_score'])
gs_df_1['CV3'] = list(grid_search_1.cv_results_['split3_test_score'])
gs_df_1['CV4'] = list(grid_search_1.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
gs_df_1.to_csv(ML_path+'EN_train_norm_CV1.csv', index = False)



[ CV 2 ]
X_train, X_test, y_train, y_test = np.array(train_data_2['Merged_features']), np.array(test_data_2['Merged_features']), np.array(train_data_2['y']), np.array(test_data_2['y'])

                    # 이름은 SVR 이지만 괜춘 
                    X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train.npy')
                    X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test.npy' )
                    y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train.npy')
                    y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test.npy' )


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9], # 
}

grid_search_2 = GridSearchCV(ElasticNet(),
                           param_grid=param_grid, verbose = 2, error_score='raise', n_jobs = 10)

grid_search_2.fit(X_train_scaled, y_train)
print(grid_search_2.best_estimator_)

gs_df_2 = pd.DataFrame(columns = ['alpha', 'l1_ratio', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_2['test_mean'] = list(grid_search_2.cv_results_['mean_test_score'])
gs_df_2['test_std'] = list(grid_search_2.cv_results_['std_test_score'])
gs_df_2['alpha'] = list(grid_search_2.cv_results_['param_alpha'])
gs_df_2['l1_ratio'] = list(grid_search_2.cv_results_['param_l1_ratio'])
gs_df_2['CV0'] = list(grid_search_2.cv_results_['split0_test_score'])
gs_df_2['CV1'] = list(grid_search_2.cv_results_['split1_test_score'])
gs_df_2['CV2'] = list(grid_search_2.cv_results_['split2_test_score'])
gs_df_2['CV3'] = list(grid_search_2.cv_results_['split3_test_score'])
gs_df_2['CV4'] = list(grid_search_2.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
gs_df_2.to_csv(ML_path+'EN_train_norm_CV2.csv', index = False)


[ CV 3 ]
X_train, X_test, y_train, y_test = np.array(train_data_3['Merged_features']), np.array(test_data_3['Merged_features']), np.array(train_data_3['y']), np.array(test_data_3['y'])

                # 이름은 SVR 이지만 괜춘 
                X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train.npy')
                X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test.npy' )
                y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train.npy')
                y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test.npy' )

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9], # 
}

grid_search_3 = GridSearchCV(ElasticNet(),
                           param_grid=param_grid, verbose = 2, error_score='raise', n_jobs = 10)
grid_search_3.fit(X_train_scaled, y_train)
print(grid_search_3.best_estimator_)

gs_df_3 = pd.DataFrame(columns = ['alpha', 'l1_ratio', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_3['test_mean'] = list(grid_search_3.cv_results_['mean_test_score'])
gs_df_3['test_std'] = list(grid_search_3.cv_results_['std_test_score'])
gs_df_3['alpha'] = list(grid_search_3.cv_results_['param_alpha'])
gs_df_3['l1_ratio'] = list(grid_search_3.cv_results_['param_l1_ratio'])
gs_df_3['CV0'] = list(grid_search_3.cv_results_['split0_test_score'])
gs_df_3['CV1'] = list(grid_search_3.cv_results_['split1_test_score'])
gs_df_3['CV2'] = list(grid_search_3.cv_results_['split2_test_score'])
gs_df_3['CV3'] = list(grid_search_3.cv_results_['split3_test_score'])
gs_df_3['CV4'] = list(grid_search_3.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
gs_df_3.to_csv(ML_path+'EN_train_norm_CV3.csv', index = False)



[ CV 4 ]
X_train, X_test, y_train, y_test = np.array(train_data_4['Merged_features']), np.array(test_data_4['Merged_features']), np.array(train_data_4['y']), np.array(test_data_4['y'])

                # 이름은 SVR 이지만 괜춘 
                X_train = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train.npy')
                X_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test.npy' )
                y_train =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train.npy')
                y_test = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test.npy' )


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

param_grid = {
    'alpha': [0.001, 0.01, 0.1, 1],
    'l1_ratio': [0.3, 0.5, 0.7, 0.9], # 
}

grid_search_4 = GridSearchCV(ElasticNet(),
                           param_grid=param_grid, verbose = 2, error_score='raise', n_jobs = 10)
grid_search_4.fit(X_train_scaled, y_train)
print(grid_search_4.best_estimator_)

gs_df_4 = pd.DataFrame(columns = ['alpha', 'l1_ratio', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])

gs_df_4['test_mean'] = list(grid_search_4.cv_results_['mean_test_score'])
gs_df_4['test_std'] = list(grid_search_4.cv_results_['std_test_score'])
gs_df_4['alpha'] = list(grid_search_4.cv_results_['param_alpha'])
gs_df_4['l1_ratio'] = list(grid_search_4.cv_results_['param_l1_ratio'])
gs_df_4['CV0'] = list(grid_search_4.cv_results_['split0_test_score'])
gs_df_4['CV1'] = list(grid_search_4.cv_results_['split1_test_score'])
gs_df_4['CV2'] = list(grid_search_4.cv_results_['split2_test_score'])
gs_df_4['CV3'] = list(grid_search_4.cv_results_['split3_test_score'])
gs_df_4['CV4'] = list(grid_search_4.cv_results_['split4_test_score'])

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
gs_df_4.to_csv(ML_path+'EN_train_norm_CV4.csv', index = False)






# 이제 이부분 다시 해야함 
from scipy import stats

def get_res(y_test_0, y_pred_0) : 
    Pcor_0, _ = stats.pearsonr(y_test_0, y_pred_0)
    Scor_0, _  = stats.spearmanr(y_test_0, y_pred_0)
    mse_0 = MSE(torch.Tensor(y_test_0), torch.Tensor(y_pred_0)).item()
    #
    return Pcor_0, Scor_0, mse_0



ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
gs_df_0 = pd.read_csv(ML_path+'EN_train_norm_CV0.csv')
gs_df_1 = pd.read_csv(ML_path+'EN_train_norm_CV1.csv')
gs_df_2 = pd.read_csv(ML_path+'EN_train_norm_CV2.csv')
gs_df_3 = pd.read_csv(ML_path+'EN_train_norm_CV3.csv')
gs_df_4 = pd.read_csv(ML_path+'EN_train_norm_CV4.csv')

test_mean = np.mean([gs_df_0.test_mean, gs_df_1.test_mean, gs_df_2.test_mean, gs_df_3.test_mean, gs_df_4.test_mean], 0)
list(test_mean).index(np.max(test_mean))

                    X_train_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_train.npy'); X_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_x_test.npy' ); y_train_0 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_train.npy') ;y_test_0 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_0_y_test.npy' ) 
                    X_train_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_train.npy'); X_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_x_test.npy' ); y_train_1 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_train.npy') ;y_test_1 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_1_y_test.npy' ) 
                    X_train_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_train.npy'); X_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_x_test.npy' ); y_train_2 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_train.npy') ;y_test_2 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_2_y_test.npy' ) 
                    X_train_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_train.npy'); X_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_x_test.npy' ); y_train_3 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_train.npy') ;y_test_3 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_3_y_test.npy' ) 
                    X_train_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_train.npy'); X_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_x_test.npy' ); y_train_4 =np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_train.npy') ;y_test_4 = np.load('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/SVR/cv_4_y_test.npy' ) 



X_train_0, X_test_0, y_train_0, y_test_0 = np.array(train_data_0['Merged_features']), np.array(test_data_0['Merged_features']), np.array(train_data_0['y']), np.array(test_data_0['y'])
X_train_1, X_test_1, y_train_1, y_test_1 = np.array(train_data_1['Merged_features']), np.array(test_data_1['Merged_features']), np.array(train_data_1['y']), np.array(test_data_1['y'])
X_train_2, X_test_2, y_train_2, y_test_2 = np.array(train_data_2['Merged_features']), np.array(test_data_2['Merged_features']), np.array(train_data_2['y']), np.array(test_data_2['y'])
X_train_3, X_test_3, y_train_3, y_test_3 = np.array(train_data_3['Merged_features']), np.array(test_data_3['Merged_features']), np.array(train_data_3['y']), np.array(test_data_3['y'])
X_train_4, X_test_4, y_train_4, y_test_4 = np.array(train_data_4['Merged_features']), np.array(test_data_4['Merged_features']), np.array(train_data_4['y']), np.array(test_data_4['y'])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_0_scaled = scaler.fit_transform(X_train_0) ; X_test_0_scaled = scaler.transform(X_test_0)
scaler = StandardScaler()
X_train_1_scaled = scaler.fit_transform(X_train_1) ; X_test_1_scaled = scaler.transform(X_test_1)
scaler = StandardScaler()
X_train_2_scaled = scaler.fit_transform(X_train_2) ; X_test_2_scaled = scaler.transform(X_test_2)
scaler = StandardScaler()
X_train_3_scaled = scaler.fit_transform(X_train_3) ; X_test_3_scaled = scaler.transform(X_test_3)
scaler = StandardScaler()
X_train_4_scaled = scaler.fit_transform(X_train_4) ; X_test_4_scaled = scaler.transform(X_test_4)


y_train_0 = np.ravel(y_train_0) ; y_test_0 = np.ravel(y_test_0)
y_train_1 = np.ravel(y_train_1) ; y_test_1 = np.ravel(y_test_1)
y_train_2 = np.ravel(y_train_2) ; y_test_2 = np.ravel(y_test_2)
y_train_3 = np.ravel(y_train_3) ; y_test_3 = np.ravel(y_test_3)
y_train_4 = np.ravel(y_train_4) ; y_test_4 = np.ravel(y_test_4)

model_final_0 = ElasticNet(alpha=0.010, l1_ratio=0.5)
model_final_1 = ElasticNet(alpha=0.010, l1_ratio=0.5)
model_final_2 = ElasticNet(alpha=0.010, l1_ratio=0.5)
model_final_3 = ElasticNet(alpha=0.010, l1_ratio=0.5)
model_final_4 = ElasticNet(alpha=0.010, l1_ratio=0.5)

model_final_0.fit(X_train_0_scaled, y_train_0)
model_final_1.fit(X_train_1_scaled, y_train_1)
model_final_2.fit(X_train_2_scaled, y_train_2)
model_final_3.fit(X_train_3_scaled, y_train_3)
model_final_4.fit(X_train_4_scaled, y_train_4)

y_pred_0 = model_final_0.predict(X_test_0_scaled)
y_pred_1 = model_final_1.predict(X_test_1_scaled)
y_pred_2 = model_final_2.predict(X_test_2_scaled)
y_pred_3 = model_final_3.predict(X_test_3_scaled)
y_pred_4 = model_final_4.predict(X_test_4_scaled)

pred_list = list(y_pred_0) + list(y_pred_1) + list(y_pred_2) + list(y_pred_3) + list(y_pred_4)
ans_list = list(y_test_0) + list(y_test_1) + list(y_test_2) + list(y_test_3) + list(y_test_4)

Pcor_0, Scor_0, mse_0 = get_res(y_test_0, y_pred_0)
Pcor_1, Scor_1, mse_1 = get_res(y_test_1, y_pred_1)
Pcor_2, Scor_2, mse_2 = get_res(y_test_2, y_pred_2)
Pcor_3, Scor_3, mse_3 = get_res(y_test_3, y_pred_3)
Pcor_4, Scor_4, mse_4 = get_res(y_test_4, y_pred_4)

ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'
RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
RF_result['Pcor'] = [Pcor_0, Pcor_1, Pcor_2, Pcor_3, Pcor_4]
RF_result['Scor'] = [Scor_0, Scor_1, Scor_2, Scor_3, Scor_4]
RF_result['mse'] = [mse_0, mse_1, mse_2, mse_3, mse_4]

RF_result.to_csv(ML_path+'leaveCombi.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

mse = np.mean((np.array(ans_list) - np.array(pred_list)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_list))

CfI = st.t.interval(alpha=0.90, df=len(pred_list)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)

RF_result2 = pd.DataFrame({'pred' : pred_list, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCombi_value.csv')





# leave cell out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []


for cell_num in range(34) :
    print(cell_num)
    train_data, test_data = prepare_data_GCN_cellout(cell_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) ; X_test_scaled = scaler.transform(X_test)
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = ElasticNet(alpha=0.010, l1_ratio=0.5)
    model_final.fit(X_train_scaled, y_train)
    y_pred = model_final.predict(X_test_scaled)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveCell.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCell_value.csv')













# leave drugs out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []


for cv_num in range(5) :
    print(cv_num)
    train_data, test_data = prepare_data_GCN_cidout(cv_num, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) ; X_test_scaled = scaler.transform(X_test)
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = ElasticNet(alpha=0.010, l1_ratio=0.5)
    model_final.fit(X_train_scaled, y_train)
    y_pred = model_final.predict(X_test_scaled)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveCID.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveCID_value.csv')







# leave tissue out 
ML_path = '/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/ElasticNet/'

RF_result = pd.DataFrame(columns = ['Pcor','Scor','mse'])
pred_results = []
ans_list = []


tissue_list = ['PROSTATE', 'BREAST', 'LARGE_INTESTINE', 'LUNG', 'OVARY', 'SKIN', 'PLEURA']

for tissue in tissue_list :
    train_data, test_data = prepare_data_GCN_tisout(tissue, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) ; X_test_scaled = scaler.transform(X_test)
    y_train = np.ravel(y_train) ; y_test = np.ravel(y_test)
    model_final = ElasticNet(alpha=0.010, l1_ratio=0.5)
    model_final.fit(X_train_scaled, y_train)
    y_pred = model_final.predict(X_test_scaled)
    Pcor_0, Scor_0, mse_0 = get_res(y_test, y_pred)
    tmp_df = pd.DataFrame({
        'Pcor' : [Pcor_0],'Scor' : [Scor_0], 
        'mse' : [mse_0] })
    RF_result = pd.concat([RF_result, tmp_df])
    pred_results = pred_results + list(y_pred)
    ans_list = ans_list + list(y_test)


RF_result.to_csv(ML_path+'leaveTis.csv')

np.round(np.mean(RF_result.mse), 4)
np.round(np.std(RF_result.mse), 4)

len(pred_results)
len(ans_list)

mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
se_mse = np.sqrt(2 * mse * mse / len(pred_results))

CfI = st.t.interval(alpha=0.90, df=len(pred_results)-1,
                loc=mse,
                scale=se_mse)

np.round(CfI, 4)

np.round(np.mean(RF_result.Pcor), 4)
np.round(np.std(RF_result.Pcor), 4)

np.round(np.mean(RF_result.Scor), 4)
np.round(np.std(RF_result.Scor), 4)


RF_result2 = pd.DataFrame({'pred' : pred_results, 'ans' : ans_list})
RF_result2.to_csv(ML_path+'leaveTis_value.csv')











