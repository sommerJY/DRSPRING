heatmap 진짜 그려봐야하나 고민 


from matplotlib import colors as mcolors
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib

NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


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

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'

file_name = 'M3V6_349_MISS2_FULL' # 0608

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


MISS_filter = ['AOBO','AXBO','AOBX', 'AXBX'] # ,'AXBO','AOBX','AXBX'

A_B_C_S_SET = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.ONEIL == 'O']




CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
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



# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_names.sort()

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['CELL'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')



CELL_CUT = 200 ####### 이것도 그렇게 되면 바꿔야하지 않을까 ##################################################################################################################


C_freq_filter = C_df[C_df.freq > CELL_CUT ]


A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.CELL)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)



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
A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re, on = 'CELL', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())



print("LEARNING")

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 182174

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


# check 

with open(PRJ_PATH+'CV_SM_list.pickle', 'rb') as f:
	CV_PICKLE = pickle.load(f)





# 주어진 환경 맞춰서 5CV만 돌릴거라서 수정버전 
def prepare_data_GCN(CV_num, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
cell_one_hot, MY_syn_RE2, norm ) :
		# 
		# CV_num = 0
		train_key = 'CV{}_train'.format(CV_num)
		test_key = 'CV{}_test'.format(CV_num)
		# 
		#
		ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[train_key])]
		ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS[test_key])]
		#
		#train_ind = list(ABCS_train.index)
		#val_ind = list(ABCS_val.index)
		tv_ind = list(ABCS_tv.index)
		random.shuffle(tv_ind)
		test_ind = list(ABCS_test.index)
		# 
		chem_feat_A_tv = MY_chem_A_feat_RE2[tv_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
		chem_feat_B_tv = MY_chem_B_feat_RE2[tv_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
		chem_adj_A_tv = MY_chem_A_adj_RE2[tv_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
		chem_adj_B_tv = MY_chem_B_adj_RE2[tv_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
		gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
		gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
		target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
		target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
		cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
		cell_tv = cell_one_hot[tv_ind];  cell_test = cell_one_hot[test_ind]
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
		tv_data['drug1_adj'] = torch.concat([chem_adj_A_tv, chem_adj_B_tv], axis = 0)
		test_data['drug1_adj'] = chem_adj_A_test
		#
		tv_data['drug2_adj'] = torch.concat([chem_adj_B_tv, chem_adj_A_tv], axis = 0)
		test_data['drug2_adj'] = chem_adj_B_test
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
		##
		tv_data['cell'] = torch.concat((cell_tv, cell_tv), axis=0)
		test_data['cell'] = cell_test
		#            
		tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
		test_data['y'] = syn_test
		#
		print(tv_data['drug1_feat'].shape, flush=True)
		print(test_data['drug1_feat'].shape, flush=True)
		return tv_data, test_data




class DATASET_GCN_W_FT(Dataset):
		def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ,
		gcn_gene_A, gcn_gene_B, target_A, target_B, cell_basal, gcn_adj, gcn_adj_weight,
		cell_info, syn_ans ):
				self.gcn_drug1_F = gcn_drug1_F
				self.gcn_drug2_F = gcn_drug2_F
				self.gcn_drug1_ADJ = gcn_drug1_ADJ
				self.gcn_drug2_ADJ = gcn_drug2_ADJ
				self.gcn_gene_A = gcn_gene_A
				self.gcn_gene_B = gcn_gene_B
				self.target_A = target_A
				self.target_B = target_B
				self.cell_basal = cell_basal
				self.gcn_adj = gcn_adj
				self.gcn_adj_weight = gcn_adj_weight
				self.syn_ans = syn_ans
				self.cell_info = cell_info
				#
		#
		def __len__(self):
				return len(self.gcn_drug1_F)
						#
		def __getitem__(self, index):
				adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
				adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
				#
				FEAT_A = torch.Tensor(np.array([ self.gcn_gene_A[index].squeeze().tolist() , self.target_A[index].tolist(), self.cell_basal[index].tolist()]).T)
				FEAT_B = torch.Tensor(np.array([ self.gcn_gene_B[index].squeeze().tolist() , self.target_B[index].tolist(), self.cell_basal[index].tolist()]).T)
				#
				return self.gcn_drug1_F[index], self.gcn_drug2_F[index],adj_re_A, adj_re_B, FEAT_A, FEAT_B, self.gcn_adj, self.gcn_adj_weight , self.cell_info[index], self.syn_ans[index]




def graph_collate_fn(batch):
		drug1_f_list = []
		drug2_f_list = []
		drug1_adj_list = []
		drug2_adj_list = []
		expA_list = []
		expB_list = []
		exp_adj_list = []
		exp_adj_w_list = []
		y_list = []
		cell_list = []
		EXP_num_nodes_seen = 0
		DRUG_1_num_nodes_seen = 0
		DRUG_2_num_nodes_seen = 0
		#
		for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
				drug1_f_list.append(drug1_f)
				drug2_f_list.append(drug2_f)
				drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
				drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
				expA_list.append(expA)
				expB_list.append(expB)
				exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
				exp_adj_w_list.append(exp_adj_w)
				y_list.append(torch.Tensor(y))
				cell_list.append(torch.Tensor(cell))
				EXP_num_nodes_seen += expA.shape[0]
				DRUG_1_num_nodes_seen += drug1_f.shape[0]
				DRUG_2_num_nodes_seen += drug2_f.shape[0]
		#
		drug1_f_new = torch.cat(drug1_f_list, 0)
		drug2_f_new = torch.cat(drug2_f_list, 0)
		drug1_adj_new = torch.cat(drug1_adj_list, 1)
		drug2_adj_new = torch.cat(drug2_adj_list, 1)
		expA_new = torch.cat(expA_list, 0)
		expB_new = torch.cat(expB_list, 0)
		exp_adj_new = torch.cat(exp_adj_list, 1)
		exp_adj_w_new = torch.cat(exp_adj_w_list, 1)
		y_new = torch.stack(y_list, 0)
		cell_new = torch.stack(cell_list, 0)
		return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, cell_new, y_new


def weighted_mse_loss(input, target, weight):
		return (weight * (input - target) ** 2).mean()


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

# CV_0
train_data_0, test_data_0 = prepare_data_GCN(0, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
cell_one_hot, MY_syn_RE2, norm)

# CV_1
train_data_1, test_data_1 = prepare_data_GCN(1, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
cell_one_hot, MY_syn_RE2, norm)

# CV_2
train_data_2, test_data_2 = prepare_data_GCN(2, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
cell_one_hot, MY_syn_RE2, norm)

# CV_3
train_data_3, test_data_3 = prepare_data_GCN(3, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
cell_one_hot, MY_syn_RE2, norm)

# CV_4
train_data_4, test_data_4 = prepare_data_GCN(4, A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
cell_one_hot, MY_syn_RE2, norm)



# WEIGHT 
def get_loss_weight(CV) :
		train_data = globals()['train_data_'+str(CV)]
		ys = train_data['y'].squeeze().tolist()
		min_s = np.amin(ys)
		loss_weight = np.log(train_data['y'] - min_s + np.e)
		return loss_weight

LOSS_WEIGHT_0 = get_loss_weight(0)
LOSS_WEIGHT_1 = get_loss_weight(1)
LOSS_WEIGHT_2 = get_loss_weight(2)
LOSS_WEIGHT_3 = get_loss_weight(3)
LOSS_WEIGHT_4 = get_loss_weight(4)

JY_IDX_WEIGHT_T = torch.Tensor(JY_IDX_WEIGHT).view(1,-1)



# DATA check  
def make_merged_data(CV) :
		train_data = globals()['train_data_'+str(CV)]
		test_data = globals()['test_data_'+str(CV)]
		#
		T_train = DATASET_GCN_W_FT(
				torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']),
				torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
				torch.Tensor(train_data['GENE_A']), torch.Tensor(train_data['GENE_B']),
				torch.Tensor(train_data['TARGET_A']), torch.Tensor(train_data['TARGET_B']), torch.Tensor(train_data['cell_BASAL']),
				JY_ADJ_IDX, JY_IDX_WEIGHT_T,
				train_data['cell'].float(),
				torch.Tensor(train_data['y'])
				)
		#
		#       
		T_test = DATASET_GCN_W_FT(
				torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']),
				torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
				torch.Tensor(test_data['GENE_A']), torch.Tensor(test_data['GENE_B']),
				torch.Tensor(test_data['TARGET_A']), torch.Tensor(test_data['TARGET_B']), torch.Tensor(test_data['cell_BASAL']),
				JY_ADJ_IDX, JY_IDX_WEIGHT_T,
				test_data['cell'].float(),
				torch.Tensor(test_data['y'])
				)
		#
		return T_train, T_test




# CV 0 
T_train_0, T_test_0 = make_merged_data(0)
#RAY_train_0 = ray.put(T_train_0)
RAY_test_0 = ray.put(T_test_0)
#RAY_loss_weight_0 = ray.put(LOSS_WEIGHT_0)


# CV 1
T_train_1, T_test_1 = make_merged_data(1)
#RAY_train_1 = ray.put(T_train_1)
RAY_test_1 = ray.put(T_test_1)
#RAY_loss_weight_1 = ray.put(LOSS_WEIGHT_1)


# CV 2 
T_train_2, T_test_2 = make_merged_data(2)
#RAY_train_2 = ray.put(T_train_2)
RAY_test_2 = ray.put(T_test_2)
#RAY_loss_weight_2 = ray.put(LOSS_WEIGHT_2)


# CV 3
T_train_3, T_test_3 = make_merged_data(3)
#RAY_train_3 = ray.put(T_train_3)
RAY_test_3 = ray.put(T_test_3)
#RAY_loss_weight_3 = ray.put(LOSS_WEIGHT_3)


# CV 4
T_train_4, T_test_4 = make_merged_data(4)
#RAY_train_4 = ray.put(T_train_4)
RAY_test_4 = ray.put(T_test_4)
#RAY_loss_weight_4 = ray.put(LOSS_WEIGHT_4)



class MY_expGCN_parallel_model(torch.nn.Module):
		def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem,
		G_layer_exp, G_indim_exp, G_hiddim_exp,
		layers_1, layers_2, layers_3, cell_dim ,
		out_dim, inDrop, drop):
				super(MY_expGCN_parallel_model, self).__init__()
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
				self.cell_dim = cell_dim
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
				self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1]+self.cell_dim , self.layers_3[0] )])
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
		def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, cell, syn ):
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
				input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
				input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
				#
				for L1 in range(len(self.Convs_1)):
						if L1 != len(self.Convs_1)-1 :
								input_drug1 = self.Convs_1[L1](input_drug1)
								input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
								input_drug1 = F.relu(input_drug1)
						else :
								input_drug1 = self.Convs_1[L1](input_drug1)
				#
				for L2 in range(len(self.Convs_1)):
						if L2 != len(self.Convs_1)-1 :
								input_drug2 = self.Convs_1[L2](input_drug2)
								input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
								input_drug2 = F.relu(input_drug2)
						else :
								input_drug2 = self.Convs_1[L2](input_drug2)
				#
				X = torch.cat(( input_drug1, input_drug2, cell ), 1)
				for L3 in range(len(self.SNPs)):
						if L3 != len(self.SNPs)-1 :
								X = self.SNPs[L3](X)
								X = F.dropout(X, p=self.drop, training = self.training)
								X = F.relu(X)
						else :
								X = self.SNPs[L3](X)
				return X



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





def inner_test( TEST_DATA, THIS_MODEL , use_cuda = False) :
	THIS_MODEL.eval()
	#
	running_loss = 0
	last_loss = 0 
	#
	ans_list = []
	pred_list = []
	with torch.no_grad() :
		for batch_idx_v, (drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(TEST_DATA) :
			expA = expA.view(-1,3)#### 다른점 
			expB = expB.view(-1,3)#### 다른점 
			adj_w = adj_w.squeeze()
			# move to GPU
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			## update the average validation loss
			output = THIS_MODEL(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y)
			MSE = torch.nn.MSELoss()
			loss = MSE(output, y) # train 이 아니라서 weight 안넣어줌. 그냥 nn.MSE 넣어주기 
			# update average validation loss 
			running_loss = running_loss + loss.item()
			pred_list = pred_list + output.squeeze().tolist()
			ans_list = ans_list + y.squeeze().tolist()
		#
	last_loss = running_loss / (batch_idx_v+1)
	val_sc, _ = stats.spearmanr(pred_list, ans_list)
	val_pc, _ = stats.pearsonr(pred_list, ans_list)
	return last_loss, val_pc, val_sc, pred_list, ans_list    






def TEST_CPU (PRJ_PATH, CV_num, my_config, model_path, model_name, model_num) :
	use_cuda = False
	#
	CV_test_dict = { 
		'CV_0': T_test_0, 'CV_1' : T_test_1, 'CV_2' : T_test_2,
		'CV_3' : T_test_3, 'CV_4': T_test_4 }
	#
	T_test = CV_test_dict[CV_num]
	test_loader = torch.utils.data.DataLoader(T_test, batch_size = my_config["config/batch_size"].item(), collate_fn = graph_collate_fn, shuffle =False, num_workers=16) # my_config['config/n_workers'].item()
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
				len(set(A_B_C_S_SET_SM.CELL)), 1,
				inDrop, Drop
				)
	#
	if torch.cuda.is_available():
		best_model = best_model.cuda()
		print('model to cuda', flush = True)
		if torch.cuda.device_count() > 1 :
			best_model = torch.nn.DataParallel(best_model)
			print('model to multi cuda', flush = True)
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
		best_model.load_state_dict(state_dict)	#
	print("state_load_done", flush = True)
	#
	#
	last_loss, val_pc, val_sc, pred_list, ans_list = inner_test(test_loader, best_model)
	R__1 , R__2 = jy_corrplot(pred_list, ans_list, PRJ_PATH, 'P.{}.{}.{}.{}_model'.format(PRJ_NAME, MISS_NAME, CV_num, model_num) )
	return  last_loss, R__1, R__2, pred_list, ans_list




def TEST_CPU (PRJ_PATH, CV_num, my_config, model_path, model_name, model_num) :
	use_cuda = False
	#
	cv = CV_num.split('_')[1]
	CV_test_dict = { 
		'CV_0': T_test_0, 'CV_1' : T_test_1, 'CV_2' : T_test_2,
		'CV_3' : T_test_3, 'CV_4': T_test_4 }
	#
	T_test = CV_test_dict[CV_num]
	test_loader = torch.utils.data.DataLoader(T_test, batch_size = 256, collate_fn = graph_collate_fn, shuffle =False) # my_config['config/n_workers'].item()
	#
	best_model = globals()['the_model_{}'.format(cv)]
	last_loss, val_pc, val_sc, pred_list, ans_list = inner_test(test_loader, best_model)
	R__1 , R__2 = jy_corrplot(pred_list, ans_list, PRJ_PATH, 'P.{}.{}.{}.{}_model'.format(PRJ_NAME, MISS_NAME, CV_num, model_num) )
	return  last_loss, R__1, R__2, pred_list, ans_list


stats.pearsonr(a, b)






PRJ_NAME = 'M3V6'
W_NAME = 'W202'
MJ_NAME = 'M3V6'
MISS_NAME = 'MIS2'
PPI_NAME = '349'
WORK_NAME = 'WORK_202' # 349###################################################################################################


#PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/' 
ANA_DF = pd.read_csv(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.csv'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
with open(PRJ_PATH+'RAY_ANA_DF.{}_{}_{}_{}.pickle'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME), 'rb') as f:
	ANA_ALL_DF = pickle.load(f)


epc_result = pd.read_csv(PRJ_PATH+"RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))


my_config = ANA_DF.loc[0]




R_0_T_CV0, R_0_1_CV0, R_0_2_CV0, pred_0_CV0, ans_0_CV0 = TEST_CPU(PRJ_PATH, 'CV_0', my_config, PRJ_PATH, 'full_CV_0_model.pth', 'full')
R_0_T_CV1, R_0_1_CV1, R_0_2_CV1, pred_0_CV1, ans_0_CV1 = TEST_CPU(PRJ_PATH, 'CV_1', my_config, PRJ_PATH, 'full_CV_1_model.pth', 'full')
R_0_T_CV2, R_0_1_CV2, R_0_2_CV2, pred_0_CV2, ans_0_CV2 = TEST_CPU(PRJ_PATH, 'CV_2', my_config, PRJ_PATH, 'full_CV_2_model.pth', 'full')
R_0_T_CV3, R_0_1_CV3, R_0_2_CV3, pred_0_CV3, ans_0_CV3 = TEST_CPU(PRJ_PATH, 'CV_3', my_config, PRJ_PATH, 'full_CV_3_model.pth', 'full')
R_0_T_CV4, R_0_1_CV4, R_0_2_CV4, pred_0_CV4, ans_0_CV4 = TEST_CPU(PRJ_PATH, 'CV_4', my_config, PRJ_PATH, 'full_CV_4_model.pth', 'full')


ABCS_test_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]
ABCS_test_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_test'])]
ABCS_test_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_test'])]
ABCS_test_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_test'])]
ABCS_test_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_test'])]

ABCS_test_concat = pd.concat([ABCS_test_0, ABCS_test_1, ABCS_test_2, ABCS_test_3, ABCS_test_4])
ABCS_test_concat['PRED'] = pred_0_CV0 + pred_0_CV1 + pred_0_CV2 + pred_0_CV3 + pred_0_CV4
ABCS_test_concat['ANS'] = ans_0_CV0 + ans_0_CV1 + ans_0_CV2 + ans_0_CV3 + ans_0_CV4
ABCS_test_concat['DIFF'] = abs(ABCS_test_concat.ANS - ABCS_test_concat.PRED)
ABCS_test_concat['D_level'] = ABCS_test_concat.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )
ABCS_test_concat['tissue'] = ABCS_test_concat.CELL.apply( lambda x : ''.join(x.split('_')[1:]) )



tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


cells_info = pd.DataFrame(
	data = np.inf,
	columns=['tissue'],
	index=list(set(ABCS_test_concat.CELL))
)

cells_info['tissue'] = ['_'.join(a.split('_')[1:]) for a in list(set(ABCS_test_concat.CELL))]


my_heatmap_dot = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(ABCS_test_concat.CELL)),
	index=list(set(ABCS_test_concat.CID_CID))
)

c_c_c_list = list(ABCS_test_concat.cid_cid_cell)
c_c_c_list_set = list(set(ABCS_test_concat.cid_cid_cell))

for c_ind in range(len(c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
		datetime.now() 
	c_c_c = c_c_c_list_set[c_ind]
	tmp_res = ABCS_test_concat[ABCS_test_concat.cid_cid_cell==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		tmp_result = tmp_res['DIFF'].item()
	else  :
		tmp_result = np.mean(tmp_res['DIFF'])
	# 
	if tmp_result < 5 : 
		my_heatmap_dot.at[c_c, c] = "under5"
	elif  tmp_result < 10 : 
		my_heatmap_dot.at[c_c, c] = "under10"
	else :
		my_heatmap_dot.at[c_c, c] = "over10"
	


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }

tissue_map = pd.DataFrame(my_heatmap_dot.columns, columns= ['cell'])
tissue_map['tissue'] = ['_'.join(a.split('_')[1:]) for a in tissue_map.cell]
tissue_map['col'] = tissue_map['tissue'].map(color_dict)

col_colors = list(tissue_map['col'])


# row color 1) 
tanimoto_map = pd.DataFrame(my_heatmap_dot.index, columns = ['cid_cid'])
tani_tmp = ABCS_test_concat[['cid_cid_cell','tani_Q']]
tani_tmp['cid_cid'] = tani_tmp.cid_cid_cell.apply(lambda x : '___'.join(x.split('___')[0:2]))
tani_tmp = tani_tmp[['cid_cid','tani_Q']].drop_duplicates()
tanimoto_map2 = pd.merge(tanimoto_map, tani_tmp, on = 'cid_cid', how = 'left' )
tanimoto_map2['col'] = ['#0DD9FE' if a == 'O' else '#383b3c' for a in list(tanimoto_map2.tani_Q)]

row_colors = list(tanimoto_map2['col'])


value_to_int = {j:i for i,j in enumerate(pd.unique(my_heatmap_dot.values.ravel()))} # like you did
n = len(value_to_int)     
cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]


gg = sns.clustermap(
	my_heatmap_dot.replace(value_to_int),  cmap=cmap, 
	figsize=(20,20),
	row_cluster=True, col_cluster = True, 
	metric = 'correlation', method = 'complete',
	col_colors = col_colors, row_colors = row_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example.W124.pdf", bbox_inches='tight')
plt.close()





















ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_train'])]
ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]


			PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

			ALL_TEST_RESULTS = pd.read_csv(PRJ_PATH + 'TEST_RESULTS.5CV.csv', sep = '\t')

			FILT_TEST_RES = ALL_TEST_RESULTS[['DrugCombCCLE','type', 'cell_onehot','ANS','pred_5_CV0', 'pred_5_CV1', 'pred_5_CV2', 'pred_5_CV3', 'pred_5_CV4']]

			FILT_TEST_RES['tissue'] = [a.split('_')[1] for a in ALL_TEST_RESULTS.DrugCombCCLE]
			FILT_TEST_RES['PRED'] = ALL_TEST_RESULTS[['pred_5_CV0', 'pred_5_CV1', 'pred_5_CV2', 'pred_5_CV3', 'pred_5_CV4']].mean(axis =1)
			FILT_TEST_RES2 = FILT_TEST_RES[['DrugCombCCLE','tissue','type', 'cell_onehot','ANS','PRED']]


# tanimoto 실험때문에 이거 넣게됨 

A_B_C_S_SET_ADD_tmp = copy.deepcopy(A_B_C_S_SET_ADD)

aaa = list(A_B_C_S_SET_ADD_tmp['CID_A'])
bbb = list(A_B_C_S_SET_ADD_tmp['CID_B'])
aa = list(A_B_C_S_SET_ADD_tmp['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_ADD_tmp['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_ADD_tmp['CELL'])

A_B_C_S_SET_ADD_tmp['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] if aa[i] < bb[i] else bb[i] + '___' + aa[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_ADD.shape[0])]
A_B_C_S_SET_ADD_tmp['CID_CID'] = [aa[i] + '___' + bb[i]  if aa[i] < bb[i] else bb[i] + '___' + aa[i] for i in range(A_B_C_S_SET_ADD.shape[0])]


ABCS_test_result = A_B_C_S_SET_ADD_tmp[['CELL','type','CID_CID', 'cid_cid_cell','tani_Q','ONEIL']][0:184331]
ABCS_test_result['tissue'] = [''.join(a.split('_')[1:]) for a in ABCS_test_result.CELL]
ABCS_test_result['ANS'] = ans_0_CV0 + ans_0_CV1 + ans_0_CV2 + ans_0_CV3 + ans_0_CV4
ABCS_test_result['PRED'] = pred_0_CV0 + pred_0_CV1 + pred_0_CV2 + pred_0_CV3 + pred_0_CV4
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}



cells_info = pd.DataFrame(
	data = np.inf,
	columns=['tissue'],
	index=list(set(ABCS_test_result.DrugCombCCLE))
)

cells_info['tissue'] = ['_'.join(a.split('_')[1:]) for a in list(set(ABCS_test_result.DrugCombCCLE))]



# 이번엔 real version
# 그래도 안이뻐서 오닐만 그려보기로 함 
ABCS_test_result = A_B_C_S_SET_SM[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE', 'ONEIL']][0:184331]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = ans_0_CV0 + ans_0_CV1 + ans_0_CV2 + ans_0_CV3 + ans_0_CV4
ABCS_test_result['PRED'] = pred_0_CV0 + pred_0_CV1 + pred_0_CV2 + pred_0_CV3 + pred_0_CV4
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )
ABCS_test_result = ABCS_test_result[ABCS_test_result.ONEIL=='O']



my_heatmap_dot = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(ABCS_test_result.DrugCombCCLE)),
	index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
		datetime.now() 
	c_c_c = c_c_c_list_set[c_ind]
	tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		tmp_result = tmp_res['DIFF'].item()
	else  :
		tmp_result = np.mean(tmp_res['DIFF'])
	# 
	if tmp_result < 5 : 
		my_heatmap_dot.at[c_c, c] = "under5"
	elif  tmp_result < 10 : 
		my_heatmap_dot.at[c_c, c] = "under10"
	else :
		my_heatmap_dot.at[c_c, c] = "over10"
	

# my_heatmap_dot.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/test.csv', index =False, sep ='\t')


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


tissue_map = pd.DataFrame(my_heatmap_dot.columns, columns= ['cell'])
tissue_map['tissue'] = ['_'.join(a.split('_')[1:]) for a in tissue_map.cell]
tissue_map['col'] = tissue_map['tissue'].map(color_dict)

col_colors = list(tissue_map['col'])


#fig = plt.subplots()

value_to_int = {j:i for i,j in enumerate(pd.unique(my_heatmap_dot.values.ravel()))} # like you did
n = len(value_to_int)     
# cmap = sns.color_palette("Pastel2", n) 
cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]

gg = sns.clustermap(
	my_heatmap_dot.replace(value_to_int),  cmap=cmap, 
	figsize=(20,20),
	row_cluster=True, col_cluster = True, 
	metric = 'cosine', method = 'complete',
	col_colors = col_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example1.on.pdf", bbox_inches='tight')
plt.close()

어째...... 왜 갈리는게 구리냐 
차라리 예측결과값으로 하는게 나을듯?














































my_heatmap = pd.DataFrame(
	data = 0, #np.inf,
	columns=list(set(ABCS_test_result.DrugCombCCLE)),
	index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : 
	if c_ind%100 == 0 : 
		print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
		datetime.now() 
	c_c_c = c_c_c_list_set[c_ind]
	tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		my_heatmap.at[c_c, c] = tmp_res['DIFF'].item()
	else  :
		my_heatmap.at[c_c, c] = np.mean(tmp_res['DIFF'])
	


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }


my_map_skin = my_heatmap[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap[tiss_cell_dict['PLEURA']]



cell_bar = pd.DataFrame(
	data = np.inf,
	columns = ['DIFF_M'],
	index = list(set(ABCS_test_result.DrugCombCCLE))
)

for a in cell_bar.index :
	meanval = np.mean(ABCS_test_result[ABCS_test_result.DrugCombCCLE==a]['DIFF'])
	cell_bar.at[a,'DIFF_M'] = meanval





col_ha = HeatmapAnnotation( TISSUE=anno_simple(cells_info.tissue, add_text=False,legend=True), axis=1,
						   MEAN=anno_barplot(cell_bar, legend=True, colors = [color_dict[tt] for tt in cells_info.tissue] ),
						   legend=True, legend_gap=5, hgap=0.5) # plot=True, 


my_heatmap2 = pd.concat([
	my_map_skin, my_map_breast, my_map_lung, my_map_ovary, my_map_Lins,
	my_map_nerv, my_map_kidn, my_map_hema, my_map_prot, my_map_bone, my_map_pleu], axis = 1)

my_heatmap3 = my_heatmap2.sort_values(list(my_heatmap2.columns))


plt.figure(figsize=(5.5, 6.5))
my_cm = ClusterMapPlotter(data=my_heatmap3, top_annotation=col_ha,
					   col_cluster=False, row_cluster=False,
					   # col_split=cells_info.tissue,
					   # col_split_gap=0.5,
					   # row_dendrogram=True, # label='values',
					   show_rownames=False, show_colnames=True,
					   verbose=0, legend_gap=5,  # tree_kws={'row_cmap': 'Set1'}, 
					   cmap='YlGn', xticklabels_kws={'labelrotation':-90,'labelcolor':'black'})
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')











# 아예 dot plot 으로 시도 

my_heatmap_dot = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(ABCS_test_result.DrugCombCCLE)),
	index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : 
	if c_ind%100 == 0 : 
		print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
		datetime.now() 
	c_c_c = c_c_c_list_set[c_ind]
	tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		tmp_result = tmp_res['DIFF'].item()
	else  :
		tmp_result = np.mean(tmp_res['DIFF'])
	# 
	if tmp_result < 5 : 
		my_heatmap_dot.at[c_c, c] = "under5"
	elif  tmp_result < 10 : 
		my_heatmap_dot.at[c_c, c] = "under10"
	else :
		my_heatmap_dot.at[c_c, c] = "over10"
	

my_map_skin = my_heatmap_dot[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot[tiss_cell_dict['PLEURA']]



cell_bar = pd.DataFrame(
	data = np.inf,
	columns = ['DIFF_M'],
	index = list(set(ABCS_test_result.DrugCombCCLE))
)

for a in cell_bar.index :
	meanval = np.mean(ABCS_test_result[ABCS_test_result.DrugCombCCLE==a]['DIFF'])
	cell_bar.at[a,'DIFF_M'] = meanval




colors = [color_dict[tt] for tt in cells_info.tissue]
col_ha = HeatmapAnnotation( TISSUE=anno_simple(cells_info.tissue, add_text=False,legend=True), axis=1,
						   MEAN=anno_barplot(cell_bar, legend=True ),
						   legend=True, legend_gap=5, hgap=0.5) # plot=True, 


my_heatmap2 = pd.concat([
	my_map_skin, my_map_breast, my_map_lung, my_map_ovary, my_map_Lins,
	my_map_nerv, my_map_kidn, my_map_hema, my_map_prot, my_map_bone, my_map_pleu], axis = 1)

my_heatmap3 = my_heatmap2.sort_values(list(my_heatmap2.columns), ascending = False)




my_dc = DotClustermapPlotter(
	data=my_heatmap3, 
	top_annotation=col_ha,
	cmap={'under5':'Reds','under10':'Purples','over10':'Blues'},
	colors={'under5':'red','under10':'purple','over10':'green'},
	#col_split_gap=0.5, #row_dendrogram=True col_split=11, 
)












ABCS_test_result2 = copy.deepcopy(ABCS_test_result)




c_c_c_list = list(ABCS_test_result2.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result2.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
		datetime.now() 
	#
	c_c_c = c_c_c_list_set[c_ind]
	tmp_res = ABCS_test_result2[ABCS_test_result2.CID_CID_CCLE==c_c_c]
	if tmp_res.shape[0] > 0 :
		new_diff = np.mean(tmp_res['DIFF'])
		indices = list(tmp_res.index)
		for ind in indices :
			ABCS_test_result2.at[ind, 'DIFF'] = new_diff


ABCS_test_result3 = ABCS_test_result2[['DrugCombCCLE', 'CID_CID', 'CID_CID_CCLE', 'tissue','DIFF']].drop_duplicates()
ABCS_test_result3['D_level'] = ABCS_test_result3.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )


my_dc = DotClustermapPlotter(
	data=ABCS_test_result3, x = 'CID_CID', y = 'DrugCombCCLE', value='DIFF',
	#top_annotation=col_ha,
	hue='D_level', #cmap={'under5':'Reds','under10':'Purples','over10':'Blues'},
	#colors={'under5':'red','under10':'purple','over10':'green'},
	#col_split_gap=0.5, #row_dendrogram=True col_split=11, 
)





# type 때문에 더 잘나왔을수도 있음... 아닌가 그래도 어차피 순서 고려까지 아니더라도 적당히 smiles 로 나눴으니까 괜찮나? 
# 고오민 










# 

onco로 마지막으로한번만 더 

cols=['under5', 'under10', 'over10']
colors=["red","blue","#008000"]

ABCS_test_result3['under5'] = ABCS_test_result3.D_level.apply(lambda x: 1 if x == 'under5' else 0)
ABCS_test_result3['under10'] = ABCS_test_result3.D_level.apply(lambda x:1 if x == 'under10' else 0)
ABCS_test_result3['over10'] = ABCS_test_result3.D_level.apply(lambda x:1 if x == 'over10' else 0)

row_vc=ABCS_test_result3.groupby('CID_CID').apply(lambda x:x.loc[:,cols].sum())
col_vc=ABCS_test_result3.groupby('DrugCombCCLE').apply(lambda x:x.loc[:,cols].sum())


plt.figure(figsize=(12,8))

my_cp = oncoPrintPlotter(
	data=ABCS_test_result3, x = 'CID_CID', y = 'DrugCombCCLE', 
	values =cols, 
	colors = colors,
   
)


plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')


df=pd.DataFrame(ABCS_test_result3[cols].iloc[1,:].values.tolist()).apply(lambda x:x/x.sum(),axis=1).fillna(0)


 #value='DIFF',
	#top_annotation=col_ha,
	#cmap={'under5':'Reds','under10':'Purples','over10':'Blues'},
	#colors={'under5':'red','under10':'purple','over10':'green'},
	#col_split_gap=0.5, #row_dendrogram=True col_split=11, 





아 진자 진심으로 모든 조합에 대한 내용이 없어서 그러니 
생각해보면 지금 이 조합이 test 에 대한거만 하고 있는데, 전체가 들어오면 그럴 필요가 없을지도?
아닌가 그래도 채워야하나 


all_ccle = list(set(ABCS_test_result3.DrugCombCCLE))
all_cidcid= list(set(ABCS_test_result3.CID_CID))

items = [all_ccle, all_cidcid]
from itertools import product
item_list = list(product(*items))

CCLE = [a for a,b in item_list]
CID_CID = [b for a,b in item_list]

ABCS_test_result4 = pd.DataFrame({'CID' : CID_CID, 'CCLE' : CCLE})
ABCS_test_result4['under5'] = 0
ABCS_test_result4['under10'] = 0
ABCS_test_result4['over10'] = 0
ABCS_test_result4['result'] = 0




for ind in range(len(item_list)) :
	if ind%1000 == 0 : 
		print(str(ind)+'/'+str(len(item_list)) )
		datetime.now() 
	a = item_list[ind][0]
	b = item_list[ind][1]
	tmp = ABCS_test_result3[ (ABCS_test_result3.DrugCombCCLE == a) & (ABCS_test_result3.CID_CID == b) ]
	if tmp.shape[0] == 1 :
		ABCS_test_result4.at[ind, 'under5'] = tmp['under5'].item()
		ABCS_test_result4.at[ind, 'under10'] = tmp['under10'].item()
		ABCS_test_result4.at[ind, 'over10'] = tmp['over10'].item()
		#print(tmp.shape)
	elif tmp.shape[0] > 1 :
		print(ind)


											a = 'A375_SKIN'
											b = '3385___46926350'

cols=['under5', 'under10', 'over10', 'result']
colors=["#ff647e","#ffc1cb","#ffe28a", '#f5f8fb']


row_vc=ABCS_test_result4.groupby('CID').apply(lambda x:x.loc[:,cols].sum())
col_vc=ABCS_test_result4.groupby('CCLE').apply(lambda x:x.loc[:,cols].sum())


# 문제는 전체를 다 그리려고 하면 뭔가 python 으로 하려는 과정에서 문제가 생긴다는것...
# 아 이거 하나 그리려고 R 을 써야하나 진짜 
# 왜자꾸 inf 문제가 생기는지 알수가 없네
# 뭐든 예시를 가져가야할것 같은디요



tmp = ABCS_test_result4[ABCS_test_result4.CCLE.isin(tiss_cell_dict['SKIN'])]

plt.figure(figsize=(12,24)) 

my_cp = oncoPrintPlotter(
	data=tmp, x = 'CCLE', y = 'CID', 
	values =cols, 
	colors = colors  
)

plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')














fig, ax = plt.subplots(1, 1, figsize=(8, 6))

vegetables = ["cucumber", "tomato", "lettuce", "asparagus",
			  "potato", "wheat", "barley"]
farmers = ["Farmer Joe", "Upland Bros.", "Smith Gardening",
		   "Agrifun", "Organiculture", "BioGoods Ltd.", "Cornylee Corp."]

harvest = np.array([[0.8, 2.4, 2.5, 3.9, 0.0, 4.0, 0.0],
					[2.4, 0.0, 4.0, 1.0, 2.7, 0.0, 0.0],
					[1.1, 2.4, 0.8, 4.3, 1.9, 4.4, 0.0],
					[0.6, 0.0, 0.3, 0.0, 3.1, 0.0, 0.0],
					[0.7, 1.7, 0.6, 2.6, 2.2, 6.2, 0.0],
					[1.3, 1.2, 0.0, 0.0, 0.0, 3.2, 5.1],
					[0.1, 2.0, 0.0, 1.4, 0.0, 1.9, 6.3]])


im, _ = heatmap(harvest, vegetables, farmers, ax=ax,
				cmap="Wistia", cbarlabel="harvest [t/year]")
annotate_heatmap(im, valfmt="{x:.1f}", size=7)


annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
				 textcolors=("red", "black"))


corr_matrix = np.corrcoef(harvest)
im, _ = heatmap(corr_matrix, vegetables, vegetables, ax=ax4,
				cmap="PuOr", vmin=-1, vmax=1,
				cbarlabel="correlation coeff.")


def func(x, pos):
	return f"{x:.2f}".replace("0.", ".").replace("1.00", "")

annotate_heatmap(im, valfmt=matplotlib.ticker.FuncFormatter(func), size=7)


plt.tight_layout()
plt.show()









def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
	#
	if not ax:
		ax = plt.gca()
	#
	# Plot the heatmap
	im = ax.imshow(data, **kwargs)
	#
	# Create colorbar
	cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
	cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
	#
	# Show all ticks and label them with the respective list entries.
	ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
	ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)
	#
	# Let the horizontal axes labeling appear on top.
	ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
	#
	# Rotate the tick labels and set their alignment.
	plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
	#
	# Turn spines off and create white grid.
	ax.spines[:].set_visible(False)
	#
	ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
	ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
	ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
	ax.tick_params(which="minor", bottom=False, left=False)
	#
	return im, cbar



def annotate_heatmap(im, data=None, valfmt="{x:.2f}", textcolors=("black", "white"), threshold=None, **textkw):
	#
	if not isinstance(data, (list, np.ndarray)):
		data = im.get_array()
	#
	# Normalize the threshold to the images color range.
	if threshold is not None:
		threshold = im.norm(threshold)
	else:
		threshold = im.norm(data.max()) / 2.
	#
	# Set default alignment to center, but allow it to be
	# overwritten by textkw.
	kw = dict(horizontalalignment="center",
			  verticalalignment="center")
	kw.update(textkw)
	#
	# Get the formatter in case a string is supplied
	if isinstance(valfmt, str):
		valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)
	#
	# Loop over the data and create a `Text` for each "pixel".
	# Change the text's color depending on the data.
	texts = []
	for i in range(data.shape[0]):
		for j in range(data.shape[1]):
			kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
			text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
			texts.append(text)
	#
	return texts    


def func(x, pos):
	return "{:.2f}".format(x).replace("0.", ".").replace("1.00", "")




my_heatmap3



fig, ax = plt.subplots(1, 1, figsize=(20, 20))

		data = np.random.randn(6, 6)
		y = ["Prod. {}".format(i) for i in range(10, 70, 10)]
		x = ["Cycle {}".format(i) for i in range(1, 7)]

data = my_heatmap3
y = list(my_heatmap3.index)
x = list(my_heatmap3.columns)



qrates = list("ABCDEFG")
norm = matplotlib.colors.BoundaryNorm(np.linspace(-3.5, 3.5, 8), 7)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: qrates[::-1][norm(x)])

im, _ = heatmap(my_heatmap3, y, x, ax=ax,
				cmap=plt.get_cmap("PiYG", 4), norm=norm,
				cbar_kw=dict(ticks=np.arange(-3, 4), format=fmt),
				cbarlabel="Quality Rating")

annotate_heatmap(im, valfmt=fmt, size=9, fontweight="bold", threshold=-1,
				 textcolors=("red", "black"))




plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')








import matplotlib.pyplot as plt

fig,ax = plt.subplots(1,1,figsize=(18,8))
my_colors=[(0.2,0.3,0.3),(0.4,0.5,0.4),(0.1,0.7,0),(0.1,0.7,0)]

sns.heatmap(dt_tweet_cnt, cmap=my_colors, square=True, linewidth=0.1, linecolor=(0.1,0.2,0.2), ax=ax)

colorbar = ax.collections[0].colorbar
M=dt_tweet_cnt.max().max()
colorbar.set_ticks([1/8*M,3/8*M,6/8*M])
colorbar.set_ticklabels(['low','med','high'])

plt.show()



for 
my_heatmap3

value_to_int = {j:i for i,j in enumerate(pd.unique(df.values.ravel()))} # like you did
n = len(value_to_int)     






















def heatmap_binary(df,
			edgecolors='w',
			#cmap=mpl.cm.RdYlGn,
			log=False):    
	width = len(df.columns)/7*10
	height = len(df.index)/7*10
	#
	fig, ax = plt.subplots(figsize=(20,10))#(figsize=(width,height))
	#
	cmap, norm = mcolors.from_levels_and_colors([0, 0.05, 1],['Teal', 'MidnightBlue'] ) # ['MidnightBlue', Teal]['Darkgreen', 'Darkred']

	heatmap = ax.pcolor(df ,
						edgecolors=edgecolors,  # put white lines between squares in heatmap
						cmap=cmap,
						norm=norm)
	data = df.values
	for y in range(data.shape[0]):
		for x in range(data.shape[1]):
			plt.text(x + 0.5 , y + 0.5, '%.4f' % data[y, x], #data[y,x] +0.05 , data[y,x] + 0.05
				 horizontalalignment='center',
				 verticalalignment='center',
				 color='w')


	ax.autoscale(tight=True)  # get rid of whitespace in margins of heatmap
	ax.set_aspect('equal')  # ensure heatmap cells are square
	ax.xaxis.set_ticks_position('top')  # put column labels at the top
	ax.tick_params(bottom='off', top='off', left='off', right='off')  # turn off ticks

	ax.set_yticks(np.arange(len(df.index)) + 0.5)
	ax.set_yticklabels(df.index, size=20)
	ax.set_xticks(np.arange(len(df.columns)) + 0.5)
	ax.set_xticklabels(df.columns, rotation=90, size= 15)

	# ugliness from http://matplotlib.org/users/tight_layout_guide.html
	from mpl_toolkits.axes_grid1 import make_axes_locatable
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", "3%", pad="1%")
	fig.colorbar(heatmap, cax=cax)

df1 = pd.DataFrame(np.random.choice([0, 0.75], size=(4,5)), columns=list('ABCDE'), index=list('WXYZ'))
heatmap_binary(df1)












# 아예 dot plot 으로 시도 




ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_train'])]
ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]


ABCS_test_result = ABCS_test[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE']]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = np.random.uniform(-10,10, 35931)
ABCS_test_result['PRED'] = np.random.uniform(-10,10, 35931)
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )


# test 에 들어간 애들로만 봐서 지금 제대로 안갈리는것 같음 (전부)

ABCS_test_result = A_B_C_S_SET_SM[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE']]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = np.random.uniform(-10,10, 184503)
ABCS_test_result['PRED'] = np.random.uniform(-10,10, 184503)
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )



# 그래도 안이뻐서 오닐만 그려보기로 함 
ABCS_test_result = A_B_C_S_SET_SM[['DrugCombCCLE','type','CID_CID', 'CID_CID_CCLE', 'ONEIL']][0:184331]
ABCS_test_result['tissue'] = [a.split('_')[1] for a in ABCS_test_result.DrugCombCCLE]
ABCS_test_result['ANS'] = ans_0_CV0 + ans_0_CV1 + ans_0_CV2 + ans_0_CV3 + ans_0_CV4
ABCS_test_result['PRED'] = pred_0_CV0 + pred_0_CV1 + pred_0_CV2 + pred_0_CV3 + pred_0_CV4
ABCS_test_result['DIFF'] = abs(ABCS_test_result.ANS - ABCS_test_result.PRED)
ABCS_test_result['D_level'] = ABCS_test_result.DIFF.apply(lambda x:'under5' if x<5 else 'under10' if x < 10 else 'over10' if x < 100 else 'NA' )
ABCS_test_result = ABCS_test_result[ABCS_test_result.ONEIL=='O']







my_heatmap_dot = pd.DataFrame(
	data = "NA", #np.inf,
	columns=list(set(ABCS_test_result.DrugCombCCLE)),
	index=list(set(ABCS_test_result.CID_CID))
)

c_c_c_list = list(ABCS_test_result.CID_CID_CCLE)
c_c_c_list_set = list(set(ABCS_test_result.CID_CID_CCLE))

for c_ind in range(len(c_c_c_list_set)) : # 대충 18만줄 대상으로 30분 소요 
	if c_ind%1000 == 0 : 
		print(str(c_ind)+'/'+str(len(c_c_c_list_set)) )
		datetime.now() 
	c_c_c = c_c_c_list_set[c_ind]
	tmp_res = ABCS_test_result[ABCS_test_result.CID_CID_CCLE==c_c_c]
	c_c = c_c_c.split('___')[0] + '___'+ c_c_c.split('___')[1]
	c = c_c_c.split('___')[2]
	if tmp_res.shape[0] ==1 :
		tmp_result = tmp_res['DIFF'].item()
	else  :
		tmp_result = np.mean(tmp_res['DIFF'])
	# 
	if tmp_result < 5 : 
		my_heatmap_dot.at[c_c, c] = "under5"
	elif  tmp_result < 10 : 
		my_heatmap_dot.at[c_c, c] = "under10"
	else :
		my_heatmap_dot.at[c_c, c] = "over10"
	

# my_heatmap_dot.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/test.csv', index =False, sep ='\t')


tissue_order = ['SKIN', 'BREAST', 'LUNG', 'OVARY', 'LARGE_INTESTINE', 'CENTRAL_NERVOUS_SYSTEM', 'KIDNEY', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','PROSTATE', 'BONE', 'PLEURA']
tiss_cell_dict ={a : list(cells_info[cells_info.tissue==a].index) for a in tissue_order }


my_map_skin = my_heatmap_dot[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot[tiss_cell_dict['PLEURA']]






# seaborn 으로 해결? 

# row index

my_heatmap_dot_c = my_heatmap_dot.replace(value_to_int)
nach = my_heatmap_dot_c.sum(axis=1)
new_ind = nach.sort_values(ascending = False).index
my_heatmap_dot_c = my_heatmap_dot_c.loc[new_ind]


# col index 만드는 중이었음 
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc
from scipy.spatial.distance import pdist

linked = linkage(my_map_skin, 'complete')

gg = sns.clustermap(my_map_skin, row_cluster=True, col_cluster = True) 

my_map_skin = my_heatmap_dot_c[tiss_cell_dict['SKIN']]
my_map_breast = my_heatmap_dot_c[tiss_cell_dict['BREAST']]
my_map_lung = my_heatmap_dot_c[tiss_cell_dict['LUNG']]
my_map_ovary = my_heatmap_dot_c[tiss_cell_dict['OVARY']]
my_map_Lins = my_heatmap_dot_c[tiss_cell_dict['LARGE_INTESTINE']]
my_map_nerv = my_heatmap_dot_c[tiss_cell_dict['CENTRAL_NERVOUS_SYSTEM']]
my_map_kidn = my_heatmap_dot_c[tiss_cell_dict['KIDNEY']]
my_map_hema = my_heatmap_dot_c[tiss_cell_dict['HAEMATOPOIETIC_AND_LYMPHOID_TISSUE']]
my_map_prot = my_heatmap_dot_c[tiss_cell_dict['PROSTATE']]
my_map_bone = my_heatmap_dot_c[tiss_cell_dict['BONE']]
my_map_pleu = my_heatmap_dot_c[tiss_cell_dict['PLEURA']]





cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]

fig, ax = plt.subplots(1,1,figsize=(18,18))
sns.heatmap(my_heatmap_dot_c, ax = ax, cmap=cmap) 
colorbar = ax.collections[0].colorbar 
r = colorbar.vmax - colorbar.vmin 
colorbar.set_ticks([colorbar.vmin + r / n * (0.5 + i) for i in range(n)])
colorbar.set_ticklabels(list(value_to_int.keys()))            


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example0.pdf", bbox_inches='tight')
plt.close()








# seaborn 으로 해결? 2 

my_heatmap_dot

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


tissue_map = pd.DataFrame(my_heatmap_dot.columns, columns= ['cell'])
tissue_map['tissue'] = ['_'.join(a.split('_')[1:]) for a in tissue_map.cell]
tissue_map['col'] = tissue_map['tissue'].map(color_dict)

col_colors = list(tissue_map['col'])


#fig = plt.subplots()

value_to_int = {j:i for i,j in enumerate(pd.unique(my_heatmap_dot.values.ravel()))} # like you did
n = len(value_to_int)     
# cmap = sns.color_palette("Pastel2", n) 
cmap = [mcolors.to_rgba('#f5f8fb'),mcolors.to_rgba('#ffe28a'),mcolors.to_rgba("#ffc1cb"),mcolors.to_rgba("#ff647e")]

gg = sns.clustermap(
	my_heatmap_dot.replace(value_to_int),  cmap=cmap, 
	figsize=(20,20),
	row_cluster=True, col_cluster = True, 
	col_colors = col_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example1.on.pdf", bbox_inches='tight')
plt.close()





# skin 이 지랄이라서 skin 만 빼고 돌려보기 

my_heatmap_dot2 = my_heatmap_dot.drop(tiss_cell_dict['SKIN'], axis =1)
my_heatmap_dot3 = my_heatmap_dot2.replace(value_to_int)
nach = my_heatmap_dot3.sum(axis=1)
naind = [a for a in my_heatmap_dot3.index if nach[a] != 0]
my_heatmap_dot4 = my_heatmap_dot3.loc[naind]

gg = sns.clustermap(
	my_heatmap_dot4,  cmap=cmap, 
	figsize=(20,20),
	row_cluster=True, col_cluster = True, 
	col_colors = col_colors, dendrogram_ratio=0.2, yticklabels=False) 


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/example2.pdf", bbox_inches='tight')
plt.close()




