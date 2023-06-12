
import pandas as pd 
import torch 
import copy 
import networkx as nx
import numpy as np
import sklearn




NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 


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
#	ID_G.add_node(nn)

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






MJ_NAME = 'M3V5'
WORK_DATE = '23.05.26' # 349 ###################################################################################################
MISS_NAME = 'MIS2'

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'

file_name = 'M3V5_349_MISS2_FULL'

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)

MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))

A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)

A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)
aaa = list(A_B_C_S_SET_ADD2['drug_row_CID'])
bbb = list(A_B_C_S_SET_ADD2['drug_col_CID'])
aa = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_ADD2['DrugCombCCLE'])
aaaa = [a if type(a) == str else 'NA' for a in list(A_B_C_S_SET_ADD2['ROW_BETA_sig_id'])]
bbbb = [a if type(a) == str else 'NA' for a in list(A_B_C_S_SET_ADD2['COL_BETA_sig_id'])]

A_B_C_S_SET_ADD2['CID_CID_CCLE'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + cc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + cc[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] if aa[i] < bb[i] else bb[i] + '___' + aa[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SIG_SIG'] = [aaaa[i] + '___' + bbbb[i] if aaaa[i] < bbbb[i] else bbbb[i] + '___' + aaaa[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]

DC_dup_check = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/DC_duplicates.csv', sep ='\t') 
DC_dup_list = list(DC_dup_check['id_id_cell'])

A_B_C_S_SET_ADD2_dup_check = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.CID_CID_CCLE.isin(DC_dup_list)]

check_dc_info = list(A_B_C_S_SET_ADD2_dup_check.CID_CID_CCLE)

# 내꺼에서는 딱히 dup 가 아닐수도 (이미 모종의 이유로 지워져서? ) if tmp.shape[0] != len(set(tmp.SIG_SIG)) :
rm_index = []
syn_prob = []
for a in check_dc_info :
	tmp = A_B_C_S_SET_ADD2_dup_check[A_B_C_S_SET_ADD2_dup_check.CID_CID_CCLE == a]
	sig_set = list(set(tmp.SIG_SIG))
	#print(len(sig_set))
	for i in sig_set :
		tmp_index = list(tmp[tmp.SIG_SIG == i].index)
		if MY_syn[tmp_index[0]].item() * MY_syn[tmp_index[1]].item() <0 :
			syn_prob.append(a)
			rm_index = rm_index + tmp_index
	tmp2 = tmp[['CID_CID_CCLE','SIG_SIG']].drop_duplicates(keep='last')
	rm_index = rm_index + list(tmp2.index)


A_B_C_S_SET_rmdup = A_B_C_S_SET_ADD2.drop(rm_index) # 302039

A_B_C_S_SET_rmdup[['CID_CID_CCLE','SIG_SIG']].drop_duplicates() # 302039 # CID-SIG 중복인것만 남음
A_B_C_S_SET_rmdup[['CID_CID_CCLE','SM_C_CHECK','SIG_SIG']].drop_duplicates() # 302039
A_B_C_S_SET_rmdup[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 301232
A_B_C_S_SET_rmdup[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() # 300997
len(set(A_B_C_S_SET_rmdup['CID_CID_CCLE'])) # 51212
len(set(A_B_C_S_SET_rmdup['SM_C_CHECK'])) # 51160




MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET = A_B_C_S_SET_rmdup[A_B_C_S_SET_rmdup.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']

## A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] ###################### old targets 
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]

CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ori_col = list( ccle_exp.columns ) # entrez!
for_gene = ori_col[1:]
for_gene2 = [int(a.split('(')[1].split(')')[0]) for a in for_gene]
new_col = ['DepMap_ID']+for_gene2 
ccle_exp.columns = new_col

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
ccle_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.DrugCombCCLE.isin(ccle_names)]

data_ind = list(A_B_C_S_SET.index)

MY_syn_RE = MY_syn[data_ind]
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)


# cell line vector 


DC_CELL_DF2 = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/'+'DC_CELL_INFO.csv', sep = '\t')

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.DrugCombCCLE)] # 38

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname']], on = 'DrugCombCCLE', how = 'left'  )


# sample number filter 

# 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['DrugCombCCLE'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')


CELL_CUT = 200 ##################################################################################

C_freq_filter = C_df[C_df.freq > CELL_CUT ] 


A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.DrugCombCCLE)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)

data_ind = list(A_B_C_S_SET_COH.index)

MY_syn_RE2 = MY_syn_RE[data_ind]


# merge 전 후로 index 달라지므로 뒤에 넣어줬음 
A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())










print("LEARNING")

A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 182174

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })
data_nodup_df2 = data_nodup_df.sort_values('cell')
data_nodup_df2 = data_nodup_df2.reset_index(drop =True) # 181419

grouped_df = data_nodup_df2.groupby('cell')


CV_1_list = []; CV_2_list = []; CV_3_list = []; CV_4_list = []; CV_5_list = []

for i, g in grouped_df:
	nums = int(.2 * len(g)) 
	bins = []
	g2 = sklearn.utils.shuffle(g, random_state=42)
	for ii in list(range(0, len(g2), nums)):
		if len(bins)< 5 :
			bins.append(ii)
	#
	bins = bins[1:]
	res = np.split(g2, bins)
	print(i)
	print(len(g2))
	len(set(sum([list(ii.setset) for ii in res],[])))
	len(set(sum([list(ii.setset) for ii in res],[]))) == len(sum([list(ii.setset) for ii in res],[]))
	CV_1_list = CV_1_list + res[0].index.tolist()
	CV_2_list = CV_2_list + res[1].index.tolist()
	CV_3_list = CV_3_list + res[2].index.tolist()
	CV_4_list = CV_4_list + res[3].index.tolist()
	CV_5_list = CV_5_list + res[4].index.tolist()





CV_ND_INDS = {
	'CV0_train' : CV_1_list + CV_2_list + CV_3_list + CV_4_list, 
	'CV0_val' : CV_5_list, 
	'CV1_train' : CV_2_list + CV_3_list + CV_4_list + CV_5_list , 
	'CV1_val' : CV_1_list, 
	'CV2_train' : CV_3_list + CV_4_list + CV_5_list + CV_1_list, 
	'CV2_val' : CV_2_list, 
	'CV3_train' : CV_4_list + CV_5_list + CV_1_list + CV_2_list,
	'CV3_val' : CV_3_list, 
	'CV4_train' : CV_5_list + CV_1_list + CV_2_list + CV_3_list,
	'CV4_val' : CV_4_list, 
}




CV_num = 0
train_key = 'CV{}_train'.format(CV_num)
val_key = 'CV{}_val'.format(CV_num)
#
train_no_dup = data_nodup_df2.loc[CV_ND_INDS[train_key]] # train val df 
val_no_dup = data_nodup_df2.loc[CV_ND_INDS[val_key]] # train val df 
#
ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(train_no_dup.setset)]
ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(val_no_dup.setset)]
#
train_ind = list(ABCS_train.index)
val_ind = list(ABCS_val.index)







