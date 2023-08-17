# Matchmaker 
# 다시 여기부터 시작인가 
# tensorflow 아니었나 
Python 3.7
Numpy 1.18.1
Scipy 1.4.1
Pandas 1.0.1
Tensorflow 2.1.0
Tensorflow-gpu 2.1.0
Scikit-Learn 0.22.1
keras-metrics 1.1.0
h5py 2.10.0
cudnn 7.6.5 (for gpu support only)



# main code 
  
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
import numpy as np
import sys


import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from helper_funcs import normalize, progress
import torch
import copy
import sklearn
import networkx as nx 

TOOL_PATH = '/home01/k040a01/04.MatchMaker/'

NETWORK_PATH = '/home01/k040a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k040a01/01.Data/LINCS/'
DC_PATH = '/home01/k040a01/01.Data/DrugComb/'



# HS Drug pathway DB 활용 -> 349
print('NETWORK')

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


SAVE_PATH = '/home01/k040a01/02.M3V6/M3V6_349_DATA/'
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'

file_name = 'M3V6_349_MISS2_ONEIL' # 0608

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
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

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O'] # 11639

#A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] # 8086 -> 이걸 빼야하나 말아야하나 #################

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

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left' )


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
no_TF_CID = [104842, 208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 24748204, 3062316, 216239, 3385, 5288382, 5311, 59691338, 60750, 5329102, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
no_TF_CELL = ['LOVO', 'A375', 'HT29', 'OVCAR3', 'SW-620', 'SK-OV-3', 'MDAMB436', 'NCIH23', 'RKO', 'UACC62', 'A2780', 'VCAP', 'A427', 'T-47D', 'ES2', 'PA1', 'RPMI7951', 'SKMES1', 'NCIH2122', 'HT144', 'NCIH1650', 'SW837', 'OV90', 'UWB1289', 'HCT116', 'A2058', 'NCIH520']


ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(no_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(no_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(no_TF_CELL)]


##### yes target filter version 

yes_TF_CID = [208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 3062316, 216239, 3385, 5288382, 5311, 60750, 5329102, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
yes_TF_CELL = ['LOVO', 'A375', 'HT29', 'OVCAR3', 'SW-620', 'SK-OV-3', 'MDAMB436', 'NCIH23', 'RKO', 'UACC62', 'A2780', 'VCAP', 'A427', 'T-47D', 'ES2', 'PA1', 'RPMI7951', 'SKMES1', 'NCIH2122', 'HT144', 'NCIH1650', 'SW837', 'OV90', 'UWB1289', 'HCT116', 'A2058', 'NCIH520']

ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(yes_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(yes_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(yes_TF_CELL)]









A_B_C_S_SET_COH = copy.deepcopy(ON_filt_3)

data_ind = list(A_B_C_S_SET_COH.index)
MY_syn_RE2 = MY_syn_RE[data_ind]


A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())

A_B_C_S_SET_COH2['syn_ans'] = MY_syn_RE2.squeeze().tolist()





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





###########################################################################################
###########################################################################################
###########################################################################################

# 일단 생 5CV orderr 

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


# 지금 다른 코드들 구조를 생각하면 이런식으로 해야하나 고민 
CV_ND_INDS = {
	'CV0_train' : CV_1_setset + CV_2_setset + CV_3_setset +CV_4_setset, 
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



A_B_C_S_SET_SM['CID_CID_CELL'] = A_B_C_S_SET_SM.CID_CID +"___"+ A_B_C_S_SET_SM.DC_cellname

ABCS_tv_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_train'])]
ABCS_test_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]

ABCS_tv_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_train'])]
ABCS_test_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_test'])]

ABCS_tv_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_train'])]
ABCS_test_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_test'])]

ABCS_tv_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_train'])]
ABCS_test_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_test'])]

ABCS_tv_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_train'])]
ABCS_test_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_test'])]


DRSPRING_CID = list(set(list(A_B_C_S_SET_SM.CID_A) + list(A_B_C_S_SET_SM.CID_B))) # 1342
DRSPRING_CELL = list(set(A_B_C_S_SET_SM.DC_cellname))

DC_DRUG_DF_FULL = pd.read_csv(DC_PATH+'DC_DRUG_DF_PC.csv', sep ='\t')



########################## MM 



MM_DATA = pd.read_csv(TOOL_PATH + 'DrugCombinationData.tsv', sep = '\t')
CEL_DATA =pd.read_csv(TOOL_PATH + 'E-MTAB-3610.sdrf.txt', sep = '\t')



def data_loader(drug1_chemicals, drug2_chemicals, cell_line_gex, comb_data_name):
	print("File reading ...")
	comb_data = pd.read_csv(comb_data_name, sep="\t")
	cell_line = pd.read_csv(cell_line_gex,header=None)
	chem1 = pd.read_csv(drug1_chemicals,header=None)
	chem2 = pd.read_csv(drug2_chemicals,header=None)
	synergies = np.array(comb_data["synergy_loewe"])
	#
	cell_line = np.array(cell_line.values)
	chem1 = np.array(chem1.values)
	chem2 = np.array(chem2.values)
	return chem1, chem2, cell_line, synergies


with open(TOOL_PATH + 'drugs_info.json') as json_file :
	MM_drug_info =json.load(json_file)


MM_DATA = TOOL_PATH + 'DrugCombinationData.tsv'
MM_comb_data = pd.read_csv(MM_DATA, sep="\t")

drug1 = TOOL_PATH + 'data/drug1_chem.csv'
drug2 = TOOL_PATH + 'data/drug2_chem.csv'
CEL_gex = TOOL_PATH + 'data/cell_line_gex.csv'

chem1, chem2, cell_line, synergies = data_loader(drug1, drug2, CEL_gex, MM_DATA )

chem1.shape # (286421, 541)
chem2.shape # (286421, 541)
cell_line.shape # (286421, 972)
synergies.shape # (286421,)



##### match 확인 

							set(MM_comb_data.cell_line_name) - set(DC_CELL_DF2.DC_cellname) # 786-O
							set(MM_comb_data.cell_line_name) - set(DRSPRING_CELL)
							set(DRSPRING_CELL) - set(MM_comb_data.cell_line_name)
							# {'SKMEL30', 'ZR751', 'MSTO', 'CAOV3', 'DLD1', 'NCI-H460', 'KPL1'}


							MM_comb_drug = list(set(list(MM_comb_data['drug_row']) + list(MM_comb_data['drug_col'])))
							set(MM_comb_drug) - set(DC_DRUG_DF_FULL.dname)

							set(DRSPRING_CELL) - set(DC_CELL_DF2.DC_cellname) # 0 




MM_comb_drug_match_name = list(set(list(MM_comb_data.drug_row) + list(MM_comb_data.drug_col))) # 3040
MM_drug_match = pd.DataFrame({'MM_drug' : MM_comb_drug_match_name  })


MM_drug_CID_name = {key : MM_drug_info[key]['name'] for key in MM_drug_info.keys()} # 3952
missing = [a for a in MM_comb_drug_match_name if a not in MM_drug_CID_name.values()] # 119?


MM_drug_CID_name_DF = pd.DataFrame.from_dict(MM_drug_CID_name, orient = 'index')
MM_drug_CID_name_DF['MM_CID'] = list(MM_drug_CID_name_DF.index)
MM_drug_CID_name_DF.columns = ['MM_drug', 'MM_CID']

MM_drug_match_2 = pd.merge(MM_drug_match, MM_drug_CID_name_DF, on='MM_drug', how='left')


for indind in range(MM_drug_match_2.shape[0]) :
	if type(list(MM_drug_match_2.MM_CID)[indind]) != str :
		MM_drug_match_2.at[indind,'MM_CID'] = '0'

MM_drug_match_2['MM_CID'] = [int(a) for a in MM_drug_match_2['MM_CID']]

MM_drug_match_ok = MM_drug_match_2[MM_drug_match_2.MM_CID>0] # 2921
MM_drug_match_miss = MM_drug_match_2[MM_drug_match_2.MM_CID==0] # 119



chem_feat_dict = {a : np.round(MM_drug_info[a]['chemicals'],4) for a in MM_drug_info.keys()}

drug_ind = MM_comb_data[MM_comb_data.drug_row == 'COSTUNOLIDE'].index[0].item()
chem1[drug_ind] # 541,




for indind in list(MM_drug_match_miss.index) : 
	drugname = MM_drug_match_miss.at[indind, 'MM_drug']
	drug_ind = MM_comb_data[MM_comb_data.drug_row == drugname].index[0].item()
	ans_feat = np.round(chem1[drug_ind].tolist(),4)
	#
	for cid in chem_feat_dict.keys() :
		tmp= []
		if all(ans_feat == chem_feat_dict[cid]) == True:
			MM_drug_match_miss.at[indind, 'MM_CID'] = int(cid )
			tmp.append(cid)
		if len(set(tmp)) > 1 :
			print(cid) # 겹치는 feat 없는걸로 
	   


MM_drug_match_miss[MM_drug_match_miss.MM_CID==0] # 전부 매치 


MM_drug_match_final = pd.concat([MM_drug_match_ok, MM_drug_match_miss])

check = MM_drug_match_final[MM_drug_match_final.MM_CID.duplicated(False)] # 다 보겠다는거 

check_cid = list(check.MM_CID)

for cid in check_cid :
	names = list(MM_drug_match_final[MM_drug_match_final.MM_CID == cid]['MM_drug'])
	if len(names) == 2 :
		drug_ind_1 = MM_comb_data[MM_comb_data.drug_row == names[0]].index[0].item()
		drug_ind_2 = MM_comb_data[MM_comb_data.drug_row == names[1]].index[0].item()
		if all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_2],4)) == False :
			print(cid)
	else :
		print(cid)


names = list(MM_drug_match_final[MM_drug_match_final.MM_CID == cid]['MM_drug'])
#drug_ind_1 = MM_comb_data[MM_comb_data.drug_row == names[0]].index[0].item()
#drug_ind_2 = MM_comb_data[MM_comb_data.drug_row == names[1]].index[0].item()
#drug_ind_3 = MM_comb_data[MM_comb_data.drug_row == names[2]].index[0].item()


#all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_2],4))
#all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_3],4))
#all(np.round(chem1[drug_ind_2],4) == np.round(chem1[drug_ind_3],4))



MM_drug_match_final.columns = ['drug_row','drug_row_cid']
MM_comb_data_RE = pd.merge(MM_comb_data, MM_drug_match_final, on ='drug_row', how = 'left')

MM_drug_match_final.columns = ['drug_col','drug_col_cid']
MM_comb_data_RE = pd.merge(MM_comb_data_RE, MM_drug_match_final, on ='drug_col', how = 'left')


#################################################
# match ours 


aaa = list(MM_comb_data_RE['drug_row_cid'])
bbb = list(MM_comb_data_RE['drug_col_cid'])
ccc = list(MM_comb_data_RE['cell_line_name'])

# 306
MM_comb_data_RE['CID_CID'] = [str(int(aaa[i])) + '___' + str(int(bbb[i])) if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i])) for i in range(MM_comb_data_RE.shape[0])]

# 10404 -- duplicated 가 이상한게 아님 
MM_comb_data_RE['CID_CID_CELL'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + ccc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + ccc[i] for i in range(MM_comb_data_RE.shape[0])]

set_data = list(set(MM_comb_data_RE['CID_CID_CELL']))
MM_comb_data_RE['ori_index'] = list(MM_comb_data_RE.index)


MM_comb_data_R2 = MM_comb_data_RE[['CID_CID_CELL','ori_index']]
rep_index = list(MM_comb_data_R2.CID_CID_CELL.drop_duplicates().index)
MM_comb_data_R2 = MM_comb_data_R2.loc[rep_index]




# CV 0 
JY_train = ABCS_tv_0 
JY_test = ABCS_test_0

# CV 1
JY_train = ABCS_tv_1
JY_test = ABCS_test_1

# CV 2
JY_train = ABCS_tv_0
JY_test = ABCS_test_0

# CV 3
JY_train = ABCS_tv_0
JY_test = ABCS_test_0

# CV 4 
JY_train = ABCS_tv_0
JY_test = ABCS_test_0



def get_data(JY_train, JY_test) :
	#
	test_idx = list(MM_comb_data_R2[MM_comb_data_R2.CID_CID_CELL.isin(JY_test.CID_CID_CELL)]['ori_index'])
	tr_idx = list(MM_comb_data_R2[MM_comb_data_R2.CID_CID_CELL.isin(JY_train.CID_CID_CELL)]['ori_index'])
	#
	ds_test_df = MM_comb_data_R2[MM_comb_data_R2.CID_CID_CELL.isin(JY_test.CID_CID_CELL)]
	ds_tr_df = MM_comb_data_R2[MM_comb_data_R2.CID_CID_CELL.isin(JY_train.CID_CID_CELL)]
	#
	ds_test_df_ccc = list(ds_test_df.CID_CID_CELL)
	ds_tr_df_ccc = list(ds_tr_df.CID_CID_CELL)
	#
	y_test = [JY_test[JY_test.CID_CID_CELL==a]['syn_ans'].item() for a in ds_test_df_ccc]
	y_tr = [JY_train[JY_train.CID_CID_CELL==a]['syn_ans'].item() for a in ds_tr_df_ccc]
	return tr_idx , test_idx, y_tr, y_test







# original 5cv 
tr_idx_0 , test_idx_0, y_tr_0, y_test_0 = get_data(ABCS_tv_0, ABCS_test_0)
tr_idx_1 , test_idx_1, y_tr_1, y_test_1 = get_data(ABCS_tv_1, ABCS_test_1)
tr_idx_2 , test_idx_2, y_tr_2, y_test_2 = get_data(ABCS_tv_2, ABCS_test_2)
tr_idx_3 , test_idx_3, y_tr_3, y_test_3 = get_data(ABCS_tv_3, ABCS_test_3)
tr_idx_4 , test_idx_4, y_tr_4, y_test_4 = get_data(ABCS_tv_4, ABCS_test_4)





# leave cell out 
CELL_0_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='LOVO'] ; CELL_0_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='LOVO']
CELL_1_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='A375']; CELL_1_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='A375']
CELL_2_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='HT29']; CELL_2_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='HT29']
CELL_3_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='OVCAR3']; CELL_3_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='OVCAR3']
CELL_4_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='SW-620']; CELL_4_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='SW-620']
CELL_5_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='SK-OV-3']; CELL_5_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='SK-OV-3']
CELL_6_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='MDAMB436']; CELL_6_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='MDAMB436']
CELL_7_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='NCIH23']; CELL_7_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='NCIH23']
CELL_8_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='RKO']; CELL_8_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='RKO']
CELL_9_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='UACC62']; CELL_9_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='UACC62']
CELL_10_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='A2780']; CELL_10_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='A2780']
CELL_11_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='VCAP']; CELL_11_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='VCAP']
CELL_12_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='A427']; CELL_12_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='A427']
CELL_13_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='T-47D']; CELL_13_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='T-47D']
CELL_14_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='ES2']; CELL_14_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='ES2']
CELL_15_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='PA1']; CELL_15_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='PA1']
CELL_16_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='RPMI7951']; CELL_16_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='RPMI7951']
CELL_17_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='SKMES1']; CELL_17_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='SKMES1']
CELL_18_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='NCIH2122']; CELL_18_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='NCIH2122']
CELL_19_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='HT144']; CELL_19_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='HT144']
CELL_20_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='NCIH1650']; CELL_20_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='NCIH1650']
CELL_21_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='SW837']; CELL_21_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='SW837']
CELL_22_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='OV90']; CELL_22_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='OV90']
CELL_23_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='UWB1289']; CELL_23_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='UWB1289']
CELL_24_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='HCT116']; CELL_24_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='HCT116']
CELL_25_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='A2058']; CELL_25_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='A2058']
CELL_26_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!='NCIH520']; CELL_26_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname=='NCIH520']


# cell 
tr_idx_C0, test_idx_C0, y_tr_C0, y_test_C0 = get_data(CELL_0_train, CELL_0_test)
tr_idx_C1, test_idx_C1, y_tr_C1, y_test_C1 = get_data(CELL_1_train, CELL_1_test)
tr_idx_C2, test_idx_C2, y_tr_C2, y_test_C2 = get_data(CELL_2_train, CELL_2_test)
tr_idx_C3, test_idx_C3, y_tr_C3, y_test_C3 = get_data(CELL_3_train, CELL_3_test)
tr_idx_C4, test_idx_C4, y_tr_C4, y_test_C4 = get_data(CELL_4_train, CELL_4_test)
tr_idx_C5, test_idx_C5, y_tr_C5, y_test_C5 = get_data(CELL_5_train, CELL_5_test)
tr_idx_C6, test_idx_C6, y_tr_C6, y_test_C6 = get_data(CELL_6_train, CELL_6_test)
tr_idx_C7, test_idx_C7, y_tr_C7, y_test_C7 = get_data(CELL_7_train, CELL_7_test)
tr_idx_C8, test_idx_C8, y_tr_C8, y_test_C8 = get_data(CELL_8_train, CELL_8_test)
tr_idx_C9, test_idx_C9, y_tr_C9, y_test_C9 = get_data(CELL_9_train, CELL_9_test)
tr_idx_C10, test_idx_C10, y_tr_C10, y_test_C10 = get_data(CELL_10_train, CELL_10_test)
tr_idx_C11, test_idx_C11, y_tr_C11, y_test_C11 = get_data(CELL_11_train, CELL_11_test)
tr_idx_C12, test_idx_C12, y_tr_C12, y_test_C12 = get_data(CELL_12_train, CELL_12_test)
tr_idx_C13, test_idx_C13, y_tr_C13, y_test_C13 = get_data(CELL_13_train, CELL_13_test)
tr_idx_C14, test_idx_C14, y_tr_C14, y_test_C14 = get_data(CELL_14_train, CELL_14_test)
tr_idx_C15, test_idx_C15, y_tr_C15, y_test_C15 = get_data(CELL_15_train, CELL_15_test)
tr_idx_C16, test_idx_C16, y_tr_C16, y_test_C16 = get_data(CELL_16_train, CELL_16_test)
tr_idx_C17, test_idx_C17, y_tr_C17, y_test_C17 = get_data(CELL_17_train, CELL_17_test)
tr_idx_C18, test_idx_C18, y_tr_C18, y_test_C18 = get_data(CELL_18_train, CELL_18_test)
tr_idx_C19, test_idx_C19, y_tr_C19, y_test_C19 = get_data(CELL_19_train, CELL_19_test)
tr_idx_C20, test_idx_C20, y_tr_C20, y_test_C20 = get_data(CELL_20_train, CELL_20_test)
tr_idx_C21, test_idx_C21, y_tr_C21, y_test_C21 = get_data(CELL_21_train, CELL_21_test)
tr_idx_C22, test_idx_C22, y_tr_C22, y_test_C22 = get_data(CELL_22_train, CELL_22_test)
tr_idx_C23, test_idx_C23, y_tr_C23, y_test_C23 = get_data(CELL_23_train, CELL_23_test)
tr_idx_C24, test_idx_C24, y_tr_C24, y_test_C24 = get_data(CELL_24_train, CELL_24_test)
tr_idx_C25, test_idx_C25, y_tr_C25, y_test_C25 = get_data(CELL_25_train, CELL_25_test)
tr_idx_C26, test_idx_C26, y_tr_C26, y_test_C26 = get_data(CELL_26_train, CELL_26_test)




# leave cid 
CID_0 = no_TF_CID[0:6]
CID_1 = no_TF_CID[6:12]
CID_2 = no_TF_CID[12:18]
CID_3 = no_TF_CID[18:24]
CID_4 = no_TF_CID[24:]

CID_0_train = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_0)==False) & (A_B_C_S_SET_SM.CID_B.isin(CID_0)==False) ]; 
CID_0_test = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_0)) | (A_B_C_S_SET_SM.CID_B.isin(CID_0))]

CID_1_train = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_1)==False) & (A_B_C_S_SET_SM.CID_B.isin(CID_1)==False) ]; 
CID_1_test = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_1)) | (A_B_C_S_SET_SM.CID_B.isin(CID_1))]

CID_2_train = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_2)==False) & (A_B_C_S_SET_SM.CID_B.isin(CID_2)==False) ]; 
CID_2_test = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_2)) | (A_B_C_S_SET_SM.CID_B.isin(CID_2))]

CID_3_train = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_3)==False) & (A_B_C_S_SET_SM.CID_B.isin(CID_3)==False) ]; 
CID_3_test = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_3)) | (A_B_C_S_SET_SM.CID_B.isin(CID_3))]

CID_4_train = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_4)==False) & (A_B_C_S_SET_SM.CID_B.isin(CID_4)==False) ]; 
CID_4_test = A_B_C_S_SET_SM[(A_B_C_S_SET_SM.CID_A.isin(CID_4)) | (A_B_C_S_SET_SM.CID_B.isin(CID_4))]


tr_idx_C0, test_idx_C0, y_tr_C0, y_test_C0 = get_data(CID_0_train, CID_0_test)
tr_idx_C1, test_idx_C1, y_tr_C1, y_test_C1 = get_data(CID_1_train, CID_1_test)
tr_idx_C2, test_idx_C2, y_tr_C2, y_test_C2 = get_data(CID_2_train, CID_2_test)
tr_idx_C3, test_idx_C3, y_tr_C3, y_test_C3 = get_data(CID_3_train, CID_3_test)
tr_idx_C4, test_idx_C4, y_tr_C4, y_test_C4 = get_data(CID_4_train, CID_4_test)







# leave tissue 

A_B_C_S_SET_SM['tissue'] = A_B_C_S_SET_SM.CELL.apply(lambda x : '_'.join(x.split('_')[1:]))

CELL_0_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='LARGE_INTESTINE'] ; CELL_0_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='LARGE_INTESTINE']
CELL_1_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='LUNG']; CELL_1_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='LUNG']
CELL_2_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='BREAST']; CELL_2_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='BREAST']
CELL_3_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='PROSTATE']; CELL_3_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='PROSTATE']
CELL_4_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='SKIN']; CELL_4_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='SKIN']
CELL_5_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='OVARY']; CELL_5_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='OVARY']

tr_idx_C0, test_idx_C0, y_tr_C0, y_test_C0 = get_data(CELL_0_train, CELL_0_test)
tr_idx_C1, test_idx_C1, y_tr_C1, y_test_C1 = get_data(CELL_1_train, CELL_1_test)
tr_idx_C2, test_idx_C2, y_tr_C2, y_test_C2 = get_data(CELL_2_train, CELL_2_test)
tr_idx_C3, test_idx_C3, y_tr_C3, y_test_C3 = get_data(CELL_3_train, CELL_3_test)
tr_idx_C4, test_idx_C4, y_tr_C4, y_test_C4 = get_data(CELL_4_train, CELL_4_test)
tr_idx_C5, test_idx_C5, y_tr_C5, y_test_C5 = get_data(CELL_5_train, CELL_5_test)












#############################################################


def normalize(X, means1=None, std1=None, means2=None, std2=None, 
	feat_filt=None, norm='tanh_norm'):
	if std1 is None:
		std1 = np.nanstd(X, axis=0) # nan 무시하고 표준편차 구하기 
	if feat_filt is None:
		feat_filt = std1!=0
	X = X[:,feat_filt]
	X = np.ascontiguousarray(X)
	if means1 is None:
		means1 = np.mean(X, axis=0)
	X = (X-means1)/std1[feat_filt]
	if norm == 'norm':
		return(X, means1, std1, feat_filt)
	elif norm == 'tanh':
		return(np.tanh(X), means1, std1, feat_filt)
	elif norm == 'tanh_norm':
		X = np.tanh(X)
		if means2 is None:
			means2 = np.mean(X, axis=0)
		if std2 is None:
			std2 = np.std(X, axis=0)
		X = (X-means2)/std2
		X[:,std2==0]=0
		return(X, means1, std1, means2, std2, feat_filt)


def pearson(y, pred):
	pear = stats.pearsonr(y, pred)
	pear_value = pear[0]
	pear_p_val = pear[1]
	print("Pearson correlation is {} and related p_value is {}".format(pear_value, pear_p_val))
	return pear_value

def spearman(y, pred):
	spear = stats.spearmanr(y, pred)
	spear_value = spear[0]
	spear_p_val = spear[1]
	print("Spearman correlation is {} and related p_value is {}".format(spear_value, spear_p_val))
	return spear_value

def mse(y, pred):
	err = mean_squared_error(y, pred)
	print("Mean squared error is {}".format(err))
	return err

def squared_error(y,pred):
	errs = []
	for i in range(y.shape[0]):
		err = (y[i]-pred[i]) * (y[i]-pred[i])
		errs.append(err)
	return np.asarray(errs)



def prepare_data(chem1, chem2, cell_line, train_syn, test_syn, norm, 
	train_ind, test_ind):
	print("Data normalization and preparation of train/validation/test data")
	#
	train_data = {}
	test_data = {}
	#
	# chem 1 이랑 2 를 붙인다고? 
	train1 = np.concatenate((chem1[train_ind,:],chem2[train_ind,:]),axis=0) # 엥 왜 
	train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
	test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train2 = np.concatenate((chem2[train_ind,:],chem1[train_ind,:]),axis=0)
	train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
	test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
	train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
	test_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(cell_line[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
	#
	train_data['drug1'] = np.concatenate((train_data['drug1'],train_cell_line),axis=1)
	train_data['drug2'] = np.concatenate((train_data['drug2'],train_cell_line),axis=1)
	#
	test_data['drug1'] = np.concatenate((test_data['drug1'],test_cell_line),axis=1)
	test_data['drug2'] = np.concatenate((test_data['drug2'],test_cell_line),axis=1)
	#
	train_data['y'] = np.concatenate((train_syn, train_syn),axis=0)
	test_data['y'] = np.array(test_syn)
	print(test_data['drug1'].shape)
	print(test_data['drug2'].shape)
	return train_data, test_data



norm = 'tanh_norm'
train_data_0, test_data_0 = prepare_data(chem1, chem2, cell_line, y_tr_0, y_test_0, norm, tr_idx_0 , test_idx_0)
train_data_1, test_data_1 = prepare_data(chem1, chem2, cell_line, y_tr_1, y_test_1, norm, tr_idx_1 , test_idx_1)
train_data_2, test_data_2 = prepare_data(chem1, chem2, cell_line, y_tr_2, y_test_2, norm, tr_idx_2 , test_idx_2)
train_data_3, test_data_3 = prepare_data(chem1, chem2, cell_line, y_tr_3, y_test_3, norm, tr_idx_3 , test_idx_3)
train_data_4, test_data_4 = prepare_data(chem1, chem2, cell_line, y_tr_4, y_test_4, norm, tr_idx_4 , test_idx_4)
											

# calculate weights for weighted MSE loss 
min_s_0 = np.amin(train_data_0['y'])
loss_weight_0 = np.log(train_data_0['y'] - min_s_0 + np.e) # log 취해서 다 고만고만한 값이 됨 

min_s_1 = np.amin(train_data_1['y'])
loss_weight_1 = np.log(train_data_1['y'] - min_s_1 + np.e) # log 취해서 다 고만고만한 값이 됨 

min_s_2 = np.amin(train_data_2['y'])
loss_weight_2 = np.log(train_data_2['y'] - min_s_2 + np.e) # log 취해서 다 고만고만한 값이 됨 

min_s_3 = np.amin(train_data_3['y'])
loss_weight_3 = np.log(train_data_3['y'] - min_s_3 + np.e) # log 취해서 다 고만고만한 값이 됨 

min_s_4 = np.amin(train_data_4['y'])
loss_weight_4 = np.log(train_data_4['y'] - min_s_4 + np.e) # log 취해서 다 고만고만한 값이 됨 






# leave cell out 

norm = 'tanh_norm'
train_data_0, test_data_0 = prepare_data(chem1, chem2, cell_line, y_tr_C0, y_test_C0, norm, tr_idx_C0 , test_idx_C0)
train_data_1, test_data_1 = prepare_data(chem1, chem2, cell_line, y_tr_C1, y_test_C1, norm, tr_idx_C1 , test_idx_C1)
train_data_2, test_data_2 = prepare_data(chem1, chem2, cell_line, y_tr_C2, y_test_C2, norm, tr_idx_C2 , test_idx_C2)
train_data_3, test_data_3 = prepare_data(chem1, chem2, cell_line, y_tr_C3, y_test_C3, norm, tr_idx_C3 , test_idx_C3)
train_data_4, test_data_4 = prepare_data(chem1, chem2, cell_line, y_tr_C4, y_test_C4, norm, tr_idx_C4 , test_idx_C4)
train_data_5, test_data_5 = prepare_data(chem1, chem2, cell_line, y_tr_C5, y_test_C5, norm, tr_idx_C5 , test_idx_C5)
train_data_6, test_data_6 = prepare_data(chem1, chem2, cell_line, y_tr_C6, y_test_C6, norm, tr_idx_C6 , test_idx_C6)
train_data_7, test_data_7 = prepare_data(chem1, chem2, cell_line, y_tr_C7, y_test_C7, norm, tr_idx_C7 , test_idx_C7)
train_data_8, test_data_8 = prepare_data(chem1, chem2, cell_line, y_tr_C8, y_test_C8, norm, tr_idx_C8 , test_idx_C8)
train_data_9, test_data_9 = prepare_data(chem1, chem2, cell_line, y_tr_C9, y_test_C9, norm, tr_idx_C9 , test_idx_C9)

train_data_10, test_data_10 = prepare_data(chem1, chem2, cell_line, y_tr_C10, y_test_C10, norm, tr_idx_C10 , test_idx_C10)
train_data_11, test_data_11 = prepare_data(chem1, chem2, cell_line, y_tr_C11, y_test_C11, norm, tr_idx_C11 , test_idx_C11)
train_data_12, test_data_12 = prepare_data(chem1, chem2, cell_line, y_tr_C12, y_test_C12, norm, tr_idx_C12 , test_idx_C12)
train_data_13, test_data_13 = prepare_data(chem1, chem2, cell_line, y_tr_C13, y_test_C13, norm, tr_idx_C13 , test_idx_C13)
train_data_14, test_data_14 = prepare_data(chem1, chem2, cell_line, y_tr_C14, y_test_C14, norm, tr_idx_C14 , test_idx_C14)
train_data_15, test_data_15 = prepare_data(chem1, chem2, cell_line, y_tr_C15, y_test_C15, norm, tr_idx_C15 , test_idx_C15)
train_data_16, test_data_16 = prepare_data(chem1, chem2, cell_line, y_tr_C16, y_test_C16, norm, tr_idx_C16 , test_idx_C16)
train_data_17, test_data_17 = prepare_data(chem1, chem2, cell_line, y_tr_C17, y_test_C17, norm, tr_idx_C17 , test_idx_C17)
train_data_18, test_data_18 = prepare_data(chem1, chem2, cell_line, y_tr_C18, y_test_C18, norm, tr_idx_C18 , test_idx_C18)
train_data_19, test_data_19 = prepare_data(chem1, chem2, cell_line, y_tr_C19, y_test_C19, norm, tr_idx_C19 , test_idx_C19)
											
train_data_20, test_data_20 = prepare_data(chem1, chem2, cell_line, y_tr_C0, y_test_C0, norm, tr_idx_C0 , test_idx_C0)
train_data_21, test_data_21 = prepare_data(chem1, chem2, cell_line, y_tr_C1, y_test_C1, norm, tr_idx_C1 , test_idx_C1)
train_data_22, test_data_22 = prepare_data(chem1, chem2, cell_line, y_tr_C2, y_test_C2, norm, tr_idx_C2 , test_idx_C2)
train_data_23, test_data_23 = prepare_data(chem1, chem2, cell_line, y_tr_C3, y_test_C3, norm, tr_idx_C3 , test_idx_C3)
train_data_24, test_data_24 = prepare_data(chem1, chem2, cell_line, y_tr_C4, y_test_C4, norm, tr_idx_C4 , test_idx_C4)
train_data_25, test_data_25 = prepare_data(chem1, chem2, cell_line, y_tr_C5, y_test_C5, norm, tr_idx_C5 , test_idx_C5)
train_data_26, test_data_26 = prepare_data(chem1, chem2, cell_line, y_tr_C6, y_test_C6, norm, tr_idx_C6 , test_idx_C6)


									

min_s_0 = np.amin(train_data_0['y']) ; loss_weight_0 = np.log(train_data_0['y'] - min_s_0 + np.e)
min_s_1 = np.amin(train_data_1['y']) ; loss_weight_1 = np.log(train_data_1['y'] - min_s_1 + np.e) 
min_s_2 = np.amin(train_data_2['y']) ; loss_weight_2 = np.log(train_data_2['y'] - min_s_2 + np.e) 
min_s_3 = np.amin(train_data_3['y']) ; loss_weight_3 = np.log(train_data_3['y'] - min_s_3 + np.e) 
min_s_4 = np.amin(train_data_4['y']) ; loss_weight_4 = np.log(train_data_4['y'] - min_s_4 + np.e) 
min_s_5 = np.amin(train_data_5['y']) ; loss_weight_5 = np.log(train_data_5['y'] - min_s_5 + np.e)
min_s_6 = np.amin(train_data_6['y']) ; loss_weight_6 = np.log(train_data_6['y'] - min_s_6 + np.e) 
min_s_7 = np.amin(train_data_7['y']) ; loss_weight_7 = np.log(train_data_7['y'] - min_s_7 + np.e) 
min_s_8 = np.amin(train_data_8['y']) ; loss_weight_8 = np.log(train_data_8['y'] - min_s_8 + np.e) 
min_s_9 = np.amin(train_data_9['y']) ; loss_weight_9 = np.log(train_data_9['y'] - min_s_9 + np.e) 

min_s_10 = np.amin(train_data_10['y']) ; loss_weight_10 = np.log(train_data_10['y'] - min_s_10 + np.e)
min_s_11 = np.amin(train_data_11['y']) ; loss_weight_11 = np.log(train_data_11['y'] - min_s_11 + np.e) 
min_s_12 = np.amin(train_data_12['y']) ; loss_weight_12 = np.log(train_data_12['y'] - min_s_12 + np.e) 
min_s_13 = np.amin(train_data_13['y']) ; loss_weight_13 = np.log(train_data_13['y'] - min_s_13 + np.e) 
min_s_14 = np.amin(train_data_14['y']) ; loss_weight_14 = np.log(train_data_14['y'] - min_s_14 + np.e) 
min_s_15 = np.amin(train_data_15['y']) ; loss_weight_15 = np.log(train_data_15['y'] - min_s_15 + np.e)
min_s_16 = np.amin(train_data_16['y']) ; loss_weight_16 = np.log(train_data_16['y'] - min_s_16 + np.e) 
min_s_17 = np.amin(train_data_17['y']) ; loss_weight_17 = np.log(train_data_17['y'] - min_s_17 + np.e) 
min_s_18 = np.amin(train_data_18['y']) ; loss_weight_18 = np.log(train_data_18['y'] - min_s_18 + np.e) 
min_s_19 = np.amin(train_data_19['y']) ; loss_weight_19 = np.log(train_data_19['y'] - min_s_19 + np.e) 

min_s_20 = np.amin(train_data_20['y']) ; loss_weight_20 = np.log(train_data_20['y'] - min_s_20 + np.e)
min_s_21 = np.amin(train_data_21['y']) ; loss_weight_21 = np.log(train_data_21['y'] - min_s_21 + np.e) 
min_s_22 = np.amin(train_data_22['y']) ; loss_weight_22 = np.log(train_data_22['y'] - min_s_22 + np.e) 
min_s_23 = np.amin(train_data_23['y']) ; loss_weight_23 = np.log(train_data_23['y'] - min_s_23 + np.e) 
min_s_24 = np.amin(train_data_24['y']) ; loss_weight_24 = np.log(train_data_24['y'] - min_s_24 + np.e) 
min_s_25 = np.amin(train_data_25['y']) ; loss_weight_25 = np.log(train_data_25['y'] - min_s_25 + np.e)
min_s_26 = np.amin(train_data_26['y']) ; loss_weight_26 = np.log(train_data_26['y'] - min_s_26 + np.e) 







# leave cid 
norm = 'tanh_norm'
train_data_0, test_data_0 = prepare_data(chem1, chem2, cell_line, y_tr_C0, y_test_C0, norm, tr_idx_C0 , test_idx_C0)
train_data_1, test_data_1 = prepare_data(chem1, chem2, cell_line, y_tr_C1, y_test_C1, norm, tr_idx_C1 , test_idx_C1)
train_data_2, test_data_2 = prepare_data(chem1, chem2, cell_line, y_tr_C2, y_test_C2, norm, tr_idx_C2 , test_idx_C2)
train_data_3, test_data_3 = prepare_data(chem1, chem2, cell_line, y_tr_C3, y_test_C3, norm, tr_idx_C3 , test_idx_C3)
train_data_4, test_data_4 = prepare_data(chem1, chem2, cell_line, y_tr_C4, y_test_C4, norm, tr_idx_C4 , test_idx_C4)


min_s_0 = np.amin(train_data_0['y']) ; loss_weight_0 = np.log(train_data_0['y'] - min_s_0 + np.e)
min_s_1 = np.amin(train_data_1['y']) ; loss_weight_1 = np.log(train_data_1['y'] - min_s_1 + np.e) 
min_s_2 = np.amin(train_data_2['y']) ; loss_weight_2 = np.log(train_data_2['y'] - min_s_2 + np.e) 
min_s_3 = np.amin(train_data_3['y']) ; loss_weight_3 = np.log(train_data_3['y'] - min_s_3 + np.e) 
min_s_4 = np.amin(train_data_4['y']) ; loss_weight_4 = np.log(train_data_4['y'] - min_s_4 + np.e) 






# leave tissue 

norm = 'tanh_norm'
train_data_0, test_data_0 = prepare_data(chem1, chem2, cell_line, y_tr_C0, y_test_C0, norm, tr_idx_C0 , test_idx_C0)
train_data_1, test_data_1 = prepare_data(chem1, chem2, cell_line, y_tr_C1, y_test_C1, norm, tr_idx_C1 , test_idx_C1)
train_data_2, test_data_2 = prepare_data(chem1, chem2, cell_line, y_tr_C2, y_test_C2, norm, tr_idx_C2 , test_idx_C2)
train_data_3, test_data_3 = prepare_data(chem1, chem2, cell_line, y_tr_C3, y_test_C3, norm, tr_idx_C3 , test_idx_C3)
train_data_4, test_data_4 = prepare_data(chem1, chem2, cell_line, y_tr_C4, y_test_C4, norm, tr_idx_C4 , test_idx_C4)
train_data_5, test_data_5 = prepare_data(chem1, chem2, cell_line, y_tr_C5, y_test_C5, norm, tr_idx_C5 , test_idx_C5)

min_s_0 = np.amin(train_data_0['y']) ; loss_weight_0 = np.log(train_data_0['y'] - min_s_0 + np.e)
min_s_1 = np.amin(train_data_1['y']) ; loss_weight_1 = np.log(train_data_1['y'] - min_s_1 + np.e) 
min_s_2 = np.amin(train_data_2['y']) ; loss_weight_2 = np.log(train_data_2['y'] - min_s_2 + np.e) 
min_s_3 = np.amin(train_data_3['y']) ; loss_weight_3 = np.log(train_data_3['y'] - min_s_3 + np.e) 
min_s_4 = np.amin(train_data_4['y']) ; loss_weight_4 = np.log(train_data_4['y'] - min_s_4 + np.e) 
min_s_5 = np.amin(train_data_5['y']) ; loss_weight_5 = np.log(train_data_5['y'] - min_s_5 + np.e)















# load architecture file
architecture = pd.read_csv(TOOL_PATH+'architecture.txt')

# prepare layers of the model and the model name
layers = {}
layers['DSN_1'] = '2048-4096-2048' # architecture['DSN_1'][0] # layers of Drug Synergy Network 1
layers['DSN_2'] = '2048-4096-2048' #architecture['DSN_2'][0] # layers of Drug Synergy Network 2
layers['SPN'] = '2048-1024' #architecture['SPN'][0] # layers of Synergy Prediction Network
modelName = 'matchmaker.w' # args.saved_model_name # name of the model to save the weights

# define constant parameters of MatchMaker
l_rate = 0.0001
inDrop = 0.2
drop = 0.5
max_epoch = 1000
batch_size = 128
earlyStop_patience = 100

dsn1_layers = layers["DSN_1"].split("-")
dsn2_layers = layers["DSN_2"].split("-")
snp_layers = layers["SPN"].split("-")






def generate_network(train, layers, inDrop, drop):
	# contruct two parallel networks
	for l in range(len(dsn1_layers)):
		if l == 0:
			input_drug1 = Input(shape=(train["drug1"].shape[1],)) # tensor 만드는 얘기인듯 train_data["drug1"].shape -> (343706, 1396)
			middle_layer = Dense(int(dsn1_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug1) # 대충 미리 정해둔 layer 수 맞춰서 dense 만듬 
			middle_layer = Dropout(float(inDrop))(middle_layer) # 중간에 dropout 도 더하기 
		elif l == (len(dsn1_layers)-1):
			dsn1_output = Dense(int(dsn1_layers[l]), activation='linear')(middle_layer)
		else:
			middle_layer = Dense(int(dsn1_layers[l]), activation='relu')(middle_layer)
			middle_layer = Dropout(float(drop))(middle_layer)
	#
	for l in range(len(dsn2_layers)):
		if l == 0:
			input_drug2    = Input(shape=(train["drug2"].shape[1],))
			middle_layer = Dense(int(dsn2_layers[l]), activation='relu', kernel_initializer='he_normal')(input_drug2)
			middle_layer = Dropout(float(inDrop))(middle_layer)
		elif l == (len(dsn2_layers)-1):
			dsn2_output = Dense(int(dsn2_layers[l]), activation='linear')(middle_layer)
		else:
			middle_layer = Dense(int(dsn2_layers[l]), activation='relu')(middle_layer)
			middle_layer = Dropout(float(drop))(middle_layer)
	#
	concatModel = concatenate([dsn1_output, dsn2_output])
	#
	for snp_layer in range(len(snp_layers)):
		if len(snp_layers) == 1:
			snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
			snp_output = Dense(1, activation='linear')(snpFC)
		else:
			# more than one FC layer at concat
			if snp_layer == 0:
				snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(concatModel)
				snpFC = Dropout(float(drop))(snpFC)
			elif snp_layer == (len(snp_layers)-1):
				snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
				snp_output = Dense(1, activation='linear')(snpFC)
			else:
				snpFC = Dense(int(snp_layers[snp_layer]), activation='relu')(snpFC)
				snpFC = Dropout(float(drop))(snpFC)
	#
	model = Model([input_drug1, input_drug2], snp_output)
	return model




num_cores = 128
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
GPU = True
config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=num_cores,
						inter_op_parallelism_threads=num_cores,
						allow_soft_placement=True,
						device_count = {'CPU' : 1,
										'GPU' : 1}
					   )



tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

tf.test.is_gpu_available()
tf.config.list_physical_devices('GPU')


# 5cv 
MM_Model_0 = generate_network(train_data_0, layers, inDrop, drop)
MM_Model_1 = generate_network(train_data_1, layers, inDrop, drop)
MM_Model_2 = generate_network(train_data_2, layers, inDrop, drop)
MM_Model_3 = generate_network(train_data_3, layers, inDrop, drop)
MM_Model_4 = generate_network(train_data_4, layers, inDrop, drop)



# leave cell 
MM_Model_0 = generate_network(train_data_0, layers, inDrop, drop)
MM_Model_1 = generate_network(train_data_1, layers, inDrop, drop)
MM_Model_2 = generate_network(train_data_2, layers, inDrop, drop)
MM_Model_3 = generate_network(train_data_3, layers, inDrop, drop)
MM_Model_4 = generate_network(train_data_4, layers, inDrop, drop)
MM_Model_5 = generate_network(train_data_5, layers, inDrop, drop)
MM_Model_6 = generate_network(train_data_6, layers, inDrop, drop)
MM_Model_7 = generate_network(train_data_7, layers, inDrop, drop)
MM_Model_8 = generate_network(train_data_8, layers, inDrop, drop)
MM_Model_9 = generate_network(train_data_9, layers, inDrop, drop)

MM_Model_10 = generate_network(train_data_10, layers, inDrop, drop)
MM_Model_11 = generate_network(train_data_11, layers, inDrop, drop)
MM_Model_12 = generate_network(train_data_12, layers, inDrop, drop)
MM_Model_13 = generate_network(train_data_13, layers, inDrop, drop)
MM_Model_14 = generate_network(train_data_14, layers, inDrop, drop)
MM_Model_15 = generate_network(train_data_15, layers, inDrop, drop)
MM_Model_16 = generate_network(train_data_16, layers, inDrop, drop)
MM_Model_17 = generate_network(train_data_17, layers, inDrop, drop)
MM_Model_18 = generate_network(train_data_18, layers, inDrop, drop)
MM_Model_19 = generate_network(train_data_19, layers, inDrop, drop)

MM_Model_20 = generate_network(train_data_20, layers, inDrop, drop)
MM_Model_21 = generate_network(train_data_21, layers, inDrop, drop)
MM_Model_22 = generate_network(train_data_22, layers, inDrop, drop)
MM_Model_23 = generate_network(train_data_23, layers, inDrop, drop)
MM_Model_24 = generate_network(train_data_24, layers, inDrop, drop)
MM_Model_25 = generate_network(train_data_25, layers, inDrop, drop)
MM_Model_26 = generate_network(train_data_26, layers, inDrop, drop)





# leave cid 

MM_Model_0 = generate_network(train_data_0, layers, inDrop, drop)
MM_Model_1 = generate_network(train_data_1, layers, inDrop, drop)
MM_Model_2 = generate_network(train_data_2, layers, inDrop, drop)
MM_Model_3 = generate_network(train_data_3, layers, inDrop, drop)
MM_Model_4 = generate_network(train_data_4, layers, inDrop, drop)




# leave tissue 
MM_Model_0 = generate_network(train_data_0, layers, inDrop, drop)
MM_Model_1 = generate_network(train_data_1, layers, inDrop, drop)
MM_Model_2 = generate_network(train_data_2, layers, inDrop, drop)
MM_Model_3 = generate_network(train_data_3, layers, inDrop, drop)
MM_Model_4 = generate_network(train_data_4, layers, inDrop, drop)
MM_Model_5 = generate_network(train_data_5, layers, inDrop, drop)










def trainer(model, l_rate, train, val, epo, batch_size, earlyStop, modelName,weights):
	cb_check = ModelCheckpoint((modelName), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
	model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=float(l_rate), beta_1=0.9, beta_2=0.999, amsgrad=False))
	model.fit([train["drug1"], train["drug2"]], train["y"], epochs=epo, shuffle=True, batch_size=batch_size, verbose=1, 
				   validation_data=([val["drug1"], val["drug2"]], val["y"]),sample_weight=weights,
				   callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = earlyStop), cb_check])
	return model




PRJ_NAME = 'W901'
PRJ_PATH = '/home01/k040a01/04.MatchMaker/02.RES/'


model_0 = trainer(MM_Model_0, l_rate, train_data_0, test_data_0, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME), loss_weight_0)


model_1 = trainer(MM_Model_1, l_rate, train_data_1, test_data_1, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME), loss_weight_1)


model_2 = trainer(MM_Model_2, l_rate, train_data_2, test_data_2, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME), loss_weight_2)


model_3 = trainer(MM_Model_3, l_rate, train_data_3, test_data_3, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME), loss_weight_3)


model_4 = trainer(MM_Model_4, l_rate, train_data_4, test_data_4, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME), loss_weight_4)


from scipy import stats



test_pred_0 = model_0.predict([test_data_0["drug1"], test_data_0["drug2"]]).squeeze().tolist()
test_pred_1 = model_1.predict([test_data_1["drug1"], test_data_1["drug2"]]).squeeze().tolist()
test_pred_2 = model_2.predict([test_data_2["drug1"], test_data_2["drug2"]]).squeeze().tolist()
test_pred_3 = model_3.predict([test_data_3["drug1"], test_data_3["drug2"]]).squeeze().tolist()
test_pred_4 = model_4.predict([test_data_4["drug1"], test_data_4["drug2"]]).squeeze().tolist()

test_Pcorr_0 , _ = stats.pearsonr(np.array(test_data_0['y']).tolist(), test_pred_0)
test_Pcorr_1 , _ = stats.pearsonr(np.array(test_data_1['y']).tolist(), test_pred_1)
test_Pcorr_2 , _ = stats.pearsonr(np.array(test_data_2['y']).tolist(), test_pred_2)
test_Pcorr_3 , _ = stats.pearsonr(np.array(test_data_3['y']).tolist(), test_pred_3)
test_Pcorr_4 , _ = stats.pearsonr(np.array(test_data_4['y']).tolist(), test_pred_4)

test_Scorr_0 , _ = stats.spearmanr(np.array(test_data_0['y']).tolist(), test_pred_0)
test_Scorr_1 , _ = stats.spearmanr(np.array(test_data_1['y']).tolist(), test_pred_1)
test_Scorr_2 , _ = stats.spearmanr(np.array(test_data_2['y']).tolist(), test_pred_2)
test_Scorr_3 , _ = stats.spearmanr(np.array(test_data_3['y']).tolist(), test_pred_3)
test_Scorr_4 , _ = stats.spearmanr(np.array(test_data_4['y']).tolist(), test_pred_4)


test_loss_0 = mse(np.array(test_data_0['y']).tolist(), test_pred_0)
test_loss_1 = mse(np.array(test_data_1['y']).tolist(), test_pred_1)
test_loss_2 = mse(np.array(test_data_2['y']).tolist(), test_pred_2)
test_loss_3 = mse(np.array(test_data_3['y']).tolist(), test_pred_3)
test_loss_4 = mse(np.array(test_data_4['y']).tolist(), test_pred_4)



test_result = pd.DataFrame({
	'PCOR' : [test_Pcorr_0, test_Pcorr_1, test_Pcorr_2, test_Pcorr_3, test_Pcorr_4],
	'SCOR' : [test_Scorr_0, test_Scorr_1, test_Scorr_2, test_Scorr_3, test_Scorr_4],
	'LOSS' : [test_loss_0, test_loss_1, test_loss_2, test_loss_3, test_loss_4]
})



model_0.save(PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME)) 


test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))


tail 04.MatchMaker/02.RES/RESULT.W901.5CV.txt -n 100












PRJ_NAME = 'W903'
PRJ_PATH = '/home01/k040a01/04.MatchMaker/02.RES/'


model_0 = trainer(MM_Model_0, l_rate, train_data_0, test_data_0, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME), loss_weight_0)
model_1 = trainer(MM_Model_1, l_rate, train_data_1, test_data_1, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME), loss_weight_1)
model_2 = trainer(MM_Model_2, l_rate, train_data_2, test_data_2, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME), loss_weight_2)
model_3 = trainer(MM_Model_3, l_rate, train_data_3, test_data_3, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME), loss_weight_3)
model_4 = trainer(MM_Model_4, l_rate, train_data_4, test_data_4, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME), loss_weight_4)
model_5 = trainer(MM_Model_5, l_rate, train_data_5, test_data_5, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV5'.format(PRJ_NAME), loss_weight_5)
model_6 = trainer(MM_Model_6, l_rate, train_data_6, test_data_6, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV6'.format(PRJ_NAME), loss_weight_6)
model_7 = trainer(MM_Model_7, l_rate, train_data_7, test_data_7, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV7'.format(PRJ_NAME), loss_weight_7)
model_8 = trainer(MM_Model_8, l_rate, train_data_8, test_data_8, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV8'.format(PRJ_NAME), loss_weight_8)
model_9 = trainer(MM_Model_9, l_rate, train_data_9, test_data_9, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV9'.format(PRJ_NAME), loss_weight_9)

model_10 = trainer(MM_Model_10, l_rate, train_data_10, test_data_10, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV10'.format(PRJ_NAME), loss_weight_10)
model_11 = trainer(MM_Model_11, l_rate, train_data_11, test_data_11, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV11'.format(PRJ_NAME), loss_weight_11)
model_12 = trainer(MM_Model_12, l_rate, train_data_12, test_data_12, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV12'.format(PRJ_NAME), loss_weight_12)
model_13 = trainer(MM_Model_13, l_rate, train_data_13, test_data_13, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV13'.format(PRJ_NAME), loss_weight_13)
model_14 = trainer(MM_Model_14, l_rate, train_data_14, test_data_14, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV14'.format(PRJ_NAME), loss_weight_14)
model_15 = trainer(MM_Model_15, l_rate, train_data_15, test_data_15, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV15'.format(PRJ_NAME), loss_weight_15)
model_16 = trainer(MM_Model_16, l_rate, train_data_16, test_data_16, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV16'.format(PRJ_NAME), loss_weight_16)
model_17 = trainer(MM_Model_17, l_rate, train_data_17, test_data_17, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV17'.format(PRJ_NAME), loss_weight_17)
model_18 = trainer(MM_Model_18, l_rate, train_data_18, test_data_18, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV18'.format(PRJ_NAME), loss_weight_18)
model_19 = trainer(MM_Model_19, l_rate, train_data_19, test_data_19, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV19'.format(PRJ_NAME), loss_weight_19)


model_20 = trainer(MM_Model_20, l_rate, train_data_20, test_data_20, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV20'.format(PRJ_NAME), loss_weight_20)
model_21 = trainer(MM_Model_21, l_rate, train_data_21, test_data_21, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV21'.format(PRJ_NAME), loss_weight_21)
model_22 = trainer(MM_Model_22, l_rate, train_data_22, test_data_22, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV22'.format(PRJ_NAME), loss_weight_22)
model_23 = trainer(MM_Model_23, l_rate, train_data_23, test_data_23, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV23'.format(PRJ_NAME), loss_weight_23)
model_24 = trainer(MM_Model_24, l_rate, train_data_24, test_data_24, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV24'.format(PRJ_NAME), loss_weight_24)
model_25 = trainer(MM_Model_25, l_rate, train_data_25, test_data_25, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV25'.format(PRJ_NAME), loss_weight_25)
model_26 = trainer(MM_Model_26, l_rate, train_data_26, test_data_26, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV26'.format(PRJ_NAME), loss_weight_26)















from scipy import stats

PCOR = []
SCOR = []

for cvnum in range(27) : 
	model = globals()['model_'+str(cvnum)]
	test_data_d1 = globals()['test_data_'+str(cvnum)]['drug1']
	test_data_d2 = globals()['test_data_'+str(cvnum)]['drug2']
	pred_result = model.predict([test_data_d1, test_data_d2]).squeeze().tolist()
	ans = globals()['test_data_'+str(cvnum)]['y']
	pcor, _ = stats.pearsonr(np.array(ans).tolist(), pred_result)
	scor, _ = stats.spearmanr(np.array(ans).tolist(), pred_result)
	PCOR.append(pcor)
	SCOR.append(scor)


test_result = pd.DataFrame({
	'PCOR' : PCOR, 
	'SCOR' : SCOR
})



model_0.save(PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME)) 
model_5.save(PRJ_PATH+'MM_{}_CV5'.format(PRJ_NAME)) 
model_6.save(PRJ_PATH+'MM_{}_CV6'.format(PRJ_NAME)) 
model_7.save(PRJ_PATH+'MM_{}_CV7'.format(PRJ_NAME)) 
model_8.save(PRJ_PATH+'MM_{}_CV8'.format(PRJ_NAME)) 
model_9.save(PRJ_PATH+'MM_{}_CV9'.format(PRJ_NAME)) 

model_10.save(PRJ_PATH+'MM_{}_CV10'.format(PRJ_NAME)) 
model_11.save(PRJ_PATH+'MM_{}_CV11'.format(PRJ_NAME)) 
model_12.save(PRJ_PATH+'MM_{}_CV12'.format(PRJ_NAME)) 
model_13.save(PRJ_PATH+'MM_{}_CV13'.format(PRJ_NAME)) 
model_14.save(PRJ_PATH+'MM_{}_CV14'.format(PRJ_NAME)) 
model_15.save(PRJ_PATH+'MM_{}_CV15'.format(PRJ_NAME)) 
model_16.save(PRJ_PATH+'MM_{}_CV16'.format(PRJ_NAME)) 
model_17.save(PRJ_PATH+'MM_{}_CV17'.format(PRJ_NAME)) 
model_18.save(PRJ_PATH+'MM_{}_CV18'.format(PRJ_NAME)) 
model_19.save(PRJ_PATH+'MM_{}_CV19'.format(PRJ_NAME)) 

model_20.save(PRJ_PATH+'MM_{}_CV20'.format(PRJ_NAME)) 
model_21.save(PRJ_PATH+'MM_{}_CV21'.format(PRJ_NAME)) 
model_22.save(PRJ_PATH+'MM_{}_CV22'.format(PRJ_NAME)) 
model_23.save(PRJ_PATH+'MM_{}_CV23'.format(PRJ_NAME)) 
model_24.save(PRJ_PATH+'MM_{}_CV24'.format(PRJ_NAME)) 
model_25.save(PRJ_PATH+'MM_{}_CV25'.format(PRJ_NAME)) 
model_26.save(PRJ_PATH+'MM_{}_CV26'.format(PRJ_NAME)) 

test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))






# leave cid 

PRJ_NAME = 'W904'
PRJ_PATH = '/home01/k040a01/04.MatchMaker/02.RES/'


model_0 = trainer(MM_Model_0, l_rate, train_data_0, test_data_0, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME), loss_weight_0)
model_1 = trainer(MM_Model_1, l_rate, train_data_1, test_data_1, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME), loss_weight_1)
model_2 = trainer(MM_Model_2, l_rate, train_data_2, test_data_2, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME), loss_weight_2)
model_3 = trainer(MM_Model_3, l_rate, train_data_3, test_data_3, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME), loss_weight_3)
model_4 = trainer(MM_Model_4, l_rate, train_data_4, test_data_4, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME), loss_weight_4)



from scipy import stats

mse = keras.losses.MeanSquaredError()


PCOR = []
SCOR = []
losses= []

for cvnum in range(5) : 
	model = globals()['model_'+str(cvnum)]
	test_data_d1 = globals()['test_data_'+str(cvnum)]['drug1']
	test_data_d2 = globals()['test_data_'+str(cvnum)]['drug2']
	pred_result = model.predict([test_data_d1, test_data_d2]).squeeze().tolist()
	ans = globals()['test_data_'+str(cvnum)]['y']
	pcor, _ = stats.pearsonr(np.array(ans).tolist(), pred_result)
	scor, _ = stats.spearmanr(np.array(ans).tolist(), pred_result)
	PCOR.append(pcor)
	SCOR.append(scor)
	mmm = mse(np.array(ans).tolist(), pred_result).numpy()
	losses.append(mmm)




test_result = pd.DataFrame({
	'PCOR' : PCOR, 
	'SCOR' : SCOR,
	'LOSS' : losses
})


test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))




model_0.save(PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME)) 












# leave tissue  

PRJ_NAME = 'W907'
PRJ_PATH = '/home01/k040a01/04.MatchMaker/02.RES/'


model_0 = trainer(MM_Model_0, l_rate, train_data_0, test_data_0, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME), loss_weight_0)
model_1 = trainer(MM_Model_1, l_rate, train_data_1, test_data_1, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME), loss_weight_1)
model_2 = trainer(MM_Model_2, l_rate, train_data_2, test_data_2, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME), loss_weight_2)
model_3 = trainer(MM_Model_3, l_rate, train_data_3, test_data_3, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME), loss_weight_3)
model_4 = trainer(MM_Model_4, l_rate, train_data_4, test_data_4, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME), loss_weight_4)
model_5 = trainer(MM_Model_5, l_rate, train_data_5, test_data_5, max_epoch, batch_size,
								earlyStop_patience, PRJ_PATH+'MM_{}_CV5'.format(PRJ_NAME), loss_weight_5)




from scipy import stats

mse = keras.losses.MeanSquaredError()


PCOR = []
SCOR = []
losses= []

for cvnum in range(6) : 
	model = globals()['model_'+str(cvnum)]
	test_data_d1 = globals()['test_data_'+str(cvnum)]['drug1']
	test_data_d2 = globals()['test_data_'+str(cvnum)]['drug2']
	pred_result = model.predict([test_data_d1, test_data_d2]).squeeze().tolist()
	ans = globals()['test_data_'+str(cvnum)]['y']
	pcor, _ = stats.pearsonr(np.array(ans).tolist(), pred_result)
	scor, _ = stats.spearmanr(np.array(ans).tolist(), pred_result)
	PCOR.append(pcor)
	SCOR.append(scor)
	mmm = mse(np.array(ans).tolist(), pred_result).numpy()
	losses.append(mmm)




test_result = pd.DataFrame({
	'PCOR' : PCOR, 
	'SCOR' : SCOR,
	'LOSS' : losses
})


test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))




model_0.save(PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'MM_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'MM_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'MM_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'MM_{}_CV4'.format(PRJ_NAME)) 
model_5.save(PRJ_PATH+'MM_{}_CV5'.format(PRJ_NAME)) 



























#################################
아니 좀 궁금해서 
loss 가 어떻게 되었는지랑, 각각 epoch 이 다른건지 확인이 좀 필요함 
-> 그러면 나도 각각 model 에서의 최대 epoch 로 확인해주면 되니까 
-> 일단 다운로드 받기 



PRJ_NAME = 'W903'
PRJ_PATH = '/home01/k040a01/04.MatchMaker/02.RES/'

new_model = tf.keras.models.load_model(PRJ_PATH+'MM_{}_CV0'.format(PRJ_NAME))
test_pred = new_model.predict([test_data_0['drug1'], test_data_0['drug2']]).squeeze().tolist()
test_pcor, _ = stats.pearsonr(np.array(test_ans).tolist(), test_pred)

PCOR = []
SCOR = []
losses = []

mse = keras.losses.MeanSquaredError()

for cell_num in range(27):
	cell_num
	test_ans = globals()['test_data_'+str(cell_num)]['y']
	test_x = globals()['test_data_'+str(cell_num)]
	#
	with tf.device('CPU'):
		new_model = tf.keras.models.load_model(PRJ_PATH+'MM_{}_CV{}'.format(PRJ_NAME, cell_num))
	test_pred = new_model.predict([test_x['drug1'], test_x['drug2']])
	test_pred = sum(test_pred.tolist(),[])
	test_pcor, _ = stats.pearsonr(np.array(test_ans).tolist(), test_pred)
	test_scor, _ = stats.spearmanr(np.array(test_ans).tolist(), test_pred)
	PCOR.append(test_pcor)
	SCOR.append(test_scor)
	mmm = mse(np.array(test_ans).tolist(), test_pred).numpy()
	losses.append(mmm)






test_result = pd.DataFrame({
	'PCOR' : PCOR, 
	'SCOR' : SCOR,
	'LOSS' : losses
})


test_result.to_csv(PRJ_PATH + 'RESULT_CORR.ver2.{}'.format(PRJ_NAME))














