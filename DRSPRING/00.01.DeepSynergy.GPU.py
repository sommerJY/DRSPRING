


# 되는지 확인해야함 
salloc --partition=1gpu -N 1 -n 1 --tasks-per-node=1 --comment="test"

conda activate DS_1
changed to tensorflow 2 


import numpy as np
import pandas as pd
import pickle 
import gzip

import os, sys

import json

#import keras 
import tensorflow
from tensorflow import keras
#import keras as K
from tensorflow import keras as K
import tensorflow as tf
from tensorflow.keras import backend
# from tensorflow.keras.backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random
import networkx as nx
import torch
import copy

import sklearn




#tf.config.list_physical_devices('GPU')

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

tf.test.is_gpu_available()
# False 

tf.config.experimental.list_physical_devices()





TOOL_PATH = '/home01/k040a01/03.DeepSynergy/01.Data/'

# JY data 

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


#############################################
# deepsynergy


hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model

norm = 'tanh'
test_fold = 0
val_fold = 1

def normalize(X, means1=None, std1=None, means2=None, std2=None, feat_filt=None, norm='tanh_norm'):
	if std1 is None:
		std1 = np.nanstd(X, axis=0)
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

file = gzip.open(os.path.join(TOOL_PATH,'X.p.gz'), 'rb')
X = pickle.load(file)
file.close()

labels = pd.read_csv(os.path.join(TOOL_PATH,'labels.csv'), index_col=0) 
labels = pd.concat([labels, labels]) 
DeepS_label = list(set(list(labels.drug_a_name) + list(labels.drug_b_name))) # 38개 


DS_labels_RE = pd.read_csv(os.path.join(TOOL_PATH,'RELABEL.csv'))


def change_cellname(old , new) :
	index_num = list(DS_labels_RE[DS_labels_RE.cell_line==old].index)
	for ind in index_num :
		DS_labels_RE.at[ind, 'cell_line'] = new


change_cellname('UWB1289BRCA1','UWB1289+BRCA1')
change_cellname('SKOV3','SK-OV-3')
change_cellname('SW620','SW-620')
change_cellname('NCIH460','NCI-H460')
change_cellname('T47D','T-47D')



				DS_labels_RE_ccc = list(set(DS_labels_RE.ON_CID_CID_CELL))

				DC_labels_RE_ccc_df = pd.DataFrame({'ccc' : DS_labels_RE_ccc})
				DC_labels_RE_ccc_df['syn'] = 0.0000

				for indd in range(len(DS_labels_RE_ccc)) : 
					ccc = DS_labels_RE_ccc[indd]
					syns = DS_labels_RE[DS_labels_RE.ON_CID_CID_CELL==ccc]['synergy']
					new_syn = np.mean(syns)
					DC_labels_RE_ccc_df.at[indd, 'syn'] = new_syn




#################### filter 



aaa = list(DS_labels_RE['drug_a_CID'])
bbb = list(DS_labels_RE['drug_b_CID'])
ccc = list(DS_labels_RE['cell_line'])

# 306
DS_labels_RE['CID_CID'] = [str(int(aaa[i])) + '___' + str(int(bbb[i])) if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i])) for i in range(DS_labels_RE.shape[0])]

# 10404 -- duplicated 가 이상한게 아님 
DS_labels_RE['CID_CID_CELL'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + ccc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + ccc[i] for i in range(DS_labels_RE.shape[0])]

set_data = list(set(DS_labels_RE['CID_CID_CELL']))



DS_labels_RE2 = DS_labels_RE[['CID_CID_CELL','ori_index']]
rep_index = list(DS_labels_RE2.CID_CID_CELL.drop_duplicates().index)
DS_labels_RE2 = DS_labels_RE2.loc[rep_index]



# CV 0 
JY_train = ABCS_tv_0 # 5380
JY_test = ABCS_test_0 # 1398

# CV 1
JY_train = ABCS_tv_0
JY_test = ABCS_test_0

# CV 2
JY_train = ABCS_tv_0
JY_test = ABCS_test_0

# CV 3
JY_train = ABCS_tv_0
JY_test = ABCS_test_0

# CV 4 
JY_train = ABCS_tv_0
JY_test = ABCS_test_0


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




# leave tissue  

A_B_C_S_SET_SM['tissue'] = A_B_C_S_SET_SM.CELL.apply(lambda x : '_'.join(x.split('_')[1:]))

CELL_0_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='LARGE_INTESTINE'] ; CELL_0_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='LARGE_INTESTINE']
CELL_1_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='LUNG']; CELL_1_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='LUNG']
CELL_2_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='BREAST']; CELL_2_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='BREAST']
CELL_3_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='PROSTATE']; CELL_3_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='PROSTATE']
CELL_4_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='SKIN']; CELL_4_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='SKIN']
CELL_5_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue!='OVARY']; CELL_5_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.tissue=='OVARY']






# 본격적 모델 

def moving_average(a, n=3):
	ret = np.cumsum(a, dtype=float)
	ret[n:] = ret[n:] - ret[:-n]
	return ret[n - 1:] / n


def make_data(JY_train, JY_test) : 
	#
	test_idx = list(DS_labels_RE2[DS_labels_RE2.CID_CID_CELL.isin(JY_test.CID_CID_CELL)]['ori_index'])
	tr_idx = list(DS_labels_RE2[DS_labels_RE2.CID_CID_CELL.isin(JY_train.CID_CID_CELL)]['ori_index'])
	#
	ds_test_df = DS_labels_RE2[DS_labels_RE2.CID_CID_CELL.isin(JY_test.CID_CID_CELL)]
	ds_tr_df = DS_labels_RE2[DS_labels_RE2.CID_CID_CELL.isin(JY_train.CID_CID_CELL)]
	#
	ds_test_df_ccc = list(ds_test_df.CID_CID_CELL)
	ds_tr_df_ccc = list(ds_tr_df.CID_CID_CELL)
	#
	#
	X_tr1 = X[tr_idx]
	X_tr2 = []
	for xx in X_tr1 : 
		a_feat = xx[0:4387]
		b_feat = xx[4387:4387*2]
		c_feat = xx[4387*2 :]
		new_feat = b_feat.tolist() + a_feat.tolist() + c_feat.tolist()
		X_tr2.append(new_feat)
	#
	#
	X_tr2 = np.array(X_tr2)
	X_tr = np.concatenate((X_tr1, X_tr2))
	X_test = X[test_idx]
	#
	y_tr = [JY_train[JY_train.CID_CID_CELL==a]['syn_ans'].item() for a in ds_tr_df_ccc]
	y_tr = y_tr + y_tr
	y_test = [JY_test[JY_test.CID_CID_CELL==a]['syn_ans'].item() for a in ds_test_df_ccc]
	#
	#
	if norm == "tanh_norm":
		X_tr, mean, std, mean2, std2, feat_filt = normalize(X_tr, norm=norm)
		X_test, mean, std, mean2, std2, feat_filt = normalize(X_test, mean, std, mean2, std2, 
															feat_filt=feat_filt, norm=norm)
	else:
		X_tr, mean, std, feat_filt = normalize(X_tr, norm=norm)
		X_test, mean, std, feat_filt = normalize(X_test, mean, std, feat_filt=feat_filt, norm=norm)
	#
	#
	X_tr = tf.convert_to_tensor(X_tr, dtype=tf.float32)
	X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
	#
	y_tr = tf.convert_to_tensor(y_tr, dtype=tf.float32)
	y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
	return X_tr, X_test, y_tr, y_test






# 기본 CV 
X_tr_0, X_test_0, y_tr_0, y_test_0 = make_data(ABCS_tv_0, ABCS_test_0)
X_tr_1, X_test_1, y_tr_1, y_test_1 = make_data(ABCS_tv_1, ABCS_test_1)
X_tr_2, X_test_2, y_tr_2, y_test_2 = make_data(ABCS_tv_2, ABCS_test_2)
X_tr_3, X_test_3, y_tr_3, y_test_3 = make_data(ABCS_tv_3, ABCS_test_3)
X_tr_4, X_test_4, y_tr_4, y_test_4 = make_data(ABCS_tv_4, ABCS_test_4)




# cell 
X_tr_C0, X_test_C0, y_tr_C0, y_test_C0 = make_data(CELL_0_train, CELL_0_test)
X_tr_C1, X_test_C1, y_tr_C1, y_test_C1 = make_data(CELL_1_train, CELL_1_test)
X_tr_C2, X_test_C2, y_tr_C2, y_test_C2 = make_data(CELL_2_train, CELL_2_test)
X_tr_C3, X_test_C3, y_tr_C3, y_test_C3 = make_data(CELL_3_train, CELL_3_test)
X_tr_C4, X_test_C4, y_tr_C4, y_test_C4 = make_data(CELL_4_train, CELL_4_test)
X_tr_C5, X_test_C5, y_tr_C5, y_test_C5 = make_data(CELL_5_train, CELL_5_test)
X_tr_C6, X_test_C6, y_tr_C6, y_test_C6 = make_data(CELL_6_train, CELL_6_test)
X_tr_C7, X_test_C7, y_tr_C7, y_test_C7 = make_data(CELL_7_train, CELL_7_test)
X_tr_C8, X_test_C8, y_tr_C8, y_test_C8 = make_data(CELL_8_train, CELL_8_test)
X_tr_C9, X_test_C9, y_tr_C9, y_test_C9 = make_data(CELL_9_train, CELL_9_test)
X_tr_C10, X_test_C10, y_tr_C10, y_test_C10 = make_data(CELL_10_train, CELL_10_test)
X_tr_C11, X_test_C11, y_tr_C11, y_test_C11 = make_data(CELL_11_train, CELL_11_test)
X_tr_C12, X_test_C12, y_tr_C12, y_test_C12 = make_data(CELL_12_train, CELL_12_test)
X_tr_C13, X_test_C13, y_tr_C13, y_test_C13 = make_data(CELL_13_train, CELL_13_test)
X_tr_C14, X_test_C14, y_tr_C14, y_test_C14 = make_data(CELL_14_train, CELL_14_test)
X_tr_C15, X_test_C15, y_tr_C15, y_test_C15 = make_data(CELL_15_train, CELL_15_test)
X_tr_C16, X_test_C16, y_tr_C16, y_test_C16 = make_data(CELL_16_train, CELL_16_test)
X_tr_C17, X_test_C17, y_tr_C17, y_test_C17 = make_data(CELL_17_train, CELL_17_test)
X_tr_C18, X_test_C18, y_tr_C18, y_test_C18 = make_data(CELL_18_train, CELL_18_test)
X_tr_C19, X_test_C19, y_tr_C19, y_test_C19 = make_data(CELL_19_train, CELL_19_test)
X_tr_C20, X_test_C20, y_tr_C20, y_test_C20 = make_data(CELL_20_train, CELL_20_test)
X_tr_C21, X_test_C21, y_tr_C21, y_test_C21 = make_data(CELL_21_train, CELL_21_test)
X_tr_C22, X_test_C22, y_tr_C22, y_test_C22 = make_data(CELL_22_train, CELL_22_test)
X_tr_C23, X_test_C23, y_tr_C23, y_test_C23 = make_data(CELL_23_train, CELL_23_test)
X_tr_C24, X_test_C24, y_tr_C24, y_test_C24 = make_data(CELL_24_train, CELL_24_test)
X_tr_C25, X_test_C25, y_tr_C25, y_test_C25 = make_data(CELL_25_train, CELL_25_test)
X_tr_C26, X_test_C26, y_tr_C26, y_test_C26 = make_data(CELL_26_train, CELL_26_test)






# cid 
X_tr_C0, X_test_C0, y_tr_C0, y_test_C0 = make_data(CID_0_train, CID_0_test)
X_tr_C1, X_test_C1, y_tr_C1, y_test_C1 = make_data(CID_1_train, CID_1_test)
X_tr_C2, X_test_C2, y_tr_C2, y_test_C2 = make_data(CID_2_train, CID_2_test)
X_tr_C3, X_test_C3, y_tr_C3, y_test_C3 = make_data(CID_3_train, CID_3_test)
X_tr_C4, X_test_C4, y_tr_C4, y_test_C4 = make_data(CID_4_train, CID_4_test)


# tissue 
X_tr_C0, X_test_C0, y_tr_C0, y_test_C0 = make_data(CELL_0_train, CELL_0_test)
X_tr_C1, X_test_C1, y_tr_C1, y_test_C1 = make_data(CELL_1_train, CELL_1_test)
X_tr_C2, X_test_C2, y_tr_C2, y_test_C2 = make_data(CELL_2_train, CELL_2_test)
X_tr_C3, X_test_C3, y_tr_C3, y_test_C3 = make_data(CELL_3_train, CELL_3_test)
X_tr_C4, X_test_C4, y_tr_C4, y_test_C4 = make_data(CELL_4_train, CELL_4_test)
X_tr_C5, X_test_C5, y_tr_C5, y_test_C5 = make_data(CELL_5_train, CELL_5_test)








exec(open(os.path.join(TOOL_PATH,hyperparameter_file)).read()) 

os.environ["CUDA_VISIBLE_DEVICES"]="0"


model_0 = Sequential()
for i in range(len(layers)):
	if i==0:
		model_0.add(Dense(layers[i], input_shape=(X_tr_0.shape[1],), activation=act_func, 
						kernel_initializer='he_normal'))
		model_0.add(Dropout(float(input_dropout)))
	elif i==len(layers)-1:
		model_0.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
	else:
		model_0.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
		model_0.add(Dropout(float(dropout)))
	model_0.compile(loss='mean_squared_error', 
	optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))


model_1 = Sequential()
for i in range(len(layers)):
	if i==0:
		model_1.add(Dense(layers[i], input_shape=(X_tr_1.shape[1],), activation=act_func, 
						kernel_initializer='he_normal'))
		model_1.add(Dropout(float(input_dropout)))
	elif i==len(layers)-1:
		model_1.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
	else:
		model_1.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
		model_1.add(Dropout(float(dropout)))
	model_1.compile(loss='mean_squared_error', 
	optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))



model_2 = Sequential()
for i in range(len(layers)):
	if i==0:
		model_2.add(Dense(layers[i], input_shape=(X_tr_2.shape[1],), activation=act_func, 
						kernel_initializer='he_normal'))
		model_2.add(Dropout(float(input_dropout)))
	elif i==len(layers)-1:
		model_2.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
	else:
		model_2.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
		model_2.add(Dropout(float(dropout)))
	model_2.compile(loss='mean_squared_error', 
	optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))



model_3 = Sequential()
for i in range(len(layers)):
	if i==0:
		model_3.add(Dense(layers[i], input_shape=(X_tr_3.shape[1],), activation=act_func, 
						kernel_initializer='he_normal'))
		model_3.add(Dropout(float(input_dropout)))
	elif i==len(layers)-1:
		model_3.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
	else:
		model_3.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
		model_3.add(Dropout(float(dropout)))
	model_3.compile(loss='mean_squared_error', 
	optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))



model_4 = Sequential()
for i in range(len(layers)):
	if i==0:
		model_4.add(Dense(layers[i], input_shape=(X_tr_4.shape[1],), activation=act_func, 
						kernel_initializer='he_normal'))
		model_4.add(Dropout(float(input_dropout)))
	elif i==len(layers)-1:
		model_4.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
	else:
		model_4.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
		model_4.add(Dropout(float(dropout)))
	model_4.compile(loss='mean_squared_error', 
	optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))



def make_model(X_tr) : 
	model = Sequential()
	for i in range(len(layers)):
		if i==0:
			model.add(Dense(layers[i], input_shape=(X_tr.shape[1],), activation=act_func, 
							kernel_initializer='he_normal'))
			model.add(Dropout(float(input_dropout)))
		elif i==len(layers)-1:
			model.add(Dense(layers[i], activation='linear', kernel_initializer="he_normal"))
		else:
			model.add(Dense(layers[i], activation=act_func, kernel_initializer="he_normal"))
			model.add(Dropout(float(dropout)))
		model.compile(loss='mean_squared_error', 
		optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))
	return model





# model.summary()
# hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_test, y_test))





from scipy import stats

hist_0 = model_0.fit(X_tr_0, y_tr_0, epochs=10, shuffle=True, batch_size=64, validation_data=(X_test_0, y_test_0))
hist_1 = model_1.fit(X_tr_1, y_tr_1, epochs=10, shuffle=True, batch_size=64, validation_data=(X_test_1, y_test_1))
hist_2 = model_2.fit(X_tr_2, y_tr_2, epochs=10, shuffle=True, batch_size=64, validation_data=(X_test_2, y_test_2))
hist_3 = model_3.fit(X_tr_3, y_tr_3, epochs=10, shuffle=True, batch_size=64, validation_data=(X_test_3, y_test_3))
hist_4 = model_4.fit(X_tr_4, y_tr_4, epochs=10, shuffle=True, batch_size=64, validation_data=(X_test_4, y_test_4))


test_loss_0 = hist_0.history['val_loss']
test_loss_1 = hist_1.history['val_loss']
test_loss_2 = hist_2.history['val_loss']
test_loss_3 = hist_3.history['val_loss']
test_loss_4 = hist_4.history['val_loss']


test_pred_0 = model_0.predict(X_test_0).squeeze().tolist()
test_pred_1 = model_1.predict(X_test_1).squeeze().tolist()
test_pred_2 = model_2.predict(X_test_2).squeeze().tolist()
test_pred_3 = model_3.predict(X_test_3).squeeze().tolist()
test_pred_4 = model_4.predict(X_test_4).squeeze().tolist()

test_Pcorr_0 , _ = stats.pearsonr(np.array(y_test_0).tolist(), test_pred_0)
test_Pcorr_1 , _ = stats.pearsonr(np.array(y_test_1).tolist(), test_pred_1)
test_Pcorr_2 , _ = stats.pearsonr(np.array(y_test_2).tolist(), test_pred_2)
test_Pcorr_3 , _ = stats.pearsonr(np.array(y_test_3).tolist(), test_pred_3)
test_Pcorr_4 , _ = stats.pearsonr(np.array(y_test_4).tolist(), test_pred_4)

test_Scorr_0 , _ = stats.spearmanr(np.array(y_test_0).tolist(), test_pred_0)
test_Scorr_1 , _ = stats.spearmanr(np.array(y_test_1).tolist(), test_pred_1)
test_Scorr_2 , _ = stats.spearmanr(np.array(y_test_2).tolist(), test_pred_2)
test_Scorr_3 , _ = stats.spearmanr(np.array(y_test_3).tolist(), test_pred_3)
test_Scorr_4 , _ = stats.spearmanr(np.array(y_test_4).tolist(), test_pred_4)

loss_result = pd.DataFrame({
	'CV0' : test_loss_0, 'CV1' : test_loss_1, 'CV2' : test_loss_2, 'CV3' : test_loss_3, 'CV4' : test_loss_4, 
	})


test_result = pd.DataFrame({
	'PCOR' : [test_Pcorr_0, test_Pcorr_1, test_Pcorr_2, test_Pcorr_3, test_Pcorr_4],
	'SCOR' : [test_Scorr_0, test_Scorr_1, test_Scorr_2, test_Scorr_3, test_Scorr_4],
})


PRJ_NAME = 'W801'
PRJ_PATH = '/home01/k040a01/03.DeepSynergy/02.RES/'

model_0.save(PRJ_PATH+'DS_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'DS_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'DS_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'DS_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'DS_{}_CV4'.format(PRJ_NAME)) 


loss_result.to_csv(PRJ_PATH + 'RESULT_LOSS.{}'.format(PRJ_NAME))
test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))









##### leave cell out 


from scipy import stats

model_0 = make_model(X_tr_C0)
model_1 = make_model(X_tr_C1)
model_2 = make_model(X_tr_C2)
model_3 = make_model(X_tr_C3)
model_4 = make_model(X_tr_C4)
model_5 = make_model(X_tr_C5)
model_6 = make_model(X_tr_C6)
model_7 = make_model(X_tr_C7)
model_8 = make_model(X_tr_C8)
model_9 = make_model(X_tr_C9)

model_10 = make_model(X_tr_C10)
model_11 = make_model(X_tr_C11)
model_12 = make_model(X_tr_C12)
model_13 = make_model(X_tr_C13)
model_14 = make_model(X_tr_C14)
model_15 = make_model(X_tr_C15)
model_16 = make_model(X_tr_C16)
model_17 = make_model(X_tr_C17)
model_18 = make_model(X_tr_C18)
model_19 = make_model(X_tr_C19)

model_20 = make_model(X_tr_C20)
model_21 = make_model(X_tr_C21)
model_22 = make_model(X_tr_C22)
model_23 = make_model(X_tr_C23)
model_24 = make_model(X_tr_C24)
model_25 = make_model(X_tr_C25)
model_26 = make_model(X_tr_C26)


max_epoch = 10

hist_0 = model_0.fit(X_tr_C0, y_tr_C0, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C0, y_test_C0))
hist_1 = model_1.fit(X_tr_C1, y_tr_C1, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C1, y_test_C1))
hist_2 = model_2.fit(X_tr_C2, y_tr_C2, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C2, y_test_C2))
hist_3 = model_3.fit(X_tr_C3, y_tr_C3, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C3, y_test_C3))
hist_4 = model_4.fit(X_tr_C4, y_tr_C4, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C4, y_test_C4))
hist_5 = model_5.fit(X_tr_C5, y_tr_C5, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C5, y_test_C5))
hist_6 = model_6.fit(X_tr_C6, y_tr_C6, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C6, y_test_C6))
hist_7 = model_7.fit(X_tr_C7, y_tr_C7, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C7, y_test_C7))
hist_8 = model_8.fit(X_tr_C8, y_tr_C8, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C8, y_test_C8))
hist_9 = model_9.fit(X_tr_C9, y_tr_C9, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C9, y_test_C9))

hist_10 = model_10.fit(X_tr_C10, y_tr_C10, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C10, y_test_C10))
hist_11 = model_11.fit(X_tr_C11, y_tr_C11, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C11, y_test_C11))
hist_12 = model_12.fit(X_tr_C12, y_tr_C12, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C12, y_test_C12))
hist_13 = model_13.fit(X_tr_C13, y_tr_C13, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C13, y_test_C13))
hist_14 = model_14.fit(X_tr_C14, y_tr_C14, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C14, y_test_C14))
hist_15 = model_15.fit(X_tr_C15, y_tr_C15, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C15, y_test_C15))
hist_16 = model_16.fit(X_tr_C16, y_tr_C16, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C16, y_test_C16))
hist_17 = model_17.fit(X_tr_C17, y_tr_C17, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C17, y_test_C17))
hist_18 = model_18.fit(X_tr_C18, y_tr_C18, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C18, y_test_C18))
hist_19 = model_19.fit(X_tr_C19, y_tr_C19, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C19, y_test_C19))

hist_20 = model_20.fit(X_tr_C20, y_tr_C20, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C20, y_test_C20))
hist_21 = model_21.fit(X_tr_C21, y_tr_C21, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C21, y_test_C21))
hist_22 = model_22.fit(X_tr_C22, y_tr_C22, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C22, y_test_C22))
hist_23 = model_23.fit(X_tr_C23, y_tr_C23, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C23, y_test_C23))
hist_24 = model_24.fit(X_tr_C24, y_tr_C24, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C24, y_test_C24))
hist_25 = model_25.fit(X_tr_C25, y_tr_C25, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C25, y_test_C25))
hist_26 = model_26.fit(X_tr_C26, y_tr_C26, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C26, y_test_C26))


test_loss_list = [globals()['hist_'+str(cell_num)].history['val_loss'] for cell_num in range(27)]

loss_result = pd.DataFrame({'CV{}'.format(cell_num) : test_loss_list[cell_num] for cell_num in range(27)})

pred_list = [globals()['model_'+str(cell_num)].predict(globals()['X_test_C'+str(cell_num)]).squeeze().tolist() for cell_num in range(27)]


PCOR = []
SCOR = []

for cell_num in range(27):
	test_ans = globals()['y_test_'+str(cell_num)]
	test_pred = pred_list[cell_num]
	test_pcor, _ = stats.pearsonr(np.array(test_ans).tolist(), test_pred)
	test_scor, _ = stats.spearmanr(np.array(test_ans).tolist(), test_pred)
	PCOR.append(test_pcor)
	SCOR.append(test_scor)


test_result = pd.DataFrame({
	'PCOR' : PCOR,
	'SCOR' : SCOR
})


PRJ_NAME = 'W803'
PRJ_PATH = '/home01/k040a01/03.DeepSynergy/02.RES/'

model_0.save(PRJ_PATH+'DS_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'DS_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'DS_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'DS_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'DS_{}_CV4'.format(PRJ_NAME)) 
model_5.save(PRJ_PATH+'DS_{}_CV5'.format(PRJ_NAME)) 
model_6.save(PRJ_PATH+'DS_{}_CV6'.format(PRJ_NAME)) 
model_7.save(PRJ_PATH+'DS_{}_CV7'.format(PRJ_NAME)) 
model_8.save(PRJ_PATH+'DS_{}_CV8'.format(PRJ_NAME)) 
model_9.save(PRJ_PATH+'DS_{}_CV9'.format(PRJ_NAME)) 

model_10.save(PRJ_PATH+'DS_{}_CV10'.format(PRJ_NAME)) 
model_11.save(PRJ_PATH+'DS_{}_CV11'.format(PRJ_NAME)) 
model_12.save(PRJ_PATH+'DS_{}_CV12'.format(PRJ_NAME)) 
model_13.save(PRJ_PATH+'DS_{}_CV13'.format(PRJ_NAME)) 
model_14.save(PRJ_PATH+'DS_{}_CV14'.format(PRJ_NAME)) 
model_15.save(PRJ_PATH+'DS_{}_CV15'.format(PRJ_NAME)) 
model_16.save(PRJ_PATH+'DS_{}_CV16'.format(PRJ_NAME)) 
model_17.save(PRJ_PATH+'DS_{}_CV17'.format(PRJ_NAME)) 
model_18.save(PRJ_PATH+'DS_{}_CV18'.format(PRJ_NAME)) 
model_19.save(PRJ_PATH+'DS_{}_CV19'.format(PRJ_NAME)) 

model_20.save(PRJ_PATH+'DS_{}_CV20'.format(PRJ_NAME)) 
model_21.save(PRJ_PATH+'DS_{}_CV21'.format(PRJ_NAME)) 
model_22.save(PRJ_PATH+'DS_{}_CV22'.format(PRJ_NAME)) 
model_23.save(PRJ_PATH+'DS_{}_CV23'.format(PRJ_NAME)) 
model_24.save(PRJ_PATH+'DS_{}_CV24'.format(PRJ_NAME)) 
model_25.save(PRJ_PATH+'DS_{}_CV25'.format(PRJ_NAME)) 
model_26.save(PRJ_PATH+'DS_{}_CV26'.format(PRJ_NAME)) 




loss_result.to_csv(PRJ_PATH + 'RESULT_LOSS.{}'.format(PRJ_NAME))
test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))










##### leave cid out 


from scipy import stats

model_0 = make_model(X_tr_C0)
model_1 = make_model(X_tr_C1)
model_2 = make_model(X_tr_C2)
model_3 = make_model(X_tr_C3)
model_4 = make_model(X_tr_C4)

max_epoch = 10

hist_0 = model_0.fit(X_tr_C0, y_tr_C0, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C0, y_test_C0))
hist_1 = model_1.fit(X_tr_C1, y_tr_C1, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C1, y_test_C1))
hist_2 = model_2.fit(X_tr_C2, y_tr_C2, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C2, y_test_C2))
hist_3 = model_3.fit(X_tr_C3, y_tr_C3, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C3, y_test_C3))
hist_4 = model_4.fit(X_tr_C4, y_tr_C4, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C4, y_test_C4))



test_loss_list = [globals()['hist_'+str(cell_num)].history['val_loss'] for cell_num in range(5)]

loss_result = pd.DataFrame({'CV{}'.format(cell_num) : test_loss_list[cell_num] for cell_num in range(5)})

pred_list = [globals()['model_'+str(cell_num)].predict(globals()['X_test_C'+str(cell_num)]).squeeze().tolist() for cell_num in range(5)]


PCOR = []
SCOR = []

for cell_num in range(5):
	test_ans = globals()['y_test_C'+str(cell_num)]
	test_pred = pred_list[cell_num]
	test_pcor, _ = stats.pearsonr(np.array(test_ans).tolist(), test_pred)
	test_scor, _ = stats.spearmanr(np.array(test_ans).tolist(), test_pred)
	PCOR.append(test_pcor)
	SCOR.append(test_scor)


test_result = pd.DataFrame({
	'PCOR' : PCOR,
	'SCOR' : SCOR
})


PRJ_NAME = 'W804'
PRJ_PATH = '/home01/k040a01/03.DeepSynergy/02.RES/'

model_0.save(PRJ_PATH+'DS_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'DS_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'DS_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'DS_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'DS_{}_CV4'.format(PRJ_NAME)) 



loss_result.to_csv(PRJ_PATH + 'RESULT_LOSS.{}'.format(PRJ_NAME))
test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))










##### leave tissue out 


from scipy import stats

model_0 = make_model(X_tr_C0)
model_1 = make_model(X_tr_C1)
model_2 = make_model(X_tr_C2)
model_3 = make_model(X_tr_C3)
model_4 = make_model(X_tr_C4)
model_5 = make_model(X_tr_C5)



max_epoch = 1000

hist_0 = model_0.fit(X_tr_C0, y_tr_C0, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C0, y_test_C0))
hist_1 = model_1.fit(X_tr_C1, y_tr_C1, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C1, y_test_C1))
hist_2 = model_2.fit(X_tr_C2, y_tr_C2, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C2, y_test_C2))
hist_3 = model_3.fit(X_tr_C3, y_tr_C3, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C3, y_test_C3))
hist_4 = model_4.fit(X_tr_C4, y_tr_C4, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C4, y_test_C4))
hist_5 = model_5.fit(X_tr_C5, y_tr_C5, epochs=max_epoch, shuffle=True, batch_size=64, validation_data=(X_test_C5, y_test_C5))



test_loss_list = [globals()['hist_'+str(cell_num)].history['val_loss'] for cell_num in range(6)]

loss_result = pd.DataFrame({'CV{}'.format(cell_num) : test_loss_list[cell_num] for cell_num in range(6)})

pred_list = [globals()['model_'+str(cell_num)].predict(globals()['X_test_C'+str(cell_num)]).squeeze().tolist() for cell_num in range(6)]


PCOR = []
SCOR = []

for cell_num in range(6):
	test_ans = globals()['y_test_C'+str(cell_num)]
	test_pred = pred_list[cell_num]
	test_pcor, _ = stats.pearsonr(np.array(test_ans).tolist(), test_pred)
	test_scor, _ = stats.spearmanr(np.array(test_ans).tolist(), test_pred)
	PCOR.append(test_pcor)
	SCOR.append(test_scor)


test_result = pd.DataFrame({
	'PCOR' : PCOR,
	'SCOR' : SCOR
})


PRJ_NAME = 'W807'
PRJ_PATH = '/home01/k040a01/03.DeepSynergy/02.RES/'

model_0.save(PRJ_PATH+'DS_{}_CV0'.format(PRJ_NAME)) 
model_1.save(PRJ_PATH+'DS_{}_CV1'.format(PRJ_NAME)) 
model_2.save(PRJ_PATH+'DS_{}_CV2'.format(PRJ_NAME)) 
model_3.save(PRJ_PATH+'DS_{}_CV3'.format(PRJ_NAME)) 
model_4.save(PRJ_PATH+'DS_{}_CV4'.format(PRJ_NAME)) 
model_5.save(PRJ_PATH+'DS_{}_CV5'.format(PRJ_NAME)) 



loss_result.to_csv(PRJ_PATH + 'RESULT_LOSS.{}'.format(PRJ_NAME))
test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))



















tail 03.DeepSynergy/02.RES/RESULT.W801.CV5.txt 






##################

SCOR 저장이 안됨 

PRJ_NAME = 'W803'
PRJ_PATH = '/home01/k040a01/03.DeepSynergy/02.RES/'


PCOR = []
SCOR = []

for cell_num in range(27):
	cell_num
	test_ans = globals()['y_test_C'+str(cell_num)]
	test_x = globals()['X_test_C'+str(cell_num)]
	with tf.device('CPU'):
		new_model = tf.keras.models.load_model(PRJ_PATH+'DS_{}_CV{}'.format(PRJ_NAME, cell_num))
	test_pred = new_model.predict(test_x)
	test_pcor, _ = stats.pearsonr(np.array(test_ans).tolist(), test_pred)
	test_scor, _ = stats.spearmanr(np.array(test_ans).tolist(), test_pred)
	PCOR.append(test_pcor)
	SCOR.append(test_scor)


test_result = pd.DataFrame({
	'PCOR' : PCOR,
	'SCOR' : SCOR
})


pcor_check = [a.tolist() for a in list(test_result.PCOR)]


test_result.to_csv(PRJ_PATH + 'RESULT_CORR.{}'.format(PRJ_NAME))











num = 816
DS_res_1 = pd.read_csv('/home01/k040a01/03.DeepSynergy/02.RES/' + 'RESULT_CORR.W{}'.format(num))
DS_res_2 = pd.read_csv('/home01/k040a01/03.DeepSynergy/02.RES/' + 'RESULT_LOSS.W{}'.format(num), index_col = 0)

np.round([np.mean(DS_res_2.loc[999]), np.std(DS_res_2.loc[999])], 4)
np.round([np.mean(DS_res_1.PCOR), np.std(DS_res_1.PCOR)], 4)
np.round([np.mean(DS_res_1.SCOR), np.std(DS_res_1.SCOR)], 4)










num = 803
DS_res_1 = pd.read_csv('/home01/k040a01/03.DeepSynergy/02.RES/' + 'RESULT_CORR.W{}'.format(num))
DS_res_2 = pd.read_csv('/home01/k040a01/03.DeepSynergy/02.RES/' + 'RESULT_LOSS.W{}'.format(num), index_col = 0)

np.round([np.mean(DS_res_2.loc[999]), np.std(DS_res_2.loc[999])], 4)


				mse = np.mean((np.array(ans_list) - np.array(pred_results)) ** 2)
				se_mse = np.sqrt(2 * mse * mse / len(pred_results))

				CfI = stats.t.interval(alpha=0.90, df=len(pred_results)-1,
								loc=mse,
								scale=se_mse)

				np.round(CfI, 4)

np.round([np.mean(DS_res_1.PCOR), np.std(DS_res_1.PCOR)], 4)
np.round([np.mean(DS_res_1.SCOR), np.std(DS_res_1.SCOR)], 4)






