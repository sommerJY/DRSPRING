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




# TOOL_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/'
# TOOL_PATH = '/home/jiyeonH/09.DATA/MATCHMAKER/'
TOOL_PATH = '/home01/k040a01/04.MatchMaker/'

MM_DATA = pd.read_csv(TOOL_PATH + 'DrugCombinationData.tsv', sep = '\t')
CEL_DATA =pd.read_csv(TOOL_PATH + 'E-MTAB-3610.sdrf.txt', sep = '\t')



# performance_metrics
# performance_metrics
# performance_metrics

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






# helper funcs 
# helper funcs 
# helper funcs 

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
drug1 = TOOL_PATH + 'data/drug1_chem.csv'
drug2 = TOOL_PATH + 'data/drug2_chem.csv'
CEL_gex = TOOL_PATH + 'data/cell_line_gex.csv'

chem1, chem2, cell_line, synergies = data_loader(drug1, drug2, CEL_gex, MM_DATA )

MM_comb_data = pd.read_csv(MM_DATA, sep="\t")
# drug_row , drug_col, cell_line_name, synergy_loewe

# >>> chem1.shape
# (286421, 541) 
# >>> chem2.shape
# (286421, 541)
# >>> cell_line.shape
# (286421, 972) # 978 중에서 고른거
# >>> synergies.shape
# (286421,)


# 여기서 우리 데이터랑 비교 시작해야함 

# final add version 

#NETWORK_PATH = '/home/jiyeonH/09.DATA/01.Data/HumanNet/'
#LINCS_PATH = '/home/jiyeonH/09.DATA/01.Data/LINCS/' 
#DATA_PATH = '/home/jiyeonH/09.DATA/'
#DC_PATH = '/home/jiyeonH/09.DATA/01.Data/DrugComb/'
#SAVE_PATH = '/home/jiyeonH/09.DATA/'


#NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
#LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
#DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'
#DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
#SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_W32_349_FULL/'



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

file_name = 'M3V6_349_MISS2_FULL' # 0608

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
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

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] # 8086 -> 이걸 빼야하나 말아야하나 #################

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

data_ind = list(A_B_C_S_SET_COH.index)

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






























































# 일단 CID 랑 cell line 부터 

# Drug info 
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]

DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')

DC_lengs = list(DC_DRUG_DF_FULL.leng)
DC_lengs2 = [int(a) for a in DC_lengs if a!= 'error']


# Cell info 
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)

DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(0,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 2319
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']



# 데이터 아이디 맞는지 확인 

set(MM_comb_data.cell_line_name) - set(DC_CELL_DF2.DC_cellname)
# {'786-0'}
아......
시봉빵봉 
이게 무슨일이죠 
그렇습니다. 얘네 덕에 잃어버린 cell line 하나 건짐 


set(MM_comb_drug) - set(DC_DRUG_DF.dname)

MM_comb_drug_match_name = list(set(list(MM_comb_data.drug_row) + list(MM_comb_data.drug_col))) # 3040
MM_drug_match = pd.DataFrame({'MM_drug' : MM_comb_drug_match_name  })


MM_drug_CID_name = {key : MM_drug_info[key]['name'] for key in MM_drug_info.keys()} # 3952
missing = [a for a in MM_comb_drug_match_name if a not in MM_drug_CID_name.values()] # 100?

MM_drug_CID_name_DF = pd.DataFrame(MM_drug_CID_name, index = [0])
MM_drug_CID_name_DF = MM_drug_CID_name_DF.T
MM_drug_CID_name_DF.columns = ['MM_drug']
MM_drug_CID_name_DF['MM_CID'] = list(MM_drug_CID_name_DF.index)

MM_drug_match_2 = pd.merge(MM_drug_match, MM_drug_CID_name_DF, on='MM_drug', how='left')

for indind in range(MM_drug_match_2.shape[0]) :
    if type(list(MM_drug_match_2.MM_CID)[indind]) != str :
        MM_drug_match_2.at[indind,'MM_CID'] = '0'

MM_drug_match_2['MM_CID'] = [int(a) for a in MM_drug_match_2['MM_CID']]

MM_drug_match_ok = MM_drug_match_2[MM_drug_match_2.MM_CID>0] # 119
MM_drug_match_miss = MM_drug_match_2[MM_drug_match_2.MM_CID==0] # 119


일단 MM 에서 사용했다고 제공한 feat 랑 내가 git 에서 가져온거랑 일치하는지? 
1) 그냥 잘 맞는 애 
M344      3994

    논문에서 제공받은거 
drug_ind = MM_comb_data[MM_comb_data.drug_row == 'M344'].index[0].item()
chem1[drug_ind] # 541,
chem1[drug_ind][0:100]

    깃헙에서 주운거 
CID = '3994'
MM_drug_info[CID]
len(MM_drug_info[CID]['chemicals']) # 541 

np.round(chem1[drug_ind].tolist(),4) == np.round(MM_drug_info[CID]['chemicals'],4)


            그래서 이 100 개에 대해서 그냥 아예 chem feat 같은걸 찾아주는게 어떨지? 


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

5284373, 36462, 5790

cid = 5790
names = list(MM_drug_match_final[MM_drug_match_final.MM_CID == cid]['MM_drug'])
drug_ind_1 = MM_comb_data[MM_comb_data.drug_row == names[0]].index[0].item()
drug_ind_2 = MM_comb_data[MM_comb_data.drug_row == names[1]].index[0].item()
drug_ind_3 = MM_comb_data[MM_comb_data.drug_row == names[2]].index[0].item()


all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_2],4))
all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_3],4))
all(np.round(chem1[drug_ind_2],4) == np.round(chem1[drug_ind_3],4))

# 다 그런거구먼 
# 겹치는 경우엔 어떻게 진행했으려나 

MM_drug_match_final.columns = ['drug_row','drug_row_cid']
MM_comb_data_RE = pd.merge(MM_comb_data, MM_drug_match_final, on ='drug_row', how = 'left')

MM_drug_match_final.columns = ['drug_col','drug_col_cid']
MM_comb_data_RE = pd.merge(MM_comb_data_RE, MM_drug_match_final, on ='drug_col', how = 'left')










이제 내 데이터랑 매칭 시켜야함 


JY_DATA_PATH = '/home01/k040a01/02.M3V5/M3V5_W32_349_DATA/'

내꺼 ADD 저장된거 가져와서 필터링 후 COH2 가지고 비교

DRSPRING_DATA_ALL = pd.concat([A_B_C_S_SET_COH])
DRSPRING_DATA_ALL['drug_row_CID'] = [int(a.split('___')[0]) for a in DRSPRING_DATA_ALL.CID_CID]
DRSPRING_DATA_ALL['drug_col_CID'] = [int(a.split('___')[1]) for a in DRSPRING_DATA_ALL.CID_CID]



DRSPRING_CID = list(set(list(DRSPRING_DATA_ALL.drug_row_CID) + list(DRSPRING_DATA_ALL.drug_col_CID))) # 1342
DRSPRING_CELL = list(set(DRSPRING_DATA_ALL.DC_cellname))

MM_redata_1 = MM_comb_data_RE[MM_comb_data_RE.cell_line_name.isin(DRSPRING_CELL)] # 40190
MM_redata_2 = MM_redata_1[MM_redata_1.drug_row_cid.isin(DRSPRING_CID)] # 28510
MM_redata_3 = MM_redata_2[MM_redata_2.drug_col_cid.isin(DRSPRING_CID)] # 21042


aaa = list(MM_redata_3['drug_row_cid'])
bbb = list(MM_redata_3['drug_col_cid'])
ccc = list(MM_redata_3['cell_line_name'])
aa = list(MM_redata_3['drug_row']) 
bb = list(MM_redata_3['drug_col']) 


# 306
MM_redata_3['CID_CID'] = [str(int(aaa[i])) + '___' + str(int(bbb[i])) if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i])) for i in range(MM_redata_3.shape[0])]

# 10404 -- duplicated 가 이상한게 아님 
MM_redata_3['CID_CID_CELL'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + ccc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + ccc[i] for i in range(MM_redata_3.shape[0])]

# 
MM_redata_3['id_id_CELL'] = [str(aa[i]) + '___' + str(bb[i])+ '___' + ccc[i] if aa[i] < bb[i] else str(bb[i]) + '___' + str(aa[i])+ '___' + ccc[i] for i in range(MM_redata_3.shape[0])]



len(set(MM_redata_3.CID_CID)) # 4963
len(set(MM_redata_3.CID_CID_CELL)) # 101878
len(set(MM_redata_3.id_id_CELL)) # 101931
# duplicated CID 여기도 있음. 논문상에서는 제거하고 한것 같기도. 

set_data = list(set(MM_redata_3['CID_CID_CELL']))
set_data.sort()

random.seed(24)
random.shuffle(set_data)
# set_data[0:10]
bins = [round(len(set_data)*0.2*a) for a in range(1,5)]
res = np.split(set_data, bins)




# CID Check 


# 데이터 나누기 

test_ind = 
val_ind = 
train_ind = 







def prepare_data(chem1, chem2, cell_line, synergies, norm, 
	train_ind_fname, val_ind_fname, test_ind_fname):
    print("Data normalization and preparation of train/validation/test data")
    test_ind = list(np.loadtxt(test_ind_fname,dtype=np.int))
    val_ind = list(np.loadtxt(val_ind_fname,dtype=np.int))
    train_ind = list(np.loadtxt(train_ind_fname,dtype=np.int))
    #
    train_data = {}
    val_data = {}
    test_data = {}
    #
    # chem 1 이랑 2 를 붙인다고? 
    train1 = np.concatenate((chem1[train_ind,:],chem2[train_ind,:]),axis=0) # 엥 왜 
    train_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(train1, norm=norm)
    val_data['drug1'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem1[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_data['drug1'], mean1, std1, mean2, std2, feat_filt = normalize(chem1[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    #
    train2 = np.concatenate((chem2[train_ind,:],chem1[train_ind,:]),axis=0)
    train_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(train2, norm=norm)
    val_data['drug2'], mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(chem2[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_data['drug2'], mean1, std1, mean2, std2, feat_filt = normalize(chem2[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    #
    train3 = np.concatenate((cell_line[train_ind,:],cell_line[train_ind,:]),axis=0)
    train_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(train3, norm=norm)
    val_cell_line, mmean1, sstd1, mmean2, sstd2, feat_filtt = normalize(cell_line[val_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    test_cell_line, mean1, std1, mean2, std2, feat_filt = normalize(cell_line[test_ind,:],mean1, std1, mean2, std2, feat_filt=feat_filt, norm=norm)
    #
    train_data['drug1'] = np.concatenate((train_data['drug1'],train_cell_line),axis=1)
    train_data['drug2'] = np.concatenate((train_data['drug2'],train_cell_line),axis=1)
    #
    val_data['drug1'] = np.concatenate((val_data['drug1'],val_cell_line),axis=1)
    val_data['drug2'] = np.concatenate((val_data['drug2'],val_cell_line),axis=1)
    #
    test_data['drug1'] = np.concatenate((test_data['drug1'],test_cell_line),axis=1)
    test_data['drug2'] = np.concatenate((test_data['drug2'],test_cell_line),axis=1)
    #
    train_data['y'] = np.concatenate((synergies[train_ind],synergies[train_ind]),axis=0)
    val_data['y'] = synergies[val_ind]
    test_data['y'] = synergies[test_ind]
    print(test_data['drug1'].shape)
    print(test_data['drug2'].shape)
    return train_data, val_data, test_data


norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data(chem1, chem2, cell_line, synergies, norm,
                                            train_indx, val_indx, test_indx)

# >>> train_data.keys()
# dict_keys(['drug1', 'drug2', 'y'])


# calculate weights for weighted MSE loss 
min_s = np.amin(train_data['y'])
loss_weight = np.log(train_data['y'] - min_s + np.e) # log 취해서 다 고만고만한 값이 됨 

# load architecture file
architecture = pd.read_csv('architecture.txt')

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



MM_Model = generate_network(train_data, layers, inDrop, drop)


def trainer(model, l_rate, train, val, epo, batch_size, earlyStop, modelName,weights):
    cb_check = ModelCheckpoint((modelName), verbose=1, monitor='val_loss',save_best_only=True, mode='auto')
    model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(lr=float(l_rate), beta_1=0.9, beta_2=0.999, amsgrad=False))
    model.fit([train["drug1"], train["drug2"]], train["y"], epochs=epo, shuffle=True, batch_size=batch_size, verbose=1, 
                   validation_data=([val["drug1"], val["drug2"]], val["y"]),sample_weight=weights,
                   callbacks=[EarlyStopping(monitor='val_loss', mode='auto', patience = earlyStop),cb_check])
    return model



def predict(model, data):
    pred = model.predict(data)
    return pred.flatten()




parser = argparse.ArgumentParser(description='REQUEST REQUIRED PARAMETERS OF MatchMaker')

parser.add_argument('--comb-data-name', default='data/DrugCombinationData.tsv',
                    help="Name of the drug combination data")

parser.add_argument('--cell_line-gex', default='data/cell_line_gex.csv',
                    help="Name of the cell line gene expression data")

parser.add_argument('--drug1-chemicals', default='data/drug1_chem.csv',
                    help="Name of the chemical features data for drug 1")

parser.add_argument('--drug2-chemicals', default='data/drug2_chem.csv',
                    help="Name of the chemical features data for drug 2")

parser.add_argument('--gpu-devices', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')

parser.add_argument('--train-test-mode', default=1, type = int,
                    help="Test of train mode (0: test, 1: train)")

parser.add_argument('--train-ind', default='data/train_inds.txt',
                    help="Data indices that will be used for training")

parser.add_argument('--val-ind', default='data/val_inds.txt',
                    help="Data indices that will be used for validation")

parser.add_argument('--test-ind', default='data/test_inds.txt',
                    help="Data indices that will be used for test")

parser.add_argument('--arch', default='data/architecture.txt',
                    help="Architecute file to construct MatchMaker layers")

parser.add_argument('--gpu-support', default=True,
                    help='Use GPU support or not')

parser.add_argument('--saved-model-name', default="matchmaker.h5",
                    help='Model name to save weights')
args = parser.parse_args()



###
num_cores = 128
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_devices
GPU = True
if args.gpu_support:
    num_GPU = 1
    num_CPU = 1
else:
    num_CPU = 2
    num_GPU = 0


config = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=True,
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
                       )



tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

# load and process data
chem1, chem2, cell_line, synergies = MatchMaker.data_loader(args.drug1_chemicals, args.drug2_chemicals,
                                                args.cell_line_gex, args.comb_data_name)
# normalize and split data into train, validation and test
norm = 'tanh_norm'
train_data, val_data, test_data = MatchMaker.prepare_data(chem1, chem2, cell_line, synergies, norm,
                                            args.train_ind, args.val_ind, args.test_ind)







model = trainer(MM_Model, l_rate, train_data, val_data, max_epoch, batch_size,
                                earlyStop_patience, modelName, loss_weight)

# 그냥 돌리면 돌아감 ->

model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

if (args.train_test_mode == 1):
    # if we are in training mode
    model = MatchMaker.trainer(model, l_rate, train_data, val_data, max_epoch, batch_size,
                                earlyStop_patience, modelName, loss_weight)
# load the best model
model.load_weights(modelName)

# predict in Drug1, Drug2 order
pred1 = MatchMaker.predict(model, [test_data['drug1'],test_data['drug2']])
mse_value = performance_metrics.mse(test_data['y'], pred1)
spearman_value = performance_metrics.spearman(test_data['y'], pred1)
pearson_value = performance_metrics.pearson(test_data['y'], pred1)
np.savetxt("pred1.txt", np.asarray(pred1), delimiter=",")
np.savetxt("y_test.txt", np.asarray(test_data['y']), delimiter=",")
# predict in Drug2, Drug1 order
pred2 = MatchMaker.predict(model, [test_data['drug2'],test_data['drug1']])
# take the mean for final prediction
pred = (pred1 + pred2) / 2

mse_value = performance_metrics.mse(test_data['y'], pred)
spearman_value = performance_metrics.spearman(test_data['y'], pred)
pearson_value = performance_metrics.pearson(test_data['y'], pred)




print(test_data['drug1'].shape)
print(test_data['drug1'].shape)
print(test_data['drug1'].shape)



