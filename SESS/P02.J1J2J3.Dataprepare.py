
import rdkit
import os
import os.path as osp
from math import ceil
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch.utils.data import Dataset

from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool #DiffPool
from torch_geometric.nn import SAGEConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pickle
import joblib
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import sklearn
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = sklearn.preprocessing.OneHotEncoder
import datetime
from datetime import *
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error
												
# Graph convolution model
import torch_geometric.nn as pyg_nn
# Graph utility function
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import torch_geometric.nn.conv
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs


import networkx as nx
import copy
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy import stats
import sklearn.model_selection

import sys
import random
import shutil
import math

import ray
from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd


PRJ_NAME = 'M1_J1'
PRJ_NAME = 'M2_J3'
PRJ_NAME = 'M3_J3'


# WORK_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M1_J1/'
# WORK_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M1_J1/'



DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'




# Drug Comb 데이터 가져오기 
# synergy info 
DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF1_re = DC_tmp_DF1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
DC_tmp_DF1_re['drug_row_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_row_id'])]
DC_tmp_DF1_re['drug_col_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_col_id'])]

# drug len check : drug_ids = list(set(list(DC_tmp_DF1.drug_row_id_re) + list(DC_tmp_DF1.drug_col_id_re))) # 15811
# cell len check : cell_ids = list(set(DC_tmp_DF1.cell_line_id)) # 2040

DC_tmp_DF2 = DC_tmp_DF1_re[DC_tmp_DF1_re['quality'] != 'bad'] # 1457561
# drug len check : drug_ids = list(set(list(DC_tmp_DF2.drug_row_id_re) + list(DC_tmp_DF2.drug_row_id_re))) # 8333
# cell len check : cell_ids = list(set(DC_tmp_DF2.cell_line_id)) # 2040


# Drug info 
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]



						# drug info merge to pubchem data (특히 smiles 아 여기 smiles 없는데..? )
						DC_DRUG_DF_FULL = pd.merge(DC_DRUG_DF, PUBCHEM_ALL , left_on = 'cid', right_on ='CID', how = 'left')


						smiles_df = DC_DRUG_DF_FULL[['cid','CAN_SMILES']].drop_duplicates()
						only_smiles = list(smiles_df['CAN_SMILES'])
						leng = []

						for i in range(len(only_smiles)) :
							smiles = only_smiles[i]
							z = 0
							try:
								iMol = Chem.MolFromSmiles(smiles.strip())
								NUM = iMol.GetNumAtoms()
								leng.append(NUM)
							except:
								leng.append("error")
								print("error",z,i)

						smiles_df['leng'] = leng # max : 115

						DC_DRUG_DF_FULL = pd.merge(DC_DRUG_DF_FULL, smiles_df[['cid','leng']], on = 'cid', how = 'left')
# DC_DRUG_DF_FULL.to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')
DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')
# pubchem canonical smiles 쓰고 length 값 들어간 버전 


DC_lengs = list(DC_DRUG_DF_FULL.leng)
DC_lengs2 = [int(a) for a in DC_lengs if a!= 'error']


#plot_dir = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/'
#plt.hist(DC_lengs2 , range = (0, 295), bins = 295)
#plt.legend()
#plt.savefig(plot_dir+"DrugComb_CIDs_LENG.png", format = 'png', dpi=300)



# Cell info 
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)

DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 2319
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']








print("DC filtering")

DC_DATA_filter = DC_tmp_DF2[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958 -> 1363698

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re > 0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re > 0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates() # 648516
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re']].drop_duplicates() # 75174
len(list(set(list(DC_DATA_filter4.drug_row_id_re) + list(DC_DATA_filter4.drug_col_id_re)))) # 4327



# cid renaming
DC_DRUG_DF2 = DC_DRUG_DF_FULL[['id_re','dname','cid','CAN_SMILES']] # puibchem 공식 smiles 
																	# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id_re','drug_row','drug_row_CID', 'drug_row_sm']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id_re', how='left' ) # 751450 -> 740884

DC_DRUG_DF2.columns = ['drug_col_id_re','drug_col','drug_col_CID', 'drug_col_sm']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id_re', how='left') # 751450 -> 740884


#  Add cell data and cid filter
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 751450 -> 740884
DC_DATA7_1 = DC_DATA7_1[['drug_row_CID','drug_col_CID','DrugCombCello','synergy_loewe']].drop_duplicates() # 740882


# filtering 
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_CID>0] # 740882 -> 737104
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_CID>0] # 737104 -> 725496
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]
DC_DATA7_4_cello = DC_DATA7_3[cello_t] #  720249
TF_check = [True if np.isnan(a)==False else False for a in DC_DATA7_4_cello.synergy_loewe] 
DC_DATA7_5_cello = DC_DATA7_4_cello[TF_check] # 719946

DC_cello_final = DC_DATA7_5_cello[['drug_row_CID','drug_col_CID','DrugCombCello']].drop_duplicates() # 554128
DC_cello_final_dup = DC_DATA7_5_cello[['drug_row_CID','drug_col_CID','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348 -> 719946

DC_cello_final_cids = list(set(list(DC_cello_final_dup.drug_row_CID) + list(DC_cello_final_dup.drug_col_CID)))
# 4302





# LINCS data filter 
# LINCS data filter 
# LINCS data filter 


BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)


# pert type 확인 
filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996 
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460 samples, 58 cells, 30456 compounds 


# 한번 더 pubchem converter 로 내가 붙인 애들 
BETA_CP_info_filt = BETA_CP_info[['pert_id','canonical_smiles']].drop_duplicates() # 34419


can_sm_re = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/can_sm_conv', sep = '\t', header = None)

can_sm_re.columns = ['canonical_smiles','CONV_CID']
can_sm_re = can_sm_re.drop_duplicates()
len(set([a for a in BETA_CP_info['pert_id'] if type(a) == str])) # 34419
len(set([a for a in can_sm_re['canonical_smiles'] if type(a) == str])) # 28575
len(set(can_sm_re[can_sm_re.CONV_CID>0]['CONV_CID'])) # 27841


can_sm_re2 = pd.merge(BETA_CP_info_filt, can_sm_re, on = 'canonical_smiles', how = 'left') # 34419 -> 1 sm 1 cid 확인 

can_sm_re3 = can_sm_re2[['pert_id','canonical_smiles','CONV_CID']].drop_duplicates() # 
# converter 가 더 많이 붙여주는듯 


# 민지가 pcp 로 붙인 애들 
BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 
# 문제는, pert id 에 붙는 cid 겹치는 애들이 있다는거.. 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles']].drop_duplicates() # 25903
BETA_MJ_RE_CK = BETA_MJ_RE[['pert_id','SMILES_cid']]
len(set([a for a in BETA_MJ_RE['pert_id'] if type(a) == str])) # 25903
len(set([a for a in BETA_MJ_RE['canonical_smiles'] if type(a) == str])) # 25864
len(set(BETA_MJ_RE_CK[BETA_MJ_RE_CK.SMILES_cid>0]['SMILES_cid'])) # 25642

check = pd.merge(can_sm_re3, BETA_MJ_RE_CK, on = 'pert_id', how = 'left' )
check2 = check[check.CONV_CID !=check.SMILES_cid]
check3 = check2[check2.SMILES_cid > 0 ]
check4 = check3[check3.CONV_CID > 0 ]
# 그래서 둘이 안맞는게 58 개 정도 있어서... CID 를 최대한 붙이기 위해서 그러면.. 
# 근데 그래서 대충 drubcomb 에 있는 CID 랑 얼마나 겹치나 보니... 그냥 저냥 conv 10 개, mj 3 개 

# sum(check4.CONV_CID.isin(request_mj.CID)) # 10 
# sum(check4.SMILES_cid.isin(request_mj.CID)) # 3 

# 버리려고 했는데, 버리면 안될지도? 
# exemplar 에 다 들어가는 기염을 토하고 있음 ㅎ 



# LINCS match final 
pert_id_match = check[check.CONV_CID == check.SMILES_cid][['pert_id','canonical_smiles','CONV_CID']]
# sum((check2.CONV_CID >0 ) &( np.isnan(check2.SMILES_cid)==True)) # 2521
# sum((check2.SMILES_cid >0 ) &( np.isnan(check2.CONV_CID)==True)) # 427
conv_win = check2[(check2.CONV_CID >0 ) & ( np.isnan(check2.SMILES_cid)==True)][['pert_id','canonical_smiles','CONV_CID']]
mj_win = check2[(check2.SMILES_cid >0 ) & ( np.isnan(check2.CONV_CID)==True)][['pert_id','canonical_smiles','SMILES_cid']]
nans = check2[(np.isnan(check2.SMILES_cid)==True ) & ( np.isnan(check2.CONV_CID)==True)] # 5995
nans2 = nans[nans.pert_id.isin(filter2.pert_id)]
nans3 = nans2[-nans2.canonical_smiles.isin(['restricted', np.nan])]
# 한 162개 정도는 아예 안붙음. 심지어 우리의 exemplar 에도 있지만.. 흑... 

pert_id_match.columns = ['pert_id','canonical_smiles','CID'] # 25418,
conv_win.columns = ['pert_id','canonical_smiles','CID'] # 2521,
mj_win.columns =['pert_id','canonical_smiles','CID']


individual_check = check4.reset_index(drop =True)

individual_check_conv = individual_check.loc[[0,4,5,6,10,11,12,13,16,17,18,19]+[a for a in range(21,34)]+[36,40,54]][['pert_id','canonical_smiles','CONV_CID']]
individual_check_mj = individual_check.loc[[1,2,3,7,8,9,14,15,20,34,35,37,38,39]+[a for a in range(41,54)]+[55,56,57]][['pert_id','canonical_smiles','SMILES_cid']]
# [a for a in individual_check_conv.index if a in individual_check_mj.index]
individual_check_conv.columns = ['pert_id','canonical_smiles','CID'] # 28
individual_check_mj.columns = ['pert_id','canonical_smiles','CID'] # 30 


LINCS_PERT_MATCH = pd.concat([pert_id_match, conv_win, mj_win, individual_check_conv,  individual_check_mj]) # 28424
len(set([a for a in LINCS_PERT_MATCH['pert_id'] if type(a) == str])) # 34419 -> 28424
len(set([a for a in LINCS_PERT_MATCH['canonical_smiles'] if type(a) == str])) # 28575 -> 28381
len(set(LINCS_PERT_MATCH[LINCS_PERT_MATCH.CID>0]['CID'])) # 27841 -> 28154
LINCS_PERT_MATCH_cids = list(set(LINCS_PERT_MATCH.CID))

# LINCS_PERT_MATCH.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')

# aa = list(LINCS_PERT_MATCH.CID)
# [a for a in aa if aa.count(a)>1]
# 10172943

# merge with exemplar sigid 
BETA_EXM = pd.merge(filter2, LINCS_PERT_MATCH, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.CID > 0] # 128038 # 이건 늘어났음 

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 128038
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','CID','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 128038


cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)] 
# sum(cello_tt) : 111012
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','CID','cellosaurus_id','sig_id']].drop_duplicates() # 111012
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.CID)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]
BETA_CID_CELLO_SIG['CID'] = [int(a) for a in list(BETA_CID_CELLO_SIG['CID']) ] # 111012 
# 늘어남! 

# -> CCLE 필터까지 완료 / DC CID 고정 완료 


# MATCH DC & LINCS 
print('DC and LINCS') 
# 무조건 pubchem 공식 파일 사용하기 (mj ver) 
# 문제는 pert_id 에 CID 가 중복되어 붙는 애들도 있다는거  -> 어떻게 해결할거? 
# pert id 로 나눠서 하는게 맞을것 같음.


DC_cello_final_dup_ROW_CHECK = list(DC_cello_final_dup.drug_row_CID)
DC_cello_final_dup_COL_CHECK = list(DC_cello_final_dup.drug_col_CID)
DC_cello_final_dup_CELL_CHECK = list(DC_cello_final_dup.DrugCombCello)

DC_cello_final_dup['ROWCHECK'] = [str(int(DC_cello_final_dup_ROW_CHECK[i]))+'__'+DC_cello_final_dup_CELL_CHECK[i] for i in range(DC_cello_final_dup.shape[0])]
DC_cello_final_dup['COLCHECK'] = [str(int(DC_cello_final_dup_COL_CHECK[i]))+'__'+DC_cello_final_dup_CELL_CHECK[i] for i in range(DC_cello_final_dup.shape[0])]



# 공식 smiles 



for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv',sep = '\t', low_memory = False)
# for_CAN_smiles = copy.deepcopy(PC_FILTER)
for_CAN_smiles = for_CAN_smiles[['CID','CAN_SMILES']]
for_CAN_smiles.columns = ['drug_row_CID','ROW_CAN_SMILES']
DC_cello_final_dup = pd.merge(DC_cello_final_dup, for_CAN_smiles, on='drug_row_CID', how ='left' )
for_CAN_smiles.columns = ['drug_col_CID','COL_CAN_SMILES']
DC_cello_final_dup = pd.merge(DC_cello_final_dup, for_CAN_smiles, on='drug_col_CID', how ='left' )
for_CAN_smiles.columns = ['CID','CAN_SMILES']




# CAN_SMILES NA 있음?  
CAN_TF_1 = [True if type(a) == float else False for a in list(DC_cello_final_dup.ROW_CAN_SMILES)]
CAN_TF_DF_1 = DC_cello_final_dup[CAN_TF_1]
CAN_TF_2 = [True if type(a) == float else False for a in list(DC_cello_final_dup.COL_CAN_SMILES)]
CAN_TF_DF_2 = DC_cello_final_dup[CAN_TF_2]
# DC 기준으로는 없음. LINCS 기준에서는 있었음 




BETA_CID_CELLO_SIG_ID_CHECK = list(BETA_CID_CELLO_SIG.CID)
BETA_CID_CELLO_SIG_CELL_CHECK = list(BETA_CID_CELLO_SIG.cellosaurus_id)

BETA_CID_CELLO_SIG['IDCHECK'] = [str(int(BETA_CID_CELLO_SIG_ID_CHECK[i]))+'__'+BETA_CID_CELLO_SIG_CELL_CHECK[i] for i in range(BETA_CID_CELLO_SIG.shape[0])]
# 이렇게 되면 , IDCHECK 에 중복이 생기긴 함. pert 때문에. 


BETA_CID_CELLO_SIG.columns=['ROW_pert_id', 'drug_row_CID', 'DrugCombCello', 'ROW_BETA_sig_id',  'ROWCHECK']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG[['ROW_pert_id', 'ROW_BETA_sig_id',  'ROWCHECK']], left_on = 'ROWCHECK', right_on = 'ROWCHECK', how = 'left') # 720619

BETA_CID_CELLO_SIG.columns=['COL_pert_id', 'drug_col_CID', 'DrugCombCello', 'COL_BETA_sig_id', 'COLCHECK']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG[['COL_pert_id', 'COL_BETA_sig_id', 'COLCHECK']], left_on = 'COLCHECK', right_on = 'COLCHECK', how = 'left') # 721206

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id', 'IDCHECK']





# (1) AO BO  
FILTER_AO_BO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.ROW_BETA_sig_id[a]) == str) & (type(CELLO_DC_BETA_2.COL_BETA_sig_id[a]) == str)]
DATA_AO_BO = CELLO_DC_BETA_2.loc[FILTER_AO_BO] # 11379 
DATA_AO_BO[['drug_row_CID','drug_col_CID','DrugCombCello']].drop_duplicates() # 8404
DATA_AO_BO[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCello']].drop_duplicates() # 8914
DATA_AO_BO[['ROW_pert_id','COL_pert_id','DrugCombCello']].drop_duplicates() #  8914
DATA_AO_BO[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCello']].drop_duplicates() #  8404
DATA_AO_BO_cids = list(set(list(DATA_AO_BO.drug_row_CID) + list(DATA_AO_BO.drug_col_CID))) # 172 



# (2) AX BO 
FILTER_AX_BO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.ROW_BETA_sig_id[a]) != str) & (type(CELLO_DC_BETA_2.COL_BETA_sig_id[a]) == str)]
DATA_AX_BO = CELLO_DC_BETA_2.loc[FILTER_AX_BO] # 11967
DATA_AX_BO[['drug_row_CID','drug_col_CID','DrugCombCello']].drop_duplicates() # 9428
DATA_AX_BO[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCello']].drop_duplicates() # 452
DATA_AX_BO[['ROW_pert_id','COL_pert_id','DrugCombCello']].drop_duplicates() #  452
DATA_AX_BO[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCello']].drop_duplicates() #  9414 -> 겹치는 애들 제거해야하는걸로! 
DATA_AX_BO_cids = list(set(list(DATA_AX_BO.drug_row_CID) + list(DATA_AX_BO.drug_col_CID))) # 635 


tmp = DATA_AX_BO[['drug_row_CID','DrugCombCello']].drop_duplicates() # 1274
tmp1 = [str(a) for a in tmp['drug_row_CID']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCello'])[a] for a in range(tmp.shape[0])] # 1274
len(tmp2)



# (3) AO BX 
FILTER_AO_BX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.ROW_BETA_sig_id[a]) == str) & (type(CELLO_DC_BETA_2.COL_BETA_sig_id[a]) != str)]
DATA_AO_BX = CELLO_DC_BETA_2.loc[FILTER_AO_BX] # 14998
DATA_AO_BX[['drug_row_CID','drug_col_CID','DrugCombCello']].drop_duplicates() # 12926
DATA_AO_BX[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCello']].drop_duplicates() # 449
DATA_AO_BX[['ROW_pert_id','COL_pert_id','DrugCombCello']].drop_duplicates() #  449
DATA_AO_BX[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCello']].drop_duplicates() #  12915 -> 겹치는 애들 제거해야하는걸로! 
DATA_AO_BX_cids = list(set(list(DATA_AO_BX.drug_row_CID) + list(DATA_AO_BX.drug_col_CID))) # 274 


tmp = DATA_AO_BX[['drug_col_CID','DrugCombCello']].drop_duplicates()
tmp1 = [str(a) for a in tmp['drug_col_CID']]
tmp2 = [tmp1[a] + "__" + list(tmp['DrugCombCello'])[a] for a in range(len(tmp1))] # 900


# (4) AX BX 
FILTER_AX_BX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.ROW_BETA_sig_id[a]) != str) & (type(CELLO_DC_BETA_2.COL_BETA_sig_id[a]) != str)]
DATA_AX_BX = CELLO_DC_BETA_2.loc[FILTER_AX_BX] # 584465
DATA_AX_BX = DATA_AX_BX[DATA_AX_BX.DrugCombCello!='NA']
DATA_AX_BX[['drug_row_CID','drug_col_CID','DrugCombCello']].drop_duplicates() # 506710
DATA_AX_BX[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCello']].drop_duplicates() # 232 의미 없음 
DATA_AX_BX[['ROW_pert_id','COL_pert_id','DrugCombCello']].drop_duplicates() #  232 의미 없음 
DATA_AX_BX[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCello']].drop_duplicates() #  8404
DATA_AX_BX_cids = list(set(list(DATA_AX_BX.drug_row_CID) + list(DATA_AX_BX.drug_col_CID))) # 4280 


tmp_r = DATA_AX_BX[['drug_row_CID','DrugCombCello']].drop_duplicates() # 18162
tmp_c = DATA_AX_BX[['drug_col_CID','DrugCombCello']].drop_duplicates() # 12702
tmp1 = [str(a) for a in tmp_r['drug_row_CID']]
tmp2 = [str(a) for a in tmp_c['drug_col_CID']]
tmp3 = [tmp1[a] + "__" + list(tmp_r['DrugCombCello'])[a] for a in range(len(tmp1))] # 
tmp4 = [tmp2[a] + "__" + list(tmp_c['DrugCombCello'])[a] for a in range(len(tmp2))] # 
len(set(tmp3+tmp4)) # 19429


DATA_AO_BO['type'] = 'AOBO' # 11379
DATA_AX_BO['type'] = 'AXBO' # 11967
DATA_AO_BX['type'] = 'AOBX' # 14998
DATA_AX_BX['type'] = 'AXBX' # 584465



(5) AXBO + AOBX

DATA_AB_ONE = pd.concat([DATA_AO_BX, DATA_AX_BO])
DATA_AB_ONE[['drug_row_CID','drug_col_CID','DrugCombCello']].drop_duplicates() # 22354
DATA_AB_ONE[['ROW_BETA_sig_id','COL_BETA_sig_id','DrugCombCello']].drop_duplicates() # 901
DATA_AB_ONE[['ROW_pert_id','COL_pert_id','DrugCombCello']].drop_duplicates() #  901
DATA_AB_ONE[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCello']].drop_duplicates() #  22329 








#################################################################################################
##################################################################################################






print('NETWORK')
# HUMANNET 사용 

hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()
lm_entrezs = list(BETA_lm_genes.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(BETA_lm_genes.gene_id)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 3885

len(set(list(hnet_L2['G_A']) + list(hnet_L2['G_B']))) # 611

ID_G = nx.from_pandas_edgelist(hnet_L2, 'G_A', 'G_B')

MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


# 유전자 이름으로 붙이기 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)


new_node_names = []
for a in ID_G.nodes():
	tmp_name = LINCS_978[LINCS_978.gene_id == a ]['gene_symbol'].item() # 6118
	new_node_name = str(a) + '__' + tmp_name
	new_node_names = new_node_names + [new_node_name]

mapping = {list(ID_G.nodes())[a]:new_node_names[a] for a in range(len(new_node_names))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE




# TARGET # 이거 이제 완료하고 그다음 단계로 넘어가야해 

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t')


TARGET_DB.columns= [ 'CID_RE','gene_symbol','DB','L_gene_symbol','EntrezID' ]
TARGET_DB['CID'] = list(TARGET_DB.CID_RE)


# OLD_TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
# OLD_TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
# OLD_TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
TARGET_DB_ori = pd.read_csv(OLD_TARGET_PATH+'combined_target.csv', low_memory=False, index_col = 0)
TARGET_DB_ori.columns = ['CID','gene_symbol','DB']
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/' 
L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')
L_check = L_matching_list[['L_gene_symbol','entrez']]

TARGET_DB_ori2 = pd.merge(TARGET_DB_ori, L_check, left_on = 'gene_symbol', right_on = 'L_gene_symbol', how = 'left')
TARGET_DB_ori3 = TARGET_DB_ori2[TARGET_DB_ori2.entrez>0]

TARGET_DB_ori3.columns= [ 'CID_RE','gene_symbol','DB','L_gene_symbol','EntrezID' ]
TARGET_DB_ori3['CID'] = list(TARGET_DB_ori3.CID_RE)

TARGET_DB = copy.deepcopy(TARGET_DB_ori3)


# 잠시 비교

CID = 57363
set(TARGET_DB_ori3[TARGET_DB_ori3.CID == CID]['entrez'])

set(TARGET_DB[TARGET_DB.CID_RE== CID]['ENTREZ_RE'])

#########################################################################
#########################################################################
#########################################################################
#########################################################################

# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE





# data construction 

(1) NONE MISSING : MISS_0

A_B_C_S = DATA_AO_BO.reset_index(drop = True) # 11379
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCello']].drop_duplicates()


(2) one missing : MISS_1
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 38344 
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCello']].drop_duplicates()


(3) two missing : MISS_2
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 622809
A_B_C_S[['drug_row_CID','drug_col_CID', 'DrugCombCello']].drop_duplicates()



# drug target filter 진행 
A_B_C_S_row = A_B_C_S[A_B_C_S.drug_row_CID.isin(list(TARGET_DB.CID_RE))]
A_B_C_S_col = A_B_C_S_row[A_B_C_S_row.drug_col_CID.isin(list(TARGET_DB.CID_RE))]
A_B_C_S_col.shape
A_B_C_S_SET = copy.deepcopy(A_B_C_S_col)
A_B_C_S_SET = A_B_C_S_SET.drop('synergy_loewe', axis = 1).drop_duplicates()
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)


# LINCS exp order 따지기 
BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)




# check length
def check_len_num(SMILES):
	tf = []
	re_try = []
	try :
		NUM = DC_DRUG_DF_FULL[DC_DRUG_DF_FULL.CAN_SMILES==SMILES]['leng'].item()
		tf.append(NUM)
	except : 
		re_try.append(SMILES)
	#
	if len(re_try) > 0: 
		try : 
			iMol = Chem.MolFromSmiles(SMILES.strip())
			NUM = iMol.GetNumAtoms()
			tf.append(NUM)
		except :
			tf.append("error")
			print("error",z,i)
	return tf




tf_list_A = []
tf_list_B = []
for a in range(A_B_C_S_SET.shape[0]):
	sm_1 = A_B_C_S_SET.ROW_CAN_SMILES[a]
	sm_2 = A_B_C_S_SET.COL_CAN_SMILES[a]
	tf_a = check_len_num(sm_1)
	tf_b = check_len_num(sm_2)
	tf_list_A = tf_list_A + tf_a
	tf_list_B = tf_list_B + tf_b


A_B_C_S_SET['ROW_len'] = [int(a) for a in tf_list_A]
A_B_C_S_SET['COL_len'] = [int(a) for a in tf_list_B]

max_len = max(list(A_B_C_S_SET['ROW_len'])+list(A_B_C_S_SET['COL_len']))

A_B_C_S_SET_rlen = A_B_C_S_SET[A_B_C_S_SET.ROW_len<=50]
A_B_C_S_SET_clen = A_B_C_S_SET_rlen[A_B_C_S_SET_rlen.COL_len<=50]

A_B_C_S_SET = A_B_C_S_SET_clen.reset_index(drop=True) # 





# Tanimoto filter 

ABCS_ori_CIDs = list(set(list(A_B_C_S.drug_row_CID) + list(A_B_C_S.drug_col_CID))) # 172 
ABCS_FILT_CIDS = list(set(list(A_B_C_S_SET.drug_row_CID) + list(A_B_C_S_SET.drug_col_CID))) # 172 

ABCS_ori_SMILEs = list(set(list(A_B_C_S.ROW_CAN_SMILES) + list(A_B_C_S.COL_CAN_SMILES))) # 171
ABCS_FILT_SMILEs = list(set(list(A_B_C_S_SET.ROW_CAN_SMILES) + list(A_B_C_S_SET.COL_CAN_SMILES))) # 171 


PC_check = for_CAN_smiles[for_CAN_smiles.CID.isin(ABCS_ori_CIDs)]


def calculate_internal_pairwise_similarities(smiles_list) :
	"""
	Computes the pairwise similarities of the provided list of smiles against itself.
		Symmetric matrix of pairwise similarities. Diagonal is set to zero.
	"""
	mols = [Chem.MolFromSmiles(x.strip()) for x in smiles_list]
	fps = [Chem.RDKFingerprint(x) for x in mols]
	nfps = len(fps)
	#
	similarities = np.zeros((nfps, nfps))
	#
	for i in range(1, nfps):
		sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
		similarities[i, :i] = sims
		similarities[:i, i] = sims
	return similarities 


sim_matrix_order = list(PC_check.CAN_SMILES)
sim_matrix = calculate_internal_pairwise_similarities(sim_matrix_order)


row_means = []
for i in range(sim_matrix.shape[0]):
	indexes = [a for a in range(sim_matrix.shape[0])]
	indexes.pop(i)
	row_tmp = sim_matrix[i][indexes]
	row_mean = np.mean(row_tmp)
	row_means = row_means + [row_mean]


means_df = pd.DataFrame({ 'CIDs' : list(PC_check.CID), 'MEAN' : row_means})
means_df = means_df.sort_values('MEAN')
means_df['cat']= 'MEAN'
# means_df['filter']=['stay' if (a in cids_all) else 'nope' for a in list(means_df.CIDs)] 
means_df['dot_col']= ['IN' if (a in ABCS_FILT_CIDS) else 'OUT' for a in list(means_df.CIDs)] 


means_df['over0.1'] = ['IN' if a > 0.1  else 'OUT' for a in list(means_df.MEAN)] 
means_df['over0.2'] = ['IN' if a > 0.2  else 'OUT' for a in list(means_df.MEAN)] 
means_df['overQ'] = ['IN' if a > means_df.MEAN.describe()['25%']  else 'OUT' for a in list(means_df.MEAN)] 



# check triads num

  over 0.1 only 
tmp_list = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.1'] == 'IN') ]['CIDs'])
tmp_row = A_B_C_S_SET[A_B_C_S_SET.drug_row_CID.isin(tmp_list)]
tmp_col = tmp_row[tmp_row.drug_col_CID.isin(tmp_list)]
tmp_col

over 0.2 only 
tmp_list = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['over0.2'] == 'IN') ]['CIDs'])
tmp_row = A_B_C_S_SET[A_B_C_S_SET.drug_row_CID.isin(tmp_list)]
tmp_col = tmp_row[tmp_row.drug_col_CID.isin(tmp_list)]
tmp_col

over Q only 
tmp_list = list(means_df[(means_df['dot_col'] == 'IN') & (means_df['overQ'] == 'IN') ]['CIDs'])
tmp_row = A_B_C_S_SET[A_B_C_S_SET.drug_row_CID.isin(tmp_list)]
tmp_col = tmp_row[tmp_row.drug_col_CID.isin(tmp_list)]
tmp_col



A_B_C_S_SET = copy.deepcopy(tmp_col)
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)




# MJ data filter - if we need 
# 그리고 50 잘리는걸로 예상됨 



MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

#MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_deside_NF.csv')
#MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_mugcn_vers2.csv')
MJ_request_ANS = pd.read_csv(MJ_DIR+'PRJ2_EXP_fugcn_vers1.csv')

MJ_request_ANS2 = copy.deepcopy(MJ_request_ANS)
MJ_request_ANS_COL = list(MJ_request_ANS2.columns)

MJ_request_ANS_re_col = MJ_request_ANS_COL[0:3]+[int(a.split('__')[0])  for a in MJ_request_ANS_COL[3:]]
MJ_request_ANS2.columns =MJ_request_ANS_re_col 
MJ_request_ANS3 = MJ_request_ANS2.T.drop_duplicates().T


# deside & mu
A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.drug_row_CID.isin(MJ_request_ANS3.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.drug_col_CID.isin(MJ_request_ANS3.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)


# fu
A_B_C_S_SET_MJ = A_B_C_S_SET[A_B_C_S_SET.ROWCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ[A_B_C_S_SET_MJ.COLCHECK.isin(MJ_request_ANS.columns)]
A_B_C_S_SET_MJ = A_B_C_S_SET_MJ.reset_index(drop = True)


# ori_cids = set(list(A_B_C_S_SET.drug_row_CID) + list(A_B_C_S_SET.drug_col_CID))
# [a for a in ori_cids if a not in list(MJ_request_ANS3.columns)]
# MISS 0 에서 없는 CID : {11683005.0, 387447.0}
# MISS 1 에서 없는 CID : {3045381.0, 16683013.0, 483407.0, 6252.0, 9818231.0, 11178236.0, 354624.0, 10607.0, 387447.0, 54677977.0, 389644.0, 154256.0, 12947.0, 16681698.0, 176871.0, 2800.0, 2900.0, 53396328.0, 56603532.0, 16755649.0, 72686.0, 49830988.0, 451668.0, 49831008.0, 3005572.0, 3003565.0, 60602.0, 11683005.0, 439525.0, 4979942.0, 3310.0, 5376.0, 3751173.0, 135421197.0, 53437714.0, 77082.0, 60699.0, 2913564.0, 62770.0, 216467.0, 57519507.0, 57519518.0, 57519523.0, 57519532.0, 57519545.0, 2733525.0, 53401057.0, 9703.0, 63009.0, 11525740.0, 25183872.0, 57441923.0, 5312137.0, 1691.0, 3793.0, 73461.0, 1805.0, 135413523.0, 20279.0, 657237.0, 54761306.0}
# MISS 2 에서 없는 CID : {5464092.0, 213023.0, 8228.0, 10174505.0, 53239854.0, 11591741.0, 483407.0, 11477084.0, 41114.0, 114869.0, 4194514.0, 8478.0, 41317.0, 45375887.0, 425.0, 5702062.0, 9929138.0, 471.0, 25050.0, 8691.0, 8695.0, 5284360.0, 66062.0, 66064.0, 66069.0, 66070.0, 8732.0, 573.0, 5702220.0, 5284441.0, 5284443.0, 4022878.0, 164521.0, 709.0, 123596.0, 66259.0, 123607.0, 44483281.0, 156418.0, 8980.0, 5284631.0, 53396327.0, 53396328.0, 9066.0, 42066809.0, 17275.0, 41867.0, 5702541.0, 9560989.0, 57336745.0, 23667630.0, 23667631.0, 23667642.0, 6398970.0, 451668.0, 17506.0, 9351.0, 45139106.0, 11683005.0, 517321.0, 9433.0, 206042.0, 1253.0, 656628.0, 656631.0, 107770.0, 53437714.0, 91430.0, 25232708.0, 135398741.0, 6710614.0, 21874004.0, 91505.0, 91513.0, 6710658.0, 107969.0, 443873.0, 443878.0, 9703.0, 108031.0, 9578005.0, 443939.0, 9782.0, 444000.0, 444030.0, 25183872.0, 157313.0, 1691.0, 165542.0, 24995523.0, 1744.0, 15558393.0, 11519741.0, 1805.0, 657201.0, 23643975.0, 657237.0, 18283.0, 83823.0, 51082.0, 11626384.0, 657308.0, 135497698.0, 23693301.0, 92151.0, 23685176.0, 23668834.0, 2154.0, 354624.0, 26964.0, 2390.0, 6433119.0, 10603.0, 10607.0, 387447.0, 59772.0, 16738693.0, 10648.0, 76219.0, 2518.0, 44591583.0, 2532.0, 2548.0, 2563.0, 24906273.0, 10866.0, 2713.0, 68304.0, 27350.0, 16681698.0, 3033832.0, 2800.0, 9906942.0, 44329754.0, 92965.0, 11057.0, 11065.0, 11079.0, 2889.0, 2900.0, 11102.0, 5311339.0, 68539.0, 16755649.0, 68546.0, 68551.0, 68553.0, 9849808.0, 11224.0, 207841.0, 3049.0, 68589.0, 3054.0, 68601.0, 68624.0, 158758.0, 68647.0, 44072.0, 68727.0, 3203.0, 60560.0, 101526.0, 49867936.0, 49867937.0, 3235.0, 5459110.0, 53398697.0, 9931953.0, 216248.0, 60602.0, 3288.0, 150762.0, 3310.0, 60656.0, 3325.0, 216322.0, 77082.0, 60699.0, 24800541.0, 60749.0, 716121.0, 44383.0, 54685047.0, 3461.0, 216457.0, 216467.0, 57519507.0, 9883029.0, 57519518.0, 60834.0, 57519523.0, 60837.0, 200103.0, 57519532.0, 57519545.0, 3541.0, 60918.0, 60953.0, 3661.0, 36431.0, 6917719.0, 6917733.0, 11570805.0, 5312137.0, 175804.0, 3793.0, 20179.0, 151289.0, 6852391.0, 20279.0, 667466.0, 36708.0, 11431811.0, 2723716.0, 2723754.0, 4046.0, 16683013.0, 73265216.0, 9818231.0, 73265274.0, 11751549.0, 53440640.0, 12456.0, 6918313.0, 77997.0, 4272.0, 54685920.0, 159968.0, 11178236.0, 176406.0, 5362123.0, 54677977.0, 4583.0, 4599.0, 4611.0, 6435335.0, 6918664.0, 11440648.0, 389644.0, 2724368.0, 4646.0, 3011155.0, 12309103.0, 5485201.0, 12947.0, 160436.0, 5018304.0, 4813.0, 119525.0, 176871.0, 4840.0, 119583.0, 56603532.0, 23671691.0, 5034.0, 5052.0, 5056.0, 9966538.0, 439246.0, 13266.0, 21467.0, 439280.0, 54259.0, 406563.0, 7566371.0, 54360.0, 5213.0, 5258.0, 23696523.0, 3003565.0, 5307.0, 21704.0, 16118986.0, 13520.0, 21718.0, 439525.0, 3052775.0, 5352.0, 5355.0, 5376.0, 9827599.0, 5402.0, 2913564.0, 5281056.0, 5281066.0, 5281067.0, 5281068.0, 5281069.0, 62770.0, 23663953.0, 23663979.0, 7271796.0, 23663996.0, 62857.0, 5527.0, 5530.0, 62878.0, 62882.0, 5538.0, 11957668.0, 11957684.0, 67089852.0, 62920.0, 13770.0, 2864586.0, 2733525.0, 62935.0, 53401057.0, 6419941.0, 636397.0, 636402.0, 62969.0, 62978.0, 63002.0, 63009.0, 6420013.0, 54840.0, 448055.0, 6420040.0, 54889.0, 54891.0, 71279.0, 5748.0, 54900.0, 9852573.0, 5790.0, 13985.0, 5795.0, 71317162.0, 5807.0, 5815.0, 71352.0, 521951.0, 71399.0, 71420.0, 5909.0, 71478.0, 54761306.0, 6014.0, 55182.0, 219025.0, 6048.0, 6100.0, 219099.0, 6126.0, 6135.0, 38911.0, 3045381.0, 6169.0, 6172.0, 6175.0, 71764.0, 6240.0, 6241.0, 6252.0, 227456.0, 6461.0, 104769.0, 5462355.0, 5282139.0, 3086685.0, 3086686.0, 47471.0, 170361.0, 16759173.0, 153997.0, 72092.0, 80311.0, 60119583.0, 5282379.0, 5282386.0, 5282402.0, 5282407.0, 5282408.0, 6759.0, 5282435.0, 47751.0, 72327.0, 64142.0, 154256.0, 5388961.0, 5282474.0, 31401.0, 31411.0, 9870009.0, 47812.0, 9910986.0, 56031.0, 23665411.0, 56069.0, 56965896.0, 154417.0, 23360.0, 39764.0, 56205.0, 56207.0, 441307.0, 441325.0, 72686.0, 31728.0, 441344.0, 441345.0, 441351.0, 56329.0, 23565.0, 46931003.0, 49830988.0, 49831008.0, 3005572.0, 64648.0, 64650.0, 146571.0, 35028115.0, 9804991.0, 64737.0, 4979942.0, 53394675.0, 515328.0, 3751173.0, 16219401.0, 135421197.0, 122125.0, 3038502.0, 71433510.0, 204100.0, 7550.0, 32169.0, 392622.0, 23666110.0, 11673085.0, 130621.0, 56843850.0, 11525740.0, 57441923.0, 24211.0, 11984591.0, 73442.0, 16760554.0, 65264.0, 73461.0, 135413494.0, 135413505.0, 135413523.0, 65335.0, 65340.0, 65341.0, 65348.0, 16760658.0, 16230.0, 163751.0, 65464.0, 65495.0}
# 

# '135449292__CVCL_0031', '135449292__CVCL_9827'



# cell line 
'CENTRAL_NERVOUS_SYSTEM', 'PLEURA', 'OVARY', 
'KIDNEY', 'PANCREAS', 'LARGE_INTESTINE', 'LIVER', 'SKIN', 
'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BREAST', 'BONE', 
'SOFT_TISSUE', 'URINARY_TRACT', 'LUNG', 'STOMACH'

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCello.isin(A_B_C_S_SET_MJ.DrugCombCello)]

DC_CELL_info_filt['cell_onehot'] = [a for a in range(len(set(A_B_C_S_SET_MJ.DrugCombCello)))]
DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) for a in list(DC_CELL_info_filt['DrugCombCCLE'])]
DC_CELL_info_filt = DC_CELL_info_filt.reset_index(drop = True)

DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
DC_CELL_info_filt.at[(DC_CELL_info_filt[DC_CELL_info_filt.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'


## TISSUE_SET = list(set(DC_CELL_info_filt['tissue']))
## DC_CELL_info_filt['tissue_onehot'] = [TISSUE_SET.index(a) for a in list(DC_CELL_info_filt['tissue'])]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET_MJ, DC_CELL_info_filt[['DrugCombCello','DC_cellname','cell_onehot']], on = 'DrugCombCello', how = 'left'  )

cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH['cell_onehot']).long())







# stack data 

sig_id = 'LJP009_A375_24H:O24'
DrugA_SIG = 'LJP009_A375_24H:O24'
DrugB_SIG = 'PBIOA016_A375_24H:D11'




def get_LINCS_data(DRUG_SIG):
	Drug_EXP = BETA_BIND[['id',DRUG_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.gene_id]
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	#
	return list(Drug_EXP_ORD[DRUG_SIG])





def get_targets(CID): # 이건 지금 필터링 한 경우임 
	#
	target_cids = list(set(TARGET_DB.CID))
	if CID in target_cids:
		tmp_df2 = TARGET_DB[TARGET_DB.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		gene_ids = list(BETA_ORDER_DF.gene_id)
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		gene_ids = list(BETA_ORDER_DF.gene_id)
		vec = [0 for a in gene_ids ]
	return vec





def get_CHEM(cid, k=1):
	maxNumAtoms = max_len
	smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
	iMol = Chem.MolFromSmiles(smiles.strip())
	iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) 
	iFeature = np.zeros((maxNumAtoms, 64)) ################## feature 크기 고정 
	iFeatureTmp = []
	for atom in iMol.GetAtoms():
		iFeatureTmp.append( atom_feature(atom) )### atom features only
	iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
	iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
	iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
	ADJ = adj_k(np.asarray(iAdj), k)
	return iFeature, ADJ




def atom_feature(atom):
	ar = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
									  ['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'F', 'Br', 'I',
									   'Na', 'Fe', 'B', 'Mg', 'Al', 'Si', 'K', 'H', 'Se', 'Ca',
									   'Zn', 'As', 'Mo', 'V', 'Cu', 'Hg', 'Cr', 'Co', 'Bi','Tc',
									   'Sb', 'Gd', 'Li', 'Ag', 'Au', 'Unknown']) +
					one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
					one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5]) +
					[atom.GetIsAromatic()])    # (36, 8, 5, 5, 9, 1) -> 64 
	return ar



def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
	#print list((map(lambda s: x == s, allowable_set)))
	return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding_unk(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))

def convertAdj(adj):
	dim = len(adj)
	a = adj.flatten()
	b = np.zeros(dim*dim)
	c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
	d = c.reshape((dim, dim))
	return d



def adj_k(adj, k): # 근데 k 가 왜 필요한거지 -> 민지 말에 의하면 
	ret = adj
	for i in range(0, k-1):
		ret = np.dot(ret, adj)  
	return convertAdj(ret)




def get_cell(IND) : 
	cell_res = cell_one_hot[IND]
	return(cell_res)




def get_synergy_data(DrugA_CID, DrugB_CID, Cell):
	ABCS1 = A_B_C_S[A_B_C_S.drug_row_CID == DrugA_CID]
	ABCS2 = ABCS1[ABCS1.drug_col_CID == DrugB_CID]
	ABCS3 = ABCS2[ABCS2.DrugCombCello == Cell]
	synergy_score = np.median(ABCS3.synergy_loewe) # 원래는 무조건 median
	return synergy_score





# DESIDE(M1) / mu1 (M2)
def get_MJ_data( CID ): 
	if CID in list(MJ_request_ANS3.columns) :
		MJ_DATA = MJ_request_ANS3[['entrez_id',CID]]
		ord = [list(MJ_DATA.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
		MJ_DATA_re = MJ_DATA.loc[ord] 
		RES = MJ_DATA_re[CID]
		OX = 'O'
	else : 
		RES = [0]*978
		OX = 'X'
	return list(RES), OX




# fu (M3)
def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_request_ANS.columns) :
		MJ_DATA = MJ_request_ANS[['entrez_id', CHECK]]
		ord = [list(MJ_DATA.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
		MJ_DATA_re = MJ_DATA.loc[ord] 
		RES = MJ_DATA_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*978
		OX = 'X'
	return list(RES), OX








max_len = 50

MY_chem_A_feat = torch.empty(size=(A_B_C_S_SET_COH.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(A_B_C_S_SET_COH.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(A_B_C_S_SET_COH.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(A_B_C_S_SET_COH.shape[0], max_len, max_len))
MY_g_feat_A = torch.empty(size=(A_B_C_S_SET_COH.shape[0], 978, 2))
MY_g_feat_B = torch.empty(size=(A_B_C_S_SET_COH.shape[0], 978, 2))
MY_Cell = torch.empty(size=(A_B_C_S_SET_COH.shape[0], cell_one_hot.shape[1]))
MY_syn =  torch.empty(size=(A_B_C_S_SET_COH.shape[0],1))


MY_chem_A_feat = torch.empty(size=(128, max_len, 64))
MY_chem_B_feat= torch.empty(size=(128, max_len, 64))
MY_chem_A_adj = torch.empty(size=(128, max_len, max_len))
MY_chem_B_adj= torch.empty(size=(128, max_len, max_len))
MY_g_feat_A = torch.empty(size=(128, 978, 2))
MY_g_feat_B = torch.empty(size=(128, 978, 2))
MY_Cell = torch.empty(size=(128, cell_one_hot.shape[1]))
MY_syn =  torch.empty(size=(128,1))








# M1 & M2 || MISS 0 / 1 / 2 version 
# M1 & M2 || MISS 0 / 1 / 2 version 
# M1 & M2 || MISS 0 / 1 / 2 version 
# M1 & M2 || MISS 0 / 1 / 2 version 

Fail_ind = []
from datetime import datetime

for IND in range(MY_chem_A_feat.shape[0]): #  
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(MY_chem_A_feat.shape[0]) )
		Fail_ind
		datetime.now()
	#
	DrugA_SIG = A_B_C_S_SET_COH.iloc[IND,]['ROW_BETA_sig_id']
	DrugB_SIG = A_B_C_S_SET_COH.iloc[IND,]['COL_BETA_sig_id']
	DrugA_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_row_CID']
	DrugB_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_col_CID']
	Cell = A_B_C_S_SET_COH.iloc[IND,]['DrugCombCello']
	dat_type = A_B_C_S_SET_COH.iloc[IND,]['type']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_CID, k)
	# 
	if dat_type == 'AOBO' :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B = get_LINCS_data(DrugB_SIG)
	elif dat_type == "AXBO" : 
		EXP_A, OX = get_MJ_data(DrugA_CID)
		EXP_B = get_LINCS_data(DrugB_SIG)
		if OX == 'X':
			Fail_ind.append(IND)
	elif dat_type == "AOBX" :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B, OX = get_MJ_data(DrugB_CID)
		if OX == 'X':
			Fail_ind.append(IND)
	elif dat_type == "AXBX" :
		EXP_A, OX1 = get_MJ_data(DrugA_CID)
		EXP_B, OX2 = get_MJ_data(DrugB_CID)
		if 'X' in [OX1, OX2]:
			Fail_ind.append(IND)   
	# 
	TGT_A = get_targets(DrugA_CID)
	TGT_B = get_targets(DrugB_CID)
	#
	ARR_A = np.array([EXP_A, TGT_A]).T
	ARR_B = np.array([EXP_B, TGT_B]).T
	#
	Cell_Vec = get_cell(IND)
	#
	AB_SYN = get_synergy_data(DrugA_CID, DrugB_CID, Cell)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_feat_A[IND] = torch.Tensor(ARR_A)	
	MY_g_feat_B[IND] = torch.Tensor(ARR_B)
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])






# M3 || MISS 0 / 1 / 2 version 
# M3 || MISS 0 / 1 / 2 version 
# M3 || MISS 0 / 1 / 2 version 
# M3 || MISS 0 / 1 / 2 version 

Fail_ind = []
from datetime import datetime

for IND in range(MY_chem_A_feat.shape[0]): #  
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(MY_chem_A_feat.shape[0]) )
		Fail_ind
		datetime.now()
	#
	DrugA_SIG = A_B_C_S_SET_COH.iloc[IND,]['ROW_BETA_sig_id']
	DrugB_SIG = A_B_C_S_SET_COH.iloc[IND,]['COL_BETA_sig_id']
	DrugA_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_row_CID']
	DrugB_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_col_CID']
	DrugA_CID_CELL = A_B_C_S_SET_COH.iloc[IND,]['ROWCHECK']
	DrugB_CID_CELL = A_B_C_S_SET_COH.iloc[IND,]['COLCHECK']	
	Cell = A_B_C_S_SET_COH.iloc[IND,]['DrugCombCello']
	dat_type = A_B_C_S_SET_COH.iloc[IND,]['type']
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_CID, k)
	# 
	if dat_type == 'AOBO' :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B = get_LINCS_data(DrugB_SIG)
	elif dat_type == "AXBO" : 
		EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
		EXP_B = get_LINCS_data(DrugB_SIG)
		if OX == 'X':
			Fail_ind.append(IND)
	elif dat_type == "AOBX" :
		EXP_A = get_LINCS_data(DrugA_SIG)
		EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
		if OX == 'X':
			Fail_ind.append(IND)
	elif dat_type == "AXBX" :
		EXP_A, OX1 = get_MJ_data(DrugA_CID_CELL)
		EXP_B, OX2 = get_MJ_data(DrugB_CID_CELL)
		if 'X' in [OX1, OX2]:
			Fail_ind.append(IND)   
	# 
	TGT_A = get_targets(DrugA_CID)
	TGT_B = get_targets(DrugB_CID)
	#
	ARR_A = np.array([EXP_A, TGT_A]).T
	ARR_B = np.array([EXP_B, TGT_B]).T
	#
	Cell_Vec = get_cell(IND)
	#
	AB_SYN = get_synergy_data(DrugA_CID, DrugB_CID, Cell)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_feat_A[IND] = torch.Tensor(ARR_A)	
	MY_g_feat_B[IND] = torch.Tensor(ARR_B)
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])






selec_ind = A_B_C_S_SET_COH.index.isin(Fail_ind)==False
A_B_C_S_SET_COH_FAFILT = A_B_C_S_SET_COH[selec_ind]
MY_chem_A_feat_re = MY_chem_A_feat[selec_ind]
MY_chem_B_feat_re = MY_chem_B_feat[selec_ind]
MY_chem_A_adj_re = MY_chem_A_adj[selec_ind]
MY_chem_B_adj_re = MY_chem_B_adj[selec_ind]
MY_g_feat_A_re = MY_g_feat_A[selec_ind]
MY_g_feat_B_re = MY_g_feat_B[selec_ind]
MY_Cell_re = MY_Cell[selec_ind]
MY_syn_re = MY_syn[selec_ind]




MY_chem_A_feat_re = []

for t1 in MY_chem_A_feat :
	t2 = t1[0:50]
	MY_chem_A_feat_re.append(t2)

MY_chem_A_feat_re = torch.stack(MY_chem_A_feat_re, 0)



MY_chem_B_feat_re = []

for t1 in MY_chem_B_feat :
	t2 = t1[0:50]
	MY_chem_B_feat_re.append(t2)

MY_chem_B_feat_re = torch.stack(MY_chem_B_feat_re, 0)



MY_chem_A_adj_re = []

for t1 in MY_chem_A_adj :
	t2 = t1[0:50]
	t2_ = []
	for tt1 in t2 :
		tt2 = tt1[0:50]
		t2_.append(tt2)
	t3 = torch.stack(t2_, 0)
	MY_chem_A_adj_re.append(t3)

MY_chem_A_adj_re = torch.stack(MY_chem_A_adj_re, 0)




MY_chem_B_adj_re = []

for t1 in MY_chem_B_adj :
	t2 = t1[0:50]
	t2_ = []
	for tt1 in t2 :
		tt2 = tt1[0:50]
		t2_.append(tt2)
	t3 = torch.stack(t2_, 0)
	MY_chem_B_adj_re.append(t3)

MY_chem_B_adj_re = torch.stack(MY_chem_B_adj_re, 0)



SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M1_J1/M1_J1_DATA/'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M2_J3/M2_J3_DATA/'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3_J3/M3_J3_DATA/'


MISS_NAME = 'MISS_0'
MISS_NAME = 'MISS_1'
MISS_NAME = 'MISS_2'

MISS_NAME = 'MISS_0_TG'
MISS_NAME = 'MISS_1_TG'
MISS_NAME = 'MISS_2_TG'



torch.save(MY_chem_A_feat_re, SAVE_PATH+'{}.{}.MY_chem_A_feat.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'{}.{}.MY_chem_B_feat.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'{}.{}.MY_chem_A_adj.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'{}.{}.MY_chem_B_adj.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'{}.{}.MY_g_feat_A.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'{}.{}.MY_g_feat_B.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'{}.{}.MY_Cell.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_syn_re, SAVE_PATH+'{}.{}.MY_syn.pt'.format(PRJ_NAME, MISS_NAME))

A_B_C_S_SET_COH_FAFILT.to_csv(SAVE_PATH+'{}.{}.A_B_C_S_SET.csv'.format(PRJ_NAME, MISS_NAME))











# M1 & M2 || MISS 3 / 4 version 
# M1 & M2 || MISS 3 / 4 version 
# M1 & M2 || MISS 3 / 4 version 
# M1 & M2 || MISS 3 / 4 version 

Fail_ind = []
from datetime import datetime

for IND in range(MY_chem_A_feat.shape[0]): #  
	if IND%100 == 0 :
		print(str(IND)+'/'+str(MY_chem_A_feat.shape[0]) )
		Fail_ind
		datetime.now()
	#
	DrugA_SIG = A_B_C_S_SET_COH.iloc[IND,]['ROW_BETA_sig_id']
	DrugB_SIG = A_B_C_S_SET_COH.iloc[IND,]['COL_BETA_sig_id']
	DrugA_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_row_CID']
	DrugB_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_col_CID']
	Cell = A_B_C_S_SET_COH.iloc[IND,]['DrugCombCello']
	dat_type = A_B_C_S_SET_COH.iloc[IND,]['type']
		#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_CID, k)
	# 
	EXP_A, OX1 = get_MJ_data(DrugA_CID)
	EXP_B, OX2 = get_MJ_data(DrugB_CID)
	if 'X' in [OX1, OX2]:
		Fail_ind.append(IND)   
	# 
	TGT_A = get_targets(DrugA_CID)
	TGT_B = get_targets(DrugB_CID)
	#
	ARR_A = np.array([EXP_A, TGT_A]).T
	ARR_B = np.array([EXP_B, TGT_B]).T
	#
	Cell_Vec = get_cell(IND)
	#
	AB_SYN = get_synergy_data(DrugA_CID, DrugB_CID, Cell)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_feat_A[IND] = torch.Tensor(ARR_A)	
	MY_g_feat_B[IND] = torch.Tensor(ARR_B)
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])










# M3 || MISS 3 / 4 version 
# M3 || MISS 3 / 4 version 
# M3 || MISS 3 / 4 version 
# M3 || MISS 3 / 4 version 

Fail_ind = []
from datetime import datetime

for IND in range(MY_chem_A_feat.shape[0]): #  
	if IND%100 == 0 :
		print(str(IND)+'/'+str(MY_chem_A_feat.shape[0]) )
		Fail_ind
		datetime.now()
	#
	DrugA_SIG = A_B_C_S_SET_COH.iloc[IND,]['ROW_BETA_sig_id']
	DrugB_SIG = A_B_C_S_SET_COH.iloc[IND,]['COL_BETA_sig_id']
	DrugA_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_row_CID']
	DrugB_CID = A_B_C_S_SET_COH.iloc[IND,]['drug_col_CID']
	DrugA_CID_CELL = A_B_C_S_SET_COH.iloc[IND,]['ROWCHECK']
	DrugB_CID_CELL = A_B_C_S_SET_COH.iloc[IND,]['COLCHECK']		
	Cell = A_B_C_S_SET_COH.iloc[IND,]['DrugCombCello']
	dat_type = A_B_C_S_SET_COH.iloc[IND,]['type']
		#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	DrugB_Feat, DrugB_ADJ = get_CHEM(DrugB_CID, k)
	# 
	EXP_A, OX1 = get_MJ_data(DrugA_CID_CELL)
	EXP_B, OX2 = get_MJ_data(DrugB_CID_CELL)
	if 'X' in [OX1, OX2]:
		Fail_ind.append(IND)   
	# 
	TGT_A = get_targets(DrugA_CID)
	TGT_B = get_targets(DrugB_CID)
	#
	ARR_A = np.array([EXP_A, TGT_A]).T
	ARR_B = np.array([EXP_B, TGT_B]).T
	#
	Cell_Vec = get_cell(IND)
	#
	AB_SYN = get_synergy_data(DrugA_CID, DrugB_CID, Cell)
	#
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_feat_A[IND] = torch.Tensor(ARR_A)	
	MY_g_feat_B[IND] = torch.Tensor(ARR_B)
	MY_Cell[IND] = Cell_Vec
	MY_syn[IND] = torch.Tensor([AB_SYN])











selec_ind = A_B_C_S_SET_COH.index.isin(Fail_ind)==False
A_B_C_S_SET_COH_FAFILT = A_B_C_S_SET_COH[selec_ind]
MY_chem_A_feat_re = MY_chem_A_feat[selec_ind]
MY_chem_B_feat_re = MY_chem_B_feat[selec_ind]
MY_chem_A_adj_re = MY_chem_A_adj[selec_ind]
MY_chem_B_adj_re = MY_chem_B_adj[selec_ind]
MY_g_feat_A_re = MY_g_feat_A[selec_ind]
MY_g_feat_B_re = MY_g_feat_B[selec_ind]
MY_Cell_re = MY_Cell[selec_ind]
MY_syn_re = MY_syn[selec_ind]


MY_chem_A_feat_re = []

for t1 in MY_chem_A_feat :
	t2 = t1[0:50]
	MY_chem_A_feat_re.append(t2)

MY_chem_A_feat_re = torch.stack(MY_chem_A_feat_re, 0)



MY_chem_B_feat_re = []

for t1 in MY_chem_B_feat :
	t2 = t1[0:50]
	MY_chem_B_feat_re.append(t2)

MY_chem_B_feat_re = torch.stack(MY_chem_B_feat_re, 0)



MY_chem_A_adj_re = []

for t1 in MY_chem_A_adj :
	t2 = t1[0:50]
	t2_ = []
	for tt1 in t2 :
		tt2 = tt1[0:50]
		t2_.append(tt2)
	t3 = torch.stack(t2_, 0)
	MY_chem_A_adj_re.append(t3)

MY_chem_A_adj_re = torch.stack(MY_chem_A_adj_re, 0)




MY_chem_B_adj_re = []

for t1 in MY_chem_B_adj :
	t2 = t1[0:50]
	t2_ = []
	for tt1 in t2 :
		tt2 = tt1[0:50]
		t2_.append(tt2)
	t3 = torch.stack(t2_, 0)
	MY_chem_B_adj_re.append(t3)

MY_chem_B_adj_re = torch.stack(MY_chem_B_adj_re, 0)




SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M1_J1/M1_J1_DATA/'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M2_J3/M2_J3_DATA/'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3_J3/M3_J3_DATA/'

MISS_NAME = 'MISS_3'
MISS_NAME = 'MISS_4'


torch.save(MY_chem_A_feat_re, SAVE_PATH+'{}.{}.MY_chem_A_feat.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_chem_B_feat_re, SAVE_PATH+'{}.{}.MY_chem_B_feat.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_chem_A_adj_re, SAVE_PATH+'{}.{}.MY_chem_A_adj.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_chem_B_adj_re, SAVE_PATH+'{}.{}.MY_chem_B_adj.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_g_feat_A_re, SAVE_PATH+'{}.{}.MY_g_feat_A.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_g_feat_B_re, SAVE_PATH+'{}.{}.MY_g_feat_B.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_Cell_re, SAVE_PATH+'{}.{}.MY_Cell.pt'.format(PRJ_NAME, MISS_NAME))
torch.save(MY_syn_re, SAVE_PATH+'{}.{}.MY_syn.pt'.format(PRJ_NAME, MISS_NAME))

A_B_C_S_SET_COH_FAFILT.to_csv(SAVE_PATH+'{}.{}.A_B_C_S_SET.csv'.format(PRJ_NAME, MISS_NAME))









