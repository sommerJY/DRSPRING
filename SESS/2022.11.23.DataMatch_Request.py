


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

# 민지랑 나눌곳 
/st06/jiyeonH/13.DD_SESS/01.PRJ2

# 필요한 곳 
# 무조건 pubchem CID 로 하기! 



WORK_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'
PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


# Pubchem Whole Data
# PUBCHEM_ALL = pd.read_csv('/st06/jiyeonH/12.HTP_DB/08.PUBCHEM/PUBCHEM_MJ_031022.csv',  low_memory = False)
# PC_FILTER = PUBCHEM_ALL[['CID','CAN_SMILES','ISO_SMILES']]
# PC_FILTER.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/CID_SMILES.csv', sep ='\t')
# PC_FILTER = PUBCHEM_ALL[['CID','filtered_synonym','unfiltered_synonym']]
# PC_FILTER.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/CID_synonyms.csv', sep ='\t')
PC_CID_SM_list = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/CID_SMILES.csv', sep ='\t')
# 162640045





# Drug Comb 데이터 가져오기 
# Drug Comb 데이터 가져오기 
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


                                생각해보니 그냥 DrugDomb 전체에 해당하는걸 요청해야함 
                                -> 왜냐면 그래야 예측된 exp 값이 더 중요하다고 얘기할 수 있거든 
                                -> 근데 brd id 는 lincs 에서만 쓰고 있어서 CID 로 줄수밖에 음슴 
                                # rr = request_mj_1[['drug_row_cid_re','drug_col_cid_re','DrugCombCello']].drop_duplicates() # 554431


                                request_mj_1 = DC_DATA7_4_cello[['drug_row_cid_re','drug_col_cid_re','DrugCombCello', 'drug_row_sm', 'drug_col_sm']]
                                request_mj_1 = DC_DATA7_4_cello[['drug_row_CID','drug_col_CID','DrugCombCello']]


                                request_mj_2 = request_mj_1[['drug_row_CID','DrugCombCello']]
                                request_mj_3 = request_mj_1[['drug_col_CID','DrugCombCello']]
                                request_mj_2.columns = ['CID','DrugCombCello','SM']
                                request_mj_3.columns = ['CID','DrugCombCello','SM']

                                request_mj = pd.concat([request_mj_2, request_mj_3]).drop_duplicates()
                                request_mj.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/EXP_request.csv', sep ='\t')
                                # request_mj = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/EXP_request.csv', sep ='\t')
                            #혹시 request 에 있는 CID 들에 대한 target 많이 다른지? 
                            # DC_CIDs = list(set(request_mj.CID)) # 4302
                            # 

                            #  CCLE 버전으로 진행하려면 중간부터 빼내야함 
                            DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 751450 -> 740884
                            DC_DATA7_1 = DC_DATA7_1[['drug_row_CID','drug_col_CID','DrugCombCello','DrugCombCCLE','synergy_loewe']].drop_duplicates() # 740882
                            DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_CID>0] 
                            DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_CID>0] 
                            tmp = DC_DATA7_3[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates()
                            tf_ch = [True if type(a) == str else False for a in tmp.DrugCombCCLE]
                            tmp2  = tmp[tf_ch]
                            tmp3 = tmp2[tmp2.DrugCombCCLE != 'NA']
                            #request_mj_1 = DC_DATA7_3[['drug_row_CID','drug_col_CID','DrugCombCCLE']]
                            request_mj_2 = tmp3[['drug_row_CID','DrugCombCCLE']]
                            request_mj_3 = tmp3[['drug_col_CID','DrugCombCCLE']]
                            request_mj_2.columns = ['CID','DrugCombCCLE']
                            request_mj_3.columns = ['CID','DrugCombCCLE']
                            request_mj = pd.concat([request_mj_2, request_mj_3]).drop_duplicates()
                            request_mj.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/EXP_request_CCLEver.csv', sep ='\t')

                            


more_request =                             



                            # 11520894     CVCL_0132

                                
 

                                tmp = DC_cello_final.reset_index(drop=True)
                                a = list(tmp.drug_row_cid)
                                b = list(tmp.drug_col_cid)
                                c = list(tmp.DrugCombCello)

                                DC_cello_final_dup_list = [str(a[i])+'_'+str(b[i])+'_'+c[i] for i in range(554128)]


                                tmp = rr.reset_index(drop=True)
                                a = [float(a) for a in list(tmp.drug_row_cid_re)]
                                b = [float(a) for a in list(tmp.drug_col_cid_re)]
                                c = list(tmp.DrugCombCello)


                                rr_list = [str(a[i])+'_'+str(b[i])+'_'+c[i] for i in range(554431)]

                                # 73013358.0_9943465.0_CVCL_0039 의 경우 synergy score 구린거 확인 






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



# SIG info 마지막으로 한번만 더 확인 
# The core cell lines are: A375, A549, HA1E, HCC515, HT29, HEPG2, MCF7, PC3, VCAP

SIG_RE = BETA_SIG_info[['sig_id', 'pert_type', 'pert_id', 'pert_idose', 'pert_itime', 'cell_iname', 'tas', 'is_exemplar_sig']]

# filt1 = SIG_RE[SIG_RE.pert_id == 'BRD-K48853221']
# filt2 = filt1[filt1.cell_iname == 'HCC515']

SIG_RE_1 = SIG_RE[SIG_RE.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
SIG_RE_2 = SIG_RE_1[SIG_RE_1.is_exemplar_sig == 1]

pert = list(SIG_RE_2.pert_id)
cell = list(SIG_RE_2.cell_iname)

SIG_RE_2['set']= [pert[i] + "_" + cell[i] for i in range(SIG_RE_2.shape[0])] # 136460
# 다행히도 pert - cell 은 일치 -> 그래서 exemplar 를 이용해도 괜춘 
# 문제는 CID - cell 은 또 아닌것 같아서..? 그건 smiles 기준으로 한번 더 봐야함 




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

sum(check4.CONV_CID.isin(request_mj.CID)) # 10 
sum(check4.SMILES_cid.isin(request_mj.CID)) # 3 

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
BETA_CID_CELLO_SIG['CID'] = [int(a) for a in list(BETA_CID_CELLO_SIG['CID']) ] # 111012 # 늘어남! 

# -> CCLE 필터까지 완료 / DC CID 고정 완료 




                    mjmj = list(set(BETA_MJ.SMILES_cid)) # 25642
                    len([a for a in mjmj if a in DC_cello_final_cids]) # 1677

                    newnew = list(set(LINCS_PERT_MATCH.CID)) # 28154
                    len([a for a in newnew if a in DC_cello_final_cids]) # 1736



# MATCH DC & LINCS 
print('DC and LINCS') 
# 무조건 pubchem 공식 파일 사용하기 (mj ver) 
# 문제는 pert_id 에 CID 가 중복되어 붙는 애들도 있다는거  -> 어떻게 해결할거? 
# pert id 로 나눠서 하는게 맞을것 같음.

        (1) TRIAL 1 코드 

# 아 매칭이 더 늘어날 줄 알았는데... 그게 아니고 이게 CID 랑 Cello 가 다 맞아야하니까 그게 안되나봄 
# 헷갈리니 아예 제대로 비교해보도록 하자 

DC_cello_final_dup_ROW_CHECK = list(DC_cello_final_dup.drug_row_CID)
DC_cello_final_dup_COL_CHECK = list(DC_cello_final_dup.drug_col_CID)
DC_cello_final_dup_CELL_CHECK = list(DC_cello_final_dup.DrugCombCello)

DC_cello_final_dup['ROWCHECK'] = [str(int(DC_cello_final_dup_ROW_CHECK[i]))+'__'+DC_cello_final_dup_CELL_CHECK[i] for i in range(DC_cello_final_dup.shape[0])]
DC_cello_final_dup['COLCHECK'] = [str(int(DC_cello_final_dup_COL_CHECK[i]))+'__'+DC_cello_final_dup_CELL_CHECK[i] for i in range(DC_cello_final_dup.shape[0])]


# 공식 smiles 
for_CAN_smiles = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/CID_SMILES.csv',sep = '\t', low_memory = False)
# for_CAN_smiles = copy.deepcopy(PC_FILTER)
for_CAN_smiles = for_CAN_smiles[['CID','CAN_SMILES']]
for_CAN_smiles.columns = ['drug_row_CID','ROW_CAN_SMILES']
DC_cello_final_dup = pd.merge(DC_cello_final_dup, for_CAN_smiles, on='drug_row_CID', how ='left' )
for_CAN_smiles.columns = ['drug_col_CID','COL_CAN_SMILES']
DC_cello_final_dup = pd.merge(DC_cello_final_dup, for_CAN_smiles, on='drug_col_CID', how ='left' )


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
1) human net 

# hunet_dir = '/home01/k006a01/01.DATA/HumanNet/'
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

new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



# TARGET # 이거 이제 완료하고 그다음 단계로 넘어가야해 
완성은 대충 됐고, drugbank 숫자만 확인하면 될듯 
문제는 string 다시 한번 더 쓸거면 확인하고 넘어가야함 

그래서 내일 할일 :
분석할꺼 맞춰서 데이터 가공하고 첫번째 데이터 돌리기 

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

# TARGET_DB = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t')


그렇게 되면 721206에서 646702 -> 581365 가 됨 일단 줄여서 시작해야게싿 
tmp1 = CELLO_DC_BETA_2[CELLO_DC_BETA_2.drug_row_CID.isin(TARGET_DB.CID_RE)]
tmp2 = tmp1[tmp1.drug_col_CID.isin(TARGET_DB.CID_RE)]












##############################################

# 데이터 만들기 위한 단계 

# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE



# 써야하는 DC set 확인 

(1) NONE MISSING : MISS_0

A_B_C_S = DATA_AO_BO.reset_index(drop = True) # 11379

(2) one missing : MISS_1
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 38344 -> 31652

(3) two missing : MISS_2
A_B_C_S = pd.concat([DATA_AO_BO, DATA_AX_BO, DATA_AO_BX, DATA_AX_BX])
A_B_C_S = A_B_C_S.reset_index(drop = True) # 622809




PRJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
MJ_request_ANS = pd.read_csv(PRJ_DIR+'PRJ2_EXP_deside_NF.csv')

