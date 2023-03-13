
data renewal 을 하고 일단 뭘 추가한다고 생각해야하는거지 


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

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


WORK_PATH = 
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/'
IDK_PATH = 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'


WORK_PATH = '/home01/k006a01/PRJ.01/TRIAL_3.1/'
DC_PATH = '/home01/k006a01/01.DATA/DrugComb/'
IDK_PATH = '/home01/k006a01/01.DATA/IDK/'
LINCS_PATH = '/home01/k006a01/01.DATA/LINCS/'
TARGET_PATH = '/home01/k006a01/01.DATA/TARGET'




# LINCS DATA
print("LINCS")


BETA_BIND = pd.read_csv(LINCS_PATH+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)


filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460

BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles',
	   'pubchem_cid', 'h_bond_acceptor_count', 'h_bond_donor_count',
	   'rotatable_bond_count', 'MolLogP', 'molecular_weight',
	   'canonical_smiles_re', 'tpsa']].drop_duplicates()

BETA_EXM = pd.merge(filter2, BETA_MJ_RE, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.SMILES_cid > 0]


BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 111916
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]


print("DC filtering")
DC_tmp_DF2=pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False)
DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']]
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates()
DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_col_id>0]
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_row_id>0]
DC_DATA_filter4.cell_line_id # unique 295


# Drug DATA
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)



# cid filter 
DC_DRUG_DF2 = DC_DRUG_DF[['id','dname','cid']]
# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서
DC_DRUG_DF2.columns = ['drug_row_id','drug_row','drug_row_cid']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id', how='left' )

DC_DRUG_DF2.columns = ['drug_col_id','drug_col','drug_col_cid']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id', how='left')



# Cell DATA
with open(DC_PATH+'cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)


DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 751450
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']



# check DC triads (DC drug, cell line data )
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left')
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_cid>0]
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_cid>0] # 735595
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]
ccle_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCCLE)]

DC_DATA7_4_cello = DC_DATA7_3[cello_t] # 730348
DC_cello_final = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 563367
DC_cello_final_dup = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348



print('DC and LINCS')
BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id']

FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
FILTER2 = [True if type(a)==float else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER2] # 11742 ??? 
FILTER3 = [True if np.isnan(a)==False else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER3] # 11701 
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 





#
print('check val data')

GDSC_data = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/'+'Original_screen_All_tissues_fitted_data.csv', low_memory=False)
GDSC_CELL = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC/'+'GDSC_CELL_0707.tsv', sep = '\t')
GDSC_ID = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC/'+'GDSC_LIST_0706.tsv', sep = '\t')
GDSC_ID_cut = GDSC_ID[['drug_id','PubCHEM']].drop_duplicates()
GDSC_ID_cut['ANCHOR_ID'] = [str(a) for a in GDSC_ID_cut.drug_id]

# 296132
GDSC_cut = GDSC_data[['ANCHOR_ID','LIBRARY_ID','CELL_LINE_NAME','COSMIC_ID','ANCHOR_VIABILITY']].drop_duplicates()
# 108259
GDSC_cut2 = GDSC_cut[['ANCHOR_ID','LIBRARY_ID','CELL_LINE_NAME','COSMIC_ID']].drop_duplicates()

GDSC_ID_cut.columns = ['drug_id','ANCHOR_PubCHEM','ANCHOR_ID']
GDSC_cut3 = pd.merge(GDSC_cut2, GDSC_ID_cut, on ='ANCHOR_ID', how = 'left')

GDSC_ID_cut.columns = ['drug_id','LIBRARY_PubCHEM','LIBRARY_ID']
GDSC_cut4 = pd.merge(GDSC_cut3, GDSC_ID_cut, on = 'LIBRARY_ID', how = 'left')

GDSC_cut5 = GDSC_cut4[['ANCHOR_ID','ANCHOR_PubCHEM','LIBRARY_ID','LIBRARY_PubCHEM','COSMIC_ID','CELL_LINE_NAME']].drop_duplicates()
check_1 = [True if type(a) == str else False for a in list(GDSC_cut5.ANCHOR_PubCHEM)]
check_2 = [True if type(a) == str else False for a in list(GDSC_cut5.LIBRARY_PubCHEM)]
check_3 = [True if ((check_1[a]==True) & (check_2[a]==True)) else False for a in range(len(check_1)) ]
GDSC_cut6 = GDSC_cut5[check_3] # 전부 pubchem int 로 변환 가능 

DC_CELLO_pass = DC_CELL_DF[['cellosaurus_accession','cell_model_passport_id','cosmic_id']].drop_duplicates()


GDSC_cut7 = pd.merge(GDSC_cut6, DC_CELLO_pass, left_on='COSMIC_ID', right_on ='cosmic_id', how ='left')
check_4 = [True if type(a)==int else False for a in GDSC_cut7.cosmic_id]
GDSC_cut8 = GDSC_cut7[check_4] # 99429
New_Set = GDSC_cut8[['ANCHOR_PubCHEM','LIBRARY_PubCHEM','cellosaurus_accession']]





# 그래서 우리가 가지고 있는 세트랑 얼마나 겹치는가
ori_uniq = DC_cello_final_dup[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates()
ori_uniq = ori_uniq.reset_index(drop = True)
# 순서 생각
original_check1 = [ str(ori_uniq['drug_row_cid'][a]) + '_' + str(ori_uniq['drug_col_cid'][a]) + '_' + str(ori_uniq['DrugCombCello'][a]) for a in range(563367)]
original_check2 = [ str(ori_uniq['drug_col_cid'][a]) + '_' + str(ori_uniq['drug_row_cid'][a]) + '_' + str(ori_uniq['DrugCombCello'][a]) for a in range(563367)]
original_check = list(set(original_check1+original_check2))

New_Set = New_Set.reset_index(drop = True)
new_check1 = [ str(New_Set['ANCHOR_PubCHEM'][a]) + '_' + str(New_Set['LIBRARY_PubCHEM'][a]) + '_' + str(New_Set['cellosaurus_accession'][a]) for a in range(99429)]
#new_check2 = [ str(New_Set['LIBRARY_PubCHEM'][a]) + '_' + str(New_Set['ANCHOR_PubCHEM'][a]) + '_' + str(New_Set['cellosaurus_accession'][a]) for a in range(99429)]
#new_check = list(set(new_check1+new_check2))

# cid 기준 체크 
original_check1 = [ str(ori_uniq['drug_row_cid'][a]) + '_' + str(ori_uniq['drug_col_cid'][a]) for a in range(563367)]
original_check2 = [ str(ori_uniq['drug_col_cid'][a]) + '_' + str(ori_uniq['drug_row_cid'][a]) for a in range(563367)]
original_check = list(set(original_check1+original_check2))
New_Set = New_Set.reset_index(drop = True)
new_check1 = [ str(New_Set['ANCHOR_PubCHEM'][a]) + '_' + str(New_Set['LIBRARY_PubCHEM'][a]) for a in range(99429)]



len(set(new_check1)-set(original_check)) # 98256 개의 여집합 (cell 기준)



# 그래서 이 새로운 애들은 얼마나 sig 가 붙을 수 있죠 

New_Set['ANCHOR_PubCHEM_flt'] = [float(a) for a in New_Set['ANCHOR_PubCHEM']]
New_Set['LIBRARY_PubCHEM_flt'] = [float(a) for a in New_Set['LIBRARY_PubCHEM']]
New_Set2 = New_Set[['ANCHOR_PubCHEM_flt','LIBRARY_PubCHEM_flt','cellosaurus_accession']]

BETA_CID_CELLO_SIG.columns=['pert_id', 'ANCHOR_PubCHEM_flt', 'cellosaurus_accession', 'BETA_sig_id']
New_Set_LINCS_1 = pd.merge(New_Set2, BETA_CID_CELLO_SIG, on = ['ANCHOR_PubCHEM_flt','cellosaurus_accession'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['pert_id', 'LIBRARY_PubCHEM_flt', 'cellosaurus_accession', 'BETA_sig_id']
New_Set_LINCS_2 = pd.merge(New_Set_LINCS_1, BETA_CID_CELLO_SIG, on = ['LIBRARY_PubCHEM_flt','cellosaurus_accession'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id']

FILTER = [a for a in range(New_Set_LINCS_2.shape[0]) if (type(New_Set_LINCS_2.BETA_sig_id_x[a]) == str) & (type(New_Set_LINCS_2.BETA_sig_id_y[a]) == str)]
New_Set_LINCS_3 = New_Set_LINCS_2.loc[FILTER] # 2127
New_Set_LINCS_3[['BETA_sig_id_x','BETA_sig_id_y']].drop_duplicates()





# sig 차이 확인 
ori_uniq = CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y']].drop_duplicates()
ori_uniq = ori_uniq.reset_index(drop = True)

original_check1 = [ str(ori_uniq['BETA_sig_id_x'][a]) + '_' + str(ori_uniq['BETA_sig_id_y'][a]) for a in range(9230)]
original_check2 = [ str(ori_uniq['BETA_sig_id_y'][a]) + '_' + str(ori_uniq['BETA_sig_id_x'][a]) for a in range(9230)]
original_check = list(set(original_check1+original_check2))
New_Set_check2 = New_Set_LINCS_3[['BETA_sig_id_x','BETA_sig_id_y']].drop_duplicates()
New_Set_check3 = New_Set_check2.reset_index(drop = True)
new_check2 = [ str(New_Set_check3['BETA_sig_id_x'][a]) + '_' + str(New_Set_check3['BETA_sig_id_y'][a]) for a in range(2127)]

len(set(new_check2)-set(original_check)) # 1988 개의 여집합 (cell 기준)




# 아니 그래서 얘네가 인정한 synergy 는 몇개임

only_set = GDSC_data[['COMBI_ID','CELL_LINE_NAME']].drop_duplicates() # 108,259
syn_set = GDSC_data[['COMBI_ID','CELL_LINE_NAME','Synergy']].drop_duplicates() # 113,000





# test 
raw 값이 fit 값이랑 같아지는지 보면 되는일? 


fit_data_ex = GDSC_data[GDSC_data.CELL_LINE_NAME == 'CAL-120']
fit_data_ex2 = fit_data_ex[fit_data_ex.ANCHOR_NAME=='AZD7762']
fit_data_ex3 = fit_data_ex2[fit_data_ex2.LIBRARY_NAME == 'Cisplatin']

#    BARCODE CELL_LINE_NAME ANCHOR_NAME ANCHOR_CONC LIBRARY_NAME  LIBRARY_CONC  ANCHOR_VIABILITY
#0     14576        CAL-120     AZD7762        0.25    Cisplatin           4.0          0.962699
#16    14576        CAL-120     AZD7762      0.0625    Cisplatin           4.0          0.968714

fit_data_ex2_re = fit_data_ex[fit_data_ex.ANCHOR_NAME=='Cisplatin']
fit_data_ex3_re = fit_data_ex2_re[fit_data_ex2_re.LIBRARY_NAME == 'AZD7762']
#     BARCODE CELL_LINE_NAME ANCHOR_NAME ANCHOR_CONC LIBRARY_NAME  LIBRARY_CONC  ANCHOR_VIABILITY
#294    14578        CAL-120   Cisplatin           1      AZD7762           1.0          0.960081
#306    14578        CAL-120   Cisplatin           4      AZD7762           1.0          0.964972


raw_data_ex = NS_raw[NS_raw.CELL_LINE_NAME == 'CAL-120']
raw_data_ex_1 = raw_data_ex[raw_data_ex.DRUG_NAME=='AZD7762']


raw_data_ex_2 = raw_data_ex[raw_data_ex.DRUG_NAME=='Cisplatin']
#         BARCODE RESEARCH_PROJECT  SCAN_ID          DATE_CREATED             SCAN_DATE  CELL_ID  ...  POSITION      TAG DRUG_ID  DRUG_NAME  CONC  INTENSITY
#122770     14576      GDSC_Breast    14423  2016-04-06T23:00:00Z  2016-04-10T23:00:00Z     2147  ...       101     A1-S  1005.0  Cisplatin   4.0      29720
#122773     14576      GDSC_Breast    14423  2016-04-06T23:00:00Z  2016-04-10T23:00:00Z     2147  ...       103  L1-D1-C  1005.0  Cisplatin   4.0      27235
# ... 1456 rows 


plate_check_1_14576 = raw_data_ex_1[raw_data_ex_1.BARCODE == 14576]
plate_check_2_14576 = raw_data_ex_2[raw_data_ex_2.BARCODE == 14576]


plt_1_ANS1 = plate_check_1_14576[plate_check_1_14576.TAG == 'A12-S'] # 28152.2 mean
plt_1_ANC1 = plate_check_1_14576[plate_check_1_14576.TAG == 'A12-C']

'L1-D1-S' 'L1-D2-S', 'L1-D3-S', 'L1-D4-S', 'L1-D5-S', 'L1-D6-S', 'L1-D7-S',
'L1-D1-C' 'L1-D2-C', 'L1-D3-C', 'L1-D4-C', 'L1-D5-C', 'L1-D6-C', 'L1-D7-C',

plt_2_LIB1 = plate_check_2_14576[plate_check_2_14576.CONC==0.25]
plt_2_LIB2 = plate_check_2_14576[plate_check_2_14576.CONC==4.0]
plt_2_LIB_sgl = plate_check_2_14576[plate_check_2_14576.TAG=='L1-D7-C']


plate_14576 = NS_raw[NS_raw.BARCODE == 14576 ] # 2610
plate_14578 = NS_raw[NS_raw.BARCODE == 14578 ] # 2610




일단 raw 다시 보기 시작해야함 

GDSC_data = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/'+'Original_screen_All_tissues_fitted_data.csv', low_memory=False)
# 296707 rows 
NS_raw = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/'+'Original_screen_All_tissues_raw_data.csv') 
# 8164070 rows
NS_day1 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/'+'Original_screen_All_tissues_day1_data.csv')
# 890880 rows 

아 day 는 그냥 control 








