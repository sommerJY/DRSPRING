W 48 & W 491 & W 492

import pandas as pd 
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
from ray.tune import Analysis
import pickle
import math
import torch
import os 
import copy
import numpy as np

MJ_NAME = 'M3V5'
PPI_NAME = '349'
MISS_NAME = 'MIS1'

WORK_NAME = 'WORK_518' # 349
W_NAME = 'W518'
WORK_DATE = '23.06.15' # 349


anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
#     anal_dir = '/home01/k040a01/ray_results/PRJ02.23.06.13.M3V5.WORK_37.349.MIS22/'
list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]
exp_json
anal_df = ExperimentAnalysis(os.path.join(anal_dir, exp_json[0]))

ANA_DF_1 = anal_df.dataframe()
ANA_ALL_DF_1 = anal_df.trial_dataframes

ANA_DF = ANA_DF_1

ANA_DF = ANA_DF.sort_values('config/CV')
ANA_DF.index = [0,1,2,3,4]
ANA_ALL_DF = ANA_ALL_DF_1


ANA_DF.to_csv('/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
import pickle
with open("/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME), "wb") as fp:
	pickle.dump(ANA_ALL_DF,fp) 

'/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
"/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)


limit = 1000


cv0_key = ANA_DF['logdir'][0] ;	cv1_key = ANA_DF['logdir'][1]; 	cv2_key = ANA_DF['logdir'][2] ;	cv3_key = ANA_DF['logdir'][3];	cv4_key = ANA_DF['logdir'][4]

epc_T_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)
epc_T_LS_std = np.std([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)

epc_T_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)
epc_T_PC_std = np.std([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)

epc_T_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)
epc_T_SC_std = np.std([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)

epc_V_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)
epc_V_LS_std = np.std([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)

epc_V_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)
epc_V_PC_std = np.std([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)

epc_V_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)
epc_V_SC_std = np.std([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)


epc_result = pd.DataFrame({
	'T_LS_mean' : epc_T_LS_mean, 'T_PC_mean' : epc_T_PC_mean, 'T_SC_mean' : epc_T_SC_mean, 
	'T_LS_std' : epc_T_LS_std, 'T_PC_std' : epc_T_PC_std, 'T_SC_std' : epc_T_SC_std, 
	'V_LS_mean' : epc_V_LS_mean, 'V_PC_mean' : epc_V_PC_mean, 'V_SC_mean' : epc_V_SC_mean, 
	'V_LS_std' : epc_V_LS_std, 'V_PC_std' : epc_V_PC_std, 'V_SC_std' : epc_V_SC_std,
})

epc_result[['T_LS_mean', 'T_LS_std', 'T_PC_mean', 'T_PC_std','T_SC_mean','T_SC_std', 'V_LS_mean', 'V_LS_std', 'V_PC_mean', 'V_PC_std','V_SC_mean','V_SC_std']].to_csv("/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))

"/home01/k040a01/02.M3V5/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
        


1) min loss

min(epc_result.sort_values('V_LS_mean')['V_LS_mean']) ; min_VLS = min(epc_result.sort_values('V_LS_mean')['V_LS_mean'])
KEY_EPC = epc_result[epc_result.V_LS_mean == min_VLS].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VLS_cv0_PATH = cv0_key + checkpoint
VLS_cv0_PATH
VLS_cv1_PATH = cv1_key + checkpoint
VLS_cv1_PATH
VLS_cv2_PATH = cv2_key + checkpoint
VLS_cv2_PATH
VLS_cv3_PATH = cv3_key + checkpoint
VLS_cv3_PATH
VLS_cv4_PATH = cv4_key + checkpoint
VLS_cv4_PATH


KEY_EPC
round(epc_result.loc[KEY_EPC].V_LS_mean,3)
round(epc_result.loc[KEY_EPC].V_LS_std,3)



2) PC best 

epc_result.sort_values('V_PC_mean', ascending = False) 
max(epc_result['V_PC_mean']); max_VPC = max(epc_result['V_PC_mean'])
KEY_EPC = epc_result[epc_result.V_PC_mean == max_VPC].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VPC_cv0_PATH = cv0_key + checkpoint
VPC_cv0_PATH
VPC_cv1_PATH = cv1_key + checkpoint
VPC_cv1_PATH
VPC_cv2_PATH = cv2_key + checkpoint
VPC_cv2_PATH
VPC_cv3_PATH = cv3_key + checkpoint
VPC_cv3_PATH
VPC_cv4_PATH = cv4_key + checkpoint
VPC_cv4_PATH


KEY_EPC
round(epc_result.loc[KEY_EPC].V_PC_mean,3)
round(epc_result.loc[KEY_EPC].V_PC_std,3)


3) SC best 

epc_result.sort_values('V_SC_mean', ascending = False) 
max(epc_result['V_SC_mean']); max_VSC = max(epc_result['V_SC_mean'])
KEY_EPC = epc_result[epc_result.V_SC_mean == max_VSC].index.item()
checkpoint = "/checkpoint_"+str(KEY_EPC).zfill(6)
VSC_cv0_PATH = cv0_key + checkpoint
VSC_cv0_PATH
VSC_cv1_PATH = cv1_key + checkpoint
VSC_cv1_PATH
VSC_cv2_PATH = cv2_key + checkpoint
VSC_cv2_PATH
VSC_cv3_PATH = cv3_key + checkpoint
VSC_cv3_PATH
VSC_cv4_PATH = cv4_key + checkpoint
VSC_cv4_PATH

KEY_EPC
round(epc_result.loc[KEY_EPC].V_SC_mean,3)
round(epc_result.loc[KEY_EPC].V_SC_std,3)












with open(file='{}/CV_SM_list.pickle'.format(WORK_PATH), mode='wb') as f:
    pickle.dump(CV_ND_INDS, f)
with open(file='{}/CV_SM_list.pickle'.format(WORK_PATH), mode='wb') as f:
        pickle.dump(CV_ND_INDS, f)




/home01/k040a01/02.M3V5/M3V5_W501_349_MIS0/
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W501_349_MIS0/RESULT.G4.CV5.txt
tail /home01/k040a01/logs/M3V5W501_GPU4_12889.log


sbatch gpu4.W502.CV5.any M3V5_WORK502.349.CV5.py
/home01/k040a01/02.M3V5/M3V5_W502_349_MIS0/
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W502_349_MIS0/RESULT.G4.CV5.txt
tail /home01/k040a01/logs/M3V5W502_GPU4_12889.log



sbatch gpu4.W503.CV5.any M3V5_WORK503.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W503_349_MIS0/RESULT.G4.CV5.txt

sbatch gpu4.W504.CV5.any M3V5_WORK504.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W504_349_MIS1/RESULT.G4.CV5.txt

sbatch gpu4.W505.CV5.any M3V5_WORK505.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W505_349_MIS1/RESULT.G4.CV5.txt

sbatch gpu4.W506.CV5.any M3V5_WORK506.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W506_349_MIS1/RESULT.G4.CV5.txt




sbatch gpu4.W507.CV5.any M3V5_WORK507.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W507_349_MIS0/RESULT.G4.CV5.txt

sbatch gpu4.W508.CV5.any M3V5_WORK508.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W508_349_MIS0/RESULT.G4.CV5.txt

sbatch gpu4.W509.CV5.any M3V5_WORK509.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W509_349_MIS0/RESULT.G4.CV5.txt



sbatch gpu4.W510.CV5.any M3V5_WORK510.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W510_349_MIS1/RESULT.G4.CV5.txt 

sbatch gpu4.W511.CV5.any M3V5_WORK511.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W511_349_MIS1/RESULT.G4.CV5.txt

sbatch gpu4.W512.CV5.any M3V5_WORK512.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W512_349_MIS1/RESULT.G4.CV5.txt









sbatch gpu4.W513.CV5.any M3V5_WORK513.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W513_349_MIS0/RESULT.G4.CV5.txt

sbatch gpu4.W514.CV5.any M3V5_WORK514.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W514_349_MIS0/RESULT.G4.CV5.txt

sbatch gpu4.W515.CV5.any M3V5_WORK515.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W515_349_MIS0/RESULT.G4.CV5.txt



sbatch gpu4.W516.CV5.any M3V5_WORK516.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W516_349_MIS1/RESULT.G4.CV5.txt 

sbatch gpu4.W517.CV5.any M3V5_WORK517.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W517_349_MIS1/RESULT.G4.CV5.txt

sbatch gpu4.W518.CV5.any M3V5_WORK518.349.CV5.py
tail -n 100 /home01/k040a01/02.M3V5/M3V5_W518_349_MIS1/RESULT.G4.CV5.txt







print('CID_CID', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])

len(set(tmp2))


print('CID_CID', flush = True)
len(set(A_B_C_S_SET_COH2.CID_CID))



print('CID_CID_CCLE', flush = True)
len(set(A_B_C_S_SET_COH2.CID_CID_CCLE))

print('DrugCombCCLE', flush = True)
len(set(A_B_C_S_SET_COH2.DrugCombCCLE))
