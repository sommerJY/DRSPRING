



# 혹시 이쁜 barplot 으로 나타낼 수 있을까 싶어서 그려보는 ablation 결과 

def get_mean(ANA_DF, ANA_ALL_DF, limit) : 
	limit = 1000
	#
	cv0_key = ANA_DF['logdir'][0] ;	cv1_key = ANA_DF['logdir'][1]; 	cv2_key = ANA_DF['logdir'][2] ;	cv3_key = ANA_DF['logdir'][3];	cv4_key = ANA_DF['logdir'][4]
	#	
	epc_T_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)
	epc_T_LS_std = np.std([ANA_ALL_DF[cv0_key]['T_LS'][0:limit], ANA_ALL_DF[cv1_key]['T_LS'][0:limit],ANA_ALL_DF[cv2_key]['T_LS'][0:limit], ANA_ALL_DF[cv3_key]['T_LS'][0:limit], ANA_ALL_DF[cv4_key]['T_LS'][0:limit]], axis = 0)
	#
	epc_T_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)
	epc_T_PC_std = np.std([ANA_ALL_DF[cv0_key]['T_PC'][0:limit], ANA_ALL_DF[cv1_key]['T_PC'][0:limit],ANA_ALL_DF[cv2_key]['T_PC'][0:limit], ANA_ALL_DF[cv3_key]['T_PC'][0:limit], ANA_ALL_DF[cv4_key]['T_PC'][0:limit]], axis = 0)
	#
	epc_T_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)
	epc_T_SC_std = np.std([ANA_ALL_DF[cv0_key]['T_SC'][0:limit], ANA_ALL_DF[cv1_key]['T_SC'][0:limit],ANA_ALL_DF[cv2_key]['T_SC'][0:limit], ANA_ALL_DF[cv3_key]['T_SC'][0:limit], ANA_ALL_DF[cv4_key]['T_SC'][0:limit]], axis = 0)
	#
	epc_V_LS_mean = np.mean([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)
	epc_V_LS_std = np.std([ANA_ALL_DF[cv0_key]['V_LS'][0:limit], ANA_ALL_DF[cv1_key]['V_LS'][0:limit],ANA_ALL_DF[cv2_key]['V_LS'][0:limit], ANA_ALL_DF[cv3_key]['V_LS'][0:limit], ANA_ALL_DF[cv4_key]['V_LS'][0:limit]], axis = 0)
	#
	epc_V_PC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)
	epc_V_PC_std = np.std([ANA_ALL_DF[cv0_key]['V_PC'][0:limit], ANA_ALL_DF[cv1_key]['V_PC'][0:limit],ANA_ALL_DF[cv2_key]['V_PC'][0:limit], ANA_ALL_DF[cv3_key]['V_PC'][0:limit], ANA_ALL_DF[cv4_key]['V_PC'][0:limit]], axis = 0)
	#
	epc_V_SC_mean = np.mean([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)
	epc_V_SC_std = np.std([ANA_ALL_DF[cv0_key]['V_SC'][0:limit], ANA_ALL_DF[cv1_key]['V_SC'][0:limit],ANA_ALL_DF[cv2_key]['V_SC'][0:limit], ANA_ALL_DF[cv3_key]['V_SC'][0:limit], ANA_ALL_DF[cv4_key]['V_SC'][0:limit]], axis = 0)
	#
	epc_result = pd.DataFrame({
		'T_LS_mean' : epc_T_LS_mean, 'T_PC_mean' : epc_T_PC_mean, 'T_SC_mean' : epc_T_SC_mean, 
		'T_LS_std' : epc_T_LS_std, 'T_PC_std' : epc_T_PC_std, 'T_SC_std' : epc_T_SC_std, 
		'V_LS_mean' : epc_V_LS_mean, 'V_PC_mean' : epc_V_PC_mean, 'V_SC_mean' : epc_V_SC_mean, 
		'V_LS_std' : epc_V_LS_std, 'V_PC_std' : epc_V_PC_std, 'V_SC_std' : epc_V_SC_std,
	})
	return (epc_result)


# 각 epoch 별 mean 으로 본거 
def get_mean_result(epc_result) :
	#
	#1) min loss
	#
	min(epc_result.sort_values('V_LS_mean')['V_LS_mean']) ; min_VLS = min(epc_result.sort_values('V_LS_mean')['V_LS_mean'])
	KEY_EPC_1 = epc_result[epc_result.V_LS_mean == min_VLS].index.item()
	epc_V_LS_mean = round(epc_result.loc[KEY_EPC_1].V_LS_mean, 4)
	epc_V_LS_std = round(epc_result.loc[KEY_EPC_1].V_LS_std, 4)
	#
	#2) PC best 
	#
	epc_result.sort_values('V_PC_mean', ascending = False) 
	max(epc_result['V_PC_mean']); max_VPC = max(epc_result['V_PC_mean'])
	KEY_EPC_2 = epc_result[epc_result.V_PC_mean == max_VPC].index.item()
	epc_V_PC_mean = round(epc_result.loc[KEY_EPC_2].V_PC_mean, 4)
	epc_V_PC_std = round(epc_result.loc[KEY_EPC_2].V_PC_std, 4)
	#
	#
	#3) SC best 
	#
	epc_result.sort_values('V_SC_mean', ascending = False) 
	max(epc_result['V_SC_mean']); max_VSC = max(epc_result['V_SC_mean'])
	KEY_EPC_3 = epc_result[epc_result.V_SC_mean == max_VSC].index.item()
	epc_V_SC_mean = round(epc_result.loc[KEY_EPC_3].V_SC_mean, 4)
	epc_V_SC_std = round(epc_result.loc[KEY_EPC_3].V_SC_std, 4)
	#
	print(KEY_EPC_1)
	print([epc_V_LS_mean, epc_V_LS_std])
	print(KEY_EPC_2)
	print([epc_V_PC_mean, epc_V_PC_std])
	print(KEY_EPC_3)
	print([epc_V_SC_mean, epc_V_SC_std])
	return(epc_V_LS_mean, epc_V_LS_std, epc_V_PC_mean, epc_V_PC_std, epc_V_SC_mean, epc_V_SC_std)


#  각 CV 결과에서 제일 좋게 나왔던거 가져다 본거 (early stop 썼다고 생각하기 )
def get_max_result(ANA_DF, ANA_ALL_DF, nan_check = 0) :
	#
	cv_keys = list(ANA_DF['logdir'])
	#
	epc_T_LS_mean = np.mean([min(ANA_ALL_DF[kk]['T_LS'][nan_check:]) for kk in cv_keys])
	epc_T_LS_std = np.std([min(ANA_ALL_DF[kk]['T_LS'][nan_check:]) for kk in cv_keys])
	#
	epc_T_PC_mean = np.mean([max(ANA_ALL_DF[kk]['T_PC'][nan_check:]) for kk in cv_keys])
	epc_T_PC_std = np.std([max(ANA_ALL_DF[kk]['T_PC'][nan_check:]) for kk in cv_keys])
	#
	epc_T_SC_mean = np.mean([max(ANA_ALL_DF[kk]['T_SC'][nan_check:]) for kk in cv_keys])
	epc_T_SC_std = np.std([max(ANA_ALL_DF[kk]['T_SC'][nan_check:]) for kk in cv_keys])
	#
	epc_V_LS_mean = np.mean([min(ANA_ALL_DF[kk]['V_LS'][nan_check:]) for kk in cv_keys])
	epc_V_LS_std = np.std([min(ANA_ALL_DF[kk]['V_LS'][nan_check:]) for kk in cv_keys])
	#
	epc_V_PC_mean = np.mean([max(ANA_ALL_DF[kk]['V_PC'][nan_check:]) for kk in cv_keys])
	epc_V_PC_std = np.std([max(ANA_ALL_DF[kk]['V_PC'][nan_check:]) for kk in cv_keys])
	#
	epc_V_SC_mean = np.mean([max(ANA_ALL_DF[kk]['V_SC'][nan_check:]) for kk in cv_keys])
	epc_V_SC_std = np.std([max(ANA_ALL_DF[kk]['V_SC'][nan_check:]) for kk in cv_keys])
	#
	print('Train')
	print(np.round([epc_T_LS_mean, epc_T_LS_std], 4))
	print(np.round([epc_T_PC_mean, epc_T_PC_std], 4))
	print(np.round([epc_T_SC_mean, epc_T_SC_std], 4))
	print('Val')
	print(np.round([epc_V_LS_mean, epc_V_LS_std], 4))
	print(np.round([epc_V_PC_mean, epc_V_PC_std], 4))
	print(np.round([epc_V_SC_mean, epc_V_SC_std], 4))
	#
	return(epc_V_LS_mean, epc_V_LS_std, epc_V_PC_mean, epc_V_PC_std, epc_V_SC_mean, epc_V_SC_std)


# 모르겠고 1000 epoch 돌린 결과로 보기 
def get_last_result(ANA_DF, ANA_ALL_DF) :
	#
	cv_keys = list(ANA_DF['logdir'])
	#
	epc_T_LS_mean = np.mean([ANA_ALL_DF[kk]['T_LS'][999] for kk in cv_keys])
	epc_T_LS_std = np.std([ANA_ALL_DF[kk]['T_LS'][999] for kk in cv_keys])
	#
	epc_T_PC_mean = np.mean([ANA_ALL_DF[kk]['T_PC'][999] for kk in cv_keys])
	epc_T_PC_std = np.std([ANA_ALL_DF[kk]['T_PC'][999] for kk in cv_keys])
	#
	epc_T_SC_mean = np.mean([ANA_ALL_DF[kk]['T_SC'][999] for kk in cv_keys])
	epc_T_SC_std = np.std([ANA_ALL_DF[kk]['T_SC'][999] for kk in cv_keys])
	#
	epc_V_LS_mean = np.mean([ANA_ALL_DF[kk]['V_LS'][999] for kk in cv_keys])
	epc_V_LS_std = np.std([ANA_ALL_DF[kk]['V_LS'][999] for kk in cv_keys])
	#
	epc_V_PC_mean = np.mean([ANA_ALL_DF[kk]['V_PC'][999] for kk in cv_keys])
	epc_V_PC_std = np.std([ANA_ALL_DF[kk]['V_PC'][999] for kk in cv_keys])
	#
	epc_V_SC_mean = np.mean([ANA_ALL_DF[kk]['V_SC'][999] for kk in cv_keys])
	epc_V_SC_std = np.std([ANA_ALL_DF[kk]['V_SC'][999] for kk in cv_keys])
	#
	print('Train')
	print(np.round([epc_T_LS_mean, epc_T_LS_std], 4))
	print(np.round([epc_T_PC_mean, epc_T_PC_std], 4))
	print(np.round([epc_T_SC_mean, epc_T_SC_std], 4))
	print('Val')
	print(np.round([epc_V_LS_mean, epc_V_LS_std], 4))
	print(np.round([epc_V_PC_mean, epc_V_PC_std], 4))
	print(np.round([epc_V_SC_mean, epc_V_SC_std], 4))
	#
	return(epc_V_LS_mean, epc_V_LS_std, epc_V_PC_mean, epc_V_PC_std, epc_V_SC_mean, epc_V_SC_std)
















# self.target_A[index].tolist()
# tail /home01/k040a01/02.M3V6/M3V6_W_2_349_MIS2/RESULT.206_2.CV5.txt





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
import seaborn as sns


MJ_NAME = 'M3V8'
PPI_NAME = '349'
MISS_NAME = 'MIS2'


			1) AOBO + AOBX + AXBX
			1-1 ) 전체 다 먹는거 
W_NAME = 'W403'
WORK_NAME = 'WORK_403' # 349
WORK_DATE = '23.08.27' # 349
WORK_DATE = '23.09.08_G1' # 349


anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D4F3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D4F3, epc_V_LS_std_D4F3, epc_V_PC_mean_D4F3, epc_V_PC_std_D4F3, epc_V_SC_mean_D4F3, epc_V_SC_std_D4F3 = get_mean_result(epc_result_D4F3)

all_V_LS_mean_D4F3, all_V_LS_std_D4F3, all_V_PC_mean_D4F3, all_V_PC_std_D4F3, all_V_SC_mean_D4F3, all_V_SC_std_D4F3 = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D4F3, last_V_LS_std_D4F3, last_V_PC_mean_D4F3, last_V_PC_std_D4F3, last_V_SC_mean_D4F3, last_V_SC_std_D4F3 = get_last_result(ANA_DF, ANA_ALL_DF)


			1) AOBO + AOBX + AXBX
			2) EXP only 
W_NAME = 'W405'
WORK_NAME = 'WORK_405' # 349
WORK_DATE = '23.08.27' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D4F1E = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D4F1E, epc_V_LS_std_D4F1E, epc_V_PC_mean_D4F1E, epc_V_PC_std_D4F1E, epc_V_SC_mean_D4F1E, epc_V_SC_std_D4F1E = get_mean_result(epc_result_D4F1E)

all_V_LS_mean_D4F1E, all_V_LS_std_D4F1E, all_V_PC_mean_D4F1E, all_V_PC_std_D4F1E, all_V_SC_mean_D4F1E, all_V_SC_std_D4F1E = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D4F1E, last_V_LS_std_D4F1E, last_V_PC_mean_D4F1E, last_V_PC_std_D4F1E, last_V_SC_mean_D4F1E, last_V_SC_std_D4F1E = get_last_result(ANA_DF, ANA_ALL_DF)








			1) AOBO + AOBX + AXBX
			1-2 ) Basal only
W_NAME = 'W406'
WORK_NAME = 'WORK_406' # 349
WORK_DATE = '23.08.27' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D4F1B = get_mean(ANA_DF, ANA_ALL_DF, 1000)
epc_result_D4F1B = epc_result_D4F1B.loc[2:,:]

epc_V_LS_mean_D4F1B, epc_V_LS_std_D4F1B, epc_V_PC_mean_D4F1B, epc_V_PC_std_D4F1B, epc_V_SC_mean_D4F1B, epc_V_SC_std_D4F1B = get_mean_result(epc_result_D4F1B)

all_V_LS_mean_D4F1B, all_V_LS_std_D4F1B, all_V_PC_mean_D4F1B, all_V_PC_std_D4F1B, all_V_SC_mean_D4F1B, all_V_SC_std_D4F1B = get_max_result(ANA_DF, ANA_ALL_DF, 2)

last_V_LS_mean_D4F1B, last_V_LS_std_D4F1B, last_V_PC_mean_D4F1B, last_V_PC_std_D4F1B, last_V_SC_mean_D4F1B, last_V_SC_std_D4F1B = get_last_result(ANA_DF, ANA_ALL_DF)


			1) AOBO + AOBX + AXBX
			1-2 ) target only
W_NAME = 'W407'
WORK_NAME = 'WORK_407' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D4F1T = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D4F1T, epc_V_LS_std_D4F1T, epc_V_PC_mean_D4F1T, epc_V_PC_std_D4F1T, epc_V_SC_mean_D4F1T, epc_V_SC_std_D4F1T = get_mean_result(epc_result_D4F1T)

all_V_LS_mean_D4F1T, all_V_LS_std_D4F1T, all_V_PC_mean_D4F1T, all_V_PC_std_D4F1T, all_V_SC_mean_D4F1T, all_V_SC_std_D4F1T = get_max_result(ANA_DF, ANA_ALL_DF, 2)

last_V_LS_mean_D4F1T, last_V_LS_std_D4F1T, last_V_PC_mean_D4F1T, last_V_PC_std_D4F1T, last_V_SC_mean_D4F1T, last_V_SC_std_D4F1T = get_last_result(ANA_DF, ANA_ALL_DF)








			1) AOBO + AOBX + AXBX
			1-2 ) Basal + target 
W_NAME = 'W408'
WORK_NAME = 'WORK_408' # 349
WORK_DATE = '23.08.28' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D4F2BT = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D4F2BT, epc_V_LS_std_D4F2BT, epc_V_PC_mean_D4F2BT, epc_V_PC_std_D4F2BT, epc_V_SC_mean_D4F2BT, epc_V_SC_std_D4F2BT = get_mean_result(epc_result_D4F2BT)

all_V_LS_mean_D4F2BT, all_V_LS_std_D4F2BT, all_V_PC_mean_D4F2BT, all_V_PC_std_D4F2BT, all_V_SC_mean_D4F2BT, all_V_SC_std_D4F2BT = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D4F2BT, last_V_LS_std_D4F2BT, last_V_PC_mean_D4F2BT, last_V_PC_std_D4F2BT, last_V_SC_mean_D4F2BT, last_V_SC_std_D4F2BT = get_last_result(ANA_DF, ANA_ALL_DF)








			1) AOBO + AOBX + AXBX
			1-2 ) EXP + target 
W_NAME = 'W409'
WORK_NAME = 'WORK_409' # 349
WORK_DATE = '23.08.28' # 349
WORK_DATE = '23.09.09' # 349

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

epc_result_D4F2ET = get_mean(ANA_DF, ANA_ALL_DF, 1000)
epc_result_D4F2ET = epc_result_D4F2ET.loc[5:,:]

epc_V_LS_mean_D4F2ET, epc_V_LS_std_D4F2ET, epc_V_PC_mean_D4F2ET, epc_V_PC_std_D4F2ET, epc_V_SC_mean_D4F2ET, epc_V_SC_std_D4F2ET = get_mean_result(epc_result_D4F2ET)

all_V_LS_mean_D4F2ET, all_V_LS_std_D4F2ET, all_V_PC_mean_D4F2ET, all_V_PC_std_D4F2ET, all_V_SC_mean_D4F2ET, all_V_SC_std_D4F2ET = get_max_result(ANA_DF, ANA_ALL_DF, 5)

last_V_LS_mean_D4F2ET, last_V_LS_std_D4F2ET, last_V_PC_mean_D4F2ET, last_V_PC_std_D4F2ET, last_V_SC_mean_D4F2ET, last_V_SC_std_D4F2ET = get_last_result(ANA_DF, ANA_ALL_DF)










			1) AOBO + AOBX + AXBX
			1-2 ) EXP + basal 아직 돌아가는중 
W_NAME = 'W410'
WORK_NAME = 'WORK_410' # 349
WORK_DATE = '23.08.28' # 349
WORK_DATE = '23.09.09' # 349

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

epc_result_D4F2EB = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D4F2EB, epc_V_LS_std_D4F2EB, epc_V_PC_mean_D4F2EB, epc_V_PC_std_D4F2EB, epc_V_SC_mean_D4F2EB, epc_V_SC_std_D4F2EB = get_mean_result(epc_result_D4F2EB)

all_V_LS_mean_D4F2EB, all_V_LS_std_D4F2EB, all_V_PC_mean_D4F2EB, all_V_PC_std_D4F2EB, all_V_SC_mean_D4F2EB, all_V_SC_std_D4F2EB = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D4F2EB, last_V_LS_std_D4F2EB, last_V_PC_mean_D4F2EB, last_V_PC_std_D4F2EB, last_V_SC_mean_D4F2EB, last_V_SC_std_D4F2EB = get_last_result(ANA_DF, ANA_ALL_DF)




########################################################
########################################################
########################################################
########################################################
########################################################


			2) AOBO
			1-1 ) 전체 다 먹는거 
W_NAME = 'W403_1'
WORK_NAME = 'WORK_403_1' # 349
WORK_DATE = '23.08.27' # 349

anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D1F3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F3, epc_V_LS_std_D1F3, epc_V_PC_mean_D1F3, epc_V_PC_std_D1F3, epc_V_SC_mean_D1F3, epc_V_SC_std_D1F3 = get_mean_result(epc_result_D1F3)

all_V_LS_mean_D1F3, all_V_LS_std_D1F3, all_V_PC_mean_D1F3, all_V_PC_std_D1F3, all_V_SC_mean_D1F3, all_V_SC_std_D1F3 = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F3, last_V_LS_std_D1F3, last_V_PC_mean_D1F3, last_V_PC_std_D1F3, last_V_SC_mean_D1F3, last_V_SC_std_D1F3 = get_last_result(ANA_DF, ANA_ALL_DF)


			2) AOBO
			1-2 ) gene exp 만 

W_NAME = 'W405_1'
WORK_NAME = 'WORK_405_1' # 349
WORK_DATE = '23.08.29' # 349

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

epc_result_D1F1E = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F1E, epc_V_LS_std_D1F1E, epc_V_PC_mean_D1F1E, epc_V_PC_std_D1F1E, epc_V_SC_mean_D1F1E, epc_V_SC_std_D1F1E = get_mean_result(epc_result_D1F1E)

all_V_LS_mean_D1F1E, all_V_LS_std_D1F1E, all_V_PC_mean_D1F1E, all_V_PC_std_D1F1E, all_V_SC_mean_D1F1E, all_V_SC_std_D1F1E = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F1E, last_V_LS_std_D1F1E, last_V_PC_mean_D1F1E, last_V_PC_std_D1F1E, last_V_SC_mean_D1F1E, last_V_SC_std_D1F1E = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AOBO
			1-2 ) Basal only / RAY_MY_train_ffb94_00000 
W_NAME = 'W406_1'
WORK_NAME = 'WORK_406_1' # 349
WORK_DATE = '23.08.29' # 349

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

epc_result_D1F1B = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F1B, epc_V_LS_std_D1F1B, epc_V_PC_mean_D1F1B, epc_V_PC_std_D1F1B, epc_V_SC_mean_D1F1B, epc_V_SC_std_D1F1B = get_mean_result(epc_result_D1F1B)

all_V_LS_mean_D1F1B, all_V_LS_std_D1F1B, all_V_PC_mean_D1F1B, all_V_PC_std_D1F1B, all_V_SC_mean_D1F1B, all_V_SC_std_D1F1B = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F1B, last_V_LS_std_D1F1B, last_V_PC_mean_D1F1B, last_V_PC_std_D1F1B, last_V_SC_mean_D1F1B, last_V_SC_std_D1F1B = get_last_result(ANA_DF, ANA_ALL_DF)



			2) AOBO
			1-2 ) TARGET only 

W_NAME = 'W407_1'
WORK_NAME = 'WORK_407_1' # 349
WORK_DATE = '23.08.29' # 349

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

epc_result_D1F1T = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F1T, epc_V_LS_std_D1F1T, epc_V_PC_mean_D1F1T, epc_V_PC_std_D1F1T, epc_V_SC_mean_D1F1T, epc_V_SC_std_D1F1T = get_mean_result(epc_result_D1F1T)

all_V_LS_mean_D1F1T, all_V_LS_std_D1F1T, all_V_PC_mean_D1F1T, all_V_PC_std_D1F1T, all_V_SC_mean_D1F1T, all_V_SC_std_D1F1T = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F1T, last_V_LS_std_D1F1T, last_V_PC_mean_D1F1T, last_V_PC_std_D1F1T, last_V_SC_mean_D1F1T, last_V_SC_std_D1F1T = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AOBO
			1-2 ) Basal + target 
W_NAME = 'W408_1'
WORK_NAME = 'WORK_408_1' # 349
WORK_DATE = '23.08.29' # 349

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

epc_result_D1F2BT = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F2BT, epc_V_LS_std_D1F2BT, epc_V_PC_mean_D1F2BT, epc_V_PC_std_D1F2BT, epc_V_SC_mean_D1F2BT, epc_V_SC_std_D1F2BT = get_mean_result(epc_result_D1F2BT)

all_V_LS_mean_D1F2BT, all_V_LS_std_D1F2BT, all_V_PC_mean_D1F2BT, all_V_PC_std_D1F2BT, all_V_SC_mean_D1F2BT, all_V_SC_std_D1F2BT = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F2BT, last_V_LS_std_D1F2BT, last_V_PC_mean_D1F2BT, last_V_PC_std_D1F2BT, last_V_SC_mean_D1F2BT, last_V_SC_std_D1F2BT = get_last_result(ANA_DF, ANA_ALL_DF)









			2) AOBO
			1-2 ) EXP + target 
W_NAME = 'W409_1'
WORK_NAME = 'WORK_409_1' # 349
WORK_DATE = '23.08.29' # 349

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

epc_result_D1F2ET = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F2ET, epc_V_LS_std_D1F2ET, epc_V_PC_mean_D1F2ET, epc_V_PC_std_D1F2ET, epc_V_SC_mean_D1F2ET, epc_V_SC_std_D1F2ET = get_mean_result(epc_result_D1F2ET)

all_V_LS_mean_D1F2ET, all_V_LS_std_D1F2ET, all_V_PC_mean_D1F2ET, all_V_PC_std_D1F2ET, all_V_SC_mean_D1F2ET, all_V_SC_std_D1F2ET = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F2ET, last_V_LS_std_D1F2ET, last_V_PC_mean_D1F2ET, last_V_PC_std_D1F2ET, last_V_SC_mean_D1F2ET, last_V_SC_std_D1F2ET = get_last_result(ANA_DF, ANA_ALL_DF)









			2) AOBO
			1-2 ) EXP + basal
W_NAME = 'W410_1'
WORK_NAME = 'WORK_410_1' # 349
WORK_DATE = '23.08.29' # 349

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

epc_result_D1F2EB = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D1F2EB, epc_V_LS_std_D1F2EB, epc_V_PC_mean_D1F2EB, epc_V_PC_std_D1F2EB, epc_V_SC_mean_D1F2EB, epc_V_SC_std_D1F2EB = get_mean_result(epc_result_D1F2EB)

all_V_LS_mean_D1F2EB, all_V_LS_std_D1F2EB, all_V_PC_mean_D1F2EB, all_V_PC_std_D1F2EB, all_V_SC_mean_D1F2EB, all_V_SC_std_D1F2EB = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D1F2EB, last_V_LS_std_D1F2EB, last_V_PC_mean_D1F2EB, last_V_PC_std_D1F2EB, last_V_SC_mean_D1F2EB, last_V_SC_std_D1F2EB = get_last_result(ANA_DF, ANA_ALL_DF)





########################################################
########################################################
########################################################
########################################################
########################################################

AXBO 시리즈 



			2) AXBO + AOBX
			1-1 ) 전체 다 먹는거 
W_NAME = 'W403_2'
WORK_NAME = 'WORK_403_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.19' # 349


anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F3, epc_V_LS_std_D2F3, epc_V_PC_mean_D2F3, epc_V_PC_std_D2F3, epc_V_SC_mean_D2F3, epc_V_SC_std_D2F3 = get_mean_result(epc_result_D2F3)

all_V_LS_mean_D2F3, all_V_LS_std_D2F3, all_V_PC_mean_D2F3, all_V_PC_std_D2F3, all_V_SC_mean_D2F3, all_V_SC_std_D2F3 = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F3, last_V_LS_std_D2F3, last_V_PC_mean_D2F3, last_V_PC_std_D2F3, last_V_SC_mean_D2F3, last_V_SC_std_D2F3 = get_last_result(ANA_DF, ANA_ALL_DF)




			2) AXBO + AOBX
			1-2 ) gene exp 만 

W_NAME = 'W405_2'
WORK_NAME = 'WORK_405_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F1E = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F1E, epc_V_LS_std_D2F1E, epc_V_PC_mean_D2F1E, epc_V_PC_std_D2F1E, epc_V_SC_mean_D2F1E, epc_V_SC_std_D2F1E = get_mean_result(epc_result_D2F1E)

all_V_LS_mean_D2F1E, all_V_LS_std_D2F1E, all_V_PC_mean_D2F1E, all_V_PC_std_D2F1E, all_V_SC_mean_D2F1E, all_V_SC_std_D2F1E = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F1E, last_V_LS_std_D2F1E, last_V_PC_mean_D2F1E, last_V_PC_std_D2F1E, last_V_SC_mean_D2F1E, last_V_SC_std_D2F1E = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AXBO + AOBX
			1-2 ) Basal only / RAY_MY_train_ffb94_00000 
W_NAME = 'W406_2'
WORK_NAME = 'WORK_406_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F1B = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F1B, epc_V_LS_std_D2F1B, epc_V_PC_mean_D2F1B, epc_V_PC_std_D2F1B, epc_V_SC_mean_D2F1B, epc_V_SC_std_D2F1B = get_mean_result(epc_result_D2F1B)

all_V_LS_mean_D2F1B, all_V_LS_std_D2F1B, all_V_PC_mean_D2F1B, all_V_PC_std_D2F1B, all_V_SC_mean_D2F1B, all_V_SC_std_D2F1B = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F1B, last_V_LS_std_D2F1B, last_V_PC_mean_D2F1B, last_V_PC_std_D2F1B, last_V_SC_mean_D2F1B, last_V_SC_std_D2F1B = get_last_result(ANA_DF, ANA_ALL_DF)



			2) AXBO + AOBX
			1-2 ) TARGET only 

W_NAME = 'W407_2'
WORK_NAME = 'WORK_407_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F1T = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F1T, epc_V_LS_std_D2F1T, epc_V_PC_mean_D2F1T, epc_V_PC_std_D2F1T, epc_V_SC_mean_D2F1T, epc_V_SC_std_D2F1T = get_mean_result(epc_result_D2F1T)

all_V_LS_mean_D2F1T, all_V_LS_std_D2F1T, all_V_PC_mean_D2F1T, all_V_PC_std_D2F1T, all_V_SC_mean_D2F1T, all_V_SC_std_D2F1T = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F1T, last_V_LS_std_D2F1T, last_V_PC_mean_D2F1T, last_V_PC_std_D2F1T, last_V_SC_mean_D2F1T, last_V_SC_std_D2F1T = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AXBO + AOBX
			1-2 ) Basal + target 
W_NAME = 'W408_2'
WORK_NAME = 'WORK_408_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F2BT = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F2BT, epc_V_LS_std_D2F2BT, epc_V_PC_mean_D2F2BT, epc_V_PC_std_D2F2BT, epc_V_SC_mean_D2F2BT, epc_V_SC_std_D2F2BT = get_mean_result(epc_result_D2F2BT)

all_V_LS_mean_D2F2BT, all_V_LS_std_D2F2BT, all_V_PC_mean_D2F2BT, all_V_PC_std_D2F2BT, all_V_SC_mean_D2F2BT, all_V_SC_std_D2F2BT = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F2BT, last_V_LS_std_D2F2BT, last_V_PC_mean_D2F2BT, last_V_PC_std_D2F2BT, last_V_SC_mean_D2F2BT, last_V_SC_std_D2F2BT = get_last_result(ANA_DF, ANA_ALL_DF)









			2) AXBO + AOBX
			1-2 ) EXP + target 
W_NAME = 'W409_2'
WORK_NAME = 'WORK_409_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F2ET = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F2ET, epc_V_LS_std_D2F2ET, epc_V_PC_mean_D2F2ET, epc_V_PC_std_D2F2ET, epc_V_SC_mean_D2F2ET, epc_V_SC_std_D2F2ET = get_mean_result(epc_result_D2F2ET)

all_V_LS_mean_D2F2ET, all_V_LS_std_D2F2ET, all_V_PC_mean_D2F2ET, all_V_PC_std_D2F2ET, all_V_SC_mean_D2F2ET, all_V_SC_std_D2F2ET = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F2ET, last_V_LS_std_D2F2ET, last_V_PC_mean_D2F2ET, last_V_PC_std_D2F2ET, last_V_SC_mean_D2F2ET, last_V_SC_std_D2F2ET = get_last_result(ANA_DF, ANA_ALL_DF)




			2) AXBO + AOBX
			1-2 ) EXP + basal
W_NAME = 'W410_2'
WORK_NAME = 'WORK_410_2' # 349
WORK_DATE = '23.08.29' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D2F2EB = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D2F2EB, epc_V_LS_std_D2F2EB, epc_V_PC_mean_D2F2EB, epc_V_PC_std_D2F2EB, epc_V_SC_mean_D2F2EB, epc_V_SC_std_D2F2EB = get_mean_result(epc_result_D2F2EB)

all_V_LS_mean_D2F2EB, all_V_LS_std_D2F2EB, all_V_PC_mean_D2F2EB, all_V_PC_std_D2F2EB, all_V_SC_mean_D2F2EB, all_V_SC_std_D2F2EB = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D2F2EB, last_V_LS_std_D2F2EB, last_V_PC_mean_D2F2EB, last_V_PC_std_D2F2EB, last_V_SC_mean_D2F2EB, last_V_SC_std_D2F2EB = get_last_result(ANA_DF, ANA_ALL_DF)



#################################
#################################


AXBO 시리즈 



			2) AXBX
			1-1 ) 전체 다 먹는거 
W_NAME = 'W403_3'
WORK_NAME = 'WORK_403_3' # 349
WORK_DATE = '23.08.30' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D3F3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D3F3, epc_V_LS_std_D3F3, epc_V_PC_mean_D3F3, epc_V_PC_std_D3F3, epc_V_SC_mean_D3F3, epc_V_SC_std_D3F3 = get_mean_result(epc_result_D3F3)

all_V_LS_mean_D3F3, all_V_LS_std_D3F3, all_V_PC_mean_D3F3, all_V_PC_std_D3F3, all_V_SC_mean_D3F3, all_V_SC_std_D3F3 = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D3F3, last_V_LS_std_D3F3, last_V_PC_mean_D3F3, last_V_PC_std_D3F3, last_V_SC_mean_D3F3, last_V_SC_std_D3F3 = get_last_result(ANA_DF, ANA_ALL_DF)



			2) AXBX
			1-2 ) gene exp 만 

W_NAME = 'W405_3'
WORK_NAME = 'WORK_405_3' # 349
WORK_DATE = '23.08.30' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D3F1E = get_mean(ANA_DF, ANA_ALL_DF, 1000)
epc_result_D3F1E = epc_result_D3F1E.loc[2:,:]

epc_V_LS_mean_D3F1E, epc_V_LS_std_D3F1E, epc_V_PC_mean_D3F1E, epc_V_PC_std_D3F1E, epc_V_SC_mean_D3F1E, epc_V_SC_std_D3F1E = get_mean_result(epc_result_D3F1E)

all_V_LS_mean_D3F1E, all_V_LS_std_D3F1E, all_V_PC_mean_D3F1E, all_V_PC_std_D3F1E, all_V_SC_mean_D3F1E, all_V_SC_std_D3F1E = get_max_result(ANA_DF, ANA_ALL_DF,2)

last_V_LS_mean_D3F1E, last_V_LS_std_D3F1E, last_V_PC_mean_D3F1E, last_V_PC_std_D3F1E, last_V_SC_mean_D3F1E, last_V_SC_std_D3F1E = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AXBX
			1-2 ) Basal only / RAY_MY_train_ffb94_00000 

W_NAME = 'W406_3'
WORK_NAME = 'WORK_406_3' # 349
WORK_DATE = '23.08.30' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D3F1B = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D3F1B, epc_V_LS_std_D3F1B, epc_V_PC_mean_D3F1B, epc_V_PC_std_D3F1B, epc_V_SC_mean_D3F1B, epc_V_SC_std_D3F1B = get_mean_result(epc_result_D3F1B)

all_V_LS_mean_D3F1B, all_V_LS_std_D3F1B, all_V_PC_mean_D3F1B, all_V_PC_std_D3F1B, all_V_SC_mean_D3F1B, all_V_SC_std_D3F1B = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D3F1B, last_V_LS_std_D3F1B, last_V_PC_mean_D3F1B, last_V_PC_std_D3F1B, last_V_SC_mean_D3F1B, last_V_SC_std_D3F1B = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AXBX
			1-2 ) TARGET only 

W_NAME = 'W407_3'
WORK_NAME = 'WORK_407_3' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D3F1T = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D3F1T, epc_V_LS_std_D3F1T, epc_V_PC_mean_D3F1T, epc_V_PC_std_D3F1T, epc_V_SC_mean_D3F1T, epc_V_SC_std_D3F1T = get_mean_result(epc_result_D3F1T)

all_V_LS_mean_D3F1T, all_V_LS_std_D3F1T, all_V_PC_mean_D3F1T, all_V_PC_std_D3F1T, all_V_SC_mean_D3F1T, all_V_SC_std_D3F1T = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D3F1T, last_V_LS_std_D3F1T, last_V_PC_mean_D3F1T, last_V_PC_std_D3F1T, last_V_SC_mean_D3F1T, last_V_SC_std_D3F1T = get_last_result(ANA_DF, ANA_ALL_DF)






			2) AXBX
			1-2 ) Basal + target 
W_NAME = 'W408_3'
WORK_NAME = 'WORK_408_3' # 349
WORK_DATE = '23.08.30' # 349

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

epc_result_D3F2BT = get_mean(ANA_DF, ANA_ALL_DF, 1000)
epc_result_D3F2BT = epc_result_D3F2BT.loc[3:,:]

epc_V_LS_mean_D3F2BT, epc_V_LS_std_D3F2BT, epc_V_PC_mean_D3F2BT, epc_V_PC_std_D3F2BT, epc_V_SC_mean_D3F2BT, epc_V_SC_std_D3F2BT = get_mean_result(epc_result_D3F2BT)

all_V_LS_mean_D3F2BT, all_V_LS_std_D3F2BT, all_V_PC_mean_D3F2BT, all_V_PC_std_D3F2BT, all_V_SC_mean_D3F2BT, all_V_SC_std_D3F2BT = get_max_result(ANA_DF, ANA_ALL_DF,3)

last_V_LS_mean_D3F2BT, last_V_LS_std_D3F2BT, last_V_PC_mean_D3F2BT, last_V_PC_std_D3F2BT, last_V_SC_mean_D3F2BT, last_V_SC_std_D3F2BT = get_last_result(ANA_DF, ANA_ALL_DF)









			2) AXBX
			1-2 ) EXP + target 
W_NAME = 'W409_3'
WORK_NAME = 'WORK_409_3' # 349
WORK_DATE = '23.08.30' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D3F2ET = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D3F2ET, epc_V_LS_std_D3F2ET, epc_V_PC_mean_D3F2ET, epc_V_PC_std_D3F2ET, epc_V_SC_mean_D3F2ET, epc_V_SC_std_D3F2ET = get_mean_result(epc_result_D3F2ET)

all_V_LS_mean_D3F2ET, all_V_LS_std_D3F2ET, all_V_PC_mean_D3F2ET, all_V_PC_std_D3F2ET, all_V_SC_mean_D3F2ET, all_V_SC_std_D3F2ET = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D3F2ET, last_V_LS_std_D3F2ET, last_V_PC_mean_D3F2ET, last_V_PC_std_D3F2ET, last_V_SC_mean_D3F2ET, last_V_SC_std_D3F2ET = get_last_result(ANA_DF, ANA_ALL_DF)




			2) AXBX
			1-2 ) EXP + basal
W_NAME = 'W410_3'
WORK_NAME = 'WORK_410_3' # 349
WORK_DATE = '23.08.30' # 349
WORK_DATE = '23.09.09' # 349

anal_dir = "/home01/k040a02/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)
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

epc_result_D3F2EB = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_D3F2EB, epc_V_LS_std_D3F2EB, epc_V_PC_mean_D3F2EB, epc_V_PC_std_D3F2EB, epc_V_SC_mean_D3F2EB, epc_V_SC_std_D3F2EB = get_mean_result(epc_result_D3F2EB)

all_V_LS_mean_D3F2EB, all_V_LS_std_D3F2EB, all_V_PC_mean_D3F2EB, all_V_PC_std_D3F2EB, all_V_SC_mean_D3F2EB, all_V_SC_std_D3F2EB = get_max_result(ANA_DF, ANA_ALL_DF)

last_V_LS_mean_D3F2EB, last_V_LS_std_D3F2EB, last_V_PC_mean_D3F2EB, last_V_PC_std_D3F2EB, last_V_SC_mean_D3F2EB, last_V_SC_std_D3F2EB = get_last_result(ANA_DF, ANA_ALL_DF)






#################################
#################################

ablation 

0921 result check 

last_V_LS_mean_D4F3, last_V_PC_mean_D4F3, last_V_SC_mean_D4F3 = 109.1905, 0.7138, 0.6524
last_V_LS_mean_D4F1E,  last_V_PC_mean_D4F1E,  last_V_SC_mean_D4F1E  = 138.2044, 0.6117, 0.5489
last_V_LS_mean_D4F1B,  last_V_PC_mean_D4F1B,  last_V_SC_mean_D4F1B  = 111.5472, 0.7030, 0.6378
last_V_LS_mean_D4F1T,  last_V_PC_mean_D4F1T,  last_V_SC_mean_D4F1T = 167.2891, 0.4888, 0.4261
last_V_LS_mean_D4F2BT, last_V_PC_mean_D4F2BT, last_V_SC_mean_D4F2BT = 108.6491, 0.7124, 0.6498
last_V_LS_mean_D4F2ET, last_V_PC_mean_D4F2ET, last_V_SC_mean_D4F2ET = 134.1778, 0.6261, 0.5596
last_V_LS_mean_D4F2EB, last_V_PC_mean_D4F2EB, last_V_SC_mean_D4F2EB = 108.7104, 0.7143, 0.6564

last_V_LS_mean_D1F3, last_V_PC_mean_D1F3, last_V_SC_mean_D1F3 = 127.8051, 0.6458, 0.6030
last_V_LS_mean_D1F1E,  last_V_PC_mean_D1F1E,  last_V_SC_mean_D1F1E  = 137.5744, 0.6088, 0.5768
last_V_LS_mean_D1F1B,  last_V_PC_mean_D1F1B,  last_V_SC_mean_D1F1B  = 135.2874, 0.6154, 0.5755
last_V_LS_mean_D1F1T,  last_V_PC_mean_D1F1T,  last_V_SC_mean_D1F1T  = 162.9079, 0.5090, 0.4738
last_V_LS_mean_D1F2BT, last_V_PC_mean_D1F2BT, last_V_SC_mean_D1F2BT = 135.5048, 0.6244, 0.5882
last_V_LS_mean_D1F2ET, last_V_PC_mean_D1F2ET, last_V_SC_mean_D1F2ET = 134.3883, 0.6235, 0.5715
last_V_LS_mean_D1F2EB, last_V_PC_mean_D1F2EB, last_V_SC_mean_D1F2EB = 127.1037, 0.6448, 0.6028

last_V_LS_mean_D2F3, last_V_PC_mean_D2F3, last_V_SC_mean_D2F3 = 126.5047, 0.6419, 0.5921
last_V_LS_mean_D2F1E,  last_V_PC_mean_D2F1E,  last_V_SC_mean_D2F1E  = 129.8790, 0.6166, 0.5617
last_V_LS_mean_D2F1B,  last_V_PC_mean_D2F1B,  last_V_SC_mean_D2F1B  = 128.5936, 0.6235, 0.5639
last_V_LS_mean_D2F1T,  last_V_PC_mean_D2F1T,  last_V_SC_mean_D2F1T  = 148.7010, 0.5284, 0.4690
last_V_LS_mean_D2F2BT, last_V_PC_mean_D2F2BT, last_V_SC_mean_D2F2BT = 129.2185, 0.6284, 0.5671
last_V_LS_mean_D2F2ET, last_V_PC_mean_D2F2ET, last_V_SC_mean_D2F2ET = 132.4848, 0.6095, 0.5526
last_V_LS_mean_D2F2EB, last_V_PC_mean_D2F2EB, last_V_SC_mean_D2F2EB = 127.1403, 0.6328, 0.5837

last_V_LS_mean_D3F3, last_V_PC_mean_D3F3, last_V_SC_mean_D3F3 = 111.1163, 0.7123, 0.6532
last_V_LS_mean_D3F1E,  last_V_PC_mean_D3F1E,  last_V_SC_mean_D3F1E  = 139.1226, 0.6123, 0.5470
last_V_LS_mean_D3F1B,  last_V_PC_mean_D3F1B,  last_V_SC_mean_D3F1B  = 111.7072, 0.7072, 0.6473
last_V_LS_mean_D3F1T,  last_V_PC_mean_D3F1T,  last_V_SC_mean_D3F1T  = 171.9117, 0.4824, 0.4218
last_V_LS_mean_D3F2BT, last_V_PC_mean_D3F2BT, last_V_SC_mean_D3F2BT = 110.9695, 0.7096, 0.6524
last_V_LS_mean_D3F2ET, last_V_PC_mean_D3F2ET, last_V_SC_mean_D3F2ET = 137.8803, 0.6177, 0.5530
last_V_LS_mean_D3F2EB, last_V_PC_mean_D3F2EB, last_V_SC_mean_D3F2EB = 109.7364, 0.7133, 0.6534

















# ALL
epc_D4 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_D4F3, epc_V_PC_mean_D4F3, epc_V_SC_mean_D4F3], 
	'LINCS+BASAL' :  [epc_V_LS_mean_D4F2EB, epc_V_PC_mean_D4F2EB, epc_V_SC_mean_D4F2EB], 
	'LINCS+TARGET' :  [epc_V_LS_mean_D4F2ET, epc_V_PC_mean_D4F2ET, epc_V_SC_mean_D4F2ET], 
	'BASAL+TARGET' :  [epc_V_LS_mean_D4F2BT, epc_V_PC_mean_D4F2BT, epc_V_SC_mean_D4F2BT], 
	'LINCS' :  [epc_V_LS_mean_D4F1E, epc_V_PC_mean_D4F1E, epc_V_SC_mean_D4F1E], 
	'BASAL' :  [epc_V_LS_mean_D4F1B, epc_V_PC_mean_D4F1B, epc_V_SC_mean_D4F1B], 
	'TARGET' : [epc_V_LS_mean_D4F1T, epc_V_PC_mean_D4F1T, epc_V_SC_mean_D4F1T]
	})

all_D4 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [all_V_LS_mean_D4F3, all_V_PC_mean_D4F3, all_V_SC_mean_D4F3], 
	'LINCS+BASAL' :  [all_V_LS_mean_D4F2EB, all_V_PC_mean_D4F2EB, all_V_SC_mean_D4F2EB], 
	'LINCS+TARGET' :  [all_V_LS_mean_D4F2ET, all_V_PC_mean_D4F2ET, all_V_SC_mean_D4F2ET], 
	'BASAL+TARGET' :  [all_V_LS_mean_D4F2BT, all_V_PC_mean_D4F2BT, all_V_SC_mean_D4F2BT], 
	'LINCS' :  [all_V_LS_mean_D4F1E, all_V_PC_mean_D4F1E, all_V_SC_mean_D4F1E], 
	'BASAL' :  [all_V_LS_mean_D4F1B, all_V_PC_mean_D4F1B, all_V_SC_mean_D4F1B], 
	'TARGET' : [all_V_LS_mean_D4F1T, all_V_PC_mean_D4F1T, all_V_SC_mean_D4F1T]
	})

last_D4 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [last_V_LS_mean_D4F3, last_V_PC_mean_D4F3, last_V_SC_mean_D4F3], 
	'LINCS+BASAL' :  [last_V_LS_mean_D4F2EB, last_V_PC_mean_D4F2EB, last_V_SC_mean_D4F2EB], 
	'LINCS+TARGET' :  [last_V_LS_mean_D4F2ET, last_V_PC_mean_D4F2ET, last_V_SC_mean_D4F2ET], 
	'BASAL+TARGET' :  [last_V_LS_mean_D4F2BT, last_V_PC_mean_D4F2BT, last_V_SC_mean_D4F2BT], 
	'LINCS' :  [last_V_LS_mean_D4F1E, last_V_PC_mean_D4F1E, last_V_SC_mean_D4F1E], 
	'BASAL' :  [last_V_LS_mean_D4F1B, last_V_PC_mean_D4F1B, last_V_SC_mean_D4F1B], 
	'TARGET' : [last_V_LS_mean_D4F1T, last_V_PC_mean_D4F1T, last_V_SC_mean_D4F1T]
	})


ab_4 = last_D4

ab_4_T = ab_4.T

ab_4_re = pd.DataFrame(pd.concat([ab_4_T.iloc[:,0], ab_4_T.iloc[:,1], ab_4_T.iloc[:,2]]))

ab_4_re.columns = ['value']
ab_4_re['ablation'] = list(ab_4_re.index)
ab_4_re['Data ablation'] = 'Data All'
ab_4_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7




# AOBO

epc_D1 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_D1F3, epc_V_PC_mean_D1F3, epc_V_SC_mean_D1F3], 
	'LINCS+BASAL' :  [epc_V_LS_mean_D1F2EB, epc_V_PC_mean_D1F2EB, epc_V_SC_mean_D1F2EB], 
	'LINCS+TARGET' :  [epc_V_LS_mean_D1F2ET, epc_V_PC_mean_D1F2ET, epc_V_SC_mean_D1F2ET], 
	'BASAL+TARGET' :  [epc_V_LS_mean_D1F2BT, epc_V_PC_mean_D1F2BT, epc_V_SC_mean_D1F2BT], 
	'LINCS' :  [epc_V_LS_mean_D1F1E, epc_V_PC_mean_D1F1E, epc_V_SC_mean_D1F1E], 
	'BASAL' :  [epc_V_LS_mean_D1F1B, epc_V_PC_mean_D1F1B, epc_V_SC_mean_D1F1B], 
	'TARGET' : [epc_V_LS_mean_D1F1T, epc_V_PC_mean_D1F1T, epc_V_SC_mean_D1F1T]
	})

all_D1 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [all_V_LS_mean_D1F3, all_V_PC_mean_D1F3, all_V_SC_mean_D1F3], 
	'LINCS+BASAL' :  [all_V_LS_mean_D1F2EB, all_V_PC_mean_D1F2EB, all_V_SC_mean_D1F2EB], 
	'LINCS+TARGET' :  [all_V_LS_mean_D1F2ET, all_V_PC_mean_D1F2ET, all_V_SC_mean_D1F2ET], 
	'BASAL+TARGET' :  [all_V_LS_mean_D1F2BT, all_V_PC_mean_D1F2BT, all_V_SC_mean_D1F2BT], 
	'LINCS' :  [all_V_LS_mean_D1F1E, all_V_PC_mean_D1F1E, all_V_SC_mean_D1F1E], 
	'BASAL' :  [all_V_LS_mean_D1F1B, all_V_PC_mean_D1F1B, all_V_SC_mean_D1F1B], 
	'TARGET' : [all_V_LS_mean_D1F1T, all_V_PC_mean_D1F1T, all_V_SC_mean_D1F1T]
	})

last_D1 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [last_V_LS_mean_D1F3, last_V_PC_mean_D1F3, last_V_SC_mean_D1F3], 
	'LINCS+BASAL' :  [last_V_LS_mean_D1F2EB, last_V_PC_mean_D1F2EB, last_V_SC_mean_D1F2EB], 
	'LINCS+TARGET' :  [last_V_LS_mean_D1F2ET, last_V_PC_mean_D1F2ET, last_V_SC_mean_D1F2ET], 
	'BASAL+TARGET' :  [last_V_LS_mean_D1F2BT, last_V_PC_mean_D1F2BT, last_V_SC_mean_D1F2BT], 
	'LINCS' :  [last_V_LS_mean_D1F1E, last_V_PC_mean_D1F1E, last_V_SC_mean_D1F1E], 
	'BASAL' :  [last_V_LS_mean_D1F1B, last_V_PC_mean_D1F1B, last_V_SC_mean_D1F1B], 
	'TARGET' : [last_V_LS_mean_D1F1T, last_V_PC_mean_D1F1T, last_V_SC_mean_D1F1T]
	})


ab_1 = last_D1

ab_1_T = ab_1.T

ab_1_re = pd.DataFrame(pd.concat([ab_1_T.iloc[:,0], ab_1_T.iloc[:,1], ab_1_T.iloc[:,2]]))

ab_1_re.columns = ['value']
ab_1_re['ablation'] = list(ab_1_re.index)
ab_1_re['Data ablation'] = 'LINCS all matched'
ab_1_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7








# AXBO
epc_D2 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_D2F3, epc_V_PC_mean_D2F3, epc_V_SC_mean_D2F3], 
	'LINCS+BASAL' :  [epc_V_LS_mean_D2F2EB, epc_V_PC_mean_D2F2EB, epc_V_SC_mean_D2F2EB], 
	'LINCS+TARGET' :  [epc_V_LS_mean_D2F2ET, epc_V_PC_mean_D2F2ET, epc_V_SC_mean_D2F2ET], 
	'BASAL+TARGET' :  [epc_V_LS_mean_D2F2BT, epc_V_PC_mean_D2F2BT, epc_V_SC_mean_D2F2BT], 
	'LINCS' :  [epc_V_LS_mean_D2F1E, epc_V_PC_mean_D2F1E, epc_V_SC_mean_D2F1E], 
	'BASAL' :  [epc_V_LS_mean_D2F1B, epc_V_PC_mean_D2F1B, epc_V_SC_mean_D2F1B], 
	'TARGET' : [epc_V_LS_mean_D2F1T, epc_V_PC_mean_D2F1T, epc_V_SC_mean_D2F1T]
	})

all_D2 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [all_V_LS_mean_D2F3, all_V_PC_mean_D2F3, all_V_SC_mean_D2F3], 
	'LINCS+BASAL' :  [all_V_LS_mean_D2F2EB, all_V_PC_mean_D2F2EB, all_V_SC_mean_D2F2EB], 
	'LINCS+TARGET' :  [all_V_LS_mean_D2F2ET, all_V_PC_mean_D2F2ET, all_V_SC_mean_D2F2ET], 
	'BASAL+TARGET' :  [all_V_LS_mean_D2F2BT, all_V_PC_mean_D2F2BT, all_V_SC_mean_D2F2BT], 
	'LINCS' :  [all_V_LS_mean_D2F1E, all_V_PC_mean_D2F1E, all_V_SC_mean_D2F1E], 
	'BASAL' :  [all_V_LS_mean_D2F1B, all_V_PC_mean_D2F1B, all_V_SC_mean_D2F1B], 
	'TARGET' : [all_V_LS_mean_D2F1T, all_V_PC_mean_D2F1T, all_V_SC_mean_D2F1T]
	})

last_D2 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [last_V_LS_mean_D2F3, last_V_PC_mean_D2F3, last_V_SC_mean_D2F3], 
	'LINCS+BASAL' :  [last_V_LS_mean_D2F2EB, last_V_PC_mean_D2F2EB, last_V_SC_mean_D2F2EB], 
	'LINCS+TARGET' :  [last_V_LS_mean_D2F2ET, last_V_PC_mean_D2F2ET, last_V_SC_mean_D2F2ET], 
	'BASAL+TARGET' :  [last_V_LS_mean_D2F2BT, last_V_PC_mean_D2F2BT, last_V_SC_mean_D2F2BT], 
	'LINCS' :  [last_V_LS_mean_D2F1E, last_V_PC_mean_D2F1E, last_V_SC_mean_D2F1E], 
	'BASAL' :  [last_V_LS_mean_D2F1B, last_V_PC_mean_D2F1B, last_V_SC_mean_D2F1B], 
	'TARGET' : [last_V_LS_mean_D2F1T, last_V_PC_mean_D2F1T, last_V_SC_mean_D2F1T]
	})


ab_2 = last_D2

ab_2_T = ab_2.T

ab_2_re = pd.DataFrame(pd.concat([ab_2_T.iloc[:,0], ab_2_T.iloc[:,1], ab_2_T.iloc[:,2]]))

ab_2_re.columns = ['value']
ab_2_re['ablation'] = list(ab_2_re.index)
ab_2_re['Data ablation'] = 'LINCS one matched'
ab_2_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7


# AXBX
epc_D3 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_D3F3, epc_V_PC_mean_D3F3, epc_V_SC_mean_D3F3], 
	'LINCS+BASAL' :  [epc_V_LS_mean_D3F2EB, epc_V_PC_mean_D3F2EB, epc_V_SC_mean_D3F2EB], 
	'LINCS+TARGET' :  [epc_V_LS_mean_D3F2ET, epc_V_PC_mean_D3F2ET, epc_V_SC_mean_D3F2ET], 
	'BASAL+TARGET' :  [epc_V_LS_mean_D3F2BT, epc_V_PC_mean_D3F2BT, epc_V_SC_mean_D3F2BT], 
	'LINCS' :  [epc_V_LS_mean_D3F1E, epc_V_PC_mean_D3F1E, epc_V_SC_mean_D3F1E], 
	'BASAL' :  [epc_V_LS_mean_D3F1B, epc_V_PC_mean_D3F1B, epc_V_SC_mean_D3F1B], 
	'TARGET' : [epc_V_LS_mean_D3F1T, epc_V_PC_mean_D3F1T, epc_V_SC_mean_D3F1T]
	})

all_D3 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [all_V_LS_mean_D3F3, all_V_PC_mean_D3F3, all_V_SC_mean_D3F3], 
	'LINCS+BASAL' :  [all_V_LS_mean_D3F2EB, all_V_PC_mean_D3F2EB, all_V_SC_mean_D3F2EB], 
	'LINCS+TARGET' :  [all_V_LS_mean_D3F2ET, all_V_PC_mean_D3F2ET, all_V_SC_mean_D3F2ET], 
	'BASAL+TARGET' :  [all_V_LS_mean_D3F2BT, all_V_PC_mean_D3F2BT, all_V_SC_mean_D3F2BT], 
	'LINCS' :  [all_V_LS_mean_D3F1E, all_V_PC_mean_D3F1E, all_V_SC_mean_D3F1E], 
	'BASAL' :  [all_V_LS_mean_D3F1B, all_V_PC_mean_D3F1B, all_V_SC_mean_D3F1B], 
	'TARGET' : [all_V_LS_mean_D3F1T, all_V_PC_mean_D3F1T, all_V_SC_mean_D3F1T]
	})

last_D3 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [last_V_LS_mean_D3F3, last_V_PC_mean_D3F3, last_V_SC_mean_D3F3], 
	'LINCS+BASAL' :  [last_V_LS_mean_D3F2EB, last_V_PC_mean_D3F2EB, last_V_SC_mean_D3F2EB], 
	'LINCS+TARGET' :  [last_V_LS_mean_D3F2ET, last_V_PC_mean_D3F2ET, last_V_SC_mean_D3F2ET], 
	'BASAL+TARGET' :  [last_V_LS_mean_D3F2BT, last_V_PC_mean_D3F2BT, last_V_SC_mean_D3F2BT], 
	'LINCS' :  [last_V_LS_mean_D3F1E, last_V_PC_mean_D3F1E, last_V_SC_mean_D3F1E], 
	'BASAL' :  [last_V_LS_mean_D3F1B, last_V_PC_mean_D3F1B, last_V_SC_mean_D3F1B], 
	'TARGET' : [last_V_LS_mean_D3F1T, last_V_PC_mean_D3F1T, last_V_SC_mean_D3F1T]
	})



ab_3 = last_D3

ab_3_T = ab_3.T

ab_3_re = pd.DataFrame(pd.concat([ab_3_T.iloc[:,0], ab_3_T.iloc[:,1], ab_3_T.iloc[:,2]]))

ab_3_re.columns = ['value']
ab_3_re['ablation'] = list(ab_3_re.index)
ab_3_re['Data ablation'] = 'LINCS none matched'
ab_3_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7





ABLATION_DF = pd.concat([ab_4_re, ab_1_re, ab_2_re, ab_3_re])


ABLATION_DF_loss = ABLATION_DF[ABLATION_DF.check=='LOSS']
ABLATION_DF_PCOR = ABLATION_DF[ABLATION_DF.check=='PCOR']
ABLATION_DF_SCOR = ABLATION_DF[ABLATION_DF.check=='SCOR']


ablation_col = {
	'LINCS+BASAL+TARGET' : '#E52B50', 'LINCS+BASAL' : '#FFE344', 'LINCS+TARGET' : '#FFA153', 
	'BASAL+TARGET': '#1ADDFF', 'LINCS': '#FF6987', 'BASAL': '#8AC8FF' , 'TARGET': '#09E07B' }

fig, ax = plt.subplots(1,3,figsize=(25, 5))
sns.barplot(ax = ax[0], data  = ABLATION_DF_loss, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_0 = ax[0]
ax_0.set_xlabel('ablation', fontsize=10)
ax_0.set_ylabel('MSE loss', fontsize=10)
ax_0.set_xticks(ax_0.get_xticks())
ax_0.set_xticklabels(ax_0.get_xticklabels(), rotation = 90)
ax_0.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[1], data  = ABLATION_DF_PCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_1 = ax[1]
ax_1.set_xlabel('ablation', fontsize=10)
ax_1.set_ylabel('PCOR', fontsize=10)
ax_1.set_xticks(ax_1.get_xticks())
ax_1.set_xticklabels(ax_1.get_xticklabels(), rotation = 90)
ax_1.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[2], data  = ABLATION_DF_SCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_2 = ax[2]
ax_2.set_xlabel('ablation', fontsize=10)
ax_2.set_ylabel('SCOR', fontsize=10)
ax_2.set_xticks(ax_2.get_xticks())
ax_2.set_xticklabels(ax_2.get_xticklabels(), rotation = 90)
ax_2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.grid(False)
plt.savefig(os.path.join('ablation1.png'), dpi = 300)
plt.savefig('ablation1.pdf', format="pdf", bbox_inches = 'tight')
plt.close()






아좀 맘에 안듬 


ABLATION_DF = pd.concat([ab_4_re, ab_1_re, ab_2_re, ab_3_re])

ABLATION_DF_data_matched = ABLATION_DF[ABLATION_DF['Data ablation']=='LINCS all matched']
ABLATION_DF_data_onematched = ABLATION_DF[ABLATION_DF['Data ablation']=='LINCS one matched'] # half 써줘야하나 
ABLATION_DF_data_unmatched = ABLATION_DF[ABLATION_DF['Data ablation']=='LINCS none matched']
ABLATION_DF_data_all = ABLATION_DF[ABLATION_DF['Data ablation']=='Data All']


ablation_col = {
	'LINCS+BASAL+TARGET' : '#E52B50', 'LINCS+BASAL' : '#FFE344', 'LINCS+TARGET' : '#FFA153', 
	'BASAL+TARGET': '#1ADDFF', 'LINCS': '#FF6987', 'BASAL': '#8AC8FF' , 'TARGET': '#09E07B' }





fig, ax = plt.subplots(1,6,figsize=(25, 5))
sns.barplot(ax = ax[0], data  = ABLATION_DF_data_matched[ABLATION_DF_data_matched.check.isin(['LOSS'])], x = 'check', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_0 = ax[0]
ax_0.set_xlabel('method', fontsize=10)
ax_0.set_ylabel('MSE Loss', fontsize=10)
ax_0.set_xticks(ax_0.get_xticks())
ax_0.set_xticklabels(ax_0.get_xticklabels(), rotation = 90)
ax_0.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[1], data  = ABLATION_DF_data_matched[ABLATION_DF_data_matched.check.isin(['PCOR', 'SCOR'])], x = 'check', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_1 = ax[1]
ax_1.set_xlabel('method', fontsize=10)
ax_1.set_ylabel('Correlation', fontsize=10)
ax_1.set_xticks(ax_1.get_xticks())
ax_1.set_xticklabels(ax_1.get_xticklabels(), rotation = 90)
ax_1.legend().set_visible(False)
sns.despine()



sns.barplot(ax = ax[2], data  = ABLATION_DF_data_unmatched[ABLATION_DF_data_unmatched.check.isin(['LOSS'])], x = 'check', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_10 = ax[2]
ax_10.set_xlabel('method', fontsize=10)
ax_10.set_ylabel('MSE Loss', fontsize=10)
ax_10.set_xticks(ax_10.get_xticks())
ax_10.set_xticklabels(ax_10.get_xticklabels(), rotation = 90)
ax_10.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[3], data  = ABLATION_DF_data_unmatched[ABLATION_DF_data_unmatched.check.isin(['PCOR', 'SCOR'])], x = 'check', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_11 = ax[3]
ax_11.set_xlabel('method', fontsize=10)
ax_11.set_ylabel('Correlation', fontsize=10)
ax_11.set_xticks(ax_11.get_xticks())
ax_11.set_xticklabels(ax_11.get_xticklabels(), rotation = 90)
ax_11.legend().set_visible(False)
sns.despine()


sns.barplot(ax = ax[4], data  = ABLATION_DF_data_all[ABLATION_DF_data_all.check.isin(['LOSS'])], x = 'check', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_20 = ax[4]
ax_20.set_xlabel('method', fontsize=10)
ax_20.set_ylabel('MSE Loss', fontsize=10)
ax_20.set_xticks(ax_20.get_xticks())
ax_20.set_xticklabels(ax_20.get_xticklabels(), rotation = 90)
ax_20.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[5], data  = ABLATION_DF_data_all[ABLATION_DF_data_all.check.isin(['PCOR', 'SCOR'])], x = 'check', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_21 = ax[5]
ax_21.set_xlabel('method', fontsize=10)
ax_21.set_ylabel('Correlation', fontsize=10)
ax_21.set_xticks(ax_21.get_xticks())
ax_21.set_xticklabels(ax_21.get_xticklabels(), rotation = 90)
ax_21.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.grid(False)
plt.savefig(os.path.join('ablation2.png'), dpi = 300)
plt.savefig('ablation2.pdf', format="pdf", bbox_inches = 'tight')
plt.close()






더 마음에 안듬 





fig, ax = plt.subplots(1,3,figsize=(25, 5))
sns.barplot(ax = ax[0], data  = ABLATION_DF_loss, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_0 = ax[0]
ax_0.set_xlabel('', fontsize=10)
ax_0.set_ylabel('MSE loss', fontsize=10)
ax_0.set_xticks(ax_0.get_xticks())
ax_0.set_xticklabels(ax_0.get_xticklabels(), fontsize=15) # , rotation = 90
ax_0.set_ylim(50, 175)
ax_0.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[1], data  = ABLATION_DF_PCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_1 = ax[1]
ax_1.set_xlabel('', fontsize=10)
ax_1.set_ylabel('PCOR', fontsize=10)
ax_1.set_xticks(ax_1.get_xticks())
ax_1.set_xticklabels(ax_1.get_xticklabels(), fontsize=15)
ax_1.set_ylim(0.4, 0.8)
ax_1.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[2], data  = ABLATION_DF_SCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_2 = ax[2]
ax_2.set_xlabel('', fontsize=10)
ax_2.set_ylabel('SCOR', fontsize=10)
ax_2.set_xticks(ax_2.get_xticks())
ax_2.set_xticklabels(ax_2.get_xticklabels(), fontsize=15)
ax_2.set_ylim(0.3, 0.7)
ax_2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.grid(False)
plt.savefig(os.path.join('/home01/k040a01/ablation3.png'), dpi = 300)
plt.close()





final version 


fig, ax = plt.subplots(1,3,figsize=(25, 5))
sns.barplot(ax = ax[0], data  = ABLATION_DF_loss, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_0 = ax[0]
ax_0.set_xlabel('', fontsize=10)
ax_0.set_ylabel('MSE loss', fontsize=10)
ax_0.set_xticks(ax_0.get_xticks())
ax_0.set_xticklabels(ax_0.get_xticklabels(), fontsize=15) # , rotation = 90
ax_0.set_ylim(50, 175)
ax_0.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[1], data  = ABLATION_DF_PCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_1 = ax[1]
ax_1.set_xlabel('', fontsize=10)
ax_1.set_ylabel('PCOR', fontsize=10)
ax_1.set_xticks(ax_1.get_xticks())
ax_1.set_xticklabels(ax_1.get_xticklabels(), fontsize=15)
ax_1.set_ylim(0.4, 0.8)
ax_1.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[2], data  = ABLATION_DF_SCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_2 = ax[2]
ax_2.set_xlabel('', fontsize=10)
ax_2.set_ylabel('SCOR', fontsize=10)
ax_2.set_xticks(ax_2.get_xticks())
ax_2.set_xticklabels(ax_2.get_xticklabels(), fontsize=15)
ax_2.set_ylim(0.3, 0.7)
ax_2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.grid(False)
#plt.savefig(os.path.join('/home01/k040a01/ablation3.png'), dpi = 300)
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/ablation.0921.png'), dpi = 300)

#plt.savefig('/home01/k040a01/ablation3.pdf', format="pdf", bbox_inches = 'tight')
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/ablation.0921.pdf', format="pdf", bbox_inches = 'tight')

plt.close()








교수님이 주문한 MSE 최대에서 빼는 버전  

ABLATION_DF_loss['m_value'] = 200-ABLATION_DF_loss.value

fig, ax = plt.subplots(1,3,figsize=(25, 5))
sns.barplot(ax = ax[0], data  = ABLATION_DF_loss, x = 'Data ablation', y = 'm_value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_0 = ax[0]
ax_0.set_xlabel('', fontsize=10)
ax_0.set_ylabel('MSE loss', fontsize=10)
ax_0.set_xticks(ax_0.get_xticks())
ax_0.set_xticklabels(ax_0.get_xticklabels(), fontsize=15) # , rotation = 90
ax_0.set_ylim(20, 100)
ax_0.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[1], data  = ABLATION_DF_PCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_1 = ax[1]
ax_1.set_xlabel('', fontsize=10)
ax_1.set_ylabel('PCOR', fontsize=10)
ax_1.set_xticks(ax_1.get_xticks())
ax_1.set_xticklabels(ax_1.get_xticklabels(), fontsize=15)
ax_1.set_ylim(0.4, 0.8)
ax_1.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[2], data  = ABLATION_DF_SCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_2 = ax[2]
ax_2.set_xlabel('', fontsize=10)
ax_2.set_ylabel('SCOR', fontsize=10)
ax_2.set_xticks(ax_2.get_xticks())
ax_2.set_xticklabels(ax_2.get_xticklabels(), fontsize=15)
ax_2.set_ylim(0.3, 0.7)
ax_2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.grid(False)
#plt.savefig(os.path.join('/home01/k040a01/ablation3.png'), dpi = 300)
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/ablation.0927.png'), dpi = 300)

#plt.savefig('/home01/k040a01/ablation3.pdf', format="pdf", bbox_inches = 'tight')
plt.savefig('/st06/jiyeonH/11.TOX/DR_SPRING/trials/ablation.0927.pdf', format="pdf", bbox_inches = 'tight')

plt.close()



round(last_D4, 4)
round(last_D1, 4)
round(last_D2, 4)
round(last_D3, 4)



round(last_std_D4, 4)
round(last_std_D1, 4)
round(last_std_D2, 4)
round(last_std_D3, 4)


















# box plot 그리는거. CV 따라서 그리면 어떨지? 
# 근데 이거 그리면 따로 다시 표를 안나타내도 되나? 
# 그림이 별로면 도루묵일듯 
# 


# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcD46' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
ax, test_results = add_stat_annotation(ax = ax, data=gdsc_all, x='CT', y='PRED',
								   box_pairs=[("LARGE_INTESTINE_O", "LARGE_INTESTINE_X"),  ("BREAST_O", "BREAST_X"), ("PANCREAS_O", "PANCREAS_X")],
								   test='Mann-Whitney', text_format='star', loc='outside', verbose=2)


sns.despine()
ax.set_xlabel('tissue, OX', fontsize=10)
ax.set_ylabel('Predicted value', fontsize=10)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=10 )
plt.legend(bbox_to_anchor=(1.02, 0.15), loc='upper left', borderaxespad=0)
#plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/heatmap/gdsc.box.png'), dpi = 300)

plt.close()
























