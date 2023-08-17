



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



def get_mean_result(epc_result) :
	#
	#1) min loss
	#
	min(epc_result.sort_values('V_LS_mean')['V_LS_mean']) ; min_VLS = min(epc_result.sort_values('V_LS_mean')['V_LS_mean'])
	KEY_EPC = epc_result[epc_result.V_LS_mean == min_VLS].index.item()
	epc_V_LS_mean = round(epc_result.loc[KEY_EPC].V_LS_mean, 4)
	epc_V_LS_std = round(epc_result.loc[KEY_EPC].V_LS_std, 4)
	#
	#2) PC best 
	#
	epc_result.sort_values('V_PC_mean', ascending = False) 
	max(epc_result['V_PC_mean']); max_VPC = max(epc_result['V_PC_mean'])
	KEY_EPC = epc_result[epc_result.V_PC_mean == max_VPC].index.item()
	epc_V_PC_mean = round(epc_result.loc[KEY_EPC].V_PC_mean, 4)
	epc_V_PC_std = round(epc_result.loc[KEY_EPC].V_PC_std, 4)
	#
	#
	#3) SC best 
	#
	epc_result.sort_values('V_SC_mean', ascending = False) 
	max(epc_result['V_SC_mean']); max_VSC = max(epc_result['V_SC_mean'])
	KEY_EPC = epc_result[epc_result.V_SC_mean == max_VSC].index.item()
	epc_V_SC_mean = round(epc_result.loc[KEY_EPC].V_SC_mean, 4)
	epc_V_SC_std = round(epc_result.loc[KEY_EPC].V_SC_std, 4)
	#
	print([epc_V_LS_mean, epc_V_LS_std])
	print([epc_V_PC_mean, epc_V_PC_std])
	print([epc_V_SC_mean, epc_V_SC_std])
	return(epc_V_LS_mean, epc_V_LS_std, epc_V_PC_mean, epc_V_PC_std, epc_V_SC_mean, epc_V_SC_std)



def get_max_result(ANA_DF, ANA_ALL_DF) :
	#
	cv_keys = list(ANA_DF['logdir'])
	#
	epc_T_LS_mean = np.mean([min(ANA_ALL_DF[kk]['T_LS']) for kk in cv_keys])
	epc_T_LS_std = np.std([min(ANA_ALL_DF[kk]['T_LS']) for kk in cv_keys])
	#
	epc_T_PC_mean = np.mean([max(ANA_ALL_DF[kk]['T_PC']) for kk in cv_keys])
	epc_T_PC_std = np.std([max(ANA_ALL_DF[kk]['T_PC']) for kk in cv_keys])
	#
	epc_T_SC_mean = np.mean([max(ANA_ALL_DF[kk]['T_SC']) for kk in cv_keys])
	epc_T_SC_std = np.std([max(ANA_ALL_DF[kk]['T_SC']) for kk in cv_keys])
	#
	epc_V_LS_mean = np.mean([min(ANA_ALL_DF[kk]['V_LS']) for kk in cv_keys])
	epc_V_LS_std = np.std([min(ANA_ALL_DF[kk]['V_LS']) for kk in cv_keys])
	#
	epc_V_PC_mean = np.mean([max(ANA_ALL_DF[kk]['V_PC']) for kk in cv_keys])
	epc_V_PC_std = np.std([max(ANA_ALL_DF[kk]['V_PC']) for kk in cv_keys])
	#
	epc_V_SC_mean = np.mean([max(ANA_ALL_DF[kk]['V_SC']) for kk in cv_keys])
	epc_V_SC_std = np.std([max(ANA_ALL_DF[kk]['V_SC']) for kk in cv_keys])
	#
	print(np.round([epc_T_LS_mean, epc_T_LS_std], 4))
	print(np.round([epc_T_PC_mean, epc_T_PC_std], 4))
	print(np.round([epc_T_SC_mean, epc_T_SC_std], 4))
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


MJ_NAME = 'M3V6'
PPI_NAME = '349'
MISS_NAME = 'MIS2'


			1) AOBO + AOBX + AXBX
			1-1 ) 전체 다 먹는거 
W_NAME = 'W203'
WORK_NAME = 'WORK_203' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W203 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W203, epc_V_LS_std_W203, epc_V_PC_mean_W203, epc_V_PC_std_W203, epc_V_SC_mean_W203, epc_V_SC_std_W203 = get_mean_result(epc_result_W203)

all_V_LS_mean_W203, all_V_LS_std_W203, all_V_PC_mean_W203, all_V_PC_std_W203, all_V_SC_mean_W203, all_V_SC_std_W203 = get_max_result(ANA_DF, ANA_ALL_DF)



			1) AOBO + AOBX + AXBX
			1-2 ) gene exp 만 먹인거 206_5인데 잘못 표기함 /home01/k040a01/ray_results/PRJ02.23.06.22.M3V6.WORK_206_1.349.MIS2 / RAY_MY_train_c6697
W_NAME = 'W206'
WORK_NAME = 'WORK_206_1' # 349
# WORK_NAME = 'WORK_206_5' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W206_5 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W206_5, epc_V_LS_std_W206_5, epc_V_PC_mean_W206_5, epc_V_PC_std_W206_5, epc_V_SC_mean_W206_5, epc_V_SC_std_W206_5 = get_mean_result(epc_result_W206_5)

all_V_LS_mean_W206_5, all_V_LS_std_W206_5, all_V_PC_mean_W206_5, all_V_PC_std_W206_5, all_V_SC_mean_W206_5, all_V_SC_std_W206_5 = get_max_result(ANA_DF, ANA_ALL_DF)









			1) AOBO + AOBX + AXBX
			1-2 ) Basal only
W_NAME = 'W207'
WORK_NAME = 'WORK_207_5' # 349
# WORK_NAME = 'WORK_206_5' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W207_5 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W207_5, epc_V_LS_std_W207_5, epc_V_PC_mean_W207_5, epc_V_PC_std_W207_5, epc_V_SC_mean_W207_5, epc_V_SC_std_W207_5 = get_mean_result(epc_result_W207_5)

all_V_LS_mean_W207_5, all_V_LS_std_W207_5, all_V_PC_mean_W207_5, all_V_PC_std_W207_5, all_V_SC_mean_W207_5, all_V_SC_std_W207_5 = get_max_result(ANA_DF, ANA_ALL_DF)







			1) AOBO + AOBX + AXBX
			1-2 ) Basal + target 
W_NAME = 'W208'
WORK_NAME = 'WORK_208_5' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W208_5 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W208_5, epc_V_LS_std_W208_5, epc_V_PC_mean_W208_5, epc_V_PC_std_W208_5, epc_V_SC_mean_W208_5, epc_V_SC_std_W208_5 = get_mean_result(epc_result_W208_5)

all_V_LS_mean_W208_5, all_V_LS_std_W208_5, all_V_PC_mean_W208_5, all_V_PC_std_W208_5, all_V_SC_mean_W208_5, all_V_SC_std_W208_5 = get_max_result(ANA_DF, ANA_ALL_DF)










			1) AOBO + AOBX + AXBX
			1-2 ) EXP + target 
W_NAME = 'W209'
WORK_NAME = 'WORK_209_5' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W209_5 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W209_5, epc_V_LS_std_W209_5, epc_V_PC_mean_W209_5, epc_V_PC_std_W209_5, epc_V_SC_mean_W209_5, epc_V_SC_std_W209_5 = get_mean_result(epc_result_W209_5)

all_V_LS_mean_W209_5, all_V_LS_std_W209_5, all_V_PC_mean_W209_5, all_V_PC_std_W209_5, all_V_SC_mean_W209_5, all_V_SC_std_W209_5 = get_max_result(ANA_DF, ANA_ALL_DF)











			1) AOBO + AOBX + AXBX
			1-2 ) EXP + basal
W_NAME = 'W210'
WORK_NAME = 'WORK_210_5' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W210_5 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W210_5, epc_V_LS_std_W210_5, epc_V_PC_mean_W210_5, epc_V_PC_std_W210_5, epc_V_SC_mean_W210_5, epc_V_SC_std_W210_5 = get_mean_result(epc_result_W210_5)

all_V_LS_mean_W210_5, all_V_LS_std_W210_5, all_V_PC_mean_W210_5, all_V_PC_std_W210_5, all_V_SC_mean_W210_5, all_V_SC_std_W210_5 = get_max_result(ANA_DF, ANA_ALL_DF)





########################################################
########################################################
########################################################
########################################################
########################################################


			2) AOBO
			1-1 ) 전체 다 먹는거 
W_NAME = 'W203'
WORK_NAME = 'WORK_203_1' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W203_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W203_1, epc_V_LS_std_W203_1, epc_V_PC_mean_W203_1, epc_V_PC_std_W203_1, epc_V_SC_mean_W203_1, epc_V_SC_std_W203_1 = get_mean_result(epc_result_W203_1)

all_V_LS_mean_W203_1, all_V_LS_std_W203_1, all_V_PC_mean_W203_1, all_V_PC_std_W203_1, all_V_SC_mean_W203_1, all_V_SC_std_W203_1 = get_max_result(ANA_DF, ANA_ALL_DF)



			2) AOBO
			1-2 ) gene exp 만 먹인거 206_5인데 잘못 표기함 /home01/k040a01/ray_results/PRJ02.23.06.23.M3V6.WORK_206_1.349.MIS2 / RAY_MY_train_28371_00000

W_NAME = 'W206'
WORK_NAME = 'WORK_206_1' # 349
# WORK_NAME = 'WORK_206_5' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W206_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W206_1, epc_V_LS_std_W206_1, epc_V_PC_mean_W206_1, epc_V_PC_std_W206_1, epc_V_SC_mean_W206_1, epc_V_SC_std_W206_1 = get_mean_result(epc_result_W206_1)

all_V_LS_mean_W206_1, all_V_LS_std_W206_1, all_V_PC_mean_W206_1, all_V_PC_std_W206_1, all_V_SC_mean_W206_1, all_V_SC_std_W206_1 = get_max_result(ANA_DF, ANA_ALL_DF)









			2) AOBO
			1-2 ) Basal only / RAY_MY_train_ffb94_00000 
W_NAME = 'W207'
WORK_NAME = 'WORK_207_1' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W207_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W207_1, epc_V_LS_std_W207_1, epc_V_PC_mean_W207_1, epc_V_PC_std_W207_1, epc_V_SC_mean_W207_1, epc_V_SC_std_W207_1 = get_mean_result(epc_result_W207_1)

all_V_LS_mean_W207_1, all_V_LS_std_W207_1, all_V_PC_mean_W207_1, all_V_PC_std_W207_1, all_V_SC_mean_W207_1, all_V_SC_std_W207_1 = get_max_result(ANA_DF, ANA_ALL_DF)







			2) AOBO
			1-2 ) Basal + target 
W_NAME = 'W208'
WORK_NAME = 'WORK_208_1' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W208_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W208_1, epc_V_LS_std_W208_1, epc_V_PC_mean_W208_1, epc_V_PC_std_W208_1, epc_V_SC_mean_W208_1, epc_V_SC_std_W208_1 = get_mean_result(epc_result_W208_1)

all_V_LS_mean_W208_1, all_V_LS_std_W208_1, all_V_PC_mean_W208_1, all_V_PC_std_W208_1, all_V_SC_mean_W208_1, all_V_SC_std_W208_1 = get_max_result(ANA_DF, ANA_ALL_DF)










			2) AOBO
			1-2 ) EXP + target 
W_NAME = 'W209'
WORK_NAME = 'WORK_209_1' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W209_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W209_1, epc_V_LS_std_W209_1, epc_V_PC_mean_W209_1, epc_V_PC_std_W209_1, epc_V_SC_mean_W209_1, epc_V_SC_std_W209_1 = get_mean_result(epc_result_W209_1)

all_V_LS_mean_W209_1, all_V_LS_std_W209_1, all_V_PC_mean_W209_1, all_V_PC_std_W209_1, all_V_SC_mean_W209_1, all_V_SC_std_W209_1 = get_max_result(ANA_DF, ANA_ALL_DF)











			2) AOBO
			1-2 ) EXP + basal
W_NAME = 'W210'
WORK_NAME = 'WORK_210_1' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W210_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W210_1, epc_V_LS_std_W210_1, epc_V_PC_mean_W210_1, epc_V_PC_std_W210_1, epc_V_SC_mean_W210_1, epc_V_SC_std_W210_1 = get_mean_result(epc_result_W210_1)

all_V_LS_mean_W210_1, all_V_LS_std_W210_1, all_V_PC_mean_W210_1, all_V_PC_std_W210_1, all_V_SC_mean_W210_1, all_V_SC_std_W210_1 = get_max_result(ANA_DF, ANA_ALL_DF)

















########################################################
########################################################
########################################################
########################################################
########################################################


			3) AXBX
			1-1 ) 전체 다 먹는거 
W_NAME = 'W203'
WORK_NAME = 'WORK_203_3' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W203_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W203_3, epc_V_LS_std_W203_3, epc_V_PC_mean_W203_3, epc_V_PC_std_W203_3, epc_V_SC_mean_W203_3, epc_V_SC_std_W203_3 = get_mean_result(epc_result_W203_3)

all_V_LS_mean_W203_3, all_V_LS_std_W203_3, all_V_PC_mean_W203_3, all_V_PC_std_W203_3, all_V_SC_mean_W203_3, all_V_SC_std_W203_3 = get_max_result(ANA_DF, ANA_ALL_DF)



			3) AXBX
			1-2 ) gene exp 만 먹인거 206_5인데 잘못 표기함 /home01/k040a01/ray_results/PRJ02.23.06.23.M3V6.WORK_206_1.349.MIS2 / RAY_MY_train_28371_00000

W_NAME = 'W206'
WORK_NAME = 'WORK_206_3' # 349
WORK_DATE = '23.06.22' # 349

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

epc_result_W206_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W206_3, epc_V_LS_std_W206_3, epc_V_PC_mean_W206_3, epc_V_PC_std_W206_3, epc_V_SC_mean_W206_3, epc_V_SC_std_W206_3 = get_mean_result(epc_result_W206_3)

all_V_LS_mean_W206_3, all_V_LS_std_W206_3, all_V_PC_mean_W206_3, all_V_PC_std_W206_3, all_V_SC_mean_W206_3, all_V_SC_std_W206_3 = get_max_result(ANA_DF, ANA_ALL_DF)









			3) AXBX
			1-2 ) Basal only / RAY_MY_train_ffb94_00000 
W_NAME = 'W207'
WORK_NAME = 'WORK_207_3' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W207_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W207_3, epc_V_LS_std_W207_3, epc_V_PC_mean_W207_3, epc_V_PC_std_W207_3, epc_V_SC_mean_W207_3, epc_V_SC_std_W207_3 = get_mean_result(epc_result_W207_3)

all_V_LS_mean_W207_3, all_V_LS_std_W207_3, all_V_PC_mean_W207_3, all_V_PC_std_W207_3, all_V_SC_mean_W207_3, all_V_SC_std_W207_3 = get_max_result(ANA_DF, ANA_ALL_DF)







			3) AXBX
			1-2 ) Basal + target 
W_NAME = 'W208'
WORK_NAME = 'WORK_208_3' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W208_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W208_3, epc_V_LS_std_W208_3, epc_V_PC_mean_W208_3, epc_V_PC_std_W208_3, epc_V_SC_mean_W208_3, epc_V_SC_std_W208_3 = get_mean_result(epc_result_W208_3)

all_V_LS_mean_W208_3, all_V_LS_std_W208_3, all_V_PC_mean_W208_3, all_V_PC_std_W208_3, all_V_SC_mean_W208_3, all_V_SC_std_W208_3 = get_max_result(ANA_DF, ANA_ALL_DF)










			3) AXBX
			1-2 ) EXP + target 
W_NAME = 'W209'
WORK_NAME = 'WORK_209_3' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W209_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W209_3, epc_V_LS_std_W209_3, epc_V_PC_mean_W209_3, epc_V_PC_std_W209_3, epc_V_SC_mean_W209_3, epc_V_SC_std_W209_3 = get_mean_result(epc_result_W209_3)

all_V_LS_mean_W209_3, all_V_LS_std_W209_3, all_V_PC_mean_W209_3, all_V_PC_std_W209_3, all_V_SC_mean_W209_3, all_V_SC_std_W209_3 = get_max_result(ANA_DF, ANA_ALL_DF)











			3) AXBX
			1-2 ) EXP + basal
W_NAME = 'W210'
WORK_NAME = 'WORK_210_3' # 349
WORK_DATE = '23.06.23' # 349

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

epc_result_W210_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W210_3, epc_V_LS_std_W210_3, epc_V_PC_mean_W210_3, epc_V_PC_std_W210_3, epc_V_SC_mean_W210_3, epc_V_SC_std_W210_3 = get_mean_result(epc_result_W210_3)

all_V_LS_mean_W210_3, all_V_LS_std_W210_3, all_V_PC_mean_W210_3, all_V_PC_std_W210_3, all_V_SC_mean_W210_3, all_V_SC_std_W210_3 = get_max_result(ANA_DF, ANA_ALL_DF)











			() AOBO
			TARGET only 
W_NAME = 'W206'
WORK_NAME = 'WORK_206_2' # 349
WORK_DATE = '23.07.09' # 349

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

epc_result_W206_2 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W206_2, epc_V_LS_std_W206_2, epc_V_PC_mean_W206_2, epc_V_PC_std_W206_2, epc_V_SC_mean_W206_2, epc_V_SC_std_W206_2 = get_mean_result(epc_result_W206_2)

all_V_LS_mean_W206_2, all_V_LS_std_W206_2, all_V_PC_mean_W206_2, all_V_PC_std_W206_2, all_V_SC_mean_W206_2, all_V_SC_std_W206_2 = get_max_result(ANA_DF, ANA_ALL_DF)








			() AXBX
			TARGET only 
W_NAME = 'W207'
WORK_NAME = 'WORK_207_2' # 349
WORK_DATE = '23.07.09' # 349

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

epc_result_W207_2 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W207_2, epc_V_LS_std_W207_2, epc_V_PC_mean_W207_2, epc_V_PC_std_W207_2, epc_V_SC_mean_W207_2, epc_V_SC_std_W207_2 = get_mean_result(epc_result_W207_2)

all_V_LS_mean_W207_2, all_V_LS_std_W207_2, all_V_PC_mean_W207_2, all_V_PC_std_W207_2, all_V_SC_mean_W207_2, all_V_SC_std_W207_2 = get_max_result(ANA_DF, ANA_ALL_DF)








			() AOBO + AXBX
			TARGET only 
W_NAME = 'W208'
WORK_NAME = 'WORK_208_2' # 349
WORK_DATE = '23.07.09' # 349

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

epc_result_W208_2 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W208_2, epc_V_LS_std_W208_2, epc_V_PC_mean_W208_2, epc_V_PC_std_W208_2, epc_V_SC_mean_W208_2, epc_V_SC_std_W208_2 = get_mean_result(epc_result_W208_2)

all_V_LS_mean_W208_2, all_V_LS_std_W208_2, all_V_PC_mean_W208_2, all_V_PC_std_W208_2, all_V_SC_mean_W208_2, all_V_SC_std_W208_2 = get_max_result(ANA_DF, ANA_ALL_DF)







			() AXBO + AOBX
			Gene exp only			

W_NAME = 'W216'
WORK_NAME = 'WORK_216_1' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_1 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_1, epc_V_LS_std_W216_1, epc_V_PC_mean_W216_1, epc_V_PC_std_W216_1, epc_V_SC_mean_W216_1, epc_V_SC_std_W216_1 = get_mean_result(epc_result_W216_1)

all_V_LS_mean_W216_1, all_V_LS_std_W216_1, all_V_PC_mean_W216_1, all_V_PC_std_W216_1, all_V_SC_mean_W216_1, all_V_SC_std_W216_1 = get_max_result(ANA_DF, ANA_ALL_DF)





			() AXBO + AOBX
			Basal only		

W_NAME = 'W216'
WORK_NAME = 'WORK_216_2' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_2 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_2, epc_V_LS_std_W216_2, epc_V_PC_mean_W216_2, epc_V_PC_std_W216_2, epc_V_SC_mean_W216_2, epc_V_SC_std_W216_2 = get_mean_result(epc_result_W216_2)

all_V_LS_mean_W216_2, all_V_LS_std_W216_2, all_V_PC_mean_W216_2, all_V_PC_std_W216_2, all_V_SC_mean_W216_2, all_V_SC_std_W216_2 = get_max_result(ANA_DF, ANA_ALL_DF)







			() AXBO + AOBX
			Target only

W_NAME = 'W216'
WORK_NAME = 'WORK_206_3' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_3 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_3, epc_V_LS_std_W216_3, epc_V_PC_mean_W216_3, epc_V_PC_std_W216_3, epc_V_SC_mean_W216_3, epc_V_SC_std_W216_3 = get_mean_result(epc_result_W216_3)

all_V_LS_mean_W216_3, all_V_LS_std_W216_3, all_V_PC_mean_W216_3, all_V_PC_std_W216_3, all_V_SC_mean_W216_3, all_V_SC_std_W216_3 = get_max_result(ANA_DF, ANA_ALL_DF)







			() AXBO + AOBX
			Basal + Target

W_NAME = 'W216'
WORK_NAME = 'WORK_216_4' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_4 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_4, epc_V_LS_std_W216_4, epc_V_PC_mean_W216_4, epc_V_PC_std_W216_4, epc_V_SC_mean_W216_4, epc_V_SC_std_W216_4 = get_mean_result(epc_result_W216_4)

all_V_LS_mean_W216_4, all_V_LS_std_W216_4, all_V_PC_mean_W216_4, all_V_PC_std_W216_4, all_V_SC_mean_W216_4, all_V_SC_std_W216_4 = get_max_result(ANA_DF, ANA_ALL_DF)







			() AXBO + AOBX
			EXP + Target

W_NAME = 'W216'
WORK_NAME = 'WORK_216_5' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_5 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_5, epc_V_LS_std_W216_5, epc_V_PC_mean_W216_5, epc_V_PC_std_W216_5, epc_V_SC_mean_W216_5, epc_V_SC_std_W216_5 = get_mean_result(epc_result_W216_5)

all_V_LS_mean_W216_5, all_V_LS_std_W216_5, all_V_PC_mean_W216_5, all_V_PC_std_W216_5, all_V_SC_mean_W216_5, all_V_SC_std_W216_5 = get_max_result(ANA_DF, ANA_ALL_DF)



			() AXBO + AOBX
			EXP + Basal

W_NAME = 'W216'
WORK_NAME = 'WORK_216_6' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_6 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_6, epc_V_LS_std_W216_6, epc_V_PC_mean_W216_6, epc_V_PC_std_W216_6, epc_V_SC_mean_W216_6, epc_V_SC_std_W216_6 = get_mean_result(epc_result_W216_6)

all_V_LS_mean_W216_6, all_V_LS_std_W216_6, all_V_PC_mean_W216_6, all_V_PC_std_W216_6, all_V_SC_mean_W216_6, all_V_SC_std_W216_6 = get_max_result(ANA_DF, ANA_ALL_DF)





			() AXBO + AOBX
			Good 5CV (total)

W_NAME = 'W216'
WORK_NAME = 'WORK_216_7' # 349
WORK_DATE = '23.07.14' # 349

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

epc_result_W216_7 = get_mean(ANA_DF, ANA_ALL_DF, 1000)

epc_V_LS_mean_W216_7, epc_V_LS_std_W216_7, epc_V_PC_mean_W216_7, epc_V_PC_std_W216_7, epc_V_SC_mean_W216_7, epc_V_SC_std_W216_7 = get_mean_result(epc_result_W216_7)

all_V_LS_mean_W216_7, all_V_LS_std_W216_7, all_V_PC_mean_W216_7, all_V_PC_std_W216_7, all_V_SC_mean_W216_7, all_V_SC_std_W216_7 = get_max_result(ANA_DF, ANA_ALL_DF)










#################################
#################################
#################################
#################################

ablation 

# ALL
ab_1 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_W203, epc_V_PC_mean_W203, epc_V_SC_mean_W203], 
	'LINCS+BASAL' :  [epc_V_LS_mean_W210_5, epc_V_PC_mean_W210_5, epc_V_SC_mean_W210_5], 
	'LINCS+TARGET' :  [epc_V_LS_mean_W209_5, epc_V_PC_mean_W209_5, epc_V_SC_mean_W209_5], 
	'BASAL+TARGET' :  [epc_V_LS_mean_W208_5, epc_V_PC_mean_W208_5, epc_V_SC_mean_W208_5], 
	'LINCS' :  [epc_V_LS_mean_W206_5, epc_V_PC_mean_W206_5, epc_V_SC_mean_W206_5], 
	'BASAL' :  [epc_V_LS_mean_W207_5, epc_V_PC_mean_W207_5, epc_V_SC_mean_W207_5], 
	'TARGET' : [epc_V_LS_mean_W208_2, epc_V_PC_mean_W208_2, epc_V_SC_mean_W208_2]
	})


ab_1_T = ab_1.T

ab_1_re = pd.DataFrame(pd.concat([ab_1_T.iloc[:,0], ab_1_T.iloc[:,1], ab_1_T.iloc[:,2]]))

ab_1_re.columns = ['value']
ab_1_re['ablation'] = list(ab_1_re.index)
ab_1_re['Data ablation'] = 'Data All'
ab_1_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7



# ALL
ab_1_std = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_std_W203, epc_V_PC_std_W203, epc_V_SC_std_W203], 
	'LINCS+BASAL' :  [epc_V_LS_std_W210_5, epc_V_PC_std_W210_5, epc_V_SC_std_W210_5], 
	'LINCS+TARGET' :  [epc_V_LS_std_W209_5, epc_V_PC_std_W209_5, epc_V_SC_std_W209_5], 
	'BASAL+TARGET' :  [epc_V_LS_std_W208_5, epc_V_PC_std_W208_5, epc_V_SC_std_W208_5], 
	'LINCS' :  [epc_V_LS_std_W206_5, epc_V_PC_std_W206_5, epc_V_SC_std_W206_5], 
	'BASAL' :  [epc_V_LS_std_W207_5, epc_V_PC_std_W207_5, epc_V_SC_std_W207_5], 
	'TARGET' : [epc_V_LS_std_W208_2, epc_V_PC_std_W208_2, epc_V_SC_std_W208_2]
	})




ab_1
ab_1_std









# AOBO
ab_2 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_W203_1, epc_V_PC_mean_W203_1, epc_V_SC_mean_W203_1], 
	'LINCS+BASAL' :  [epc_V_LS_mean_W210_1, epc_V_PC_mean_W210_1, epc_V_SC_mean_W210_1], 
	'LINCS+TARGET' :  [epc_V_LS_mean_W209_1, epc_V_PC_mean_W209_1, epc_V_SC_mean_W209_1], 
	'BASAL+TARGET' :  [epc_V_LS_mean_W208_1, epc_V_PC_mean_W208_1, epc_V_SC_mean_W208_1], 
	'LINCS' :  [epc_V_LS_mean_W206_1, epc_V_PC_mean_W206_1, epc_V_SC_mean_W206_1], 
	'BASAL' :  [epc_V_LS_mean_W207_1, epc_V_PC_mean_W207_1, epc_V_SC_mean_W207_1], 
	'TARGET' : [epc_V_LS_mean_W206_2, epc_V_PC_mean_W206_2, epc_V_SC_mean_W206_2]
	})


ab_2_T = ab_2.T

ab_2_re = pd.DataFrame(pd.concat([ab_2_T.iloc[:,0], ab_2_T.iloc[:,1], ab_2_T.iloc[:,2]]))

ab_2_re.columns = ['value']
ab_2_re['ablation'] = list(ab_2_re.index)
ab_2_re['Data ablation'] = 'LINCS all matched'
ab_2_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7


ab_2_std = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_std_W203_1, epc_V_PC_std_W203_1, epc_V_SC_std_W203_1], 
	'LINCS+BASAL' :  [epc_V_LS_std_W210_1, epc_V_PC_std_W210_1, epc_V_SC_std_W210_1], 
	'LINCS+TARGET' :  [epc_V_LS_std_W209_1, epc_V_PC_std_W209_1, epc_V_SC_std_W209_1], 
	'BASAL+TARGET' :  [epc_V_LS_std_W208_1, epc_V_PC_std_W208_1, epc_V_SC_std_W208_1], 
	'LINCS' :  [epc_V_LS_std_W206_1, epc_V_PC_std_W206_1, epc_V_SC_std_W206_1], 
	'BASAL' :  [epc_V_LS_std_W207_1, epc_V_PC_std_W207_1, epc_V_SC_std_W207_1], 
	'TARGET' : [epc_V_LS_std_W206_2, epc_V_PC_std_W206_2, epc_V_SC_std_W206_2]
	})



ab_2
ab_2_std








# AXBX
ab_25 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_W216_7, epc_V_PC_mean_W216_7, epc_V_SC_mean_W216_7], 
	'LINCS+BASAL' :  [epc_V_LS_mean_W216_6, epc_V_PC_mean_W216_6, epc_V_SC_mean_W216_6], 
	'LINCS+TARGET' :  [epc_V_LS_mean_W216_5, epc_V_PC_mean_W216_5, epc_V_SC_mean_W216_5], 
	'BASAL+TARGET' :  [epc_V_LS_mean_W216_4, epc_V_PC_mean_W216_4, epc_V_SC_mean_W216_4], 
	'LINCS' :  [epc_V_LS_mean_W216_1, epc_V_PC_mean_W216_1, epc_V_SC_mean_W216_1], 
	'BASAL' :  [epc_V_LS_mean_W216_2, epc_V_PC_mean_W216_2, epc_V_SC_mean_W216_2], 
	'TARGET' : [epc_V_LS_mean_W216_3, epc_V_PC_mean_W216_3, epc_V_SC_mean_W216_3]
	})



ab_25_T = ab_25.T

ab_25_re = pd.DataFrame(pd.concat([ab_25_T.iloc[:,0], ab_25_T.iloc[:,1], ab_25_T.iloc[:,2]]))

ab_25_re.columns = ['value']
ab_25_re['ablation'] = list(ab_25_re.index)
ab_25_re['Data ablation'] = 'LINCS one matched'
ab_25_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7


ab_25_std = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_std_W216_7, epc_V_PC_std_W216_7, epc_V_SC_std_W216_7], 
	'LINCS+BASAL' :  [epc_V_LS_std_W216_6, epc_V_PC_std_W216_6, epc_V_SC_std_W216_6], 
	'LINCS+TARGET' :  [epc_V_LS_std_W216_5, epc_V_PC_std_W216_5, epc_V_SC_std_W216_5], 
	'BASAL+TARGET' :  [epc_V_LS_std_W216_4, epc_V_PC_std_W216_4, epc_V_SC_std_W216_4], 
	'LINCS' :  [epc_V_LS_std_W216_1, epc_V_PC_std_W216_1, epc_V_SC_std_W216_1], 
	'BASAL' :  [epc_V_LS_std_W216_2, epc_V_PC_std_W216_2, epc_V_SC_std_W216_2], 
	'TARGET' : [epc_V_LS_std_W216_3, epc_V_PC_std_W216_3, epc_V_SC_std_W216_3]
	})




ab_25
ab_25_std














# AXBX
ab_3 = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_mean_W203_3, epc_V_PC_mean_W203_3, epc_V_SC_mean_W203_3], 
	'LINCS+BASAL' :  [epc_V_LS_mean_W210_3, epc_V_PC_mean_W210_3, epc_V_SC_mean_W210_3], 
	'LINCS+TARGET' :  [epc_V_LS_mean_W209_3, epc_V_PC_mean_W209_3, epc_V_SC_mean_W209_3], 
	'BASAL+TARGET' :  [epc_V_LS_mean_W208_3, epc_V_PC_mean_W208_3, epc_V_SC_mean_W208_3], 
	'LINCS' :  [epc_V_LS_mean_W206_3, epc_V_PC_mean_W206_3, epc_V_SC_mean_W206_3], 
	'BASAL' :  [epc_V_LS_mean_W207_3, epc_V_PC_mean_W207_3, epc_V_SC_mean_W207_3], 
	'TARGET' : [epc_V_LS_mean_W207_2, epc_V_PC_mean_W207_2, epc_V_SC_mean_W207_2]
	})



ab_3_T = ab_3.T

ab_3_re = pd.DataFrame(pd.concat([ab_3_T.iloc[:,0], ab_3_T.iloc[:,1], ab_3_T.iloc[:,2]]))

ab_3_re.columns = ['value']
ab_3_re['ablation'] = list(ab_3_re.index)
ab_3_re['Data ablation'] = 'LINCS none matched'
ab_3_re['check'] = ['LOSS']*7 + ['PCOR']*7 + ['SCOR']*7


ab_3_std = pd.DataFrame({
	'LINCS+BASAL+TARGET' :  [epc_V_LS_std_W203_3, epc_V_PC_std_W203_3, epc_V_SC_std_W203_3], 
	'LINCS+BASAL' :  [epc_V_LS_std_W210_3, epc_V_PC_std_W210_3, epc_V_SC_std_W210_3], 
	'LINCS+TARGET' :  [epc_V_LS_std_W209_3, epc_V_PC_std_W209_3, epc_V_SC_std_W209_3], 
	'BASAL+TARGET' :  [epc_V_LS_std_W208_3, epc_V_PC_std_W208_3, epc_V_SC_std_W208_3], 
	'LINCS' :  [epc_V_LS_std_W206_3, epc_V_PC_std_W206_3, epc_V_SC_std_W206_3], 
	'BASAL' :  [epc_V_LS_std_W207_3, epc_V_PC_std_W207_3, epc_V_SC_std_W207_3], 
	'TARGET' : [epc_V_LS_std_W207_2, epc_V_PC_std_W207_2, epc_V_SC_std_W207_2]
	})





ab_3
ab_3_std






ABLATION_DF = pd.concat([ab_1_re, ab_2_re, ab_25_re, ab_3_re])


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
plt.close()






아좀 맘에 안듬 


ABLATION_DF = pd.concat([ab_1_re, ab_2_re, ab_3_re])

ABLATION_DF_data_matched = ABLATION_DF[ABLATION_DF['Data ablation']=='LINCS all matched']
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
plt.close()






더 마음에 안듬 





fig, ax = plt.subplots(1,3,figsize=(25, 5))
sns.barplot(ax = ax[0], data  = ABLATION_DF_loss, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_0 = ax[0]
ax_0.set_xlabel('', fontsize=10)
ax_0.set_ylabel('MSE loss', fontsize=15)
ax_0.set_xticks(ax_0.get_xticks())
ax_0.set_xticklabels(ax_0.get_xticklabels(), fontsize=15, rotation = 45) # , rotation = 90
ax_0.set_ylim(50, 175)
ax_0.set_yticklabels(["{:.2f}".format(a) for a in np.round(ax_0.get_yticks(),2)], fontsize=15)
ax_0.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[1], data  = ABLATION_DF_PCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_1 = ax[1]
ax_1.set_xlabel('', fontsize=10)
ax_1.set_ylabel('PCOR', fontsize=15)
ax_1.set_xticks(ax_1.get_xticks())
ax_1.set_xticklabels(ax_1.get_xticklabels(), fontsize=15, rotation = 45)
ax_1.set_ylim(0.4, 0.8)
ax_1.set_yticklabels(["{:.2f}".format(a) for a in np.round(ax_1.get_yticks(),2)], fontsize=15)
ax_1.legend().set_visible(False)
sns.despine()

sns.barplot(ax = ax[2], data  = ABLATION_DF_SCOR, x = 'Data ablation', y = 'value', hue = 'ablation', linewidth=1,  edgecolor="white", palette = ablation_col) # width = 3,
ax_2 = ax[2]
ax_2.set_xlabel('', fontsize=10)
ax_2.set_ylabel('SCOR', fontsize=15)
ax_2.set_xticks(ax_2.get_xticks())
ax_2.set_xticklabels(ax_2.get_xticklabels(), fontsize=15, rotation = 45)
ax_2.set_ylim(0.3, 0.7)
ax_2.set_yticklabels(["{:.2f}".format(a) for a in np.round(ax_2.get_yticks(),2)], fontsize=15)
ax_2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
sns.despine()

plt.tight_layout()
plt.grid(False)
plt.savefig(os.path.join('/home01/k040a01/ablation3.png'), dpi = 300)
plt.close()













# box plot 그리는거. CV 따라서 그리면 어떨지? 
# 근데 이거 그리면 따로 다시 표를 안나타내도 되나? 
# 그림이 별로면 도루묵일듯 
# 


# order=my_order,
fig, ax = plt.subplots(figsize=(13, 10)) 
sns.boxplot(ax = ax , data = gdsc_all, x = 'CT', y = 'PRED', width = 0.3, linewidth=1, hue = 'Tissue', palette = {'LARGE_INTESTINE' : '#ffcd36' , 'BREAST' : '#025669', 'PANCREAS' : '#ff009bff'} ) # width = 3,
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

