

내꺼 214 시리즈 비교하기 위함 

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

MJ_NAME = 'M3V6'
PPI_NAME = '349'
MISS_NAME = 'MIS2'

W_NAME = 'W214'
WORK_NAME = 'WORK_214_6' # 349

WORK_DATE = '23.07.10' # 349



anal_dir = "/home01/k040a01/ray_results/PRJ02.{}.{}.{}.{}.{}".format(WORK_DATE, MJ_NAME,  WORK_NAME, PPI_NAME, MISS_NAME)

list_dir = os.listdir(anal_dir)
exp_json = [a for a in list_dir if 'experiment_state' in a]

anal_df_list = [ExperimentAnalysis(os.path.join(anal_dir, a)) for a in exp_json]

for a in range(len(anal_df_list)):
    a
    ad = anal_df_list[a]
    if ad.dataframe().shape[0] > 1 :
        ad.dataframe().trial_id


# RAY_MY_train_89149_00000 / RAY_MY_train_5dc7c_00000 / RAY_MY_train_5b985_00000 / RAY_MY_train_abf17_00000
# RAY_MY_train_232c2 / RAY_MY_train_6514c / RAY_MY_train_61d4d

DF_re = pd.DataFrame(columns = ['T_LS', 'T_PC', 'T_SC', 'V_LS', 'V_PC', 'V_SC', 'time_this_iter_s',
       'should_checkpoint', 'done', 'timesteps_total', 'episodes_total',
       'training_iteration', 'trial_id', 'experiment_id', 'date', 'timestamp',
       'time_total_s', 'pid', 'hostname', 'node_ip', 'time_since_restore',
       'timesteps_since_restore', 'iterations_since_restore', 'warmup_time',
       'config/CV', 'config/G_chem_hdim', 'config/G_chem_layer',
       'config/G_exp_hdim', 'config/G_exp_layer', 'config/batch_size',
       'config/dropout_1', 'config/dropout_2', 'config/dsn_layer',
       'config/epoch', 'config/lr', 'config/n_workers', 'config/snp_layer',
       'logdir'])


ANA_DF_1 = anal_df_list[0].dataframe()

ANA_ALL_DF_1 = anal_df_list[0].trial_dataframes

ANA_DF_1 = ANA_DF_1.sort_values('config/CV')

ANA_DF_1.index = [a for a in range(5)]

ANA_DF = ANA_DF_1

ANA_ALL_DF = ANA_ALL_DF_1




            ANA_DF_1 = anal_df_list[1].dataframe()
            ANA_DF_2 = anal_df_list[0].dataframe()
            ANA_DF_3 = anal_df_list[2].dataframe()
            ANA_DF_4 = anal_df_list[3].dataframe()

            ANA_ALL_DF_1 = anal_df_list[1].trial_dataframes
            ANA_ALL_DF_2 = anal_df_list[0].trial_dataframes
            ANA_ALL_DF_3 = anal_df_list[2].trial_dataframes
            ANA_ALL_DF_4 = anal_df_list[3].trial_dataframes

            ANA_DF_1 = ANA_DF_1.sort_values('config/CV')
            ANA_DF_2 = ANA_DF_2.sort_values('config/CV')
            ANA_DF_3 = ANA_DF_3.sort_values('config/CV')
            ANA_DF_4 = ANA_DF_4.sort_values('config/CV')


            ANA_DF_1.index = [a for a in range(10)]
            ANA_DF_2.index = [a for a in range(10, 18)]
            ANA_DF_3.index = [a for a in range(18, 26)]
            ANA_DF_4.index = [a for a in range(20, 28)]

            ANA_DF = pd.concat([ANA_DF_1, ANA_DF_2, ANA_DF_3, ANA_DF_4])

            ANA_ALL_DF = {**ANA_ALL_DF_1, **ANA_ALL_DF_2, **ANA_ALL_DF_3, **ANA_ALL_DF_4}
            











                    ANA_DF.to_csv('/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))
                    import pickle
                    with open("/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME), "wb") as fp:
                        pickle.dump(ANA_ALL_DF,fp) 

                    '/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.csv'.format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME,  MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
                    "/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.pickle".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)


limit = 1000

cv0_key = ANA_DF['logdir'][0] ;	cv1_key = ANA_DF['logdir'][1]; 	cv2_key = ANA_DF['logdir'][2] ;	cv3_key = ANA_DF['logdir'][3];	cv4_key = ANA_DF['logdir'][4]

cv_keys = list(ANA_DF['logdir'])

epc_T_LS_mean = np.mean([ANA_ALL_DF[kk]['T_LS'] for kk in cv_keys], axis = 0)
epc_T_LS_std = np.std([ANA_ALL_DF[kk]['T_LS'] for kk in cv_keys], axis = 0)

epc_T_PC_mean = np.mean([ANA_ALL_DF[kk]['T_PC'] for kk in cv_keys], axis = 0)
epc_T_PC_std = np.std([ANA_ALL_DF[kk]['T_PC'] for kk in cv_keys], axis = 0)

epc_T_SC_mean = np.mean([ANA_ALL_DF[kk]['T_SC'] for kk in cv_keys], axis = 0)
epc_T_SC_std = np.std([ANA_ALL_DF[kk]['T_SC'] for kk in cv_keys], axis = 0)

epc_V_LS_mean = np.mean([ANA_ALL_DF[kk]['V_LS'] for kk in cv_keys], axis = 0)
epc_V_LS_std = np.std([ANA_ALL_DF[kk]['V_LS'] for kk in cv_keys], axis = 0)

epc_V_PC_mean = np.mean([ANA_ALL_DF[kk]['V_PC'] for kk in cv_keys], axis = 0)
epc_V_PC_std = np.std([ANA_ALL_DF[kk]['V_PC'] for kk in cv_keys], axis = 0)

epc_V_SC_mean = np.mean([ANA_ALL_DF[kk]['V_SC'] for kk in cv_keys], axis = 0)
epc_V_SC_std = np.std([ANA_ALL_DF[kk]['V_SC'] for kk in cv_keys], axis = 0)


epc_result = pd.DataFrame({
	'T_LS_mean' : epc_T_LS_mean, 'T_PC_mean' : epc_T_PC_mean, 'T_SC_mean' : epc_T_SC_mean, 
	'T_LS_std' : epc_T_LS_std, 'T_PC_std' : epc_T_PC_std, 'T_SC_std' : epc_T_SC_std, 
	'V_LS_mean' : epc_V_LS_mean, 'V_PC_mean' : epc_V_PC_mean, 'V_SC_mean' : epc_V_SC_mean, 
	'V_LS_std' : epc_V_LS_std, 'V_PC_std' : epc_V_PC_std, 'V_SC_std' : epc_V_SC_std,
})

epc_result[[
	'T_LS_mean', 'T_LS_std', 'T_PC_mean', 'T_PC_std',
	'T_SC_mean','T_SC_std', 'V_LS_mean', 'V_LS_std', 
	'V_PC_mean', 'V_PC_std','V_SC_mean','V_SC_std']].to_csv("/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME))

"/home01/k040a01/02.M3V6/{}_{}_{}_{}/RAY_ANA_DF.{}_{}_{}_{}.resDF".format(MJ_NAME, W_NAME, PPI_NAME, MISS_NAME, MJ_NAME, W_NAME, PPI_NAME, MISS_NAME)
		


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
round(epc_result.loc[KEY_EPC].V_LS_mean,4)
round(epc_result.loc[KEY_EPC].V_LS_std,4)



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
round(epc_result.loc[KEY_EPC].V_PC_mean,4)
round(epc_result.loc[KEY_EPC].V_PC_std,4)


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
round(epc_result.loc[KEY_EPC].V_SC_mean,4)
round(epc_result.loc[KEY_EPC].V_SC_std,4)





epc = 450

ANA_ALL_DF[cv0_key]['V_PC'][epc], ANA_ALL_DF[cv1_key]['V_PC'][epc],ANA_ALL_DF[cv2_key]['V_PC'][epc], ANA_ALL_DF[cv3_key]['V_PC'][epc], ANA_ALL_DF[cv4_key]['V_PC'][epc]








max 로 다시 한번만 확인 


epc_T_LS_mean = np.mean([min(ANA_ALL_DF[kk]['T_LS']) for kk in cv_keys])
epc_T_LS_std = np.std([min(ANA_ALL_DF[kk]['T_LS']) for kk in cv_keys])

epc_T_PC_mean = np.mean([max(ANA_ALL_DF[kk]['T_PC']) for kk in cv_keys])
epc_T_PC_std = np.std([max(ANA_ALL_DF[kk]['T_PC']) for kk in cv_keys])

epc_T_SC_mean = np.mean([max(ANA_ALL_DF[kk]['T_SC']) for kk in cv_keys])
epc_T_SC_std = np.std([max(ANA_ALL_DF[kk]['T_SC']) for kk in cv_keys])

epc_V_LS_mean = np.mean([min(ANA_ALL_DF[kk]['V_LS']) for kk in cv_keys])
epc_V_LS_std = np.std([min(ANA_ALL_DF[kk]['V_LS']) for kk in cv_keys])

epc_V_PC_mean = np.mean([max(ANA_ALL_DF[kk]['V_PC']) for kk in cv_keys])
epc_V_PC_std = np.std([max(ANA_ALL_DF[kk]['V_PC']) for kk in cv_keys])

epc_V_SC_mean = np.mean([max(ANA_ALL_DF[kk]['V_SC']) for kk in cv_keys])
epc_V_SC_std = np.std([max(ANA_ALL_DF[kk]['V_SC']) for kk in cv_keys])


np.round([epc_V_LS_mean, epc_V_LS_std], 4)
np.round([epc_V_PC_mean, epc_V_PC_std], 4)
np.round([epc_V_SC_mean, epc_V_SC_std], 4)

