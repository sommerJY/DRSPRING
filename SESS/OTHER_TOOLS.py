
내꺼 test input 다시 가져와야할듯 
문제는 나도 내꺼 저장할때 형식을 CID 를 같이 저장하고 그런게 아니라서 엄 음 


WORK_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_GPU/PJ01.TRIAL.4_1_3/'
JY_ABCS = pd.read_csv(WORK_PATH+'A_B_C_S_SET.csv' , sep = '\t')

G_NAME='HN_GSP'

MY_chem_A_feat = torch.load(WORK_PATH+'0920.{}.MY_chem_A_feat.pt'.format(G_NAME))
MY_chem_B_feat = torch.load(WORK_PATH+'0920.{}.MY_chem_B_feat.pt'.format(G_NAME))
MY_chem_A_adj = torch.load(WORK_PATH+'0920.{}.MY_chem_A_adj.pt'.format(G_NAME))
MY_chem_B_adj = torch.load(WORK_PATH+'0920.{}.MY_chem_B_adj.pt'.format(G_NAME))
MY_exp_A = torch.load(WORK_PATH+'0920.{}.MY_exp_A.pt'.format(G_NAME))
MY_exp_B = torch.load(WORK_PATH+'0920.{}.MY_exp_B.pt'.format(G_NAME))
MY_exp_AB = torch.load(WORK_PATH+'0920.{}.MY_exp_AB.pt'.format(G_NAME))
MY_Cell = torch.load(WORK_PATH+'0920.{}.MY_Cell.pt'.format(G_NAME))
MY_tgt_A = torch.load(WORK_PATH+'0920.{}.MY_tgt_A.pt'.format(G_NAME))
MY_tgt_B = torch.load(WORK_PATH+'0920.{}.MY_tgt_B.pt'.format(G_NAME))
MY_syn = torch.load(WORK_PATH+'0920.{}.MY_syn.pt'.format(G_NAME))

JY_ABCS['syn_score'] = MY_syn.detach().squeeze().tolist()



# normalizing 과정에 대한 고민 필요 ... GCN 이니까 상관 없지 않을까
def prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_A, MY_exp_B, MY_tgt_A, MY_tgt_B, MY_syn, MY_cell, norm ) :
	chem_feat_A_train, chem_feat_A_tv, chem_feat_B_train, chem_feat_B_tv, chem_adj_A_train, chem_adj_A_tv, chem_adj_B_train, chem_adj_B_tv, exp_A_train, exp_A_tv, exp_B_train, exp_B_tv, tgt_A_train, tgt_A_tv, tgt_B_train, tgt_B_tv, syn_train, syn_tv, cell_train, cell_tv = sklearn.model_selection.train_test_split(
			MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_A, MY_exp_B, MY_tgt_A, MY_tgt_B, MY_syn, MY_Cell,
			test_size= 0.2 , random_state=42 )
	chem_feat_A_val, chem_feat_A_test, chem_feat_B_val, chem_feat_B_test, chem_adj_A_val, chem_adj_A_test, chem_adj_B_val, chem_adj_B_test, exp_A_val, exp_A_test, exp_B_val, exp_B_test, tgt_A_val, tgt_A_test, tgt_B_val, tgt_B_test, syn_val, syn_test, cell_val, cell_test  = sklearn.model_selection.train_test_split(
			chem_feat_A_tv, chem_feat_B_tv, chem_adj_A_tv, chem_adj_B_tv, exp_A_tv, exp_B_tv, tgt_A_tv, tgt_B_tv, syn_tv, cell_tv,
			test_size=0.5, random_state=42 )
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	train_data['drug1_feat'] = torch.concat([chem_feat_A_train, chem_feat_B_train], axis = 0)
	val_data['drug1_feat'] = chem_feat_A_val
	test_data['drug1_feat'] = chem_feat_A_test
	#
	train_data['drug2_feat'] = torch.concat([chem_feat_B_train, chem_feat_A_train], axis = 0)
	val_data['drug2_feat'] = chem_feat_B_val
	test_data['drug2_feat'] = chem_feat_B_test
	#
	train_data['drug1_adj'] = torch.concat([chem_adj_A_train, chem_adj_B_train], axis = 0)
	val_data['drug1_adj'] = chem_adj_A_val
	test_data['drug1_adj'] = chem_adj_A_test
	#
	train_data['drug2_adj'] = torch.concat([chem_adj_B_train, chem_adj_A_train], axis = 0)
	val_data['drug2_adj'] = chem_adj_B_val
	test_data['drug2_adj'] = chem_adj_B_test
	#
	train_data['EXP_A'] = torch.concat([exp_A_train, exp_B_train], axis = 0)
	val_data['EXP_A'] = exp_A_val
	test_data['EXP_A'] = exp_A_test
	#
	train_data['EXP_B'] = torch.concat([exp_B_train, exp_A_train], axis = 0)
	val_data['EXP_B'] = exp_B_val
	test_data['EXP_B'] = exp_B_test
	#
	train_data['TGT_A'] = torch.concat([tgt_A_train, tgt_B_train], axis = 0)
	val_data['TGT_A'] = tgt_A_val
	test_data['TGT_A'] = tgt_A_test
	#
	train_data['TGT_B'] = torch.concat([tgt_B_train, tgt_A_train], axis = 0)
	val_data['TGT_B'] = tgt_B_val
	test_data['TGT_B'] = tgt_B_test
	#               
	train_data['y'] = np.concatenate((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	train_data['cell'] = np.concatenate((cell_train, cell_train), axis=0)
	val_data['cell'] = cell_val
	test_data['cell'] = cell_test
	#
	print(train_data['drug1_feat'].shape, flush=True)
	print(val_data['drug1_feat'].shape, flush=True)
	print(test_data['drug1_feat'].shape, flush=True)
	return train_data, val_data, test_data


seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

norm = 'tanh_norm'
train_data, val_data, test_data = prepare_data_GCN(MY_chem_A_feat, MY_chem_B_feat, MY_chem_A_adj, MY_chem_B_adj, MY_exp_A, MY_exp_B, MY_tgt_A, MY_tgt_B, MY_syn, MY_Cell, norm)

ABCS_train, ABCS_tv = sklearn.model_selection.train_test_split(JY_ABCS, test_size= 0.2 , random_state=42)
ABCS_val, ABCS_test = sklearn.model_selection.train_test_split(ABCS_tv, test_size= 0.5 , random_state=42)

# train_data 와 ABCS_train 에 있는 y 가 같은거 확인했음 

ABCS_test 그러면 다른 애들에 넣어보기 





0) Openbabel 

ABCS_test['drug_col_sm'].to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_in_colSM.csv', index= False)
ABCS_test['drug_row_sm'].to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_in_rowSM.csv', index= False)

(JY_6) (근데 header 제거하고 진행해야함)
obabel -i can /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_in_colSM.csv -o sdf -O /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_out_colSM.sdf
obabel -i can /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_in_rowSM.csv -o sdf -O /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_out_rowSM.sdf



# smiles 로 넣고 싶을때 
obabel -:'C1CN(CCC1C(C2=CC3=C(C=C2)OCO3)(C4=CC5=C(C=C4)OCO5)O)C(=O)OC6=CC=C(C=C6)[N+](=O)[O-]' -o sdf -O test.sdf


1) jCompoundMapper
java -jar /home/jiyeonH/utils/jCMapperCLI.jar -f /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_out_colSM.sdf -c ECFP -o /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/jchem_out_colSM.csv -ff FULL_CSV
java -jar /home/jiyeonH/utils/jCMapperCLI.jar -f /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/obabel_out_rowSM.sdf -c ECFP -o /st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/jchem_out_rowSM.csv -ff FULL_CSV


2) chemopy
import pychem as PYC
import os, sys
####sys.path.append("/home/jiyeonH/.conda/envs/JY_6/lib/python3.7/site-packages/")
####sys.path.append("/home/jiyeonH/utils/pychem-1.0/src/pychem")
####import pychem
# from pychem import pychem
from pychem import PyChem2d, PyChem3d

(1)
#import pandas as pd 
#import sys
#sys.path.append("/home/jiyeonH/utils/pychem-1.0/src/pychem")
#from pychem import PyChem2d, PyChem3d
#import pychem

(2) 
import pandas as pd 
import sys
import pychem 
from pychem import pychem as PYC
#from pychem import PyChem2d, PyChem3d

id = '11960529'

def get_chemical_data(id): # takes cid of drug as input
d1 = PYC.PyChem2d()
smi = d1.GetMolFromNCBI(id) 
mol = d1.ReadMolFromSmile(smi)
chemical_feat = {}
chemical_feat.update(PYC.constitution.GetConstitutional(mol))
chemical_feat.update(PYC.connectivity.GetConnectivity(mol))
chemical_feat.update(PYC.kappa.GetKappa(mol))
chemical_feat.update(PYC.bcut.CalculateBurdenVDW(mol))
chemical_feat.update(PYC.bcut.CalculateBurdenPolarizability(mol))
chemical_feat.update(PYC.estate.GetEstate(mol))
chemical_feat.update(PYC.basak.Getbasak(mol)) ### 
chemical_feat.update(PYC.geary.GetGearyAuto(mol))
chemical_feat.update(PYC.moran.GetMoranAuto(mol))
chemical_feat.update(PYC.moreaubroto.GetMoreauBrotoAuto(mol))
chemical_feat.update(PYC.molproperty.GetMolecularProperty(mol))
chemical_feat.update(PYC.moe.GetMOE(mol))
    return(chemical_feat)


    File "<stdin>", line 1, in <module>
File "/home/jiyeonH/utils/pychem-1.0/src/pychem/basak.py", line 520, in Getbasak
    result[DesLabel]=round(_basak[DesLabel](mol),3)
File "/home/jiyeonH/utils/pychem-1.0/src/pychem/basak.py", line 376, in CalculateBasakCIC1
    IC=CalculateBasakIC1(mol)
File "/home/jiyeonH/utils/pychem-1.0/src/pychem/basak.py", line 188, in CalculateBasakIC1
    return _CalculateBasakICn(mol,NumPath=2)
File "/home/jiyeonH/utils/pychem-1.0/src/pychem/basak.py", line 155, in _CalculateBasakICn
    value.sort()
    TypeError: '<' not supported between instances of 'list' and 'int'








3) FARMS 
RMA_data = 
BiocManager::install("farms")






#######################################################################################
(1) DeepSynergy 

import numpy as np
import pandas as pd
import pickle 
import gzip

import os, sys
import pandas as pd
import numpy as np
import pickle
import gzip

import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"]="3" #specify GPU 
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.keras import backend as K
from tensorflow.keras import backend
from tensorflow.keras.backend.tensorflow_backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# first, data generation 
norm = 'tanh'
test_fold = 0
val_fold = 1

# 아 이게 여기서 나온거였냐 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 
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


path_dir = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/1_DeepSynergy-master/'


file = gzip.open(path_dir+'X.p.gz', 'rb')
X = pickle.load(file) # 오 근데 이런 방식으로 gz 파일 읽을 수 있는건 또 처음 알았네 
# (46104, 12758) : feature 구조는 openbabel 을 이용해서 chemical + gene array exp 인듯 아니 근데 이걸 우리가 만들어줘야함?
file.close()


# open babel 의 경우 tarball 이용해서 ㄲ라아야하는듯 
# conda 이용해서 깔긴 했는데 흠 
아 생각해보니까 이게 A-B-C 랑 B-A-C 가 연속으로 섞임
그래서 만약에 내꺼 돌리려면 전체 데이터가지고 돌리고 (얘네는 best weight 안주므로)



#contains synergy values and fold split (numbers 0-4) -> 아예 처음부터 fold split 나눠서 주기로 함
labels = pd.read_csv(path_dir+'labels.csv', index_col=0) 
#labels are duplicated for the two different ways of ordering in the data
# (23052, 5)
# 그런데 애들 이름이 써져 있기는 함 -> 그럼 나중에 우리꺼 test 에 있는 애들이 있는지만 확인하면 되는거 아닌가 
# 아 헷갈리는군 
labels = pd.concat([labels, labels])  


# 하나 헷갈리는게 왜 data spliting 방법을 두가지로 쓰는 거지 
#indices of training data for hyperparameter selection: fold 2, 3, 4
idx_tr = np.where(np.logical_and(labels['fold']!=test_fold, labels['fold']!=val_fold))
#indices of validation data for hyperparameter selection: fold 1
idx_val = np.where(labels['fold']==val_fold)

#indices of training data for model testing: fold 1, 2, 3, 4
idx_train = np.where(labels['fold']!=test_fold)
#indices of test data for model testing: fold 0
idx_test = np.where(labels['fold']==test_fold)



# split data 
X_tr = X[idx_tr] # (27768, 12758)
X_val = X[idx_val] # (9228, 12758)
X_train = X[idx_train] # (36996, 12758)
X_test = X[idx_test] # (9108, 12758)

y_tr = labels.iloc[idx_tr]['synergy'].values # 27768
y_val = labels.iloc[idx_val]['synergy'].values
y_train = labels.iloc[idx_train]['synergy'].values
y_test = labels.iloc[idx_test]['synergy'].values


# #### Normalize training and validation data for hyperparameter selection
if norm == "tanh_norm":
    X_tr, mean, std, mean2, std2, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, mean2, std2, feat_filt = normalize(X_val, mean, std, mean2, std2, 
                                                          feat_filt=feat_filt, norm=norm)
else:
    X_tr, mean, std, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, feat_filt = normalize(X_val, mean, std, feat_filt=feat_filt, norm=norm)


# Normalize training and test data for methods comparison
if norm == "tanh_norm":
    X_train, mean, std, mean2, std2, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, mean2, std2, feat_filt = normalize(X_test, mean, std, mean2, std2, 
                                                          feat_filt=feat_filt, norm=norm)
else:
    X_train, mean, std, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, feat_filt = normalize(X_test, mean, std, feat_filt=feat_filt, norm=norm)


pickle.dump((X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test), 
            open('data_test_fold%d_%s.p'%(test_fold, norm), 'wb'))


### 데이터 확인은 끝 
# 본모델 시작 
one cross validation run : 지금 5개로 나눠놨으니가 5번 돌아감 
여기 예시에서는 fold 0 을 testing 에 사용 
60% of the data for training (folds 2, 3, 4) 
20% for validation (fold 1)
parameter selection 을 진행하고 나서는 training & validation 을 다 합치고 나머지 fold 0 만 testing 에 사용 

import os, sys
import pandas as pd
import numpy as np
import pickle
import gzip
import matplotlib
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"]="3" #specify GPU 
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras import backend
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


hyperparameter_file = 'cv_example/hyperparameters' # textfile which contains the hyperparameters of the model
data_file = 'data_test_fold0_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)

# early stopping 을 위한거라고 함 
def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


exec(open(path_dir+hyperparameter_file).read()) 


file = gzip.open(path_dir+data_file, 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()


# 이건 tensorflow 1 에서 해당되는 내용이라고 함 

config = tf.ConfigProto(
         allow_soft_placement=True
         ) # gpu_options = tf.GPUOptions(allow_growth=True)
set_session(tf.Session(config=config))



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
    model.compile(loss='mean_squared_error', optimizer=K.optimizers.SGD(lr=float(eta), momentum=0.5))



hist = model.fit(X_tr, y_tr, epochs=epochs, shuffle=True, batch_size=64, validation_data=(X_val, y_val))

val_loss = hist.history['val_loss']
model.reset_states()


# 돌아가는건 확인했는데 문제는 input 
# 우리 서버에서 1epoch 에 620s 걸림 
-> 일단 돌아가라 

model.save(path_dir+'ORIGINAL_MODEL')
model.save_weights(path_dir+'ORIGINAL_MODEL_WEIGHT')



# 그래서 내꺼 데이터 확인해보려면 
JY_T_loss, JY_T_mse  = model.evaluate(jy_test_X, jy_test_Y)





average_over = 15
mov_av = moving_average(np.array(val_loss), average_over)
smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
epo = np.argmin(smooth_val_loss)


























(3) TranSynergy - 아.. 생각한것보다 너무 복잡한데... 일단 cuda 문제에서 걸리고 있음 


import numpy as np
import pandas as pd
from os import path, mkdir, environ
import sys

from time import time
import torch
from torch import save, load
from torch.utils import data

import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import pdb
import shap # shapley additive explanation : 기계학습결과 설명하기위한 게임이론 접근방식 
# 모델에 트정 변수가 있는 경우와 없는 경우를 비교하여 변수의 중요도 계산. 
# 모델이 변수를 보는 순서가 예측에 영향을 줄 수 있으므로 변수를 공정하게 비교하기 위해 
# 모든 순서로 수행하여 모든 변수를 공정하게 비교하고자 함 
import pickle
from sklearn.cluster import MiniBatchKMeans
import wandb # weight and bias 
# tensorboard 랑 비슷한데, 얘는 tensorflow 랑 pytorch 둘다 사용 가능 
import concurrent.futures # dj 
import random

import data_utils

from src import attention_model, drug_drug, setting, my_data, logger, device2



random_seed = 913

def set_seed(seed=random_seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


import os
from time import time
import shutil

unit_test = False

working_dir = '/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/3_TranSynergy/'
src_dir = os.path.join(working_dir, 'src')
data_src_dir = os.path.join(working_dir, 'data')
# propagation_methods: target_as_1, RWlike, random_walk
propagation_method = 'random_walk'
# feature type: LINCS1000, others, determine whether or not ignoring drugs without hidden representation
feature_type = 'more' # 
F_repr_feature_length = 1000

activation_method =["relu"]
dropout = [0.2, 0.1, 0.1]
start_lr = 0.00003
lr_decay = 0.00002
model_type = 'mlp' # 모델 타입이 MLP 임? 
FC_layout = [256] * 1 + [64] * 1
n_epochs = 800
batch_size = 128
loss = 'mse'
NBS_logfile = os.path.join(working_dir, 'NBS_logfile')
data_specific = '_0.3_cv0_alpha4_gene_dep_and_expr_rwr3' # 이건 뭘까 
# _0.3_cv0_alpha3_gene_dep_and_expr_rwr3
data_folder = os.path.join(working_dir, 'datas' + data_specific)
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
    open(os.path.join(data_folder, "__init__.py"), 'w+').close()

uniq_part = "_run_{!r}".format(int(time()))
run_dir = os.path.join(working_dir, uniq_part)
logfile = os.path.join(run_dir, 'logfile')

run_specific_setting = os.path.join(run_dir, "setting.py")
cur_dir_setting = os.path.join(src_dir, "setting.py")

if not os.path.exists(run_dir):
    os.makedirs(run_dir)
    open(os.path.join(run_dir, "__init__.py"), 'w+').close()
    shutil.copyfile(cur_dir_setting, run_specific_setting)

update_final_index = True
final_index = os.path.join(data_src_dir, "synergy_score/final_index.csv")
update_xy = False
old_x = os.path.join(data_src_dir,"synergy_score/x.npy")
old_x_lengths = os.path.join(data_src_dir,"synergy_score/old_x_lengths.pkl")
old_y = os.path.join(data_src_dir,"synergy_score/y.pkl")

# 확인용 
old_x = np.load('/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/3_TranSynergy/data/x.npy') # (37104, 9608)
with open('/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/3_TranSynergy/data/old_x_lengths.pkl', mode= 'rb') as f :
    old_x_lengths = pickle.load(f) # ([2402], [2402, 2402])

with open('/st06/jiyeonH/11.TOX/MY_TRIAL_6/OTHER_TOOLS/3_TranSynergy/data/y.pkl', mode= 'rb') as f :
    old_y = pickle.load(f) # (37104, 1)



y_labels_file = os.path.join(src_dir, 'y_labels.p')
### ecfp, phy, ge, gd
catoutput_output_type = data_specific + "_dt"
save_final_pred = True
catoutput_intput_type = [data_specific + "_dt"] #["ecfp", "phy", "ge", "gd"]
#{"ecfp": 2048, "phy": 960, "single": 15, "proteomics": 107}
dir_input_type = {}#{"single": 15, "proteomics": 107}

neural_fp = True
chemfp_drug_feature_file = os.path.join(data_src_dir, 'chemicals', 'drug_features_all_three_tanh.csv')
chem_linear_layers = [1024]
drug_input_dim = {'atom': 62, 'bond': 6}
conv_size = [16, 16]
degree = [0, 1, 2, 3, 4, 5]
drug_emb_dim = 512

genes = os.path.join(data_src_dir, 'Genes', 'genes_2401_df.csv')
synergy_score = os.path.join(data_src_dir, 'synergy_score', 'synergy_score.csv')
pathway_dataset = os.path.join(data_src_dir, 'pathways', 'genewise.p')
cl_genes_dp = os.path.join(data_src_dir, 'cl_gene_dp', 'new_gene_dependencies_35.csv')
#genes_network = '../genes_network/genes_network.csv'
#drugs_profile = '../drugs_profile/drugs_profile.csv'
L1000_upregulation = os.path.join(data_src_dir, 'F_repr', 'sel_F_drug_sample.csv')
L1000_downregulation = os.path.join(data_src_dir, 'F_repr', 'sel_F_drug_sample_1.csv')
add_single_response_to_drug_target = True
F_cl = os.path.join(data_src_dir, 'F_repr', 'sel_F_cl_sample.csv')
single_response = os.path.join(data_src_dir, 'chemicals', 'single_response_features.csv')

drug_ECFP = os.path.join(data_src_dir, 'chemicals', 'ECFP6.csv')
drug_physicochem = os.path.join(data_src_dir, 'chemicals', 'physicochemical_des.csv')
cl_ECFP = os.path.join(data_src_dir, 'RF_features', 'features_importance_df.csv')
cl_physicochem = os.path.join(data_src_dir, 'RF_features', 'features_importance_df_phychem.csv')
inchi_merck = os.path.join(data_src_dir, 'chemicals', 'inchi_merck.csv')

# networks: string_network, all_tissues_top
network_update = True
network_prop_normalized = True
network_path = os.path.join(data_src_dir, 'network')
network = os.path.join(data_src_dir, 'network', 'string_network')
network_matrix = os.path.join(data_src_dir, 'network', 'string_network_matrix.csv')
split_random_seed = 3
index_in_literature = True
index_renewal = True
train_index = os.path.join(data_src_dir, 'train_index_' + str(split_random_seed))
test_index = os.path.join(data_src_dir, 'test_index_' + str(split_random_seed))

renew = False
gene_expression_simulated_result_matrix = os.path.join(data_src_dir, 'chemicals', 'gene_expression_simulated_result_matrix_string.csv')
random_walk_simulated_result_matrix = os.path.join(data_src_dir, 'chemicals', 'random_walk_simulated_result_matrix_2401_0.3_norm_36_whole_network_no_mean')
intermediate_ge_target0_matrix = os.path.join(data_src_dir, 'chemicals', 'intermediate_ge_target0_matrix')

ml_train = False
test_ml_train = False

# estimators: RandomForest, GradientBoosting
estimator = "RandomForest"

if not os.path.exists(os.path.join(src_dir, 'tensorboard_logs')):
    os.mkdir(os.path.join(src_dir, 'tensorboard_logs'))
tensorboard_log = os.path.join(src_dir, "tensorboard_logs/{}".format(time()))

combine_gene_expression_renew = False
gene_expression = "Gene_expression_raw/normalized_gene_expession_35_norm.tsv" #"CCLE.tsv"
backup_expression = "Gene_expression_raw/normalized_gene_expession_35_norm.tsv" #"GDSC.tsv"
netexpress_df = "Gene_expression_raw/netexpress_norm_35.tsv"

raw_expression_data_renew = False
processed_expression_raw = os.path.join(data_src_dir, 'Gene_expression_raw', 'processed_expression_raw_norm')

combine_drug_target_renew = False
combine_drug_target_matrix = os.path.join(data_src_dir, 'chemicals', 'combine_drug_target_matrix.csv')

drug_profiles_renew = False
drug_profiles = os.path.join(data_src_dir, 'chemicals','new_dedup_drug_profile.csv')

python_interpreter_path = '/Users/QiaoLiu1/anaconda3/envs/pynbs_env/bin/python'

y_transform = True

### ['drug_target_profile', 'drug_ECFP', 'drug_physiochemistry', 'drug_F_repr']
drug_features = ['drug_target_profile']
#drug_features = ['drug_F_repr']
ecfp_phy_drug_filter_only = True
save_each_ecfp_phy_data_point = True

### ['gene_dependence', 'netexpress','gene_expression', 'cl_F_repr', 'cl_ECFP', 'cl_drug_physiochemistry', 'combine_drugs_for_cl']
cellline_features = ['gene_dependence', 'gene_expression']
#cellline_features = ['cl_F_repr' ]

one_linear_per_dim = True

single_response_feature = []#['single_response']

#arrangement = [[1,5,11],[2,6,12],[0,4,8],[0,4,9]]
expression_dependencies_interaction = False
arrangement = [[0,1,2,3]]
update_features = False
output_FF_layers = [2000, 1000,  1]
n_feature_type = [4]
single_repsonse_feature_length = 10 * 2
if 'single_response' not in single_response_feature:
    single_repsonse_feature_length = 0
d_model_i = 1
d_model_j = 512
d_model = d_model_i * d_model_j
attention_heads = 1
attention_dropout = 0.1
n_layers = 1 # This has to be 1

load_old_model = False
old_model_path = os.path.join(working_dir, "_run_1582753440/best_model__2401_0.8_norm_drug_target_36_norm_net_single")

get_feature_imp = False
save_feature_imp_model = True
save_easy_input_only = (len(n_feature_type) == 1)
save_out_imp = False
save_inter_imp = False
best_model_path = os.path.join(run_dir, "best_model_" + data_specific)
perform_importance_study = False
input_importance_path = os.path.join(working_dir, "input_importance_" + data_specific)
out_input_importance_path = os.path.join(working_dir, "out_input_importance_" + data_specific)
transform_input_importance_path = os.path.join(working_dir, "transform_input_importance_" +data_specific)
feature_importance_path = os.path.join(working_dir, 'all_features_importance_' + data_specific )




def get_final_index():
    if not setting.update_final_index and path.exists(setting.final_index):
        final_index = pd.read_csv(setting.final_index, header=None)[0]
    else:
        final_index = my_data.SynergyDataReader.get_final_index()
    return final_index    


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

Attention 활용하고 있는건 알겠는데
자체로 


class Norm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.size = d_model
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
               / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm




def attention(q, k, v, d_k = 1, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output





class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)
        return output



















(4) prodeepsyn

데이터 다운로드부터 일임 ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
brew install git-lfs
sudo apt install git-lfs
git lfs install
git lfs pull
git lfs ls-files

13697aaa7e - cell/data/cell_feat.npy
cd2e2573d7 - cell/data/mdl_ge_128x384_sample/embeddings.npy
566f30c429 - cell/data/mdl_mut_128x384_sample/embeddings.npy
4cdc6d959d - cell/data/node_features.npy
0572493b79 - cell/data/nodes_ge.npy
246df8a09a - cell/data/nodes_mut.npy
728eae2ea2 - cell/data/ppi.coo.npy
b7fd2c3b52 - cell/data/target_ge.npy
4b53f50fe0 - cell/data/target_mut.npy
eedc80ba39 - drug/data/drug_feat.npy





