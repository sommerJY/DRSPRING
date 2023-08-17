

# DeepSynergy

# 특정 

conda activate JY_9


# DATA normalizing 
import numpy as np
import pandas as pd
import pickle 
import gzip


import os, sys

import json

import matplotlib.pyplot as plt

import keras 
import tensorflow
import tensorflow.keras as K
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.backend import set_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import random


TOOL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/01.DeepSynergy/DeepSynergy-master'

hyperparameter_file = 'hyperparameters' # textfile which contains the hyperparameters of the model
data_file = 'data_test_fold0_tanh.p.gz' # pickle file which contains the data (produced with normalize.ipynb)


# in this example tanh normalization is used
# fold 0 is used for testing and fold 1 for validation (hyperparamter selection)
norm = 'tanh'
test_fold = 0
val_fold = 1


# It normalizes the input data X. 
# If X is used for training the mean and the standard deviation is calculated during normalization. 
# If X is used for validation or testing, 
# the previously calculated mean and standard deviation of the training data should be used. 
# If "tanh_norm" is used as normalization strategy, 
# then the mean and standard deviation are calculated twice. 
# The features with a standard deviation of 0 are filtered out.

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


# 이미 앞뒤로 섞여있음 
# feature 정의도 이미 되어있는 상황
file = gzip.open(os.path.join(TOOL_PATH,'X.p.gz'), 'rb')
X = pickle.load(file)
file.close()



#contains synergy values and fold split (numbers 0-4)
labels = pd.read_csv(os.path.join(TOOL_PATH,'labels.csv'), index_col=0) 
#labels are duplicated for the two different ways of ordering in the data
labels = pd.concat([labels, labels]) 
DeepS_label = list(set(list(labels.drug_a_name) + list(labels.drug_b_name))) # 38개 






# 내꺼 drug 랑 Cell line 비교하기 
DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')

DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/'

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



my_cid_list = [str(int(a)) if a > 0 else 'NA' for a in list(DC_DRUG_DF_FULL.CID)]
my_sym_list = [a if type(a) == str else 'NA' for a in list(DC_DRUG_DF_FULL.filtered_synonym)]

cid_sym_dict = {my_cid_list[a] : list(my_sym_list[a].split('\n')) for a in range(DC_DRUG_DF_FULL.shape[0])}

DeepS_result = []
for key in cid_sym_dict.keys() :
    tmp_list = cid_sym_dict[key]
    for query in DeepS_label :
        if query in tmp_list :
            DeepS_result.append( (query, key) )
        #
        smaller_1 = query[0] + query[1:].lower()
        if smaller_1 in tmp_list :
            DeepS_result.append( (query, key) )
        #
        smaller_2 = query.lower()
        if smaller_2 in tmp_list :
            DeepS_result.append( (query, key) )



DeepS_match_1 = [a for a,b in DeepS_result]
DeepS_match_2 = [b for a,b in DeepS_result]
DeepS_match_df = pd.DataFrame({'DS' : DeepS_match_1, 'MM' : DeepS_match_2})

set(DeepS_label) - set(DeepS_match_df.DS)
set(DeepS_match_df.DS) - set(DeepS_label) 

# MITOMYCINE
# E 때문에 안들어가 
DeepS_match_df = pd.concat([DeepS_match_df, pd.DataFrame({'DS' : ['MITOMYCINE'], 'MM' : ['5746']})])
DeepS_match_df = DeepS_match_df.sort_values('DS')

                        #33      GELDANAMYCIN   5288382 # 원본에서 canonical 이걸로 
                        #41      GELDANAMYCIN  13017912 # 원본에서 alternative  
                        #-> 또옥같다. 시벌

                        #36           MK-8776  16224745 # 원본에서 canonical 이걸로 
                        #19           MK-8776  46239015 # 원본에서 canonical 

                        #37       VINORELBINE     60780 # 원본에서 canonical 
                        #38       VINORELBINE  44424639 # 원본에서 canonical 
                        #31       VINORELBINE   5311497 # 원본에서 canonical 이걸로 

DeepS_match_df = DeepS_match_df.drop([41,19,37,38])
DeepS_match_df = DeepS_match_df.drop_duplicates()
DeepS_match_df['MM_re'] = [int(a) for a in list(DeepS_match_df.MM)]
DeepS_match_df = DeepS_match_df[['DS','MM_re']]

labels['ori_index'] = [a for a in range(labels.shape[0])]
DeepS_match_df.columns = ['drug_a_name','drug_a_CID']
DS_labels_RE = pd.merge(labels, DeepS_match_df, on = 'drug_a_name', how = 'left')

DeepS_match_df.columns = ['drug_b_name','drug_b_CID']
DS_labels_RE = pd.merge(DS_labels_RE, DeepS_match_df, on = 'drug_b_name', how = 'left')

DeepS_match_df.columns = ['DS','MM']
# 46104

DS_labels_RE.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/01.DeepSynergy/RELABEL.csv', index= False)



# cell check 
DeepS_cellline = list(set(DS_labels_RE.cell_line)) # 39
my_cell_list = list(DC_CELL_DF2.DC_cellname)

[a for a in DeepS_cellline if a not in my_cell_list]
         
            #UWB1289BRCA1 -> UWB1289+BRCA1
            #SKOV3 -> SK-OV-3
            #SW620 -> SW-620
            #NCIH460 -> NCI-H460
            #T47D -> T-47D


def change_cellname(old , new) :
    index_num = list(DS_labels_RE[DS_labels_RE.cell_line==old].index)
    for ind in index_num :
        DS_labels_RE.at[ind, 'cell_line'] = new


change_cellname('UWB1289BRCA1','UWB1289+BRCA1')
change_cellname('SKOV3','SK-OV-3')
change_cellname('SW620','SW-620')
change_cellname('NCIH460','NCI-H460')
change_cellname('T47D','T-47D')





# all matched !
# 만약에 내 데이터와 맞는 내용으로 돌리려면...
# 겹치는걸 일단 찾아야함 
DRSPRING_DATA_ALL = pd.concat([ABCS_train,ABCS_val])
DRSPRING_DATA_Train = copy.deepcopy(ABCS_train)
DRSPRING_DATA_Val = copy.deepcopy(ABCS_val)

DRSPRING_CID = list(set(list(DRSPRING_DATA_ALL.drug_row_CID) + list(DRSPRING_DATA_ALL.drug_col_CID))) # 1342
DRSPRING_CELL = list(set(DRSPRING_DATA_ALL.DC_cellname))

DS_redata_1 = DS_labels_RE[DS_labels_RE.cell_line.isin(DRSPRING_CELL)] # 40190
DS_redata_2 = DS_redata_1[DS_redata_1.drug_a_CID.isin(DRSPRING_CID)] # 28510
DS_redata_3 = DS_redata_2[DS_redata_2.drug_b_CID.isin(DRSPRING_CID)] # 21042


aaa = list(DS_redata_3['drug_a_CID'])
bbb = list(DS_redata_3['drug_b_CID'])
ccc = list(DS_redata_3['cell_line'])

# 306
DS_redata_3['CID_CID'] = [str(int(aaa[i])) + '___' + str(int(bbb[i])) if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i])) for i in range(DS_redata_3.shape[0])]

# 10404 -- duplicated 가 이상한게 아님 
DS_redata_3['CID_CID_CELL'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + ccc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + ccc[i] for i in range(DS_redata_3.shape[0])]

set_data = list(set(DS_redata_3['CID_CID_CELL']))

random.seed(24)
random.shuffle(set_data)
# set_data[0:10]
bins = [round(10404*0.2*a) for a in range(1,5)]
res = np.split(set_data, bins)

tr_idx = list(DS_redata_3[DS_redata_3.CID_CID_CELL.isin(res[2].tolist()+res[3].tolist()+res[4].tolist())]['ori_index'])
val_idx = list(DS_redata_3[DS_redata_3.CID_CID_CELL.isin(res[1].tolist())]['ori_index'])
train_idx = list(DS_redata_3[DS_redata_3.CID_CID_CELL.isin(res[1].tolist()+res[2].tolist()+res[3].tolist()+res[4].tolist())]['ori_index'])
test_idx = list(DS_redata_3[DS_redata_3.CID_CID_CELL.isin(res[0])]['ori_index'])

X_tr = X[tr_idx]
X_val = X[val_idx]
X_train = X[train_idx]
X_test = X[test_idx]

y_tr = labels.iloc[tr_idx]['synergy'].values
y_val = labels.iloc[val_idx]['synergy'].values
y_train = labels.iloc[train_idx]['synergy'].values
y_test = labels.iloc[test_idx]['synergy'].values

if norm == "tanh_norm":
    X_tr, mean, std, mean2, std2, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, mean2, std2, feat_filt = normalize(X_val, mean, std, mean2, std2, 
                                                        feat_filt=feat_filt, norm=norm)
else:
    X_tr, mean, std, feat_filt = normalize(X_tr, norm=norm)
    X_val, mean, std, feat_filt = normalize(X_val, mean, std, feat_filt=feat_filt, norm=norm)


if norm == "tanh_norm":
    X_train, mean, std, mean2, std2, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, mean2, std2, feat_filt = normalize(X_test, mean, std, mean2, std2, 
                                                          feat_filt=feat_filt, norm=norm)
else:
    X_train, mean, std, feat_filt = normalize(X_train, norm=norm)
    X_test, mean, std, feat_filt = normalize(X_test, mean, std, feat_filt=feat_filt, norm=norm)




# 본격적 모델 

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


exec(open(os.path.join(TOOL_PATH,hyperparameter_file)).read()) 



config = tf.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.GPUOptions(allow_growth=True))
set_session(tf.Session(config=config))


model = Sequential()
for i in range(len(layers)):
    if i==0:
        model.add(Dense(layers[i], input_shape=(X_train.shape[1],), activation=act_func, 
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



average_over = 15
mov_av = moving_average(np.array(val_loss), average_over)
smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
epo = np.argmin(smooth_val_loss)


hist = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
test_loss = hist.history['val_loss']





# 데이터만 확인하면 돌리기 가능 




