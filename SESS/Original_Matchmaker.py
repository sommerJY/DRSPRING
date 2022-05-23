#MatchMaker: A Deep Learning Framework for Drug Synergy Prediction


#	<	DrugComb data 	>

#286421
MM_DATA = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/DrugCombinationData.tsv', sep = '\t')
CEL_DATA =pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/E-MTAB-3610.sdrf.txt', sep = '\t')

# DC 랑 MM 이랑 왜 차이가 나는가 (filtering)
#-> 일단 우리껀 1.4 가 아니고 얘네는 array 특정 데이터 가지고 함
#-> 그리고 PPI 도 안썼어  
#-> 일단 따라해볼때는 그냥 애매하게 값 대입해주기 




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


# 
import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from helper_funcs import normalize, progress



MM_DATA , drug1_chem.csv , drug2_chem.csv


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


MM_DATA = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/DrugCombinationData.tsv'
drug1 = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/drug1_chem.csv'
drug2 = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/drug2_chem.csv'
CEL_gex = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/cell_line_gex.csv'


chem1, chem2, cell_line, synergies = data_loader(drug1, drug2, CEL_gex, MM_DATA )
# >>> chem1.shape
# (286421, 541)
# >>> chem2.shape
# (286421, 541)
# >>> cell_line.shape
# (286421, 972)
# >>> synergies.shape
# (286421,)


# 데이터 나누기 

train_indx = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/train_inds.txt'
val_indx = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/val_inds.txt'
test_indx = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/test_inds.txt'

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



MM_Model =generate_network(train_data, layers, inDrop, drop)


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








# main.py
# main.py
# main.py
# main.py



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
num_cores = 8
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
                        #device_count = {'CPU' : num_CPU,
                        #                'GPU' : num_GPU}
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
                                earlyStop_patience, modelName,loss_weight)

# 그냥 돌리면 돌아감 ->

model = MatchMaker.generate_network(train_data, layers, inDrop, drop)

if (args.train_test_mode == 1):
    # if we are in training mode
    model = MatchMaker.trainer(model, l_rate, train_data, val_data, max_epoch, batch_size,
                                earlyStop_patience, modelName,loss_weight)
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






















