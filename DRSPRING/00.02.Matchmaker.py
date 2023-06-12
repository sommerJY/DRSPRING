# Matchmaker 
# 다시 여기부터 시작인가 
# tensorflow 아니었나 


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

TOOL_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/'
TOOL_PATH = '/home/jiyeonH/09.DATA/MATCHMAKER/'

MM_DATA = pd.read_csv(TOOL_PATH + 'DrugCombinationData.tsv', sep = '\t')
CEL_DATA =pd.read_csv(TOOL_PATH + 'E-MTAB-3610.sdrf.txt', sep = '\t')



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




import pandas as pd
import numpy as np
import json
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, concatenate, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
# from helper_funcs import normalize, progress



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



with open(TOOL_PATH + 'drugs_info.json') as json_file :
	MM_drug_info =json.load(json_file)


MM_DATA = TOOL_PATH + 'DrugCombinationData.tsv'
drug1 = TOOL_PATH + 'data/drug1_chem.csv'
drug2 = TOOL_PATH + 'data/drug2_chem.csv'
CEL_gex = TOOL_PATH + 'data/cell_line_gex.csv'

chem1, chem2, cell_line, synergies = data_loader(drug1, drug2, CEL_gex, MM_DATA )

MM_comb_data = pd.read_csv(MM_DATA, sep="\t")
# drug_row , drug_col, cell_line_name, synergy_loewe

# >>> chem1.shape
# (286421, 541) 
# >>> chem2.shape
# (286421, 541)
# >>> cell_line.shape
# (286421, 972) # 978 중에서 고른거
# >>> synergies.shape
# (286421,)


# 여기서 우리 데이터랑 비교 시작해야함 

# final add version 

NETWORK_PATH = '/home/jiyeonH/09.DATA/01.Data/HumanNet/'
LINCS_PATH = '/home/jiyeonH/09.DATA/01.Data/LINCS/' 
DATA_PATH = '/home/jiyeonH/09.DATA/'
DC_PATH = '/home/jiyeonH/09.DATA/01.Data/DrugComb/'
SAVE_PATH = '/home/jiyeonH/09.DATA/'

file_name = 'M3V5_349_MISS2_FULL_RE2' # 0608

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)


# 일단 CID 랑 cell line 부터 

DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 

# Drug info 
with open(DC_PATH+'drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF['id_re'] = [float(a) for a in list(DC_DRUG_DF['id'])]

DC_DRUG_DF_FULL = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_6/DC_DRUG_DF_PC.csv', sep ='\t')

DC_lengs = list(DC_DRUG_DF_FULL.leng)
DC_lengs2 = [int(a) for a in DC_lengs if a!= 'error']


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



# 데이터 아이디 맞는지 확인 

set(MM_comb_data.cell_line_name) - set(DC_CELL_DF2.DC_cellname)
# {'786-0'}
아......
시봉빵봉 
이게 무슨일이죠 
그렇습니다. 얘네 덕에 잃어버린 cell line 하나 건짐 


set(MM_comb_drug) - set(DC_DRUG_DF.dname)

MM_comb_drug_match_name = list(set(list(MM_comb_data.drug_row) + list(MM_comb_data.drug_col))) # 3040
MM_drug_match = pd.DataFrame({'MM_drug' : MM_comb_drug_match_name  })


MM_drug_CID_name = {key : MM_drug_info[key]['name'] for key in MM_drug_info.keys()} # 3952
missing = [a for a in MM_comb_drug_match_name if a not in MM_drug_CID_name.values()] # 100?

MM_drug_CID_name_DF = pd.DataFrame(MM_drug_CID_name, index = [0])
MM_drug_CID_name_DF = MM_drug_CID_name_DF.T
MM_drug_CID_name_DF.columns = ['MM_drug']
MM_drug_CID_name_DF['MM_CID'] = list(MM_drug_CID_name_DF.index)

MM_drug_match_2 = pd.merge(MM_drug_match, MM_drug_CID_name_DF, on='MM_drug', how='left')

for indind in range(MM_drug_match_2.shape[0]) :
    if type(list(MM_drug_match_2.MM_CID)[indind]) != str :
        MM_drug_match_2.at[indind,'MM_CID'] = '0'

MM_drug_match_2['MM_CID'] = [int(a) for a in MM_drug_match_2['MM_CID']]

MM_drug_match_miss = MM_drug_match_2[MM_drug_match_2.MM_CID==0] # 119



DC_CID_list = list(DC_DRUG_DF.cid)
DC_dname = list(DC_DRUG_DF.dname)



miss_name = list(MM_drug_match_miss.MM_drug)
miss_cid = list(MM_drug_match_miss.MM_CID)


# 일단 동일한거 매칭 
for dd in list(MM_drug_match_miss.index):
    dc_name = MM_drug_match_miss.at[dd, 'MM_drug']
    dc_name_l = dc_name.lower()
    dc_name_ul = dc_name[0]+dc_name[1:].lower()
    cid_candidate = []
    for dname in range(8396):
        tmp_name = DC_dname[dname]
        tmp_cid = DC_CID_list[dname]
        if (dc_name == tmp_name) or (dc_name_l == tmp_name) or (dc_name_ul == tmp_name):
            cid_candidate.append(tmp_cid)
    #
    if len(set(cid_candidate)) == 1 :
        MM_drug_match_miss.at[dd, 'MM_CID'] = cid_candidate[0]
    else :
        MM_drug_match_miss.at[dd, 'MM_CID'] = 0  



error = MM_drug_match_miss[MM_drug_match_miss.MM_CID == 0 ] # 56
error_drug = list(error.MM_drug)
error_index = list(error.index)

DC_sym_list = [a for a in DC_DRUG_DF.synonyms]

for err_ind in error_index :
    drug = error.loc[err_ind]['MM_drug']
    dc_name_l_1 = drug.lower()
    dc_name_ul_1 = drug[0]+drug[1:].lower()
    drug_re = drug.replace(' HCL',' HYDROCHLORIDE')
    dc_name_l_2 = drug_re.lower()
    dc_name_ul_2 = drug_re[0]+drug_re[1:].lower()
    cid_candidate = []
    for synns in range(8396):
        tmp = DC_sym_list[synns].split('; ')
        tmp_cid = DC_CID_list[synns]
        if (drug_re in tmp) or (dc_name_l_1 in tmp) or (dc_name_ul_1 in tmp) or (dc_name_l_2 in tmp) or (dc_name_ul_2 in tmp):
            cid_candidate.append(tmp_cid)
    if len(set(cid_candidate)) == 1 :
        MM_drug_match_miss.at[err_ind, 'MM_CID'] = cid_candidate[0]






error = MM_drug_match_miss[MM_drug_match_miss.MM_CID == 0 ] # 56
error_drug = list(error.MM_drug)
error_index = list(error.index)

for err_ind in error_index :
    drug = error.loc[err_ind]['MM_drug']
    dc_name_l_1 = drug.lower()
    dc_name_ul_1 = drug[0]+drug[1:].lower()
    drug_re = drug.replace(' HCL',' HYDROCHLORIDE')
    dc_name_l_2 = drug_re.lower()
    dc_name_ul_2 = drug_re[0]+drug_re[1:].lower()
    cid_candidate = []
    for synns in range(8396):
        tmp_name = DC_dname[synns]
        tmp_name_l = tmp_name.lower()
        tmp_cid = DC_CID_list[synns]
        if (drug_re == tmp_name_l) or (dc_name_l_1 == tmp_name_l) or (dc_name_ul_1 == tmp_name_l) or (dc_name_l_2 == tmp_name_l) or (dc_name_ul_2 == tmp_name_l):
            cid_candidate.append(tmp_cid)
    if len(set(cid_candidate)) == 1 :
        MM_drug_match_miss.at[err_ind, 'MM_CID'] = cid_candidate[0]




error = MM_drug_match_miss[MM_drug_match_miss.MM_CID == 0 ] # 52
error_drug = list(error.MM_drug)
error_index = list(error.index)

for err_ind in error_index :
    drug = error.loc[err_ind]['MM_drug']
    dc_name_l_1 = drug.lower()
    dc_name_ul_1 = drug[0]+drug[1:].lower()
    if len(drug.split(' (')) >1 : 
        drug_re_1 = drug.split(' (')[0]
        drug_re_2 = drug.split(' (')[1]
        dc_name_l_2 = drug_re_1.lower() + ' (' + drug_re_2
        dc_name_ul_2 = drug_re_1[0]+drug_re_1[1:].lower() + ' (' + drug_re_2
        cid_candidate = []
        for synns in range(8396):
            tmp = DC_sym_list[synns].split('; ')
            tmp_cid = DC_CID_list[synns]
            if (drug_re in tmp) or (dc_name_l_1 in tmp) or (dc_name_ul_1 in tmp) or (dc_name_l_2 in tmp) or (dc_name_ul_2 in tmp):
                cid_candidate.append(tmp_cid)
        if len(set(cid_candidate)) == 1 :
            MM_drug_match_miss.at[err_ind, 'MM_CID'] = cid_candidate[0]




error = MM_drug_match_miss[MM_drug_match_miss.MM_CID == 0 ] # 50
error_drug = list(error.MM_drug)
error_index = list(error.index)

for err_ind in error_index :
    drug = error.loc[err_ind]['MM_drug']
    dc_name_l_1 = drug.lower()
    dc_name_ul_1 = drug[0]+drug[1:].lower()
    drug_re = drug.replace(' HCL',' HCl')
    if len(drug.split(' ')) >1 : 
        drug_re_1 = drug_re.split(' ')[0]
        drug_re_2 = drug_re.split(' ')[1]
        dc_name_l_2 = drug_re_1.lower() + ' ' + drug_re_2
        dc_name_ul_2 = drug_re_1[0]+drug_re_1[1:].lower() + ' ' + drug_re_2
        cid_candidate = []
        for synns in range(8396):
            tmp = DC_sym_list[synns].split('; ')
            tmp_cid = DC_CID_list[synns]
            if (drug_re in tmp) or (dc_name_l_1 in tmp) or (dc_name_ul_1 in tmp) or (dc_name_l_2 in tmp) or (dc_name_ul_2 in tmp):
                cid_candidate.append(tmp_cid)
        if len(set(cid_candidate)) == 1 :
            MM_drug_match_miss.at[err_ind, 'MM_CID'] = cid_candidate[0]


error = MM_drug_match_miss[MM_drug_match_miss.MM_CID == 0 ] # 47
error_drug = list(error.MM_drug)
error_index = list(error.index)

for err_ind in error_index :
    drug = error.loc[err_ind]['MM_drug']
    dc_name_l_1 = drug.lower()
    dc_name_ul_1 = drug[0]+drug[1:].lower()
    drug_re = drug.replace('-',' ')
    dc_name_l_2 = drug_re.lower()
    dc_name_ul_2 = drug_re[0]+drug_re[1:].lower()
    drug_rere = drug.replace('-','')
    dc_name_l_3 = drug_re.lower()
    dc_name_ul_3 = drug_re[0]+drug_re[1:].lower()
    cid_candidate = []
    for synns in range(8396):
        tmp = DC_sym_list[synns].split('; ')
        tmp_cid = DC_CID_list[synns]
        if (dc_name_l_1 in tmp) or (dc_name_ul_1 in tmp) or (drug_re in tmp) or (dc_name_l_2 in tmp) or (dc_name_ul_2 in tmp) or (drug_rere in tmp) or (dc_name_l_3 in tmp) or (dc_name_ul_3 in tmp):
            cid_candidate.append(tmp_cid)
    if len(set(cid_candidate)) == 1 :
        MM_drug_match_miss.at[err_ind, 'MM_CID'] = cid_candidate[0]



error = MM_drug_match_miss[MM_drug_match_miss.MM_CID == 0 ] # 37

def give_cid(ori_name, key_name) :
    df_ind = error[error.MM_drug==ori_name].index.item()
    key_cid = MM_drug_CID_name_DF[MM_drug_CID_name_DF.MM_drug==key_name]['MM_CID'].item()
    MM_drug_match_miss.at[df_ind, MM_CID]

give_cid('Flavopiridol', 'FLAVOPIRIDOL');
give_cid('NSC 23766', 'Nsc 23766');
give_cid('BIX 02188', 'BIX02188');
give_cid('AK-77283', '');
give_cid('mefloquine', 'MEFLOQUINE');
give_cid('AG-490 (TYRPHOSTIN B42)', '');
give_cid('NYSTATIN (FUNGICIDIN)', 'nystatin');
give_cid('COSTUNOLIDE', 'Costunolide');
give_cid('Bilobalide', 'BILOBALIDE');
give_cid('ITRACONAZOLE', 'itraconazole');
give_cid('cefdinir', 'CEFDINIR');
give_cid('APATINIB', 'Apatinib');
give_cid('Cloxacillin sodium', 'CLOXACILLIN SODIUM');
give_cid('actinomycin D', 'ACTINOMYCIN D');
give_cid('HYODEOXYCHOLIC ACID', 'HYODEOXYCHOLIC ACID (HDCA)');
give_cid('DACARBAZINE', 'dacarbazine');
give_cid('BREFELDIN A', 'brefeldin A');
give_cid('TENOFOVIR DISOPROXIL FUMARATE', 'Tenofovir Disoproxil Fumarate');
give_cid('CYCLOSPORIN A', 'CYCLOSPORINE');
give_cid('CEFDITOREN PIVOXIL', 'cefditoren pivoxil');
give_cid('RIFAMPIN', ''); ? Rifampicin
give_cid('famotidine', '');
give_cid('OLIGOMYCIN A', '');
give_cid('PACLITAXEL', '');
give_cid('naringenin', '');
give_cid('Brinderdin', '');
give_cid('Apilimod', '');
give_cid('Varenicline', '');
give_cid('SILYMARIN', '');
give_cid('PIRARUBICIN', '');
give_cid('cyclosporin A', '');
give_cid('GELDANAMYCIN', '');
give_cid('Solaraze', '');
give_cid('AT101', '');
give_cid('TAK-700 (ORTERONEL)', '');
give_cid('roxithromycin', '');
give_cid('AZTREONAM', '')



# CID Check 


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



MM_Model = generate_network(train_data, layers, inDrop, drop)


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
                        device_count = {'CPU' : num_CPU,
                                        'GPU' : num_GPU}
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



