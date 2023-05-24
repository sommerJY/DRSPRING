STATE OF ART papers

0) prepare test data 

import networkx as nx
import copy 
import pandas as pd 
import numpy as np
import sklearn

NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DATA_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V3_FULL/'
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'


# HS Drug pathway DB 활용 -> 349
print('NETWORK')

hunet_gsp = pd.read_csv(NETWORK_PATH+'HS-DB.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B','SC']

LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)
lm_entrezs = list(LINCS_978.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(lm_entrezs)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(lm_entrezs)] # 3885
hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5]

len(set(list(hnet_L3['G_A']) + list(hnet_L3['G_B']))) # 611

ID_G = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')


SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V5_349_FULL/'

file_name = 'M3V5_349_MISS2_FULL'

MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.Basal_Exp == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] ####################### new targets 
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]


CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ori_col = list( ccle_exp.columns ) # entrez!
for_gene = ori_col[1:]
for_gene2 = [int(a.split('(')[1].split(')')[0]) for a in for_gene]
new_col = ['DepMap_ID']+for_gene2 
ccle_exp.columns = new_col

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
ccle_names = [a for a in ccle_exp2.DrugCombCCLE if type(a) == str]

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.DrugCombCCLE.isin(ccle_names)]

data_ind = list(A_B_C_S_SET.index)
A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)


DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.DrugCombCCLE)] # 38
A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt[['DrugCombCCLE','DC_cellname']], on = 'DrugCombCCLE', how = 'left'  )



# sample number filter # 빈도 확인 

C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq })
C_df = C_df.sort_values('freq')

CELL_CUT = 200 ############ WORK 20 ##############
C_freq_filter = C_df[C_df.freq > CELL_CUT ] 
A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.DrugCombCCLE)))]
DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)
data_ind = list(A_B_C_S_SET_COH.index)

A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')

# 0328 added.... hahahahaha
A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2)

A_B_C_S_SET_SM['ori_index'] = list(A_B_C_S_SET_SM.index)
aaa = list(A_B_C_S_SET_SM['drug_row_CID'])
bbb = list(A_B_C_S_SET_SM['drug_col_CID'])
aa = list(A_B_C_S_SET_SM['ROW_CAN_SMILES'])
bb = list(A_B_C_S_SET_SM['COL_CAN_SMILES'])
cc = list(A_B_C_S_SET_SM['DrugCombCCLE'])

A_B_C_S_SET_SM['CID_CID_CCLE'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + cc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]
A_B_C_S_SET_SM['SM_C_CHECK'] = [aa[i] + '___' + bb[i]+ '___' + cc[i] if aa[i] < bb[i] else bb[i] + '___' + aa[i]+ '___' + cc[i] for i in range(A_B_C_S_SET_SM.shape[0])]

A_B_C_S_SET_SM[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates() # 52152
A_B_C_S_SET_SM[['ROW_CAN_SMILES','COL_CAN_SMILES','DrugCombCCLE']].drop_duplicates() # 52120
len(set(A_B_C_S_SET_SM['CID_CID_CCLE'])) # 51212
len(set(A_B_C_S_SET_SM['SM_C_CHECK'])) # 51160

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({'setset' : data_no_dup.tolist(), 'cell' : data_no_dup_cells })
data_nodup_df2 = data_nodup_df.sort_values('cell')
data_nodup_df2 = data_nodup_df2.reset_index(drop =True)

grouped_df = data_nodup_df2.groupby('cell')


# 10% test 
TrainVal_list = []; Test_list =[]

for i, g in grouped_df:
	if len(g) > CELL_CUT :
		nums = int(.10 * len(g)) 
		bins = []
		g2 = sklearn.utils.shuffle(g, random_state=42)
		for ii in list(range(0, len(g2), nums)):
			if len(bins)< 10 :
				bins.append(ii)
		#
		bins = bins[1:]
		res = np.split(g2, bins)
		TrainVal_list = TrainVal_list + res[0].index.tolist() + res[1].index.tolist() + res[2].index.tolist() + res[3].index.tolist() + res[4].index.tolist() + res[5].index.tolist() + res[6].index.tolist() + res[7].index.tolist() + res[8].index.tolist()   
		Test_list = Test_list + res[9].index.tolist()
	else :
		print(i)


test_no_dup = data_nodup_df2.loc[Test_list] 
ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(test_no_dup.setset)]

ABCS_test.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/my_test.csv', sep = '\t')

ABCS_test = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/my_test.csv', sep = '\t')

test_cids = list(set(list(ABCS_test.drug_row_CID) + list(ABCS_test.drug_col_CID)))








1) DeepSynergy 다시 해보자 휴 시발 

conda create JY_4 python=3.7 -y
conda activate JY_4
which pip
pip install tensorflow
pip install keras 
conda install -c conda-forge matplotlib==3.5.2 -y
conda install seaborn==0.11.2 -y
conda install pandas==1.3.5 -y


문제는 
jCompoundMapper 로 ECFP_6 -> 1309 feat 
ChemoPy 로 physico-chemical : 802 feat 

# DATA normalizing 
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

import keras 
import tensorflow
import keras as K
import tensorflow as tf
from keras import backend
from tensorflow.keras.backend.tensorflow_backend import set_session
from keras.models import Sequential
from keras.layers import Dense, Dropout

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

#indices of training data for hyperparameter selection: fold 2, 3, 4
idx_tr = np.where(np.logical_and(labels['fold']!=test_fold, labels['fold']!=val_fold))
#indices of validation data for hyperparameter selection: fold 1
idx_val = np.where(labels['fold']==val_fold)

#indices of training data for model testing: fold 1, 2, 3, 4
idx_train = np.where(labels['fold']!=test_fold)
#indices of test data for model testing: fold 0
idx_test = np.where(labels['fold']==test_fold)

X_tr = X[idx_tr]
X_val = X[idx_val]
X_train = X[idx_train]
X_test = X[idx_test]

y_tr = labels.iloc[idx_tr]['synergy'].values
y_val = labels.iloc[idx_val]['synergy'].values
y_train = labels.iloc[idx_train]['synergy'].values
y_test = labels.iloc[idx_test]['synergy'].values

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

file = gzip.open(os.path.join(TOOL_PATH,data_file), 'rb')
X_tr, X_val, X_train, X_test, y_tr, y_val, y_train, y_test = pickle.load(file)
file.close()

config = tf.ConfigProto(
         allow_soft_placement=True,
         gpu_options = tf.GPUOptions(allow_growth=True))
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



average_over = 15
mov_av = moving_average(np.array(val_loss), average_over)
smooth_val_loss = np.pad(mov_av, int(average_over/2), mode='edge')
epo = np.argmin(smooth_val_loss)


hist = model.fit(X_train, y_train, epochs=epo, shuffle=True, batch_size=64, validation_data=(X_test, y_test))
test_loss = hist.history['val_loss']

-> 이렇게 돌리면 끝임.. 



##### feature 를 찾기위한 여정 

# gene exp feature
gene_exp_path = '/st06/jiyeonH/11.TOX/GDSC/' 
cell_rma = pd.read_csv(os.path.join(gene_exp_path, 'Cell_line_RMA_proc_basalExp.txt'), sep = '\t')
cell_anno = pd.read_csv(os.path.join(gene_exp_path, 'GDSC_CELL_0707.tsv'), sep = '\t')

cosmic_id = list(set(cell_anno[cell_anno.Name=='A2058']['COSMIC_ID']))[0]

cell_rma[['GENE_SYMBOLS']+['DATA.'+str(cosmic_id)]]








# R 3.6.1

library(farms)
library(affydata)
library(affy)


Data <- ReadAffy()
            AffyBatch object
            size of arrays=744x744 features (388 kb)
            cdf=HG-U219 (49386 affyids)
            number of samples=1018
            number of genes=49386
            annotation=hgu219
            notes=
eset <- rma(Data) # 이게 원래 RMA norm 진행하는 방식임 
write.exprs(eset, file="mydata.txt")
# 근데, 여기서 진행한 내용은 음 quantile norm 이용하고 나서 farms 이용해서 3984 개 유의미한 gene feature 가져왔다는거
# 근데 quantile norm 은 뭐로진행한거지 
# farms uses quantile normalization as default normalization procedure because it is computational efficient
# qFarms is a wrapper function to expresso and uses no background correction and quantile
# normalization as default normalization procedure.
# 옹 그럼 그냥 qFarms 쓰면 됨 
# Data_df = data.frame(Data) -> 이런식으로 활용은 불가능 

1) 기본진행 
eset_qfarm = qFarms(Data)
INIs <- INIcalls(eset_qfarm)
I_data <- getI_Eset(INIs)
I_data_DF = data.frame(I_data) # 1018 27728 으잉 3984 개 아닌데..? 왜 27728 개가 남는겨 

2) 음.. 다시 진행
eset_qfarm_L = qFarms(Data,  laplacian=TRUE)
INIs_L <- INIcalls(eset_qfarm_L)
I_data_L <- getI_Eset(INIs_L)
I_data_L_DF = data.frame(I_data_L) # 1018 * 39868 ???? ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ 아니 대체 왜 

3) RMA 해보고 진행해야하나..?  아닌것 같음. 일단 이렇게 되면 cel 데이터인걸 인식을 못해 
eset_qfarm_3 = qFarms(eset)
INIs_3 <- INIcalls(eset_qfarm_3)
I_data_3 <- getI_Eset(INIs_3)
I_data_DF_3 = data.frame(I_data_3)






일단 튜토리얼 내용은 이랬어. 
# 1) 
data(Dilution)
test_eset_1 <- qFarms(Dilution)
test_eset_1 <- expFarms(Dilution , bgcorrect.method = "rma", pmcorrect.method = "pmonly", 
normalize.method = "constant")

# 2) 
data(Dilution)
test_eset_2 <- qFarms(Dilution, laplacian=TRUE)


# informative / non informative 
data(Dilution)
test_eset_3<-qFarms(Dilution)
INIs <- INIcalls(test_eset_3)
I_data <- getI_Eset(INIs) # affybatch containing only informative probe sets
NI_data <- getNI_Eset(INIs) # affybatch containing only non-informative probe sets

I_data_DF = data.frame(I_data)

자 이게 대체 왜 안되는지 확인해보자 












##### pychem 사용하기 위한 방법 
sys.path.append('/home/jiyeonH/utils/pychem-1.0/src/pychem')
import pychem
from pychem import pychem
from pychem import Chem
from pychem import pybel







/home/jiyeonH/utils
jCMapperCLI.jar

conda install -c conda-forge openbabel





##### 아예 다른 논문에서 가져온 toxicophore 랑 ECFP 확인 
-> then we don't have to make all things 

toxicophore = pd.read_csv(os.path.join(TOOL_PATH, 'DATA_dir', 'toxicophores.csv')) # 1456020* 2276
all_chembl = list(toxicophore['Unnamed: 0'])

dense_feat = pd.read_csv(os.path.join(TOOL_PATH, 'DATA_dir', 'dense.csv')) # 1456020 * 833????

ecfp6_feat = pd.read_csv(os.path.join(TOOL_PATH, 'DATA_dir', 'ECFC6_ES.fpf'), header = None) # 
와 근데 이거 포맷을 읽기가 너무 어려운데? -> 미친 C++ 코드가 뭐 또 따로 있음 시바 
아닠ㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋㅋ
다른 사람들 대체 이걸 어떻게 해결해서 돌린거야 
해결해보자 진짜 후 하 
아니 이거만 해결하면 될것 같은데 


normal ECFP6 format?
from rdkit.Chem import AllChem
from rdkit import Chem, DataStructs

# CHEMBL153534
# -1588406296:1 -1588406285:2 -1578356240:1 -1578236978:1 -1458478794:1 -1435092110:1 -1189050329:1 -817203721:1 -644218583:1 -415800397:1 -392806765:1 -363457846:1 -279414466:1 -275714175:1 -147573862:1 1028:10 1039:5 1044:1 1942248:1 1942259:1 1952819:2 29860525:1 144538471:1 790372325:1 1347803534:1 1426021514:1 1454565402:1 1731858544:1 1866620644:1 1866620655:1 1866620660:1 1876660118:1 1876779375:1 1881396980:1 1962993589:1 2026111378:1 2041133097:1
sm = 'CC1=CC(=CN1C)C2=CSC(=N2)N=C(N)N'
smsm = rdkit.Chem.MolFromSmiles(sm)
ecfp = AllChem.GetMorganFingerprintAsBitVect(smsm, radius = 6)

self.mols = [Chem.MolFromSmiles(i) for i in smiles]
fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius = radius)
array = np.zeros((1,))
DataStructs.ConvertToNumpyArray(fp, array)
bit_headers = ['bit' + str(i) for i in range(2048)]
arr = np.empty((0,2048), int).astype(int)



PyData_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/01.DeepSynergy/DeepSynergy-master/DATA_dir/dataPythonReduced/'

with open(PyData_PATH+'ecfp6.pckl', 'rb') as f:
	ecfp6 = pickle.load(f)

with open(PyData_PATH+'samples.pckl', 'rb') as f:
	samples = pickle.load(f)


python_data = 









##### 
그러고 CID - chemblID 확인 



cid_synonyms = pd.read_csv(os.path.join(TOOL_PATH, 'DATA_dir', 'CID-Synonym-filtered'), sep='\t')
cid_synonyms.columns = ['CID','SYNS']

tmp = pd.DataFrame({"CID" : [1], "SYNS" : ['Acetyl-DL-carnitine']})
cid_synonyms_re = pd.concat([cid_synonyms,tmp])

syns_all = list(cid_synonyms_re['SYNS'])
syns_index = [a for a in range(111632803) if syns_all[a].startswith('CHEMBL') ]

cid_synonyms_cmbl = cid_synonyms_re.iloc[syns_index]
# cid_synonyms_cmbl.to_csv(os.path.join(TOOL_PATH, 'DATA_dir', 'CID_CHEMBL.csv'), sep = '\t')

cid_synonyms_cmbl_filt = cid_synonyms_cmbl[cid_synonyms_cmbl.CID.isin(test_cids)]

set(test_cids) - set(cid_synonyms_cmbl_filt.CID)

{
3549 :['CHEMBL288542'], # SCHEMBL1427458
 122724 : ['CHEMBL301982'], # SCHEMBL14954
 16683013 : ['CHEMBL2068504'], # Stibogluconate 679.80 MW 동일 C(C(C1C(C(O[Sb](=O)(O1)O[Sb]2(=O)OC(C(C(O2)C(=O)O)O)C(CO)O)C(=O)O)O)O)O
 14888 : ['CHEMBL2362016'], # SID : 144205808
 53398697 : ['CHEMBL180022'], # 93473937 랑 CID 동일 SCHEMBL571763 , 	SID: 124950160 == 93473937, CCOC1=C(C=C2C(=C1)N=CC(=C2NC3=CC(=C(C=C3)OCC4=CC=CC=N4)Cl)C#N)NC(=O)C=CCN(C)C
 6505803 : ['CHEMBL109480'], 
 57353643 : ['CHEMBL408194'], # CID 11404337 랑 동일 smiles
 441203 : ['CHEMBL11359'?????], # N.N.Cl[Pt+2]Cl SCHEMBL3789  이것도 cisplatin 이름이긴 함 MW 300.05, SID 따라가면 보통 2767
 9902100 : ['CHEMBL2143507'], # SCHEMBL5725799 CID 9923630
 5702198 : [''], # N.N.Cl[Pt]Cl... 그냥 cisplatin(5460033, N.N.Cl[Pt]Cl) 이랑 동일하게 봐야하나 -> 근데 관련된거 넣어봐도 일단 chembl 에 없어 시박  
 51039095 : ['CHEMBL3184679'], # SCHEMBL63884 # CC1CN(CC(N1)C)C2=CC=C(C=C2)C(=O)NC3=NNC(=C3)CCC4=CC(=CC(=C4)OC)OC -> CID 57430866 동일 
 2713 : ['CHEMBL790'], # C1=CC(=CC=C1NC(=NC(=NCCCCCCN=C(N)N=C(N)NC2=CC=C(C=C2)Cl)N)N)Cl
 159324 : ['CHEMBL289228'], # CN1C=NC=C1C(C2=CC=C(C=C2)Cl)(C3=CC4=C(C=C3)N(C(=O)C=C4C5=CC(=CC=C5)Cl)C)N -> 159324동일
 4829 : ['CHEMBL595'], 
 5288382 : ['CHEMBL278315']}


[ pubchem 기준 ]
cisplatin
5702198
cisplatine
441203  

N.N.Cl[Pt+2]Cl  
23644016   MW 300.05



[ chembl 기준 ]
CHEMBL11359   MW 298.03
CHEMBL2068237 MW 298.03





CHEMBL2068386 MW 424.19
CHEMBL2068387 MW 424.19




##########################
다 떠나서 Oneil 만 진행했으면 되는거였네 
갑자기 현타오네 
