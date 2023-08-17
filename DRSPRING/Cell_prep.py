

A_B_C_S_SET_SM 





######## cell line rank check (from P02.cellline_ABC.py)


avail_cell_list = list(set(A_B_C_S_SET_SM.DrugCombCCLE))

# total drugcomb data 확인 
DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 

DC_tmp_DF1 = pd.read_csv(DC_PATH+'TOTAL_BLOCK.csv', low_memory=False) # 1469182
DC_tmp_DF1_re = DC_tmp_DF1[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe','quality']]
DC_tmp_DF1_re['drug_row_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_row_id'])]
DC_tmp_DF1_re['drug_col_id_re'] = [float(a) for a in list(DC_tmp_DF1_re['drug_col_id'])]

DC_tmp_DF2 = DC_tmp_DF1_re[DC_tmp_DF1_re['quality'] != 'bad'] # 1457561
DC_tmp_DF3 = DC_tmp_DF2[(DC_tmp_DF2.drug_row_id_re > 0 ) & (DC_tmp_DF2.drug_col_id_re > 0 )] # 740932
DC_tmp_DF4 = DC_tmp_DF3[DC_tmp_DF3.cell_line_id>0].drop_duplicates() # 740884


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






print("DC filtering")

DC_DATA_filter = DC_tmp_DF2[['drug_row_id_re','drug_col_id_re','cell_line_id','synergy_loewe']] # 1457561
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates() # 1374958 -> 1363698

DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_row_id_re > 0] # 1374958 -> 1363698
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_col_id_re > 0] # 751450 -> 740884
DC_DATA_filter4.cell_line_id # unique 295
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re','cell_line_id']].drop_duplicates() # 648516
DC_DATA_filter4[['drug_row_id_re','drug_col_id_re']].drop_duplicates() # 75174
len(list(set(list(DC_DATA_filter4.drug_row_id_re) + list(DC_DATA_filter4.drug_col_id_re)))) # 4327

DC_DATA_filter4 = DC_DATA_filter4.reset_index(drop = False)

DC_DATA_filter5 = pd.merge(DC_DATA_filter4, DC_CELL_DF2[['cell_line_id','DrugCombCCLE']], on = 'cell_line_id', how ='left' )



# 그래서 drugcomb 에서 일단 사용된 내용들 CID 기준 
DC_DATA_filter6 = DC_DATA_filter5[DC_DATA_filter5.DrugCombCCLE.isin(avail_cell_list)]

good_ind = [a for a in range(DC_DRUG_DF_FULL.shape[0]) if type(DC_DRUG_DF_FULL.CAN_SMILES[a]) == str ]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL.loc[good_ind]

DC_DRUG_DF_FULL_filt['leng2'] = [int(a) for a in list(DC_DRUG_DF_FULL_filt.leng)]
DC_DRUG_DF_FULL_filt = DC_DRUG_DF_FULL_filt[DC_DRUG_DF_FULL_filt.leng2 <=50] # 7775

DC_DRUG_DF_FULL_cut = DC_DRUG_DF_FULL_filt[['id','CID','CAN_SMILES']] # DrugComb 에서 combi 할 수 있는 총 CID : 7775개 cid 
DC_DRUG_DF_FULL_cut.columns = ['drug_row_id_re','ROW_CID','ROW_CAN_SMILES']

# 있는 combi 에 대한 CID 붙이기 
DC_re_1 = pd.merge(DC_DATA_filter6, DC_DRUG_DF_FULL_cut, on = 'drug_row_id_re', how = 'left') # 146942

DC_DRUG_DF_FULL_cut.columns = ['drug_col_id_re','COL_CID', 'COL_CAN_SMILES']
DC_re_2 = pd.merge(DC_re_1, DC_DRUG_DF_FULL_cut, on = 'drug_col_id_re', how = 'left')

DC_DRUG_DF_FULL_cut.columns = ['id','CID','CAN_SMILES']

DC_re_3 = DC_re_2[['ROW_CID','COL_CID','DrugCombCCLE']].drop_duplicates()
DC_re_4 = DC_re_3.reset_index(drop = True)




목적 : training 에 활용한 DC 애들 말고, 그 CID 들에 해당하는 모든 조합중에서 우리 train 에 안쓰인 애들. 맞나? 

from itertools import combinations
from itertools import product
from itertools import permutations

DC_all_cids = list(set(DC_DRUG_DF_FULL_cut[DC_DRUG_DF_FULL_cut.CID>0]['CID'])) # 7775개 (SM 있고, 50 이하에 leng 붙는 애들 )
DC_pairs = list(combinations(DC_all_cids, 2)) 
# permutation : 모든 cid - cid 양면 
# combination : unique 한 cid - cid 


# 그러고 나서 DC 안에 있는 모든 CID - CID - Cello triads 조사
IN_DC_pairs_1 = [(DC_re_4.ROW_CID[a] ,DC_re_4.COL_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs_2 = [(DC_re_4.COL_CID[a] ,DC_re_4.ROW_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs = IN_DC_pairs_1 + IN_DC_pairs_2 # 239,044

# 혹시 내가 썼던 모델에 대한 파일에 관련된 애들은 빠져있나? 
# A_B_C_S_SET_ADD_CH  = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.DrugCombCCLE.isin(avail_cell_list)]
# A_B_C_S_SET_ADD_CH = A_B_C_S_SET_ADD_CH.reset_index(drop = True)
# ADD_CHECK_1 = [(A_B_C_S_SET_ADD_CH.drug_row_CID[a] ,A_B_C_S_SET_ADD_CH.drug_col_CID[a], A_B_C_S_SET_ADD_CH.DrugCombCCLE[a]) for a in range(A_B_C_S_SET_ADD_CH.shape[0])]
# ADD_CHECK_2 = [(A_B_C_S_SET_ADD_CH.drug_col_CID[a] ,A_B_C_S_SET_ADD_CH.drug_row_CID[a], A_B_C_S_SET_ADD_CH.DrugCombCCLE[a]) for a in range(A_B_C_S_SET_ADD_CH.shape[0])]
# ADD_CHECK = ADD_CHECK_1 + ADD_CHECK_2
# che = list(set(ADD_CHECK) - set(IN_DC_pairs) )






# 사용하는 cell line 별로 test 대상 선별해서 저장하기 
# 오래걸려! 
# c = 'CVCL_0031' # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import json

# mkdir PRJ_PATH+'VAL/'
CELVAL_PATH = PRJ_PATH + 'VAL/'
os.makedirs(CELVAL_PATH, exist_ok=True)

def save_cell_json (cell_name) :
	this_list = [(a,b,cell_name) for a,b in DC_pairs]
	NOT_in_DC_pairs = set(this_list) - set(IN_DC_pairs)
	VAL_LIST = list(NOT_in_DC_pairs)
	with open(CELVAL_PATH+'{}.json'.format(cell_name), 'w') as f:
		json.dump(VAL_LIST, f)


for cell_name in avail_cell_list :
	save_cell_json(cell_name)






# 1) 그거에 맞게 drug feature 저장하기 -> DC 에 있는 전체 CID 에 대해서 그냥 진행한거니까 
# 1) 그거에 맞게 drug feature 저장하기 -> 이제 다시 안만들어도 됨 그냥 복사하셈 
# 1) 그거에 맞게 drug feature 저장하기 -> 전체 7775 개 
# 1) 그거에 맞게 drug feature 저장하기 

PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)



def get_CHEM(cid, k=1):
	maxNumAtoms = max_len
	smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
	iMol = Chem.MolFromSmiles(smiles.strip())
	iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) 
	iFeature = np.zeros((maxNumAtoms, 64)) ################## feature 크기 고정 
	iFeatureTmp = []
	for atom in iMol.GetAtoms():
		iFeatureTmp.append( atom_feature(atom) )### atom features only
	iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
	iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
	iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
	ADJ = adj_k(np.asarray(iAdj), k)
	return iFeature, ADJ


def atom_feature(atom):
	ar = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
									  ['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'F', 'Br', 'I',
									   'Na', 'Fe', 'B', 'Mg', 'Al', 'Si', 'K', 'H', 'Se', 'Ca',
									   'Zn', 'As', 'Mo', 'V', 'Cu', 'Hg', 'Cr', 'Co', 'Bi','Tc',
									   'Sb', 'Gd', 'Li', 'Ag', 'Au', 'Unknown']) +
					one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
					one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5]) +
					[atom.GetIsAromatic()])    # (36, 8, 5, 5, 9, 1) -> 64 
	return ar



def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
	#print list((map(lambda s: x == s, allowable_set)))
	return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding_unk(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))

def convertAdj(adj):
	dim = len(adj)
	a = adj.flatten()
	b = np.zeros(dim*dim)
	c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
	d = c.reshape((dim, dim))
	return d



def adj_k(adj, k): # 근데 k 가 왜 필요한거지 -> 민지 말에 의하면 
	ret = adj
	for i in range(0, k-1):
		ret = np.dot(ret, adj)  
	return convertAdj(ret)



DC_DRUG_DF_FULL_filt2 = DC_DRUG_DF_FULL_filt.reset_index(drop = True)

max_len = 50
MY_chem_A_feat = torch.empty(size=(DC_DRUG_DF_FULL_filt2.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(DC_DRUG_DF_FULL_filt2.shape[0], max_len, max_len))


for IND in range(DC_DRUG_DF_FULL_filt2.shape[0]): #  
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(DC_DRUG_DF_FULL_filt2.shape[0]) )
		datetime.now()
	#
	DrugA_CID = DC_DRUG_DF_FULL_filt2['CID'][IND]
	#
	k=1
	DrugA_Feat, DrugA_ADJ = get_CHEM(DrugA_CID, k)
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)


SAVE_PATH = CELVAL_PATH

torch.save(MY_chem_A_feat, SAVE_PATH+'DC_ALL.MY_chem_feat.pt')
torch.save(MY_chem_A_adj, SAVE_PATH+'DC_ALL.MY_chem_adj.pt')

DC_DRUG_DF_FULL_filt2.to_csv(SAVE_PATH+'DC_ALL_7555_ORDER.csv')



# 2) 그거에 맞게 MJ EXP feauture 저장하기 # 읽는데 한세월 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 


MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'

# train 에서 활용해서 주는거
#MJ_request_ANS_for_train = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_fugcn_hsc50.csv') 
#MJ_request_ANS_for_train = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_fugcn_hu50.csv') 
# 내가 요청한 전부 

#MJ_request_ANS_FULL = pd.read_csv(MJ_DIR+'PRJ2ver2_EXP_ccle_fugcn_hsc50.csv') # 349 
#MJ_request_ANS_FULL = pd.read_csv(MJ_DIR+'PRJ2ver2_EXP_ccle_fugcn_hu50.csv') # 978 

#MJ_request_ANS_FULL = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_cellall_fugcn_hhh3.csv')
#MJ_request_ANS_FULL_786O = pd.read_csv(MJ_DIR+'PRJ2_EXP_ccle_cell786O_fugcn_hhh3.csv')



set(MJ_request_ANS_FULL.columns) - set(MJ_request_ANS_for_train.columns)
set(MJ_request_ANS_for_train.columns) - set(MJ_request_ANS_FULL.columns)


ORD = [list(MJ_request_ANS_FULL.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_request_ANS_FULL = MJ_request_ANS_FULL.iloc[ORD]


colnana = list(MJ_request_ANS_FULL.columns)[3:]
MJ_tuples_1 = [a for a in colnana if '__' in a]
MJ_tuples_2 = [(a.split('__')[0], a.split('__')[1]) for a in colnana ]

MJ_tup_df = pd.DataFrame()

MJ_tup_df['sample'] = MJ_tuples_1 
MJ_tup_df['tuple'] = MJ_tuples_2

MJ_exp_list = []

for IND in range(MJ_tup_df.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(MJ_tup_df.shape[0]) )
		datetime.now()
	Column = MJ_tup_df['sample'][IND]
	MJ_vector = MJ_request_ANS_FULL[Column].values.tolist()
	MJ_exp_list.append(MJ_vector)


MJ_TENSOR = torch.Tensor(MJ_exp_list)

SAVE_PATH = CELVAL_PATH

torch.save(MJ_TENSOR, SAVE_PATH+'AVAIL_EXP_TOT.pt')

MJ_tup_df.to_csv(SAVE_PATH+'AVAIL_EXP_TOT.csv')










# 3) 그거에 맞게 Target 저장하기 # target 종류 따라서 저장해줘야함 
# 3) 그거에 맞게 Target 저장하기 # 그래도 얼마 안걸림 
# 3) 그거에 맞게 Target 저장하기 
# 3) 그거에 맞게 Target 저장하기 




(2) 14 & 15 NEW TARGET 

DC_TOT_CIDS = DC_DRUG_DF_FULL[['CID','leng']]
DC_TOT_CIDS = DC_TOT_CIDS[DC_TOT_CIDS.CID>0]
total_DC_CIDs = set(DC_TOT_CIDS['CID'])
gene_ids = list(BETA_ORDER_DF.gene_id)

DC_TOT_CIDS = DC_TOT_CIDS.reset_index(drop = True)

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

#A_B_C_S_SET_CIDS = list(set(list(A_B_C_S_SET_ADD.drug_row_CID)+list(A_B_C_S_SET_ADD.drug_col_CID)))
#TARGET_DB_RE = TARGET_DB[TARGET_DB.CID_RE.isin(A_B_C_S_SET_CIDS)] # 없는 애도 데려가야해 
#TARGET_DB_RE = TARGET_DB_RE[TARGET_DB_RE.ENTREZ_RE.isin(gene_ids)]




target_cids = copy.deepcopy(total_DC_CIDs)
gene_ids = list(BETA_ORDER_DF.gene_id)


def get_targets(CID): # 데려 가기로 함 
	if CID in target_cids:
		tmp_df2 = TARGET_DB[TARGET_DB.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		vec = [0] * 978
		#vec = [0] * 349
	return vec


TARGETs = []

for IND in range(DC_TOT_CIDS.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(DC_TOT_CIDS.shape[0]) )
		datetime.now()
	CID = DC_TOT_CIDS['CID'][IND]
	target_vec = get_targets(CID)
	TARGETs.append(target_vec)
	

TARGET_TENSOR = torch.Tensor(TARGETs)

SAVE_PATH = CELVAL_PATH
torch.save(TARGET_TENSOR, SAVE_PATH+'DC_ALL_TARGET.pt')

DC_TOT_CIDS.to_csv(SAVE_PATH+'DC_ALL_TARGET.csv')






# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기 
# 4) 그거에 맞게 Cell Basal 저장하기
# 4) 그거에 맞게 Cell Basal 저장하기 
# avail_cell_list 필요 



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
ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
ccle_cello_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]


DC_CELL_DF_ids = set(DC_CELL_DF.ccle_name) # 1659
ccle_cell_ids = set(ccle_cell_info.DrugCombCCLE) # 1672
# DC_CELL_DF_ids - ccle_cell_ids = 205
# ccle_cell_ids - DC_CELL_DF_ids = 218




DC_CELL_DF3 = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(avail_cell_list)]
DC_CELL_DF3 = DC_CELL_DF3.reset_index(drop = True)

cell_df = ccle_exp3[ccle_exp3.DrugCombCCLE.isin(avail_cell_list)]

# ccle supporting cvcl : len(set(cell_df.DrugCombCCLE)) 25! all in

cell_basal_exp_list = []
# give vector 
for i in range(DC_CELL_DF3.shape[0]) :
	if i%100 == 0 :
		print(str(i)+'/'+str(DC_CELL_DF3.shape[0]) )
		datetime.now()
	cello = DC_CELL_DF3['DrugCombCCLE'][i]
	if cello in ccle_cello_names : 
		ccle_exp_df = cell_df[cell_df.DrugCombCCLE==cello][BETA_ENTREZ_ORDER]
		ccle_exp_vector = ccle_exp_df.values[0].tolist()
		cell_basal_exp_list.append(ccle_exp_vector)
	else : # no worries here. 
		ccle_exp_vector = [0]*978
		#ccle_exp_vector = [0]*349
		cell_basal_exp_list.append(ccle_exp_vector)

cell_base_tensor = torch.Tensor(cell_basal_exp_list)

# 
SAVE_PATH = CELVAL_PATH
torch.save(cell_base_tensor, SAVE_PATH+'AVAIL_CLL_MY_CellBase.pt')

DC_CELL_DF3.to_csv(SAVE_PATH + 'AVAIL_CELL_DF.csv')








# 5) 그거에 맞게 LINCS EXP 는 또 따로 저장하기 

BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)






filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
filter2 = filter1[filter1.is_exemplar_sig==1]
BETA_CP_info_filt = BETA_CP_info[['pert_id','canonical_smiles']].drop_duplicates() # 34419
can_sm_re = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/can_sm_conv', sep = '\t', header = None)
can_sm_re.columns = ['canonical_smiles','CONV_CID']
can_sm_re = can_sm_re.drop_duplicates()
len(set([a for a in BETA_CP_info['pert_id'] if type(a) == str])) # 34419
len(set([a for a in can_sm_re['canonical_smiles'] if type(a) == str])) # 28575
len(set(can_sm_re[can_sm_re.CONV_CID>0]['CONV_CID'])) # 27841

can_sm_re2 = pd.merge(BETA_CP_info_filt, can_sm_re, on = 'canonical_smiles', how = 'left') # 34419 -> 1 sm 1 cid 확인 
can_sm_re3 = can_sm_re2[['pert_id','canonical_smiles','CONV_CID']].drop_duplicates() # 


BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles']].drop_duplicates() # 25903
BETA_MJ_RE_CK = BETA_MJ_RE[['pert_id','SMILES_cid']]
len(set([a for a in BETA_MJ_RE['pert_id'] if type(a) == str])) # 25903
len(set([a for a in BETA_MJ_RE['canonical_smiles'] if type(a) == str])) # 25864
len(set(BETA_MJ_RE_CK[BETA_MJ_RE_CK.SMILES_cid>0]['SMILES_cid'])) # 25642

check = pd.merge(can_sm_re3, BETA_MJ_RE_CK, on = 'pert_id', how = 'left' )
check2 = check[check.CONV_CID != check.SMILES_cid]
check3 = check2[check2.SMILES_cid > 0 ]
check4 = check3[check3.CONV_CID > 0 ] # 순수하게 다른 애들 

pert_id_match = check[check.CONV_CID == check.SMILES_cid][['pert_id','canonical_smiles','CONV_CID']]
conv_win = check2[(check2.CONV_CID >0 ) & ( np.isnan(check2.SMILES_cid)==True)][['pert_id','canonical_smiles','CONV_CID']]
mj_win = check2[(check2.SMILES_cid >0 ) & ( np.isnan(check2.CONV_CID)==True)][['pert_id','canonical_smiles','SMILES_cid']]
nans = check2[(np.isnan(check2.SMILES_cid)==True ) & ( np.isnan(check2.CONV_CID)==True)] # 5995
nans2 = nans[nans.pert_id.isin(filter2.pert_id)]
nans3 = nans2[-nans2.canonical_smiles.isin(['restricted', np.nan])]

pert_id_match.columns = ['pert_id','canonical_smiles','CID'] # 25418,
conv_win.columns = ['pert_id','canonical_smiles','CID'] # 2521,
mj_win.columns =['pert_id','canonical_smiles','CID']

individual_check = check4.reset_index(drop =True)

# 하나하나 결국 살펴봄 
individual_check_conv = individual_check.loc[[0,4,5,6,10,11,12,13,16,17,18,19]+[a for a in range(21,34)]+[36,40,54]][['pert_id','canonical_smiles','CONV_CID']]
individual_check_mj = individual_check.loc[[1,2,3,7,8,9,14,15,20,34,35,37,38,39]+[a for a in range(41,54)]+[55,56,57]][['pert_id','canonical_smiles','SMILES_cid']]
individual_check_conv.columns = ['pert_id','canonical_smiles','CID'] # 28
individual_check_mj.columns = ['pert_id','canonical_smiles','CID'] # 30 

LINCS_PERT_MATCH = pd.concat([pert_id_match, conv_win, mj_win, individual_check_conv,  individual_check_mj]) # 28424
len(set([a for a in LINCS_PERT_MATCH['pert_id'] if type(a) == str])) # 34419 -> 28424
len(set([a for a in LINCS_PERT_MATCH['canonical_smiles'] if type(a) == str])) # 28575 -> 28381
len(set(LINCS_PERT_MATCH[LINCS_PERT_MATCH.CID>0]['CID'])) # 27841 -> 28154
LINCS_PERT_MATCH_cids = list(set(LINCS_PERT_MATCH.CID))

BETA_EXM = pd.merge(filter2, LINCS_PERT_MATCH, on='pert_id', how = 'left')
BETA_EXM2 = BETA_EXM[BETA_EXM.CID > 0] # 128038 # 이건 늘어났음 

BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 128038
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','CID','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 128038

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)] 
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','CID','ccle_name','sig_id']].drop_duplicates() # 111012
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.CID)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]
BETA_CID_CELLO_SIG['CID'] = [int(a) for a in list(BETA_CID_CELLO_SIG['CID']) ] # 111012 




ORD = [list(BETA_BIND.id).index(a) for a in BETA_ENTREZ_ORDER]
BETA_BIND_ORD = BETA_BIND.iloc[ORD]


BETA_CID_CELLO_SIG_re = BETA_CID_CELLO_SIG.reset_index(drop = True)

BETA_CID_CELLO_SIG_tup = [(BETA_CID_CELLO_SIG_re.CID[a], BETA_CID_CELLO_SIG_re.ccle_name[a]) for a in range(BETA_CID_CELLO_SIG_re.shape[0])]
BETA_CID_CELLO_SIG_re['tuple'] = BETA_CID_CELLO_SIG_tup
BETA_CID_CELLO_SIG_tup_re = [(str(BETA_CID_CELLO_SIG_re.CID[a]), BETA_CID_CELLO_SIG_re.ccle_name[a]) for a in range(BETA_CID_CELLO_SIG_re.shape[0])]
BETA_CID_CELLO_SIG_re['tuple_re'] = BETA_CID_CELLO_SIG_tup_re

BETA_CID_CELLO_SIG_re_re = BETA_CID_CELLO_SIG_re[BETA_CID_CELLO_SIG_re['ccle_name'].isin(avail_cell_list)]

BETA_CID_CELLO_SIG_re_re = BETA_CID_CELLO_SIG_re_re.reset_index(drop=True)

LINCS_exp_list = []

for IND in range(BETA_CID_CELLO_SIG_re_re.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(BETA_CID_CELLO_SIG_re_re.shape[0]) )
		datetime.now()
	Column = BETA_CID_CELLO_SIG_re_re['sig_id'][IND]
	L_vector = BETA_BIND_ORD[Column].values.tolist()
	LINCS_exp_list.append(L_vector)


L_TENSOR = torch.Tensor(LINCS_exp_list)

SAVE_PATH = CELVAL_PATH

torch.save(L_TENSOR, SAVE_PATH+'AVAIL_LINCS_EXP_cell.pt')

BETA_CID_CELLO_SIG_re_re.to_csv(SAVE_PATH+'AVAIL_LINCS_EXP_cell.csv')















/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/VAL

'A549_LUNG' -> TB01 Batch 128 / work 12 / 
'DLD1_LARGE_INTESTINE' -> INT04 Batch 64 / work 6 / 
'VCAP_PROSTATE'-> AMD05 Batch 64 / work 4 / 
'ES2_OVARY' -> -> INT05 Batch 64 / work 6 / 



'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'A375_SKIN'
'MDAMB231_BREAST'
'HCT116_LARGE_INTESTINE'
'PC3_PROSTATE'
'SKMEL5_SKIN'
'SKMEL28_SKIN'
'SW620_LARGE_INTESTINE'
'RKO_LARGE_INTESTINE'
'LOVO_LARGE_INTESTINE'
'U251MG_CENTRAL_NERVOUS_SYSTEM'
'OVCAR8_OVARY'
'MCF7_BREAST'
'HT29_LARGE_INTESTINE'
'MDAMB468_BREAST'
'BT474_BREAST'
'A673_BONE'
'T47D_BREAST'
'ZR751_BREAST'
'MELHO_SKIN'
'HS578T_BREAST'









############## cell line 하나만 보기 

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/VAL/'

import glob

CELL_NAME = 'VCAP_PROSTATE'
CELL_pred_file = pd.read_csv(cell_path+'PRED_{}.FINAL_ing.csv'.format(CELL_NAME)) # ing 파일이라서 의미가 음슴 


CELL_pred_file['pred_mean'] = np.mean(CELL_pred_file[['CV0', 'CV1','CV2','CV3','CV4']], 1)
CELL_pred_file['pred_std'] = np.std(CELL_pred_file[['CV0', 'CV1','CV2','CV3','CV4']], 1)

np.max(CELL_pred_file.pred_std)
np.min(CELL_pred_file.pred_std)

근데 지금 그래서 pred 한게 
pred_row_cid = len(set(CELL_pred_file.ROW_CID))






pred_df = pd.DataFrame(columns = ['CV0', 'CV1','CV2','CV3','CV4','ROW_CID','COL_CID','CCLE','Y'])
for fifi in CVCL_files :
	print(fifi.split('/')[8].split('.')[0])
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CELLO', 'PRED_RES', 'Y']]
	pred_df = pd.concat([pred_df, tmp_df2])


merged_CVCL = pred_df.drop_duplicates()

# merged_CVCL.sort_values('PRED_RES')
# merged_CVCL.to_csv(cell_path+'/PRED_RESULT.csv', index=False)
# merged_CVCL = pd.read_csv(cell_path+'/PRED_RESULT.csv')

# 그놈의 색깔 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(PRJ_NAME, MISS_NAME, W_NAME)
all_CellBase_DF = pd.read_csv(cell_path+'/AVAIL_CELL_DF.csv')
all_CellBase_DF['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(all_CellBase_DF['DrugCombCCLE'])]

all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG',  'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','BREAST','LARGE_INTESTINE', 'BONE',  'SKIN', 'PROSTATE',  'OVARY' ] # list(set(test_cell_df['tissue']))
color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"]
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

merged_CVCL_RE = pd.merge(merged_CVCL, all_CellBase_DF[['DC_cellname','DrugCombCello','tissue']], left_on = 'CELLO', right_on='DrugCombCello', how = 'left')


# BOX plot -> 너무 퍼져서 violin plot 이 의미가 없었음 

cell_list = list(set(merged_CVCL_RE.DrugCombCello))

fig, ax = plt.subplots(figsize=(30, 15))
x_pos = [a+1 for a in range(25)]
data_list = []
color_list = []
cell_renames = []
for ind in range(25) : 
	cell = cell_list[ind]
	tmp_per = merged_CVCL_RE[merged_CVCL_RE.DrugCombCello==cell]
	data_list.append(np.array(tmp_per['PRED_RES']))
	color = color_dict[list(set(tmp_per['tissue']))[0]]
	cell_rename = list(set(tmp_per['DC_cellname']))[0]
	cell_renames.append(cell_rename)
	color_list.append(color)

box = ax.boxplot(data_list, patch_artist=True ) # x_pos,
 
for patch, color in zip(box['boxes'], color_list):
	patch.set_facecolor(color)

plt.xticks(x_pos, cell_renames, rotation=90, fontsize=18)

ax.set_xlabel('cell names')
ax.set_ylabel('pred_synergy')
ax.set_title(WORK_NAME)
plt.tight_layout()

plotname = 'FINAL_cell_box'
fig.savefig('{}/{}.png'.format(cell_path, plotname), bbox_inches = 'tight')
plt.close()



############## cell line wise violin plot 

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_MIS2_W20v1/VAL'

import glob

CVCL_files_raw = glob.glob(cell_path+'/PRED_CVCL*')
CVCL_files = [a for a in CVCL_files_raw if '_ing' not in a]
CVCL_files_0553 = [a for a in CVCL_files if '0553' in a]

pred_df = pd.DataFrame(columns = ['ROW_CID','COL_CID','CELLO', 'PRED_RES', 'Y'])
for fifi in CVCL_files :
	print(fifi.split('/')[8].split('.')[0])
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CELLO', 'PRED_RES', 'Y']]
	pred_df = pd.concat([pred_df, tmp_df2])


merged_CVCL = pred_df.drop_duplicates()

# merged_CVCL.sort_values('PRED_RES')
# merged_CVCL.to_csv(cell_path+'/PRED_RESULT.csv', index=False)
# merged_CVCL = pd.read_csv(cell_path+'/PRED_RESULT.csv')

# 그놈의 색깔 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}/'.format(PRJ_NAME, MISS_NAME, W_NAME)
all_CellBase_DF = pd.read_csv(cell_path+'/AVAIL_CELL_DF.csv')
all_CellBase_DF['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(all_CellBase_DF['DrugCombCCLE'])]

all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0395']).index.item(), 'tissue'] = 'PROSTATE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_A442']).index.item(), 'tissue'] = 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
all_CellBase_DF.at[(all_CellBase_DF[all_CellBase_DF.DrugCombCello=='CVCL_0219']).index.item(), 'tissue'] = 'LARGE_INTESTINE'

tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG',  'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','BREAST','LARGE_INTESTINE', 'BONE',  'SKIN', 'PROSTATE',  'OVARY' ] # list(set(test_cell_df['tissue']))
color_set = ["#FFA420","#826C34","#D36E70","#705335","#57A639","#434B4D","#C35831","#B32821","#FAD201","#20603D","#828282","#1E1E1E"]
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

merged_CVCL_RE = pd.merge(merged_CVCL, all_CellBase_DF[['DC_cellname','DrugCombCello','tissue']], left_on = 'CELLO', right_on='DrugCombCello', how = 'left')


# BOX plot -> 너무 퍼져서 violin plot 이 의미가 없었음 

cell_list = list(set(merged_CVCL_RE.DrugCombCello))

fig, ax = plt.subplots(figsize=(30, 15))
x_pos = [a+1 for a in range(25)]
data_list = []
color_list = []
cell_renames = []
for ind in range(25) : 
	cell = cell_list[ind]
	tmp_per = merged_CVCL_RE[merged_CVCL_RE.DrugCombCello==cell]
	data_list.append(np.array(tmp_per['PRED_RES']))
	color = color_dict[list(set(tmp_per['tissue']))[0]]
	cell_rename = list(set(tmp_per['DC_cellname']))[0]
	cell_renames.append(cell_rename)
	color_list.append(color)

box = ax.boxplot(data_list, patch_artist=True ) # x_pos,
 
for patch, color in zip(box['boxes'], color_list):
	patch.set_facecolor(color)

plt.xticks(x_pos, cell_renames, rotation=90, fontsize=18)

ax.set_xlabel('cell names')
ax.set_ylabel('pred_synergy')
ax.set_title(WORK_NAME)
plt.tight_layout()

plotname = 'FINAL_cell_box'
fig.savefig('{}/{}.png'.format(cell_path, plotname), bbox_inches = 'tight')
plt.close()














