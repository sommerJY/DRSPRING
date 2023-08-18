# /st06/jiyeonH/13.DD_SESS/01.PRJ2/fugcn_hhhdt3
# cell line 별 train 짜야할것 같음 


# prep 단계 


######## cell line rank check (from P02.cellline_ABC.py)


avail_cell_list = ['CAMA1_BREAST','VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG']

avail_cell_list = avail_cell_list[0:45]
avail_cell_list = avail_cell_list[45:]

# avail_cell_list = ['786O_KIDNEY']



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
DC_pairs = list(combinations(DC_all_cids, 2))  # 30221425
# permutation : 모든 cid - cid 양면 
# combination : unique 한 cid - cid 




# 그러고 나서 DC 안에 있는 모든 CID - CID - Cello triads 조사
IN_DC_pairs_1 = [(DC_re_4.ROW_CID[a] ,DC_re_4.COL_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs_2 = [(DC_re_4.COL_CID[a] ,DC_re_4.ROW_CID[a], DC_re_4.DrugCombCCLE[a]) for a in range(DC_re_4.shape[0])]
IN_DC_pairs = IN_DC_pairs_1 + IN_DC_pairs_2 # 239,044








# 사용하는 cell line 별로 test 대상 선별해서 저장하기 
# 오래걸려! 
# c = 'CVCL_0031' # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import json

# mkdir PRJ_PATH+'VAL/'
CELVAL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/'
#os.makedirs(CELVAL_PATH, exist_ok=True)

def save_cell_json (cell_name) :
	this_list = [(a,b,cell_name) for a,b in DC_pairs]
	NOT_in_DC_pairs = set(this_list) - set(IN_DC_pairs)
	VAL_LIST = list(NOT_in_DC_pairs)
	with open(CELVAL_PATH+'{}.json'.format(cell_name), 'w') as f:
		json.dump(VAL_LIST, f)


for cell_name in avail_cell_list :
	save_cell_json(cell_name)



																				# 786O 빼먹음 시바 
																				cell_name = '786O_KIDNEY'
																				this_list = [(a,b,cell_name) for a,b in DC_pairs]
																				NOT_in_DC_pairs = set(this_list) - set(IN_DC_pairs)
																				VAL_LIST = list(NOT_in_DC_pairs)
																				with open(CELVAL_PATH+'{}.re.json'.format(cell_name), 'w') as f:
																					json.dump(VAL_LIST, f)







# 1) 그거에 맞게 drug feature 저장하기 -> DC 에 있는 전체 CID 에 대해서 그냥 진행한거니까 
# 1) 그거에 맞게 drug feature 저장하기 -> 이제 다시 안만들어도 됨 그냥 복사하셈 
# 1) 그거에 맞게 drug feature 저장하기 -> 전체 7775 개 
# 1) 그거에 맞게 drug feature 저장하기 

PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)




# 2) 그거에 맞게 MJ EXP feauture 저장하기 # 읽는데 한세월 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 
# 2) 그거에 맞게 MJ EXP feauture 저장하기 


MJ_DIR = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/fugcn_hhhdt3/'

TISSUE_LIST = [
	'BONE','BREAST','CENTRAL_NERVOUS_SYSTEM','HAEMATOPOIETIC_AND_LYMPHOID_TISSUE','KIDNEY',
	'LARGE_INTESTINE','LIVER','LUNG','OVARY', 'PANCREAS','PLEURA','PROSTATE',
	'SKIN','SOFT_TISSUE','STOMACH','URINARY_TRACT']

for TISSUE in TISSUE_LIST:
	print(TISSUE)
	MJ_request_ANS_FULL = pd.read_csv(MJ_DIR + 'PRJ2_EXP_ccle_cellall_fugcn_hhhdt3_tvt_{}.csv'.format(TISSUE))
	#
	#
	ORD = [list(MJ_request_ANS_FULL.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
	MJ_request_ANS_FULL = MJ_request_ANS_FULL.iloc[ORD]
	#
	#
	colnana = list(MJ_request_ANS_FULL.columns)[3:]
	MJ_tuples_1 = [a for a in colnana if '__' in a]
	MJ_tuples_2 = [(a.split('__')[0], a.split('__')[1]) for a in colnana ]
	#
	MJ_tup_df = pd.DataFrame()
	#
	MJ_tup_df['sample'] = MJ_tuples_1 
	MJ_tup_df['tuple'] = MJ_tuples_2
	#
	MJ_exp_list = []
	#
	for IND in range(MJ_tup_df.shape[0]) :
		if IND%100 == 0 :
			print(str(IND)+'/'+str(MJ_tup_df.shape[0]) )
			datetime.now()
		Column = MJ_tup_df['sample'][IND]
		MJ_vector = MJ_request_ANS_FULL[Column].values.tolist()
		MJ_exp_list.append(MJ_vector)
	#
	MJ_TENSOR = torch.Tensor(MJ_exp_list)
	#
	SAVE_PATH = CELVAL_PATH
	#
	torch.save(MJ_TENSOR, SAVE_PATH+'{}.AVAIL_EXP_TOT.pt'.format(TISSUE))
	#
	MJ_tup_df.to_csv(SAVE_PATH+'{}.AVAIL_EXP_TOT.csv'.format(TISSUE))





# 786O
MJ_request_ANS_FULL = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/fugcn_hhhdt3/' + 'PRJ2_EXP_ccle_cellall_fugcn_hhhdt3_tvt_jy786O.csv')
#
#
ORD = [list(MJ_request_ANS_FULL.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_request_ANS_FULL = MJ_request_ANS_FULL.iloc[ORD]
#
#
colnana = list(MJ_request_ANS_FULL.columns)[3:]
MJ_tuples_1 = [a for a in colnana if '__' in a]
MJ_tuples_2 = [(a.split('__')[0], a.split('__')[1]) for a in colnana ]
#
MJ_tup_df = pd.DataFrame()
#
MJ_tup_df['sample'] = MJ_tuples_1 
MJ_tup_df['tuple'] = MJ_tuples_2
#
MJ_exp_list = []
#
for IND in range(MJ_tup_df.shape[0]) :
	if IND%100 == 0 :
		print(str(IND)+'/'+str(MJ_tup_df.shape[0]) )
		datetime.now()
	Column = MJ_tup_df['sample'][IND]
	MJ_vector = MJ_request_ANS_FULL[Column].values.tolist()
	MJ_exp_list.append(MJ_vector)
#
MJ_TENSOR = torch.Tensor(MJ_exp_list)
#
SAVE_PATH = CELVAL_PATH
#
torch.save(MJ_TENSOR, SAVE_PATH+'{}.AVAIL_EXP_TOT.pt'.format('786O'))
#
MJ_tup_df.to_csv(SAVE_PATH+'{}.AVAIL_EXP_TOT.csv'.format('786O'))


























# 3) 그거에 맞게 Target 저장하기 # target 종류 따라서 저장해줘야함 
# 3) 그거에 맞게 Target 저장하기 # 그래도 얼마 안걸림 
# 3) 그거에 맞게 Target 저장하기 
# 3) 그거에 맞게 Target 저장하기 

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


# 786O
torch.save(TARGET_TENSOR, SAVE_PATH+'786O.DC_ALL_TARGET.pt')

DC_TOT_CIDS.to_csv(SAVE_PATH+'786O.DC_ALL_TARGET.csv')





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


avail_cell_list =  ['VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG', 'CAMA1_BREAST']


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


# 786O
torch.save(cell_base_tensor, SAVE_PATH+'786O.AVAIL_CLL_MY_CellBase.pt')
DC_CELL_DF3.to_csv(SAVE_PATH + '786O.AVAIL_CELL_DF.csv')







# 5) 그거에 맞게 LINCS EXP 는 또 따로 저장하기 
# changed version 

LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
BETA_BIND_ORI = torch.load(LINCS_ALL_PATH+'BETA_BIND.349.pt')
BETA_BIND_ORI_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND.349.siglist.csv')
BETA_BIND_NEW = torch.load(LINCS_ALL_PATH+'BETA_BIND2.349.pt')
BETA_BIND_NEW_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND2.349.siglist.csv')


BETA_BIND = torch.concat([BETA_BIND_ORI, BETA_BIND_NEW])
BETA_BIND_SIG_df = pd.concat([BETA_BIND_ORI_DF, BETA_BIND_NEW_DF])
BETA_BIND_SIG_df = BETA_BIND_SIG_df.reset_index(drop = True)

	
LINCS_PERT_MATCH = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')
LINCS_PERT_MATCH = LINCS_PERT_MATCH[['pert_id','CID']]


BETA_BIND_SIG_df_CID = pd.merge(BETA_BIND_SIG_df, LINCS_PERT_MATCH, on = 'pert_id', how = 'left')

BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID[BETA_BIND_SIG_df_CID.CID>0]
BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID2[BETA_BIND_SIG_df_CID2.pert_idose == '10 uM']
BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID2[BETA_BIND_SIG_df_CID2.pert_itime == '24 h']


ccle_info_filt = ccle_info[['stripped_cell_line_name', 'CCLE_Name']]
ccle_info_filt.columns = ['cell_iname','CCLE_Name']

BETA_BIND_SIG_df_CID_ccle = pd.merge(BETA_BIND_SIG_df_CID2, ccle_info_filt, on = 'cell_iname', how='left')
BETA_BIND_SIG_df_CID_ccle = BETA_BIND_SIG_df_CID_ccle[BETA_BIND_SIG_df_CID_ccle.CCLE_Name.isin(avail_cell_list)]


BETA_CID_CELLO_SIG_tup = [(BETA_BIND_SIG_df_CID_ccle.CID[a], BETA_BIND_SIG_df_CID_ccle.CCLE_Name[a]) for a in list(BETA_BIND_SIG_df_CID_ccle.index)]
BETA_BIND_SIG_df_CID_ccle['tuple'] = BETA_CID_CELLO_SIG_tup
BETA_CID_CELLO_SIG_tup_re = [(str(BETA_BIND_SIG_df_CID_ccle.CID[a]), BETA_BIND_SIG_df_CID_ccle.CCLE_Name[a]) for a in list(BETA_BIND_SIG_df_CID_ccle.index)]
BETA_BIND_SIG_df_CID_ccle['tuple_re'] = BETA_CID_CELLO_SIG_tup_re


L_TENSOR = BETA_BIND[BETA_BIND_SIG_df_CID_ccle.index]

SAVE_PATH = CELVAL_PATH

torch.save(L_TENSOR, SAVE_PATH+'AVAIL_LINCS_EXP_cell.pt')

BETA_BIND_SIG_df_CID_ccle.to_csv(SAVE_PATH+'AVAIL_LINCS_EXP_cell.csv')





# 786O
torch.save(L_TENSOR, SAVE_PATH+'786O.AVAIL_LINCS_EXP_cell.pt')
BETA_BIND_SIG_df_CID_ccle.to_csv(SAVE_PATH+'786O.AVAIL_LINCS_EXP_cell.csv')









# 돌리기 전에 json 이랑 
DC_ALL_7555_ORDER.csv
DC_ALL.MY_chem_feat.pt
DC_ALL.MY_chem_adj.pt
AVAIL_LINCS_EXP_cell.csv
AVAIL_LINCS_EXP_cell.pt
AVAIL_EXP_TOT.csv
AVAIL_EXP_TOT.pt
AVAIL_CELL_DF.csv
AVAIL_CLL_MY_CellBase.pt

# tissue wise check 


import rdkit
import os
import sys
import os.path as osp
from math import ceil
import pandas as pd
import numpy as np
import json
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.data import DenseDataLoader
from torch.utils.data import Dataset

from torch_geometric.nn import DenseGCNConv as GCNConv, dense_diff_pool #DiffPool
from torch_geometric.nn import SAGEConv, TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import pickle
import joblib
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import sklearn
from sklearn.preprocessing import OneHotEncoder
onehot_encoder = sklearn.preprocessing.OneHotEncoder
import datetime
from datetime import *
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import mean_squared_error

# Graph convolution model
import torch_geometric.nn as pyg_nn
# Graph utility function
import torch_geometric.utils as pyg_utils
import torch.optim as optim
import torch_geometric.nn.conv
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs

import networkx as nx
import copy
from scipy.sparse import coo_matrix
from scipy import sparse
from scipy import stats
import sklearn.model_selection

import sys
import random

import shutil
import math

import ray
from ray import tune
from functools import partial
from ray.tune.schedulers import ASHAScheduler
from ray.tune import CLIReporter
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune import ExperimentAnalysis

import numpy as np
from rdkit import Chem
from rdkit.Chem.QED import qed
import sys
import os
import pandas as pd
import pubchempy as pcp 

#NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
#LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
#DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
#TARGET_PATH = '/home01/k020a01/01.Data/TARGET/'



NETWORK_PATH = '/home01/k040a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k040a01/01.Data/LINCS/'
DC_PATH = '/home01/k040a01/01.Data/DrugComb/'
TARGET_PATH = '/home01/k040a01/01.Data/TARGET/'



print('NETWORK', flush = True)

# NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

# 349 
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

#MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

#for nn in list(MSSNG):
#	ID_G.add_node(nn)


# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]



# 유전자 이름으로 붙이기 

new_node_names = []
for a in ID_G.nodes():
	tmp_name = LINCS_978[LINCS_978.gene_id == a ]['gene_symbol'].item() # 6118
	new_node_name = str(a) + '__' + tmp_name
	new_node_names = new_node_names + [new_node_name]

mapping = {list(ID_G.nodes())[a]:new_node_names[a] for a in range(len(new_node_names))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



# Graph 확인 

JY_GRAPH = MY_G
JY_GRAPH_ORDER = MY_G.nodes()
JY_ADJ = nx.adjacency_matrix(JY_GRAPH)

JY_ADJ_tmp = torch.LongTensor(JY_ADJ.toarray())
JY_ADJ_IDX = JY_ADJ_tmp.to_sparse().indices()
JY_IDX_WEIGHT = MY_WEIGHT_SCORE


# LINCS exp order 따지기 
BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)


# Cell info

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')










import pandas as pd
import matplotlib.pyplot as plt
from ray.tune import ExperimentAnalysis
##from ray.tune import Analysis
import pickle
import math
import torch
import os






class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, 
	out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_chem, self.G_hiddim_chem)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_Common_dim)])
		self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		##
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_Common_dim)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		##
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_Common_dim+self.G_Common_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		##
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		for conv in self.G_convs_1_chem :
			conv.reset_parameters()
		for bns in self.G_bns_1_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def calc_batch_label (self, syn, feat) :
		batchnum = syn.shape[0]
		nodenum = feat.shape[0]/batchnum
		Num = [a for a in range(batchnum)]
		Rep = np.repeat(Num, nodenum)
		batch_labels = torch.Tensor(Rep).long()
		if torch.cuda.is_available():
			batch_labels = batch_labels.cuda()
		return batch_labels
	#
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, cell, syn ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
		Exp_batch_label = self.calc_batch_label(syn, EXP1)
		#
		for G_1_C in range(len(self.G_convs_1_chem)):
			if G_1_C == len(self.G_convs_1_chem)-1 :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = F.dropout(Drug1_F, p=self.inDrop, training=self.training)
				Drug1_F = self.pool(Drug1_F, Drug_batch_label )
				Drug1_F = self.tanh(Drug1_F)
				G_1_C_out = Drug1_F
			else :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = self.G_bns_1_chem[G_1_C](Drug1_F)
				Drug1_F = F.elu(Drug1_F)
		#
		for G_2_C in range(len(self.G_convs_1_chem)):
			if G_2_C == len(self.G_convs_1_chem)-1 :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = F.dropout(Drug2_F, p=self.inDrop, training=self.training)
				Drug2_F = self.pool(Drug2_F, Drug_batch_label )
				Drug2_F = self.tanh(Drug2_F)
				G_2_C_out = Drug2_F
			else :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = self.G_bns_1_chem[G_2_C](Drug2_F)
				Drug2_F = F.elu(Drug2_F)
		#
		for G_1_E in range(len(self.G_convs_1_exp)):
			if G_1_E == len(self.G_convs_1_exp)-1 :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = F.dropout(EXP1, p=self.inDrop, training=self.training)
				EXP1 = self.pool(EXP1, Exp_batch_label )
				EXP1 = self.tanh(EXP1)
				G_1_E_out = EXP1
			else :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = self.G_bns_1_exp[G_1_E](EXP1)
				EXP1 = F.elu(EXP1)
		#
		for G_2_E in range(len(self.G_convs_1_exp)):
			if G_2_E == len(self.G_convs_1_exp)-1 :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, Exp_batch_label )
				EXP2 = self.tanh(EXP2)
				G_2_E_out = EXP2
			else :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = self.G_bns_1_exp[G_2_E](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.elu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.elu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2 ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.elu(X)
			else :
				X = self.SNPs[L3](X)
		return X




PRJ_PATH = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/'
# PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/'

# CELVAL_PATH = PRJ_PATH + 'CELL_VAL/'
CELVAL_PATH = PRJ_PATH

os.makedirs(CELVAL_PATH, exist_ok = True)

all_chem_DF = pd.read_csv(CELVAL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(CELVAL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(CELVAL_PATH+'DC_ALL.MY_chem_adj.pt')

avail_LINCS_DF = pd.read_csv(CELVAL_PATH+'AVAIL_LINCS_EXP_cell.csv')
avail_LINCS_TS = torch.load(CELVAL_PATH+'AVAIL_LINCS_EXP_cell.pt')

#avail_LINCS_DF = pd.read_csv(CELVAL_PATH+'786O.AVAIL_LINCS_EXP_cell.csv')
#avail_LINCS_TS = torch.load(CELVAL_PATH+'786O.AVAIL_LINCS_EXP_cell.pt')


avail_LINCS_DF['tuple'] = [(int(avail_LINCS_DF['CID'][a]), avail_LINCS_DF['CCLE_Name'][a]) for a in range(avail_LINCS_DF.shape[0]) ]
avail_LINCS_TPs = list(avail_LINCS_DF['tuple'])



targets_DF = pd.read_csv(CELVAL_PATH+'DC_ALL_TARGET.csv')
targets_TS = torch.load(CELVAL_PATH+'DC_ALL_TARGET.pt')


# targets_DF = pd.read_csv(CELVAL_PATH+'786O.DC_ALL_TARGET.csv')
# targets_TS = torch.load(CELVAL_PATH+'786O.DC_ALL_TARGET.pt')


all_CellBase_DF = pd.read_csv(CELVAL_PATH+'AVAIL_CELL_DF.csv')
all_CellBase_TS = torch.load(CELVAL_PATH+'AVAIL_CLL_MY_CellBase.pt')


# all_CellBase_DF = pd.read_csv(CELVAL_PATH+'786O.AVAIL_CELL_DF.csv')
# all_CellBase_TS = torch.load(CELVAL_PATH+'786O.AVAIL_CLL_MY_CellBase.pt')




TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)



# LINCS 값을 우선시 하는 버전 (마치 MISS 2)
def check_exp_f_ts(CID, CELLO) :
	TUPLE = (int(CID), CELLO)
	# Gene EXP
	if TUPLE in avail_LINCS_TPs:
		L_index = avail_LINCS_DF[avail_LINCS_DF['tuple'] == TUPLE].index[0].item() # 이건 나중에 고쳐야해 
		EXP_vector = avail_LINCS_TS[L_index]
	elif TUPLE in mj_exp_TPs :
		M_index = mj_exp_DF[mj_exp_DF['tuple'] == TUPLE].index.item()
		EXP_vector = mj_exp_TS[M_index]
	else :
		print('error')
	#
	# TARGET 
	T_index = targets_DF[targets_DF['CID'] == CID].index.item()
	TG_vector = targets_TS[T_index]
	#
	# BASAL EXP 
	B_index = all_CellBase_DF[all_CellBase_DF.DrugCombCCLE == CELLO].index.item()
	B_vector = all_CellBase_TS[B_index]
	#
	#
	FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector.squeeze().tolist(), B_vector.squeeze().tolist()]).T)
	return FEAT.view(-1,3)





def check_drug_f_ts(CID) :
	INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
	adj_pre = all_chem_feat_adj[INDEX]
	adj_proc = adj_pre.long().to_sparse().indices()
	return all_chem_feat_TS[INDEX], adj_proc



class CellTest_Dataset(Dataset): 
	def __init__(self, tuple_list):
		self.tuple_list = tuple_list
	#
	def __len__(self): 
		return len(self.tuple_list)
	def __getitem__(self, idx): 
		ROW_CID, COL_CID, CELLO = self.tuple_list[idx]
		TUP_1 = (int(ROW_CID), CELLO)
		TUP_2 = (int(COL_CID), CELLO)
		#
		if (TUP_1 in TPs_all) & (TUP_2 in TPs_all) :
			drug1_f , drug1_a = check_drug_f_ts(ROW_CID)
			drug2_f , drug2_a = check_drug_f_ts(COL_CID)
			#
			expA = check_exp_f_ts(ROW_CID, CELLO)
			expB = check_exp_f_ts(COL_CID, CELLO)
			#
			adj = copy.deepcopy(JY_ADJ_IDX).long()
			adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
			#
			cell = torch.zeros(size = (1, 25)) # no need 
			#
			y = torch.Tensor([1]).float().unsqueeze(1)
		#
		else :
			drug1_f , drug1_a = torch.zeros(size = (50, 64)), torch.zeros(size = (2, 1))
			drug2_f , drug2_a = torch.zeros(size = (50, 64)), torch.zeros(size = (2, 1))
			adj = copy.deepcopy(JY_ADJ_IDX).long()
			adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
			#expA, expB = torch.zeros(size = (978, 3)), torch.zeros(size = (978, 3))
			expA, expB = torch.zeros(size = (349, 3)), torch.zeros(size = (349, 3)) ###################################
			cell = torch.zeros(size = (1, 25)) # no need
			y = torch.Tensor([0]).float().unsqueeze(1)
		#
		return ROW_CID, COL_CID, CELLO, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y




def graph_collate_fn(batch):
	tup_list = []
	#
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	expA_list = []
	expB_list = []
	exp_adj_list = []
	exp_adj_w_list = []
	y_list = []
	cell_list = []
	EXP_num_nodes_seen = 0
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	#
	for ROW_CID, COL_CID, CELLO, drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, cell, y in batch :
		tup_list.append( (ROW_CID, COL_CID, CELLO) )
		#
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w.unsqueeze(0))
		y_list.append(torch.Tensor(y))
		cell_list.append(cell)
		EXP_num_nodes_seen += expA.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	#
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	expA_new = torch.cat(expA_list, 0)
	expB_new = torch.cat(expB_list, 0)
	exp_adj_new = torch.cat(exp_adj_list, 1)
	exp_adj_w_new = torch.cat(exp_adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	cell_new = torch.stack(cell_list, 0)
	return tup_list, drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, cell_new, y_new



def Cell_Test(CELLO, MODEL_NAME, use_cuda = True) :
	print(CELLO)
	print(datetime.now(), flush = True)
	#
	#with open(CELVAL_PATH+'786O_KIDNEY.re.json') as f: #########################  
	with open(CELVAL_PATH+'{}.json'.format(CELLO)) as f: #########################  
		lst_check = [tuple(x) for x in json.load(f)]
	#
	#
	tt_df = pd.DataFrame()
	tt_df['tuple'] = lst_check
	tt_df['cid1'] = [str(int(a[0])) for a in lst_check]
	tt_df['cid2'] = [str(int(a[1])) for a in lst_check]
	tt_df['cello'] = [a[2] for a in lst_check]
	#
	tt_df['cid1_celo'] = tt_df.cid1 +'__' +tt_df.cello
	tt_df['cid2_celo'] = tt_df.cid2 +'__' +tt_df.cello
	#
	tt_df_re1 = tt_df[tt_df.cid1_celo.isin(TPs_all_2)]
	tt_df_re2 = tt_df_re1[tt_df_re1.cid2_celo.isin(TPs_all_2)]
	#
	tg_lists = [str(int(a)) for a in list(set(TARGET_DB.CID))]
	tt_df_re3 = tt_df_re2[tt_df_re2.cid1.isin(tg_lists)] # 5289702
	tt_df_re4 = tt_df_re3[tt_df_re3.cid2.isin(tg_lists)] # 2359767
	#
	tt_df_re4 = tt_df_re4.reset_index(drop=True)
	#
	tuple_list = tt_df_re4['tuple']
	#
	#
	#
	#
	#
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item()
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#      
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, 64 , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
				inDrop, Drop
				) 
	#
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		state_dict = torch.load(MODEL_NAME) #### change ! 
	else:
		state_dict = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
	# 
	# 
	print("state_dict_done", flush = True)
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)
	#
	print("state_load_done", flush = True)
	#
		#
	best_model.to(device)
	best_model.eval()
	#
	dataset = CellTest_Dataset(tuple_list)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 2048 , collate_fn = graph_collate_fn, shuffle = False, num_workers = 128 )# , num_workers=my_config['config/n_workers'].item()
	# 약간 헷갈리는게, 1gpu 기준으로 2048 도 작음 --> 그리고 자꾸 0 % 로 내려감 -> 4096 -> 8192 //-> 16384 -> 32768
	CELL_PRED_DF = pd.DataFrame(columns = ['PRED','ROW_CID','COL_CID','CCLE','Y'])
	CELL_PRED_DF.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL_ing.csv'.format(CELLO), index=False)
	#
	with torch.no_grad():
		for batch_idx_t, (tup_list, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(dataloader):
			print("{} / {}".format(batch_idx_t, len(dataloader)) , flush = True)
			print(datetime.now(), flush = True)
			list_ROW_CID = [a[0] for a in tup_list]
			list_COL_CID = [a[1] for a in tup_list]
			list_CELLO = [a[2] for a in tup_list]
			#
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			#
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y) 
			outputs = output.squeeze().tolist() # [output.squeeze().item()]
			#print(outputs)
			#
			tmp_df = pd.DataFrame({
			'PRED': outputs,
			'ROW_CID' : list_ROW_CID,
			'COL_CID' : list_COL_CID,
			'CCLE' : list_CELLO,
			'Y' : y.squeeze().tolist()
			})
			CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])
			tmp_df.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL_ing.csv'.format(CELLO), mode='a', index=False, header = False)
	return CELL_PRED_DF


PRJ_PATH = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/'
# PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/'

os.makedirs( os.path.join(PRJ_PATH,'CELL_VAL'), exist_ok = True)


OLD_PATH = '/home01/k040a01/02.M3V6/M3V6_W204_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V6_W204_349_MIS2')))

my_config = ANA_DF_CSV.loc[0]



KEY_EPC = 963
checkpoint = "checkpoint_"+str(KEY_EPC).zfill(6)


CKP_PATH = os.path.join( ANA_DF_CSV.logdir.item(), checkpoint, 'checkpoint')



# 다시 돌려보기 
'BONE'
1
'PLEURA'
1
'PROSTATE'
2
'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
5
'KIDNEY'
5
'CENTRAL_NERVOUS_SYSTEM'
6
'LARGE_INTESTINE'
9
'OVARY'
12
'LUNG'
13
'BREAST'
16
'SKIN'
22




avail_cell_dict = {
	'PROSTATE': ['VCAP', 'PC3'], 
	'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE': ['UHO1', 'K562', 'L1236', 'SR786', 'RPMI8226'], 
	'BREAST': ['BT549', 'BT474', 'MDAMB361', 'MDAMB231', 'UACC812', 'KPL1', 'MDAMB468', 'MDAMB175VII', 'HCC1500', 'HCC1419', 'MDAMB436', 'ZR751', 'MCF7', 'HS578T', 'T47D', 'CAMA1'], 
	'CENTRAL_NERVOUS_SYSTEM': ['T98G', 'SF539', 'U251MG', 'SNB75', 'SF268', 'SF295'], 
	'LUNG': ['A549', 'A427', 'HOP92', 'NCIH23', 'NCIH1650', 'NCIH520', 'HOP62', 'NCIH460', 'NCIH2122', 'NCIH226', 'EKVX', 'SKMES1', 'NCIH522'], 
	'LARGE_INTESTINE': ['SW620', 'LOVO', 'RKO', 'KM12', 'HCT15', 'HCT116', 'HT29', 'DLD1', 'SW837'], 
	'OVARY': ['NIHOVCAR3', 'OVCAR4', 'OVCAR5', 'IGROV1', 'OVCAR8', 'PA1', 'UWB1289', 'SKOV3', 'CAOV3', 'ES2', 'A2780', 'OV90'], 
	'SKIN': ['SKMEL5', 'HT144', 'RVH421', 'SKMEL28', 'A375', 'A101D', 'COLO792', 'UACC257', 'COLO800', 'MEWO', 'MELHO', 'A2058', 'RPMI7951', 'IPC298', 'MALME3M', 'SKMEL2', 'UACC62', 'LOXIMVI', 'WM115', 'G361', 'SKMEL30', 'COLO829'], 
	'BONE': ['A673'], 
	'KIDNEY': ['CAKI1', 'UO31', 'ACHN', 'A498', '786O'], 
	'PLEURA': ['MSTO211H']}













for tissue in avail_cell_dict.keys():
	avail_cell_list = avail_cell_dict[tissue]
	TISSUE = tissue
	mj_exp_DF = pd.read_csv(CELVAL_PATH+'{}.AVAIL_EXP_TOT.csv'.format(TISSUE))
	mj_exp_TS = torch.load(CELVAL_PATH+'{}.AVAIL_EXP_TOT.pt'.format(TISSUE))
	mj_exp_DF['tuple'] = [( int(a.split('__')[0]) , a.split('__')[1]) for a in mj_exp_DF['sample']]
	mj_exp_TPs = list(mj_exp_DF['tuple'])
	TPs_all = avail_LINCS_TPs + mj_exp_TPs
	TPs_all_2 = [str(a[0])+"__"+a[1] for a in TPs_all]
	for CELLO in avail_cell_list : 
		CELLO_re = CELLO+'_'+tissue
		MODEL_NAME = CKP_PATH
		CELL_PRED_DF = Cell_Test(CELLO_re, MODEL_NAME, use_cuda = True)
		CELL_PRED_DF.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL.csv'.format(CELLO_re), index=False)





# 786O
tissue = 'KIDNEY'
TISSUE = 'KIDNEY'
mj_exp_DF = pd.read_csv(CELVAL_PATH+'786O.AVAIL_EXP_TOT.csv')
mj_exp_TS = torch.load(CELVAL_PATH+'786O.AVAIL_EXP_TOT.pt')
mj_exp_DF['tuple'] = [( int(a.split('__')[0]) , a.split('__')[1]) for a in mj_exp_DF['sample']]
mj_exp_TPs = list(mj_exp_DF['tuple'])
TPs_all = avail_LINCS_TPs + mj_exp_TPs
TPs_all_2 = [str(a[0])+"__"+a[1] for a in TPs_all]
CELLO = '786O'
CELLO_re = CELLO+'_'+tissue
MODEL_NAME = CKP_PATH
CELL_PRED_DF = Cell_Test(CELLO_re, MODEL_NAME, use_cuda = True)
CELL_PRED_DF.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL.csv'.format(CELLO_re), index=False)








####################################################
그림그리기 



#######################
cell visualization 
#######################


# 그놈의 색깔 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/'

cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/'
cell_path2 = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/'

# kisti 
cell_path = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/CELL_VAL' 



# 다시 진행해주기 
import glob
from matplotlib import colors as mcolors

CVCL_files_raw = glob.glob(cell_path+'/PRED_*.FINAL.csv')
tissues = ['_'.join(a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[1:]) for a in CVCL_files_raw]
strips = [a.split('/')[-1].split('.')[0].split('PRED_')[1].split('_')[0] for a in CVCL_files_raw]

pred_df = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])
pred_df_pos = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])
pred_df_neg = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y','tissue','strip'])

for indd in range(len(CVCL_files_raw)) :
	fifi = CVCL_files_raw[indd]
	tiss = tissues[indd]
	stripname = strips[indd]
	print(stripname)
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
	tmp_df2['tissue'] = tiss
	tmp_df2['strip'] = stripname
	#tmp_df_pos = tmp_df2[tmp_df2.PRED>0]
	#tmp_df_neg = tmp_df2[tmp_df2.PRED<0]
	#pred_df_pos = pd.concat([pred_df_pos, tmp_df_pos])
	#pred_df_neg = pd.concat([pred_df_neg, tmp_df_neg])
	pred_df = pd.concat([pred_df, tmp_df2])
	
	#tmp_df3.to_csv(cell_path + 'TOP.{}.csv'.format(fifi.split('/')[8].split('.')[0]))



tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}


my_order = pred_df.groupby(by=["CCLE"])["PRED"].median().iloc[::-1].index
my_order2 = my_order.iloc[::-1].index

order_tissue = ['_'.join(a.split('_')[1:]) for a in my_order2]
order_tissue_col = [color_dict[a] for a in order_tissue]

fig, ax = plt.subplots(figsize=(40, 8))
sns.violinplot(ax = ax, data  = pred_df, x = 'CCLE', y = 'PRED', linewidth=1,  edgecolor="dimgrey", order=my_order2)
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
for tiss_num in range(len(my_order2)) : 
	violins[tiss_num].set_facecolor(mcolors.to_rgba(order_tissue_col[tiss_num]))

ax.set_xlabel('cell names', fontsize=20)
ax.set_ylabel('pred_synergy', fontsize=20)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.grid(False)
plt.savefig(os.path.join(PRJ_PATH,'cell_pred.png'), dpi = 300)








# positive only plot 

pred_df_pos
my_order2 = pred_df_pos.groupby(by=["tissue"])["PRED"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25,14))

sns.violinplot(ax = ax, data  = pred_df_pos, x = 'tissue', y = 'PRED', 
linewidth=2,  edgecolor="black", inner = None, order = my_order2) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 

for vv in range(11) : 
	tissue = my_order2[vv]
	violins[vv].set_facecolor(mcolors.to_rgba(color_dict[tissue], 0.8))


ax.set_xlabel('tissue', fontsize=15)
ax.set_ylabel('predicted value', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=15) # ax.get_xticklabels()
ax.set_yticklabels(ax.get_yticks(),  fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
#plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(cell_path,'pred_check.tissue2.png'), dpi = 300)

plt.close()



# positive only plot  re 


pred_df['tissue'] = pred_df.CCLE.apply(lambda x : '_'.join(x.split('_')[1:]) )

my_order2 = pred_df.groupby(by=["tissue"])["PRED"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25,14))

sns.violinplot(ax = ax, data  = pred_df, x = 'tissue', y = 'PRED', 
linewidth=2,  edgecolor="black", inner = None, order = my_order2) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 

for vv in range(11) : 
	tissue = my_order2[vv]
	violins[vv].set_facecolor(mcolors.to_rgba(color_dict[tissue], 0.8))


ax.set_xlabel('tissue', fontsize=15)
ax.set_ylabel('predicted value', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=15) # ax.get_xticklabels()
ax.set_yticklabels(ax.get_yticks(),  fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
#plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(cell_path,'pred_check.tissue3.png'), dpi = 300)

plt.close()




특정 cell line 에서의 위치를 따지는 경우
아예 이걸 cell line 별로 그려놓는게 나을듯 



WORK_NAME = 'WORK_203' # 349
W_NAME = 'W203'
PRJ_NAME = 'M3V6'
MJ_NAME = 'M3V6'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

ABCS_test_result = pd.read_csv(PRJ_PATH+'ABCS_test_result.csv')

ABCS_test_result['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(ABCS_test_result['CELL'])]

test_cell_df = pd.DataFrame({'DC_cellname' : list(set(ABCS_test_result.DC_cellname))})

cell_P = []
cell_S = []
cell_num = []

for cell in list(test_cell_df.DC_cellname) :
	tmp_test_re = ABCS_test_result[ABCS_test_result.DC_cellname == cell]
	cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED_3)
	cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED_3)
	cell_nums = tmp_test_re.shape[0]
	cell_P.append(cell_P_corr)
	cell_S.append(cell_S_corr)
	cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num



ABCS_test_result_tissue = ABCS_test_result[ABCS_test_result.tissue=='LARGE_INTESTINE']

ABCS_test_result_filt1 = ABCS_test_result_tissue[['DC_cellname','CELL','tissue','ANS']]
ABCS_test_result_filt1.columns = ['DC_cellname','CELL','tissue','value']
ABCS_test_result_filt1['label'] = 'tissue'

ABCS_test_result_cell = ABCS_test_result_tissue[ABCS_test_result_tissue.CELL== 'SW837_LARGE_INTESTINE']

ABCS_test_result_filt2 = ABCS_test_result_cell[['DC_cellname','CELL','tissue','ANS']]
ABCS_test_result_filt2.columns = ['DC_cellname','CELL','tissue','value']
ABCS_test_result_filt2['label'] = 'cell'


for_split_vio = pd.concat([ABCS_test_result_filt1, ABCS_test_result_filt2])

fig, ax = plt.subplots(figsize=(8, 8))
sns.violinplot(
	ax =ax, data  = for_split_vio,
	x = 'label', y = 'value',
	inner = 'quart',
	linewidth=2,  edgecolor="black")


violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 # if i%2 == 0
violins[0].set_facecolor(mcolors.to_rgba('#ffcd36'))
violins[1].set_facecolor(mcolors.to_rgba('grey'))

plt.axhline(y=6.77,color = 'r') 


ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Loewe', fontsize=20)
ax.set_xticks(ax.get_xticks())
#ax.set_xticklabels(np.array(['LARGE_INTESTINE','SW837']), fontsize = 25, rotation = 90)
ax.set_xticklabels(['',''])
ax.tick_params(axis='y', which='major', labelsize=20 )
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/cell_compare.intestine.png'), dpi = 300)
plt.close()



pred_pos_10_re = pred_pos_10.reset_index(drop = True)

res_dict = {}

all_cells = list(set(ABCS_test_result.CELL))

for cell in all_cells : 
	quart = ABCS_test_result[ABCS_test_result.CELL==cell]['ANS'].describe()['75%']
	pred_tmp = pred_pos_10_re[pred_pos_10_re.CCLE == cell]
	pred_tmp_re = pred_tmp[pred_tmp.PRED>quart]
	indx = list(pred_tmp_re.index)
	res_dict[cell] = indx








# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
# cell line 별로 살펴보는 방법.... cell line 이 92 개라는걸 생각하고 진행해주렴
tissue 별로 상위 cell line 보는건 어떻게 생각함?
괜찮은듯?


indd = 78 # IPC298

fifi = CVCL_files_raw[indd]
tiss = tissues[indd]
stripname = strips[indd]
print(stripname)
tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
tmp_df2['tissue'] = tiss
tmp_df2['strip'] = stripname
#tmp_df_pos = tmp_df2[tmp_df2.PRED>0]
#tmp_df_neg = tmp_df2[tmp_df2.PRED<0]
#pred_df_pos = pd.concat([pred_df_pos, tmp_df_pos])
#pred_df_neg = pd.concat([pred_df_neg, tmp_df_neg])
#pred_df = pd.concat([pred_df, tmp_df2])


ori_cid_cid = list(set(A_B_C_S_SET_COH2.CID_CID)) # 9448
ori_cid_cid2 = list(set(A_B_C_S_SET.CID_CID)) # 9607  # 아예 전부 제거? 

cid_a = list(tmp_df2['ROW_CID'])
cid_b = list(tmp_df2['COL_CID'])

tmp_df2['CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(tmp_df2.shape[0])]

tmp_df3 = tmp_df2[tmp_df2.CID_CID.isin(ori_cid_cid)==False]
tmp_df4 = tmp_df2[tmp_df2.CID_CID.isin(ori_cid_cid2)==False]

tmp_df5 = tmp_df4.sort_values('PRED', ascending = False)

tmp_df5.iloc[0:20,]
tmp_df5.iloc[-20:,]



10126189
11178236
71462653


my_ans = ABCS_test_result[ABCS_test_result.CELL=='IPC298_SKIN']
my_ans = ABCS_test_result[ABCS_test_result.CELL=='IPC298_SKIN']

my_ans[(my_ans.CID_A==10126189)]
my_ans[(my_ans.CID_B==10126189)]




									fifi = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/PRED_COLO829_SKIN.FINAL.csv'
									tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
									tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
									#pred_df = pd.concat([pred_df, tmp_df2])
									tmp_df3 = tmp_df2[tmp_df2.PRED>0]
									tmp_df3.to_csv(cell_path + 'TOP.{}.csv'.format(fifi.split('/')[8].split('.')[0]))


									fifi = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/PRED_WM115_SKIN.FINAL.csv'
									tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
									tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
									#pred_df = pd.concat([pred_df, tmp_df2])
									tmp_df3 = tmp_df2[tmp_df2.PRED>0]
									tmp_df3.to_csv(cell_path + 'TOP.{}.csv'.format(fifi.split('/')[8].split('.')[0]))



merged_CVCL = pred_df.drop_duplicates()

# 아.. 저장은 좀 에바였다. 너무 크다 
# merged_CVCL = merged_CVCL.sort_values('PRED')
# merged_CVCL.to_csv(cell_path+'/PRED_RESULT_MERGED.csv', index=False)
# merged_CVCL = pd.read_csv(cell_path+'/PRED_RESULT.csv')


all_CellBase_DF = pd.read_csv(cell_path2+'AVAIL_CELL_DF.csv')
all_CellBase_DF['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(all_CellBase_DF['DrugCombCCLE'])]


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}



# 아... 너무 커서 TB 기준에서는 그림이 아무것도 안그려짐 
# 적어도 INT 나 그런데로 가야할듯 
# 일단 높은 값으로 나오는거 저장하기 



ccle_list = list(pred_df['CCLE'])


merged_CVCL_RE = pd.merge(pred_df, all_CellBase_DF[['DC_cellname','DrugCombCCLE','tissue']], left_on = 'CCLE', right_on='DrugCombCCLE', how = 'left')

merged_CVCL_RE.DC_cellname = merged_CVCL_RE.DC_cellname.astype('category')

merged_CVCL_RE_filt = merged_CVCL_RE[['CCLE','DC_cellname','tissue']].drop_duplicates()
# tiss_cat = merged_CVCL_RE_filt.DC_cellname.cat.categories













# BOX plot -> 너무 퍼져서 violin plot 이 의미가 없었음 
PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/for_png/'
PRJ_PATH = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/CELL_VAL'


#cell_list = list(set(merged_CVCL_RE.DrugCombCCLE)) # 
my_order = pred_df.groupby(by=["CCLE"])["PRED"].median().iloc[::-1].index
my_order2 = my_order.iloc[::-1].index

# 이게 무슨 순서더라 
# 안돼 그래도 median 순서로 가고싶음 
#order = ['A-673', 'MDA-MB-361', 'CAMA-1', 'MDA-MB-175-VII', 'L-1236', 'UACC-812', 'HCC1500', 'HCC1419', 'U-HO1', 'T98G', 'BT-474', 'LOX IMVI', 'ZR751', 'NCIH2122', 'MDAMB436', 'COLO 829', 'KPL1', 'EKVX', 'A498', 'MDA-MB-231', 'HT144', 'T-47D', 'A375', 'RPMI7951', 'A2058', 'G-361', 'KM12', 'MCF7', 'CAOV3', 'COLO 800', 'Mel Ho', 'NCI-H226', 'UACC-257', 'OVCAR3', 'SF-539', 'PC-3', 'CAKI-1', 'UACC62', 'SR', 'UO-31', 'NCI-H522', 'SK-MEL-5', 'HOP-92', 'ACHN', '786O', 'HS 578T', 'RVH-421', 'K-562', 'HOP-62', 'OVCAR-5', 'MSTO', 'U251', 'COLO 792', 'NCIH23', 'OV90', 'LOVO', 'RPMI-8226', 'A101D', 'NCIH520', 'SK-MEL-2', 'MALME-3M', 'IPC-298', 'HT29', 'SF-295', 'UWB1289', 'SW-620', 'OVCAR-4', 'SKMES1', 'SW837', 'MDA-MB-468', 'HCT-15', 'OVCAR-8', 'SNB-75', 'SF-268', 'HCT116', 'BT-549', 'A549', 'IGROV1', 'A427', 'SK-MEL-28', 'SK-OV-3', 'NCIH1650', 'WM115', 'NCI-H460', 'MeWo', 'ES2', 'SKMEL30', 'A2780', 'RKO', 'VCAP', 'PA1', 'DLD1']
#order2= ['A673_BONE', 'MDAMB361_BREAST', 'CAMA1_BREAST', 'MDAMB175VII_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC812_BREAST', 'HCC1500_BREAST', 'HCC1419_BREAST', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'BT474_BREAST', 'LOXIMVI_SKIN', 'ZR751_BREAST', 'NCIH2122_LUNG', 'MDAMB436_BREAST', 'COLO829_SKIN', 'KPL1_BREAST', 'EKVX_LUNG', 'A498_KIDNEY', 'MDAMB231_BREAST', 'HT144_SKIN', 'T47D_BREAST', 'A375_SKIN', 'RPMI7951_SKIN', 'A2058_SKIN', 'G361_SKIN', 'KM12_LARGE_INTESTINE', 'MCF7_BREAST', 'CAOV3_OVARY', 'COLO800_SKIN', 'MELHO_SKIN', 'NCIH226_LUNG', 'UACC257_SKIN', 'NIHOVCAR3_OVARY', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'PC3_PROSTATE', 'CAKI1_KIDNEY', 'UACC62_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UO31_KIDNEY', 'NCIH522_LUNG', 'SKMEL5_SKIN', 'HOP92_LUNG', 'ACHN_KIDNEY', '786O_KIDNEY', 'HS578T_BREAST', 'RVH421_SKIN', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HOP62_LUNG', 'OVCAR5_OVARY', 'MSTO211H_PLEURA', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'COLO792_SKIN', 'NCIH23_LUNG', 'OV90_OVARY', 'LOVO_LARGE_INTESTINE', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'NCIH520_LUNG', 'SKMEL2_SKIN', 'MALME3M_SKIN', 'IPC298_SKIN', 'HT29_LARGE_INTESTINE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'UWB1289_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'SKMES1_LUNG', 'SW837_LARGE_INTESTINE', 'MDAMB468_BREAST', 'HCT15_LARGE_INTESTINE', 'OVCAR8_OVARY', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'BT549_BREAST', 'A549_LUNG', 'IGROV1_OVARY', 'A427_LUNG', 'SKMEL28_SKIN', 'SKOV3_OVARY', 'NCIH1650_LUNG', 'WM115_SKIN', 'NCIH460_LUNG', 'MEWO_SKIN', 'ES2_OVARY', 'SKMEL30_SKIN', 'A2780_OVARY', 'RKO_LARGE_INTESTINE', 'VCAP_PROSTATE', 'PA1_OVARY', 'DLD1_LARGE_INTESTINE']



order_tissue = ['_'.join(a.split('_')[1:]) for a in my_order]
order_tissue_col = [color_dict[a] for a in order_tissue]


fig, ax = plt.subplots(figsize=(40, 8))
sns.violinplot(ax = ax, data  = pred_df, x = 'CCLE', y = 'PRED', linewidth=1,  
edgecolor="dimgrey", order=my_order)
violins = [c for i,c in enumerate(ax.collections) if i%2 == 0] # pollycollection 가져오기 위함 
for tiss_num in range(len(my_order)) : 
	violins[tiss_num].set_facecolor(mcolors.to_rgba(order_tissue_col[tiss_num]))

ax.set_xlabel('cell names', fontsize=20)
ax.set_ylabel('pred_synergy', fontsize=20)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=20)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.grid(False)
# plt.savefig(os.path.join(PRJ_PATH,'cell_pred.png'), dpi = 300)
fig.savefig('{}/{}.pdf'.format(PRJ_PATH, 'cell_pred'), format="pdf", bbox_inches = 'tight')
fig.savefig('{}/{}.pdf'.format(PRJ_PATH, 'cell_pred1'), format="pdf", bbox_inches = 'tight')

plt.close()

# row num : 84,153,008














# positive only violin plot 

pred_pos_10 = pred_df[pred_df.PRED > 10]

cell_path = '/home01/k040a01/02.M3V6/M3V6_W213_349_MIS2/CELL_VAL' 


cell_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL'
import glob
from matplotlib import colors as mcolors

CVCL_files_raw = glob.glob(cell_path+'/TOP.PRED_*.csv')

pred_df = pd.DataFrame(columns = ['PRED', 'ROW_CID','COL_CID','CCLE', 'Y'])
for fifi in CVCL_files_raw :
	print(fifi.split('/')[-1].split('.')[0])
	tmp_df = pd.read_csv(fifi,  low_memory=False, index_col = False)
	tmp_df2 = tmp_df[['ROW_CID','COL_CID','CCLE', 'PRED', 'Y']]
	pred_df = pd.concat([pred_df, tmp_df2])
	#tmp_df3 = tmp_df2[tmp_df2.PRED>10]
	#tmp_df3.to_csv(cell_path + 'TOP.{}.csv'.format(fifi.split('/')[8].split('.')[0]))

pred_pos_10['tissue'] = pred_pos_10.CCLE.apply(lambda x : '_'.join(x.split('_')[1:]) )

my_order = pred_pos_10.groupby(by=["CCLE"])["PRED"].mean().sort_values().iloc[::-1].index



fig, ax = plt.subplots(figsize=(40, 8))

sns.violinplot(ax = ax, data  = pred_pos_10, x = 'CCLE', y = 'PRED', 
linewidth=2,  edgecolor="black", inner = None, order = my_order) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 

for vv in range(90) : 
	cell = my_order[vv]
	tissue = '_'.join(cell.split('_')[1:])
	violins[vv].set_facecolor(mcolors.to_rgba(color_dict[tissue], 0.8))

sns.swarmplot(ax = ax, data  = pred_pos_10, order = my_order,
x = 'CCLE', y = 'PRED', hue='CCLE', palette=sns.color_palette(['black']*90))


ax.set_xlabel('ccle', fontsize=15)
ax.set_ylabel('predicted value', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=15) # ax.get_xticklabels()
ax.set_yticklabels(ax.get_yticks(),  fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
#plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(cell_path,'pred_check.{}.png'.format(W_NAME)), dpi = 300)

plt.close()









pred_pos = pred_df[pred_df.PRED > 0]
pred_pos['tissue'] = pred_pos.CCLE.apply(lambda x : '_'.join(x.split('_')[1:]) )

my_order2 = pred_pos.groupby(by=["tissue"])["PRED"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25,14))

sns.violinplot(ax = ax, data  = pred_pos, x = 'tissue', y = 'PRED', 
linewidth=2,  edgecolor="black", inner = None, order = my_order2) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 

for vv in range(11) : 
	tissue = my_order2[vv]
	violins[vv].set_facecolor(mcolors.to_rgba(color_dict[tissue], 0.8))


ax.set_xlabel('tissue', fontsize=15)
ax.set_ylabel('predicted value', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=15) # ax.get_xticklabels()
ax.set_yticklabels(ax.get_yticks(),  fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
#plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(cell_path,'pred_check.tissue2.png'), dpi = 300)

plt.close()









pred_df['tissue'] = pred_df.CCLE.apply(lambda x : '_'.join(x.split('_')[1:]) )

my_order2 = pred_df.groupby(by=["tissue"])["PRED"].mean().sort_values().iloc[::-1].index

fig, ax = plt.subplots(figsize=(25,14))

sns.violinplot(ax = ax, data  = pred_df, x = 'tissue', y = 'PRED', 
linewidth=2,  edgecolor="black", inner = None, order = my_order2) # width = 3,,  inner = 'point'
violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 

for vv in range(11) : 
	tissue = my_order2[vv]
	violins[vv].set_facecolor(mcolors.to_rgba(color_dict[tissue], 0.8))


ax.set_xlabel('tissue', fontsize=15)
ax.set_ylabel('predicted value', fontsize=15)
ax.set_xticks(ax.get_xticks())
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90, fontsize=15) # ax.get_xticklabels()
ax.set_yticklabels(ax.get_yticks(),  fontsize=15)
ax.tick_params(axis='both', which='major', labelsize=20 )
plt.grid(True)
#plt.legend(bbox_to_anchor=(1.2, 1.2), loc="upper right", fontsize=15)
plt.tight_layout()

plt.savefig(os.path.join(cell_path,'pred_check.tissue3.png'), dpi = 300)

plt.close()













######################################
######################################
######################################
######################################
######################################
######################################

cell line 에서의 위치를 좀 따지기로 했다 

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

ABCS_test_result = pd.read_csv(PRJ_PATH+'ABCS_test_result.csv')

ABCS_test_result['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(ABCS_test_result['CELL'])]

test_cell_df = pd.DataFrame({'DC_cellname' : list(set(ABCS_test_result.DC_cellname))})

cell_P = []
cell_S = []
cell_num = []

for cell in list(test_cell_df.DC_cellname) :
	tmp_test_re = ABCS_test_result[ABCS_test_result.DC_cellname == cell]
	cell_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.PRED_3)
	cell_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.PRED_3)
	cell_nums = tmp_test_re.shape[0]
	cell_P.append(cell_P_corr)
	cell_S.append(cell_S_corr)
	cell_num.append(cell_nums)


test_cell_df['P_COR'] = cell_P
test_cell_df['S_COR'] = cell_S
test_cell_df['cell_num'] = cell_num


tissue = 'LARGE_INTESTINE'
Y_value = 16.274403
title = 'DLD1_3394_11692123'

tissue = 'OVARY'
cell = 'PA1_OVARY'
Y_value = 12.766936
title = 'PA1_68060125_22291652'

tissue = 'LUNG'
cell = 'HOP62_LUNG'
Y_value = 10.016934
title = 'HOP62_176870_22291652'

tissue = 'PROSTATE'
cell = 'VCAP_PROSTATE'
Y_value = 12.332431
title = 'VCAP_1973_70817911'

ABCS_test_result_tissue = ABCS_test_result[ABCS_test_result.tissue==tissue]

ABCS_test_result_filt1 = ABCS_test_result_tissue[['DC_cellname','CELL','tissue','ANS']]
ABCS_test_result_filt1.columns = ['DC_cellname','CELL','tissue','value']
ABCS_test_result_filt1['label'] = 'tissue'

ABCS_test_result_cell = ABCS_test_result_tissue[ABCS_test_result_tissue.CELL== cell]

ABCS_test_result_filt2 = ABCS_test_result_cell[['DC_cellname','CELL','tissue','ANS']]
ABCS_test_result_filt2.columns = ['DC_cellname','CELL','tissue','value']
ABCS_test_result_filt2['label'] = 'cell'


for_split_vio = pd.concat([ABCS_test_result_filt1, ABCS_test_result_filt2])

fig, ax = plt.subplots(figsize=(8, 8))
sns.violinplot(
	ax =ax, data  = for_split_vio,
	x = 'label', y = 'value',
	inner = 'quart',
	linewidth=2,  edgecolor="black")


tissue_set = ['CENTRAL_NERVOUS_SYSTEM', 'LUNG', 'BREAST', 'BONE', 'OVARY', 'PROSTATE', 'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'LARGE_INTESTINE', 'SKIN', 'PLEURA', 'KIDNEY' ] # list(set(test_cell_df['tissue']))
color_set = ['#CF3476','#FF7514','#025669','#308446','#84C3BE','#D53032','#4ddcfd','#ffcd36','#ac8cff',"#0000ffff","#7bff68ff"] # "#1E1E1E"
color_dict = {a : color_set[tissue_set.index(a)] for a in tissue_set}

violins = [c for i,c in enumerate(ax.collections) ] # pollycollection 가져오기 위함 # if i%2 == 0
violins[0].set_facecolor(mcolors.to_rgba(color_dict[tissue]))
violins[1].set_facecolor(mcolors.to_rgba('grey'))

plt.axhline(y=Y_value , color = 'r') 


ax.set_xlabel('', fontsize=20)
ax.set_ylabel('Loewe', fontsize=20)
ax.set_xticks(ax.get_xticks())
#ax.set_xticklabels(np.array(['LARGE_INTESTINE','SW837']), fontsize = 25, rotation = 90)
ax.set_xticklabels(['',''])
ax.tick_params(axis='y', which='major', labelsize=20 )
plt.tight_layout()
plt.savefig(os.path.join('/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/CELL_VAL/cell_compare.{}.png'.format(title)), dpi = 300)
plt.close()



pred_pos_10_re = pred_pos_10.reset_index(drop = True)

res_dict = {}

all_cells = list(set(ABCS_test_result.CELL))

for cell in all_cells : 
	quart = ABCS_test_result[ABCS_test_result.CELL==cell]['ANS'].describe()['75%']
	pred_tmp = pred_pos_10_re[pred_pos_10_re.CCLE == cell]
	pred_tmp_re = pred_tmp[pred_tmp.PRED>quart]
	indx = list(pred_tmp_re.index)
	res_dict[cell] = indx







PRED



































ccc_a = list(pred_pos_10['ROW_CID'])
ccc_b = list(pred_pos_10['COL_CID'])
ccc_t = list(pred_pos_10['tissue'])

pred_pos_10['cct'] = [str(int(ccc_a[i])) + '___' + str(int(ccc_b[i]))+ '___' + str(ccc_t[i]) if ccc_a[i] < ccc_b[i] else str(int(ccc_b[i])) + '___' + str(int(ccc_a[i]))+ '___' + str(ccc_t[i]) for i in range(len(ccc_a))]
pred_pos_10['cc'] = [str(int(ccc_a[i])) + '___' + str(int(ccc_b[i])) if ccc_a[i] < ccc_b[i] else str(int(ccc_b[i])) + '___' + str(int(ccc_a[i])) for i in range(len(ccc_a))]


cct_mean = pred_pos_10.groupby('cct').mean().sort_values('PRED')
cct_count = pred_pos_10.groupby('cct').count()['Y']
cct_std = pred_pos_10.groupby('cct').std()['Y']

cct_count_5 = list(cct_count[cct_count==4].index)
cct_count_5 = [a for a in cct_count_5 if 'SKIN' not in a]
cct_mean.loc[cct_count_5].sort_values('PRED')

cct_count_2 = list(cct_count[cct_count>2].index)
cct_mean_2 = pred_pos_10[pred_pos_10.cct.isin(cct_count_2)]
cct_mean_22 = cct_mean_2.groupby('cct').mean().sort_values('PRED', ascending = False)






cc_mean = pred_pos_10.groupby('cc').mean().sort_values('PRED')
cc_count = pred_pos_10.groupby('cc').count()['Y']
cc_count_2 = list(cc_count[cc_count>2].index)
cc_mean_2 = pred_pos_10[pred_pos_10.cc.isin(cc_count_2)]
cc_mean_22 = cc_mean_2.groupby('cc').mean().sort_values('PRED', ascending = False)


pred_df[(pred_df.ROW_CID==46220502) & (pred_df.COL_CID==208908)]
pred_df[(pred_df.ROW_CID==208908) & (pred_df.COL_CID==46220502)]


pred_df[(pred_df.ROW_CID==176870) & (pred_df.COL_CID==11167602)]
pred_df[(pred_df.ROW_CID==11167602) & (pred_df.COL_CID==176870)]



SW837_LARGE_INTESTINE
pred_df[(pred_df.ROW_CID==208908) & (pred_df.COL_CID==11167602)]
pred_df[(pred_df.ROW_CID==11167602) & (pred_df.COL_CID==208908)]
6.765147



pred_df[(pred_df.ROW_CID==5394) & (pred_df.COL_CID==11977753)]
pred_df[(pred_df.ROW_CID==11977753) & (pred_df.COL_CID==5394)]



pred_df[(pred_df.ROW_CID==135539077) & (pred_df.COL_CID==11977753)]
pred_df[(pred_df.ROW_CID==11977753) & (pred_df.COL_CID==135539077)]


pred_df[(pred_df.ROW_CID==60838) & (pred_df.COL_CID==3385)]
pred_df[(pred_df.ROW_CID==3385) & (pred_df.COL_CID==60838)]



pred_df[(pred_df.ROW_CID==60838) & (pred_df.COL_CID==9887053)]
pred_df[(pred_df.ROW_CID==9887053) & (pred_df.COL_CID==60838)]


SKMEL28_SKIN
pred_df[(pred_df.ROW_CID==84691) & (pred_df.COL_CID==24856436)]
pred_df[(pred_df.ROW_CID==24856436) & (pred_df.COL_CID==84691)]
8.812208


KPL1_BREAST
pred_df[(pred_df.ROW_CID==84691) & (pred_df.COL_CID==60750)]
pred_df[(pred_df.ROW_CID==60750) & (pred_df.COL_CID==84691)]
2.128601

DLD1_LARGE_INTESTINE
pred_df[(pred_df.ROW_CID==24856436) & (pred_df.COL_CID==60838)]
pred_df[(pred_df.ROW_CID==60838) & (pred_df.COL_CID==24856436)]
4.245074

pred_df[(pred_df.ROW_CID==11520894) & (pred_df.COL_CID==11977753)]
pred_df[(pred_df.ROW_CID==11977753) & (pred_df.COL_CID==11520894)]



HOP62_LUNG : 10.016934
pred_df[(pred_df.ROW_CID==176870) & (pred_df.COL_CID==24964624)]
pred_df[(pred_df.ROW_CID==24964624) & (pred_df.COL_CID==176870)]
In addition, Lara et al. [69] argued that the therapeutic effect of the combination of 
MK-2206 and Erlotinib on patients with non-small cell lung cancer (NSCLC) is worthy of further exploration. 
We check the prediction results of PRODeepSyn for three NSCLC cell lines included in the dataset, 
namely SKMES1, NCIH460, NCIH520, which are 38.36, 21.88 and 19.75, respectively.


pred_df[(pred_df.ROW_CID==42611257 ) & (pred_df.COL_CID==457193   )]
pred_df[(pred_df.ROW_CID==457193   ) & (pred_df.COL_CID==42611257 )]


pred_df[(pred_df.ROW_CID==11640390 ) & (pred_df.COL_CID==6914657)]
pred_df[(pred_df.ROW_CID==6914657) & (pred_df.COL_CID==11640390 )]






pred_df[(pred_df.ROW_CID==208908 ) & (pred_df.COL_CID==10113978   )]
pred_df[(pred_df.ROW_CID==10113978   ) & (pred_df.COL_CID==208908 )]


pred_df[(pred_df.ROW_CID==176870 ) & (pred_df.COL_CID==2244   )]
pred_df[(pred_df.ROW_CID==2244   ) & (pred_df.COL_CID==176870 )]

pred_df[(pred_df.ROW_CID==176870 ) & (pred_df.COL_CID==11626560   )]
pred_df[(pred_df.ROW_CID==11626560   ) & (pred_df.COL_CID==176870 )]



pred_df[(pred_df.ROW_CID==60749 ) & (pred_df.COL_CID==24964624   )]
pred_df[(pred_df.ROW_CID==24964624   ) & (pred_df.COL_CID==60749 )]


pred_df[(pred_df.ROW_CID==41867 ) & (pred_df.COL_CID==5702198   )]
pred_df[(pred_df.ROW_CID==5702198   ) & (pred_df.COL_CID==41867 )]





ccle_list = [ 

'KM12_LARGE_INTESTINE', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'SKMEL30_SKIN', 
'EKVX_LUNG', 'UACC62_SKIN', 'A673_BONE', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'SW837_LARGE_INTESTINE', 
'SF295_CENTRAL_NERVOUS_SYSTEM', 'NCIH520_LUNG', 'A549_LUNG', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'RVH421_SKIN', 'HS578T_BREAST', 'SW620_LARGE_INTESTINE', 'PC3_PROSTATE', 'UO31_KIDNEY', 'WM115_SKIN',

'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH522_LUNG', 'MDAMB436_BREAST', 'HOP92_LUNG', 
'HCC1500_BREAST', 'RKO_LARGE_INTESTINE', 'MALME3M_SKIN', 'MDAMB361_BREAST', 'G361_SKIN', 
'MDAMB468_BREAST', 'ES2_OVARY', 'A2058_SKIN', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 
'RPMI7951_SKIN', 'HCT15_LARGE_INTESTINE', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'VCAP_PROSTATE', 
'KPL1_BREAST', 'SKMES1_LUNG', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'SKMEL5_SKIN', 'A2780_OVARY', 
'UACC257_SKIN', 'NCIH2122_LUNG', 'MDAMB175VII_BREAST', 'MDAMB231_BREAST', 'SKOV3_OVARY', 
'NCIH23_LUNG', 'CAMA1_BREAST', 'A498_KIDNEY', 'MELHO_SKIN', 'NCIH1650_LUNG', 'IPC298_SKIN', 
'OVCAR4_OVARY', 'LOXIMVI_SKIN', 'COLO829_SKIN', 'SKMEL28_SKIN', 'OVCAR5_OVARY', 'COLO800_SKIN', 
'HOP62_LUNG', 'HT144_SKIN', 'HCC1419_BREAST', 'IGROV1_OVARY', 'NCIH460_LUNG', 'SKMEL2_SKIN', 
'SNB75_CENTRAL_NERVOUS_SYSTEM', 'BT474_BREAST', 'PA1_OVARY', 'UWB1289_OVARY', 'T47D_BREAST', 
'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', '786O_KIDNEY', 'OV90_OVARY', 'HT29_LARGE_INTESTINE', 
'A101D_SKIN', 'COLO792_SKIN', 'NCIH226_LUNG', 'MCF7_BREAST', 'OVCAR8_OVARY', 'CAKI1_KIDNEY',
 'ZR751_BREAST', 'CAOV3_OVARY', 'A427_LUNG', 'DLD1_LARGE_INTESTINE', 'MSTO211H_PLEURA', 
 'LOVO_LARGE_INTESTINE', 'ACHN_KIDNEY', 'MEWO_SKIN', 'HCT116_LARGE_INTESTINE', 'BT549_BREAST', 
 'SF539_CENTRAL_NERVOUS_SYSTEM', 'A375_SKIN', 'UACC812_BREAST', 'NIHOVCAR3_OVARY']

for ccle in ccle_list :
	print(ccle)
	tmp = pred_df[pred_df.CCLE == ccle]
	tmp2 = tmp.sort_values('PRED', ascending = False)
	tmp2.iloc[:20]
	tmp2.iloc[-20:]












################3 SEARCH 
SEARCH_1 = merged_CVCL[merged_CVCL.CCLE == 'DLD1_LARGE_INTESTINE']

SEARCH_1 = SEARCH_1.sort_values('PRED')



SEARCH_2 = merged_CVCL[merged_CVCL.CCLE == 'ES2_OVARY']

SEARCH_2 = SEARCH_2.sort_values('PRED')


SEARCH_3 = merged_CVCL[merged_CVCL.CCLE == 'VCAP_PROSTATE']

SEARCH_3 = SEARCH_3.sort_values('PRED')





SEARCH_1[(SEARCH_1.ROW_CID==208908) & (SEARCH_1.COL_CID==11167602)]
SEARCH_1[(SEARCH_1.ROW_CID==24856436) & (SEARCH_1.COL_CID==60838)]




SEARCH_2[(SEARCH_2.ROW_CID==11520894) & (SEARCH_2.COL_CID==11977753)]
SEARCH_2[(SEARCH_2.ROW_CID==11977753) & (SEARCH_2.COL_CID==11520894)]
