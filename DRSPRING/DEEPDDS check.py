DEEPDDS check 

# input 만들어야함 

original_input = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/04.DeepDDS/DeepDDs-master/data/independent_set/independent_cell_features_954.csv')


# 필요 1 ccle 

CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
ccle_names = [a for a in ccle_cell_info.DrugCombCCLE if type(a) == str]


ccle_ori_col = list(ccle_exp.columns)
ccle_new_col =['DepMap_ID'] + [int(a.split(')')[0].split('(')[1]) for a in ccle_ori_col[1:]]

ccle_exp.columns = ccle_new_col


# CCLE ver! 
ccle_cell_info = ccle_info[['DepMap_ID','stripped_cell_line_name','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','STR_ID','DrugCombCCLE']
ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')

# 아무래도 저쪽이 쓰는 내용을 가져다 써야할것 같음 
BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')

ori_order = list(original_input.columns)[1:]
BETA_GENE_954 = BETA_GENE[BETA_GENE.ensembl_id.isin(ori_order)]
len(set(BETA_GENE_954.ensembl_id))
BETA_GENE_954 = BETA_GENE_954.reset_index(drop = True)

ord = [list(BETA_GENE_954.ensembl_id).index(a) for a in ori_order]
BETA_GENE_954_re = BETA_GENE_954.loc[ord] 

ori_order_entrez = list(BETA_GENE_954_re.gene_id)


ccle_exp2_modi1 = ccle_exp2.loc[:,['DrugCombCCLE']+ori_order_entrez]
ccle_exp2_modi1.columns = list(original_input.columns)
ccle_exp2_modi2 = pd.concat([original_input.loc[0:0,], ccle_exp2_modi1], 0)
ccle_exp2_modi2.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/04.DeepDDS/DeepDDs-master/data/independent_set/independent_cell_features_954.csv', index = False)


===============================================

그러고 나서 내꺼 데이터 활용

WORK_NAME = 'WORK_403' # 349
W_NAME = 'W403'
PRJ_NAME = 'M3V8'
MJ_NAME = 'M3V8'
MISS_NAME = 'MIS2'
PPI_NAME = '349'


SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V8_349_FULL/'

file_name = 'M3V8_349_MISS2_FULL' # 0608

# 307644
A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)
MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))
A_B_C_S_SET_ADD['label'] = MY_syn.squeeze().tolist() 

sm_a = list(A_B_C_S_SET_ADD['ROW_CAN_SMILES'])
sm_b = list(A_B_C_S_SET_ADD['COL_CAN_SMILES'])
ccle = list(A_B_C_S_SET_ADD['CELL'])
A_B_C_S_SET_ADD['SM_C_CHECK'] = [sm_a[i] + '___' + sm_b[i]+ '___' + ccle[i] if sm_a[i] < sm_b[i] else sm_b[i] + '___' + sm_a[i]+ '___' + ccle[i] for i in range(A_B_C_S_SET_ADD.shape[0])]


MISS_filter = ['AOBO','AXBX','AXBO','AOBX'] # 

A_B_C_S_SET = A_B_C_S_SET_ADD[A_B_C_S_SET_ADD.Basal_Exp == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O']
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]
A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(ccle_names)]

CELL_92 = ['VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG', 'CAMA1_BREAST']

A_B_C_S_SET_ADD2= A_B_C_S_SET[A_B_C_S_SET.CELL.isin(CELL_92)]


# chem only 
chem_a = A_B_C_S_SET_ADD2[['CID_A','ROW_CAN_SMILES']]
chem_b = A_B_C_S_SET_ADD2[['CID_B','COL_CAN_SMILES']]
chem_a.columns = ['name','smile']
chem_b.columns = ['name','smile']

sm_list = pd.concat([chem_a, chem_b]).drop_duplicates()
sm_list.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/04.DeepDDS/DeepDDs-master/data/smiles.csv', index = False)



filter_data = A_B_C_S_SET_ADD2[['ROW_CAN_SMILES','COL_CAN_SMILES','CELL', 'label','SM_C_CHECK']]
filter_data_re = filter_data.reset_index(drop = True)

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

# 저장해둔거랑 같은지 확인 
with open('{}/CV_SM_list.{}.pickle'.format(PRJ_PATH, WORK_NAME), 'rb') as f:
	CV_ND_INDS_ray = pickle.load(f)


CV_0_train_sm = CV_ND_INDS_ray['CV0_train'] # sm : 236127, in : 236316 
CV_0_test_sm = CV_ND_INDS_ray['CV0_test'] # sm : 58028, in : 58128
CV_0_train_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_0_train_sm)].index)
CV_0_test_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_0_test_sm)].index)

CV_1_train_sm = CV_ND_INDS_ray['CV1_train'] # sm : 235024, in : 235253
CV_1_test_sm = CV_ND_INDS_ray['CV1_test'] # sm : 59131, in : 59191
CV_1_train_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_1_train_sm)].index)
CV_1_test_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_1_test_sm)].index)

CV_2_train_sm = CV_ND_INDS_ray['CV2_train'] # sm : 235355, in : 235602
CV_2_test_sm = CV_ND_INDS_ray['CV2_test'] # sm : 58800, in : 58842
CV_2_train_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_2_train_sm)].index)
CV_2_test_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_2_test_sm)].index)

CV_3_train_sm = CV_ND_INDS_ray['CV3_train'] # sm : 234600, in : 234835
CV_3_test_sm = CV_ND_INDS_ray['CV3_test'] # sm : 59555, in : 59609
CV_3_train_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_3_train_sm)].index)
CV_3_test_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_3_test_sm)].index)

CV_4_train_sm = CV_ND_INDS_ray['CV4_train'] # sm : 235514, in : 235770
CV_4_test_sm = CV_ND_INDS_ray['CV4_test'] # sm : 58641, in : 58674
CV_4_train_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_4_train_sm)].index)
CV_4_test_ind = list(filter_data_re[filter_data_re.SM_C_CHECK.isin(CV_4_test_sm)].index)

index_dict = {
    'CV0_train' : CV_0_train_ind, 'CV0_test' : CV_0_test_ind, 
    'CV1_train' : CV_1_train_ind, 'CV1_test' : CV_1_test_ind, 
    'CV2_train' : CV_2_train_ind, 'CV2_test' : CV_2_test_ind, 
    'CV3_train' : CV_3_train_ind, 'CV3_test' : CV_3_test_ind, 
    'CV4_train' : CV_4_train_ind, 'CV4_test' : CV_4_test_ind
}  

with open(file='/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/04.DeepDDS/DeepDDs-master/JY_index_dict.pickle', mode='wb') as f:



filter_data_re2 = filter_data_re[['ROW_CAN_SMILES','COL_CAN_SMILES','CELL','label']]
#filter_data_re2['cell'] = filter_data_re2.CELL.apply(lambda x : x.split('_')[0] )
#filter_data_re3 = filter_data_re2[['ROW_CAN_SMILES','COL_CAN_SMILES','cell','label']]

filter_data_re2.columns = ['drug1', 'drug2', 'cell', 'label']


filter_data_re2.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/04.DeepDDS/DeepDDs-master/data/new_labels_0_10_leave.csv', index = False)
                                                                              

################# 에러 생기는건 atom ion 문제였다고 함

tup_list = []
for a in range(37) :
    tmp = TS.loc[a]
    chem = tmp['Name']
    entrez = tmp['combin_entrez']
    if type(entrez) == str :
        ent_list = entrez.split(',')
        for ent in ent_list :
            tup_list.append( (chem, ent) )
    else :
        print(a)
        
        
# synpathy

synpathy_path = '/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/06.SynPathy/dataset/train/'
with open('{}/TRAIN.pkl'.format(synpathy_path), 'rb') as f:
	synpathy_train = pickle.load(f)


/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/06.SynPathy/dataset/train/TRAIN.pkl


# ours 

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_DB = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

A_B_C_S_cids = list(A_B_C_S_SET_ADD2.CID_A ) + list(A_B_C_S_SET_ADD2.CID_B )

TARGET_DB_RE = TARGET_DB[TARGET_DB.CID_RE.isin(A_B_C_S_cids)]
TARGET_DB_RE = TARGET_DB_RE[TARGET_DB_RE.ENTREZ_RE.isin(BETA_ENTREZ_ORDER)]
