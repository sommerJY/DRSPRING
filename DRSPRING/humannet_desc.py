

len(set(list(hnet_L2['G_A']) + list(hnet_L2['G_B']))) # 611

3885 



민지한테 추가 부탁한 내용


FN = pd.read_csv('/st06/jiyeonH/13.DD_SESS/HumanNetV3/HumanNet-FN.tsv', sep = '\t', header = None)
FN.columns = ['G_A','G_B','SC']

FN_L1 = FN[FN['G_A'].isin(BETA_lm_genes.gene_id)]
FN_L2 = FN_L1[FN_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 9300
FN_L3 = FN_L2[FN_L2.SC>=2.0] # 9300

len(set(list(FN['G_A']) + list(FN['G_B']))) # 8540
len(set(list(FN_L2['G_A']) + list(FN_L2['G_B']))) # 947
len(set(list(FN_L3['G_A']) + list(FN_L3['G_B']))) # 845








HSDB = pd.read_csv('/st06/jiyeonH/13.DD_SESS/HumanNetV3/HS-DB.tsv', sep = '\t', header = None) #  nodes
HSDB.columns = ['G_A','G_B','SC']

HSDB_L1 = HSDB[HSDB['G_A'].isin(BETA_lm_genes.gene_id)]
HSDB_L2 = HSDB_L1[HSDB_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 1278
HSDB_L3 = HSDB_L2[HSDB_L2.SC>=3.0] # 1071
HSDB_L3 = HSDB_L2[HSDB_L2.SC>=3.5] # 507

len(set(list(HSDB['G_A']) + list(HSDB['G_B']))) # 8540
len(set(list(HSDB_L2['G_A']) + list(HSDB_L2['G_B']))) # 554
len(set(list(HSDB_L3['G_A']) + list(HSDB_L3['G_B']))) # 516
len(set(list(HSDB_L3['G_A']) + list(HSDB_L3['G_B']))) # 349



# 신경쓰이는데, TEST code 한번만 확인해보기 

            349 에서 ADD 
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_349_FULL/'
file_name = 'M3V4_349_MISS2_FULL'
A_B_C_S_SET_ADD_349 = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False) # 90676

MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET_349 = A_B_C_S_SET_ADD_349[A_B_C_S_SET_ADD_349.Basal_Exp == 'O'] # 90676
A_B_C_S_SET_349 = A_B_C_S_SET_349[A_B_C_S_SET_349.SYN_OX == 'O'] # 88755
## A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] ###################### old targets 
A_B_C_S_SET_349 = A_B_C_S_SET_349[A_B_C_S_SET_349.T1OX == 'O'] ####################### new targets # 53282
A_B_C_S_SET_349 = A_B_C_S_SET_349[A_B_C_S_SET_349.type.isin(MISS_filter)] # 53282
A_B_C_S_SET_349 = A_B_C_S_SET_349[A_B_C_S_SET_349.DrugCombCCLE.isin(ccle_names)] # 53282





            978 ADD

SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V4_978_FULL/'
file_name = 'M3V4_978_MISS2_FULL'
A_B_C_S_SET_ADD_978 = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False) # 90676

MISS_filter = ['AOBO','AXBO','AOBX','AXBX']

A_B_C_S_SET_978 = A_B_C_S_SET_ADD_978[A_B_C_S_SET_ADD_978.Basal_Exp == 'O'] # 90676
A_B_C_S_SET_978 = A_B_C_S_SET_978[A_B_C_S_SET_978.SYN_OX == 'O'] # 88755
## A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T2OX == 'O'] ###################### old targets 
A_B_C_S_SET_978 = A_B_C_S_SET_978[A_B_C_S_SET_978.T1OX == 'O'] ####################### new targets # 59041
A_B_C_S_SET_978 = A_B_C_S_SET_978[A_B_C_S_SET_978.type.isin(MISS_filter)] # 59041
A_B_C_S_SET_978 = A_B_C_S_SET_978[A_B_C_S_SET_978.DrugCombCCLE.isin(ccle_names)]# 59041


