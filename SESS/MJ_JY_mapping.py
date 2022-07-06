
# 민지랑 내꺼 확인 과정




BETA_MJ = pd.read_csv(LINCS_PATH+"Lincs_pubchem_mj.csv")
BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv")
BETA_MJ2= BETA_MJ[['pert_id', 'SMILES_cid', 'pubchem_cid', 'h_bond_acceptor_count', 'h_bond_donor_count','rotatable_bond_count', 'MolLogP', 'molecular_weight','canonical_smiles_re', 'tpsa']]
BETA_MJ3 = BETA_MJ2.drop_duplicates()
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)

BETA_SELEC_SIG_pre = pd.read_csv(LINCS_PATH+'SIG_INFO.220405') # cell 58가지, 129116, cid  25589
BETA_SELEC_SIG_pre2 = BETA_SELEC_SIG_pre[['sig_id','cell_iname','pert_id']]
BETA_SELEC_SIG = pd.merge(BETA_SELEC_SIG_pre2, BETA_MJ3, left_on ='pert_id', right_on='pert_id', how = 'left' )






# 원래 내 original version 
# 원래 내 original version 
# 원래 내 original version 
# 원래 내 original version 
# 원래 내 original version 
# 원래 내 original version 

BETA_SELEC_SIG 
MT_ORI_SMILES_INDEX = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_4/BETA_SMILES_CP_total_pcp.csv')
ORI_CP_merge = pd.merge(BETA_CP_info, MT_ORI_SMILES_INDEX, left_on='canonical_smiles', right_on='canonical_smiles', how='left' )
ORI_CP_merge2 = ORI_CP_merge[['pert_id','canonical_smiles','pubchem_cid']].drop_duplicates()
# 34419 row, 34129 cids
# nan filter ori_cid = [str(float(a)) for a in ORI_CP_merge2.pubchem_cid if a >0] 
# 28349 -> uniq 28059




# BETA_CP_info 다시 pcp 돌리면 달라지는지 한번만 더 확인해볼까  NEW JY
# BETA_CP_info 다시 pcp 돌리면 달라지는지 한번만 더 확인해볼까   NEW JY
# BETA_CP_info 다시 pcp 돌리면 달라지는지 한번만 더 확인해볼까   NEW JY
# BETA_CP_info 다시 pcp 돌리면 달라지는지 한번만 더 확인해볼까   NEW JY
# BETA_CP_info 다시 pcp 돌리면 달라지는지 한번만 더 확인해볼까   NEW JY

BETA_SIG_info # (1201944, 37)

filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460


BETA_SMILES_INDEX = pd.DataFrame(list(set(BETA_CP_info['canonical_smiles'])))
BETA_SMILES_INDEX['SMILES_cid'] = ""
BETA_SMILES_INDEX.columns = ['canonical_smiles','SMILES_cid']

SMILES_LIST = [] 
for smile in list(BETA_SMILES_INDEX['canonical_smiles']) :
	try : 
		SMILES = pcp.get_compounds(smile, 'smiles')
		SMILES_LIST.append(SMILES)
	except :
		SMILES = "NA"
		SMILES_LIST.append(SMILES)

BETA_SMILES_INDEX['SMILES_cid'] = SMILES_LIST

BETA_SMILES_INDEX['pubchem_cid'] = [a[0].cid if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
# BETA_SMILES_INDEX['canonical_smiles'] = list(set(CP_info['canonical_smiles']))
BETA_SMILES_INDEX['canonical_smiles_re'] = [a[0].canonical_smiles if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
BETA_SMILES_INDEX['molecular_weight'] = [a[0].molecular_weight if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
BETA_SMILES_INDEX['h_bond_donor_count'] = [a[0].h_bond_donor_count if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
BETA_SMILES_INDEX['h_bond_acceptor_count'] = [a[0].h_bond_acceptor_count if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
BETA_SMILES_INDEX['tpsa'] = [a[0].tpsa if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
BETA_SMILES_INDEX['rotatable_bond_count'] = [a[0].rotatable_bond_count if len(a) ==1 else np.nan for a in list(BETA_SMILES_INDEX.SMILES_cid)  ]
BETA_SMILES_INDEX['MolLogP'] = [MolLogP(Chem.MolFromSmiles(a)) if type(a) == str else np.nan for a in list(BETA_SMILES_INDEX.canonical_smiles)]

BETA_SMILES_INDEX.to_csv('')

# BETA_SMILES_INDEX.to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/CID_RE_20220705.csv')





CP_merge = pd.merge(BETA_CP_info, BETA_SMILES_INDEX, left_on='canonical_smiles', right_on='canonical_smiles', how='left' )
# CP_merge_check = CP_merge[['pert_id','canonical_smiles','pubchem_cid']].drop_duplicates()
# 34419 row, 34130 cids 
# nan filter new_cid = [str(float(a)) for a in CP_merge_check.pubchem_cid if a >0] 
# 28389 -> uniq 28100

filter2_merge = pd.merge(filter2, CP_merge, on='pert_id', how = 'left' )
check = [False if a == 'nan' else True for a in list(filter2_merge.smiles_cid_re)]
filter2_merge_nonan = filter2_merge[check ]
filter2_merge_nonan2 = filter2_merge_nonan[['pert_id','sig_id', 'pert_type', 'cell_iname', 'is_exemplar_sig', 'pubchem_cid','canonical_smiles_re', 'molecular_weight', 'h_bond_donor_count', 'h_bond_acceptor_count', 'tpsa', 'rotatable_bond_count', 'smiles_cid_re']]
filter2_merge_nonan3 = filter2_merge_nonan2.drop_duplicates() # 127524

new_pc_filter = pd.merge(filter2_merge_nonan3, MJ_PC, left_on="smiles_cid_re", right_on="CID_re",how="inner") #127591



BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_SELEC_SIG, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116




MJ_PC = pd.read_csv("/st06/jiyeonH/12.HTP_DB/08.PUBCHEM/PUBCHEM_MJ_031022.csv", low_memory=False, 
usecols=["CID", "CAN_SMILES"])
#l_sc2 저희가 pubchempy돌린 파일이용

BETA_SMILES_INDEX['smiles_cid_re'] = BETA_SMILES_INDEX.pubchem_cid.astype('str')
MJ_PC['CID_re'] = MJ_PC.CID.astype('float').astype('str')

new_pc_filter = pd.merge(BETA_SMILES_INDEX, MJ_PC, left_on="smiles_cid_re", right_on="CID_re",how="inner") #127591
28336



### 다시 sig 부터 filtering 민지껄로 진행 
### 다시 sig 부터 filtering 민지껄로 진행 
### 다시 sig 부터 filtering 민지껄로 진행 
### 다시 sig 부터 filtering 민지껄로 진행 
### 다시 sig 부터 filtering 민지껄로 진행 
### 다시 sig 부터 filtering 민지껄로 진행 

BETA_SIG_info # (1201944, 37)

filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460

MJ_PC = pd.read_csv("/st06/jiyeonH/12.HTP_DB/08.PUBCHEM/PUBCHEM_MJ_031022.csv", low_memory=False, 
usecols=["CID", "CAN_SMILES"])

MJ_raw = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/LINCS_SMILES_CID_filtered_mj.tsv', sep = '\t')
MJ_raw2 = MJ_raw[['pert_id','canonical_smiles','smiles_cid']].drop_duplicates()
MJ_raw3 = MJ_raw2[MJ_raw2.smiles_cid != 'None']
# MJ_nonfil_cids = [str(float(a)) for a in list(MJ_raw.smiles_cid) if a != 'None']

filter3 = pd.merge(filter2, MJ_raw3, left_on = 'pert_id', right_on = 'pert_id', how = 'left')
check3 = [True if type(a) == str else False for a in list(filter3.smiles_cid)] # nan 제거 
filter4 = filter3[check3]
filter5 = filter4[['pert_id','sig_id','cell_iname','is_exemplar_sig','canonical_smiles','smiles_cid']]

# 동일! 

MJ_bf_filter = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/Lincs_pubchem_mj_bf_filter.tsv', sep = '\t')

MJ_bf_filter2 = MJ_bf_filter[['pert_id','canonical_smiles','smiles_cid']].drop_duplicates()
# 25913 row 25651 cids 
mj_cid = [str(float(a)) for a in list(MJ_bf_filter2.smiles_cid)]

BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 



################################
아니 근데 나만 이상함? 

ori_cid = set([str(float(a)) for a in ORI_CP_merge2.pubchem_cid if a >0] ) # 28059
MJ_nonfil_cids = set([str(float(a)) for a in list(MJ_raw.smiles_cid) if a != 'None']) # 28129
new_cid = set([str(float(a)) for a in CP_merge_check.pubchem_cid if a >0] ) # 28100

new_cid - MJ_nonfil_cids




#########################################################################################
#########################################################################################
#########################################################################################

그냥 exempler 랑 con 이랑 해서 
978 적용한거 가져와보기로 함 

filter_con = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt'])]

filter_re = pd.concat([filter2, filter_con])
filter_re.to_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_FILTER.20220705.csv')


#########################################################################################
#########################################################################################
#########################################################################################




# R ! 
library(data.table)
GSEPATH = '/st06/jiyeonH/11.TOX/LINCS'

GCT_BIG_BETA.v1 = fread(paste0(GSEPATH,"/L_2020/BETA_1.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  18959 
GCT_BIG_BETA.v2 = fread(paste0(GSEPATH,"/L_2020/BETA_2.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  50272
GCT_BIG_BETA.v3 = fread(paste0(GSEPATH,"/L_2020/BETA_3.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  23107
GCT_BIG_BETA.v4 = fread(paste0(GSEPATH,"/L_2020/BETA_4.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  5038
GCT_BIG_BETA.v5 = fread(paste0(GSEPATH,"/L_2020/BETA_5.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  23898
GCT_BIG_BETA.v6 = fread(paste0(GSEPATH,"/L_2020/BETA_6.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  7137
GCT_BIG_BETA.v7 = fread(paste0(GSEPATH,"/L_2020/BETA_7.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  6820
GCT_BIG_BETA.v8 = fread(paste0(GSEPATH,"/L_2020/BETA_8.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  1237
GCT_BIG_BETA.con = fread(paste0(GSEPATH,"/L_2020/level5_beta_ctl_n58022x12328.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  40570

# 그냥 control 들이랑 cp 가져온거 
sig_for_MJ = read.csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_FILTER.20220705.csv',  stringsAsFactors=FALSE)

for (NUM in c(1:8)){ # 1, 3,4, 5,8
	print(NUM)
	BETA = get(paste0('GCT_BIG_BETA.v',NUM))
	DF_NAME = paste0("GCT_BIG_BETA.v",NUM,".DF")
	FI_NAME = paste0("GCT_BIG_BETA.TOTAL_",NUM)
	DF = data.frame(BETA)
	FI = DF[,c(1, which(DF[1,] %in% sig_for_MJ$sig_id))]
	assign(DF_NAME, DF )
	assign(FI_NAME, FI )
	#colnames(FI)[1] = 'pr_gene_id'
	print(dim(DF))
	print(dim(FI))
	#LIST = list(DF,FI )
}


BETA = GCT_BIG_BETA.con
DF_NAME = paste0("GCT_BIG_BETA.v","Con",".DF")
FI_NAME = paste0("GCT_BIG_BETA.TOTAL_","Con")
DF = data.frame(BETA)
FI = DF[,c(1, which(DF[1,] %in% sig_for_MJ$sig_id))] # 43953
assign(DF_NAME, DF )
assign(FI_NAME, FI )
colnames(FI)[1] = 'pr_gene_id'
print(FI[1:3,1:3])
print(dim(DF))
print(dim(FI))


BETA_BIND = cbind(GCT_BIG_BETA.TOTAL_1,
	GCT_BIG_BETA.TOTAL_2[,1:ncol(GCT_BIG_BETA.TOTAL_2)],
	GCT_BIG_BETA.TOTAL_3[,1:ncol(GCT_BIG_BETA.TOTAL_3)],
	GCT_BIG_BETA.TOTAL_4[,1:ncol(GCT_BIG_BETA.TOTAL_4)],
	GCT_BIG_BETA.TOTAL_5[,1:ncol(GCT_BIG_BETA.TOTAL_5)],
	GCT_BIG_BETA.TOTAL_6[,1:ncol(GCT_BIG_BETA.TOTAL_6)],
	GCT_BIG_BETA.TOTAL_7[,1:ncol(GCT_BIG_BETA.TOTAL_7)],
	GCT_BIG_BETA.TOTAL_8[,1:ncol(GCT_BIG_BETA.TOTAL_8)],
	GCT_BIG_BETA.TOTAL_Con[,1:ncol(GCT_BIG_BETA.TOTAL_Con)]
	)

BETA_BIND_2 = BETA_BIND[,c(1, which(BETA_BIND[1,] %in% sig_for_MJ$sig_id))]
dim(BETA_BIND_2) # 180415


my_sigs = sig_for_MJ$sig_id # 181240 # untrt 때문에 차이 남 

new_sigs = BETA_BIND[1,1:180421]
new_sigs2 = as.vector(unlist(new_sigs))
sig_check = my_sigs[which(my_sigs %in% new_sigs2 == FALSE)] 
sig_check = my_sigs[which(my_sigs %in% new_sigs2 == FALSE)]

colnames(BETA_BIND_2)[2:ncol(BETA_BIND_2)] = paste0("COL_",2:ncol(BETA_BIND_2))

# write.csv(BETA_BIND_2, "/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_20220705.csv" )

landmark = read.csv(paste0(GSEPATH,'/L_2020/geneinfo_beta.txt'), sep = '\t' , stringsAsFactors=FALSE)
landmark_lm = landmark[landmark$feature_space == 'landmark' ,]
BETA_BIND_3 = BETA_BIND_2[BETA_BIND_2$V1 %in% landmark_lm$gene_id,]
colnames(BETA_BIND_3) = BETA_BIND_2[1,]

# write.csv(BETA_BIND_3, "/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_20220705_978.csv" )


#########################################################################################
#########################################################################################
#########################################################################################


filter1 = BETA_SIG_info[BETA_SIG_info.pert_type.isin(['ctl_vehicle', 'ctl_untrt' ,'trt_cp' ])]
# 764996
filter2 = filter1[filter1.is_exemplar_sig==1]
# 136460

BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

BETA_MJ_RE = BETA_MJ[['pert_id','SMILES_cid','canonical_smiles',
       'pubchem_cid', 'h_bond_acceptor_count', 'h_bond_donor_count',
       'rotatable_bond_count', 'MolLogP', 'molecular_weight',
       'canonical_smiles_re', 'tpsa']].drop_duplicates()

BETA_EXM = pd.merge(filter2, BETA_MJ_RE, on='pert_id', how = 'left')
BETA_EXM_re = pd.merge(filter2, BETA_MJ_RE, on='pert_id', how = 'inner')

BETA_EXM2 = BETA_EXM[BETA_EXM.SMILES_cid > 0]


BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_EXM2, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pert_id','pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116

cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pert_id','pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 111916
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]

ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)]
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pert_id','pubchem_cid','ccle_name','sig_id']].drop_duplicates() # 110620
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[BETA_CID_CCLE_SIG.ccle_name!='NA']
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.pubchem_cid)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]



# cello
BETA_CID_CELLO_SIG.columns=['drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

BETA_CID_CELLO_SIG.columns=['pubchem_cid', 'cellosaurus_id', 'sig_id']


FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
FILTER2 = [True if type(a)==float else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER2] # 11742 ??? 
FILTER3 = [True if np.isnan(a)==False else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER3] # 11701 
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 






















