import os
import numpy as np
import pandas as pd
import tensorflow as tf
import argparse
from scipy import stats
from sklearn.metrics import mean_squared_error
import numpy as np
import sys
import json 


# (1) 데이터 정리 


1) DrugComb 에서 요약 데이터 받은 경우 

# DC_DATA = pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/summary_v_1_5.csv', low_memory=False)
# DC_DATA2 = DC_DATA[['drug_row','drug_col','cell_line_name','synergy_loewe']]
# DC_DATA2 = DC_DATA2.drop_duplicates()
# LOC = [True if type(a) == str else False for a in DC_DATA2['drug_col']]
# DC_DATA3 = DC_DATA2.loc[LOC, :]
# LOC2 = [True if type(a) == str else False for a in DC_DATA3['drug_row']]
# DC_DATA4 = DC_DATA3.loc[LOC2, :] # 739916
# DC_DATA4.cell_line_name # unique 288



2) 그러다가 CID 랑 matching 시키는게 애매해져서 
API 다운로드 해야한다는 사실을 깨달음 

qq_list = list(range(1,1469182, 10000)) + [1469182]
error_idx = []
for Q in range(0,147) : # 
	try :
		fst = qq_list[Q]
		sec = qq_list[Q+1]
		os.system("wget -O ALL_{}.json 'https://api.drugcomb.org/summary?from={}&to={}' --no-check-certificate".format(Q,fst,sec))
		time.sleep(2)
	except :
		error_idx.append(Q)

ALL_FILES = glob.glob("/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/ALL*")

with open(ALL_FILES[0]) as json_file :
	DC_tmp =json.load(json_file)

DC_tmp_K = list(DC_tmp[0].keys())
DC_tmp_DF = pd.DataFrame(columns=DC_tmp_K)

for fifi in ALL_FILES[87:]:
	with open(fifi) as json_file :
		DC_tmp = json.load(json_file)
	#
	for DD in range(0,len(DC_tmp)):
		tmpdf = pd.DataFrame({k:[DC_tmp[DD][k]] for k in DC_tmp_K})
		DC_tmp_DF = pd.concat([DC_tmp_DF, tmpdf], axis = 0)

DC_tmp_DF2 = DC_tmp_DF.drop_duplicates()
DC_tmp_DF2 = DC_tmp_DF2.sort_values('block_id')

DC_tmp_DF2.to_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/TOTAL_BLOCK.csv')
DC_tmp_DF2=pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/TOTAL_BLOCK.csv', low_memory=False)

# 다시 필터링
DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']]
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates()
DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_col_id>0]
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_row_id>0]
DC_DATA_filter4.cell_line_id # unique 295
# 아 근데 나중에 알아버림. 여기에 {<class 'NoneType'>, <class 'int'>, <class 'float'>}
# 섞여있음 




# Drug data 
with open('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/drugs.json') as json_file :
	DC_DRUG =json.load(json_file)

DC_DRUG_K = list(DC_DRUG[0].keys())
DC_DRUG_DF = pd.DataFrame(columns=DC_DRUG_K)

for DD in range(1,len(DC_DRUG)):
	tmpdf = pd.DataFrame({k:[DC_DRUG[DD][k]] for k in DC_DRUG_K})
	DC_DRUG_DF = pd.concat([DC_DRUG_DF, tmpdf], axis = 0)

DC_DRUG_DF2 = DC_DRUG_DF[['id','dname','cid']] 
# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서 
# dnames_dup =[a for a in dnames if dnames.count(a)>1]
DC_DRUG_DF2.columns = ['drug_row_id','drug_row','drug_row_cid']
DC_DATA5 = pd.merge(DC_DATA_filter4, DC_DRUG_DF2, on ='drug_row_id', how='left' )

DC_DRUG_DF2.columns = ['drug_col_id','drug_col','drug_col_cid']
DC_DATA6 = pd.merge(DC_DATA5, DC_DRUG_DF2, on ='drug_col_id', how='left')




# Cell line data
with open('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/cell_lines.json') as json_file :
	DC_CELL =json.load(json_file)

DC_CELL_K = list(DC_CELL[0].keys())
DC_CELL_DF = pd.DataFrame(columns=DC_CELL_K)

for DD in range(1,len(DC_CELL)):
	tmpdf = pd.DataFrame({k:[DC_CELL[DD][k]] for k in DC_CELL_K})
	DC_CELL_DF = pd.concat([DC_CELL_DF, tmpdf], axis = 0)

#DC_CELL_DF.to_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/cell_lines.csv')

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 751450
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']

ONLY_CELLO_CCLE = DC_CELL_DF[['cellosaurus_accession','ccle_name']].drop_duplicates()


# check DC triads (DC drug, cell line data )
DC_DATA7_1 = pd.merge(DC_DATA6, DC_CELL_DF2 , on= 'cell_line_id', how = 'left')
DC_DATA7_2 = DC_DATA7_1[DC_DATA7_1.drug_row_cid>0]
DC_DATA7_3 = DC_DATA7_2[DC_DATA7_2.drug_col_cid>0] # 735595
cello_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCello)]
ccle_t=[True if type(a)==str else False for a in list(DC_DATA7_3.DrugCombCCLE)]

DC_DATA7_4_cello = DC_DATA7_3[cello_t] # 730348
DC_cello_final = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello']].drop_duplicates() # 563367
DC_cello_final_dup = DC_DATA7_4_cello[['drug_row_cid','drug_col_cid','DrugCombCello', 'synergy_loewe']].drop_duplicates() # 730348

DC_DATA7_4_ccle = DC_DATA7_3[ccle_t] # 730348
DC_DATA7_4_ccle = DC_DATA7_4_ccle[DC_DATA7_4_ccle.DrugCombCCLE != 'NA'] # 540037
DC_ccle_final = DC_DATA7_4_ccle[['drug_row_cid','drug_col_cid','DrugCombCCLE']].drop_duplicates() # 464137
DC_ccle_final_dup = DC_DATA7_4_ccle[['drug_row_cid','drug_col_cid','DrugCombCCLE', 'synergy_loewe']].drop_duplicates() # 540037






# make cello input 

# pubchem data 
PC_DATA = pd.read_csv('/st06/jiyeonH/12.HTP_DB/08.PUBCHEM/PUBCHEM_MJ_031022.csv', low_memory=False)

# GEX data 
CEL_gex = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/data/cell_line_gex.csv'
cell_line = pd.read_csv(CEL_gex,header=None) # 근데 이 내용이 그냥 expression 값으로만 되어 있음. 이름도 ㄴ 내가 보기엔 일일히 matching 해준것 같은데 그 filtering 과정 제공을 안해줌  


# CCLE data 
CCLE_TPM = '/st06/jiyeonH/13.DD_SESS/CCLE.0222/CCLE_expression.csv'
CCLE = pd.read_csv(CCLE_TPM)
CCLE_info = pd.read_csv('/st06/jiyeonH/13.DD_SESS/CCLE.0222/sample_info.csv')
CCLE_sample_info = pd.read_csv('/st06/jiyeonH/13.DD_SESS/CCLE.0222/sample_info.csv')



# LINCS_DATA
# 원래 그냥 smiles -> cid 는 unique 28059
BETA_BIND = pd.read_csv("/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_SS_df.978.csv")
BETA_SELEC_SIG = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/SIG_INFO.220405') # cell 58가지, 129116, cid  25589
BETA_CP_info = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/compoundinfo_beta.txt')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/siginfo_beta.txt', low_memory = False)

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()

# 일단 cell 부터 맞추기 
BETA_CEL_info2 = BETA_CEL_info[['cell_iname','cellosaurus_id','ccle_name']] # 240 
BETA_SELEC_SIG_wCell = pd.merge(BETA_SELEC_SIG, BETA_CEL_info2, on = 'cell_iname', how = 'left') # 129116
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell[['pubchem_cid','cellosaurus_id','ccle_name','sig_id']]
BETA_SELEC_SIG_wCell2 = BETA_SELEC_SIG_wCell2.drop_duplicates() # 129116


cello_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.cellosaurus_id)]
BETA_CID_CELLO_SIG = BETA_SELEC_SIG_wCell2[cello_tt][['pubchem_cid','cellosaurus_id','sig_id']].drop_duplicates() # 111916
beta_cid_cello_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CELLO_SIG.pubchem_cid)]
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf] # 110555

ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)]
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pubchem_cid','ccle_name','sig_id']].drop_duplicates() # 110620
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[BETA_CID_CCLE_SIG.ccle_name!='NA'] 
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.pubchem_cid)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]




# DrugComb & LINCS 비교
# 한번만 다시 확인 LINCS A - LINCS B - DC_result - Cell
DC_cello_final_dup # 730348
DC_ccle_final_dup
BETA_CID_CCLE_SIG
BETA_CID_CELLO_SIG # 110555

# cello
BETA_CID_CELLO_SIG.columns=['drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'inner') # 731051


BETA_CID_CELLO_SIG.columns=['drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'inner') # 731644


FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
FILTER2 = [True if type(a)==float else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER2] # 11742 ??? 
FILTER3 = [True if np.isnan(a)==False else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER3] # 11701 
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 


# ccle 
BETA_CID_CCLE_SIG.columns=['drug_row_cid', 'DrugCombCCLE', 'BETA_sig_id']
CCLE_DC_BETA_1 = pd.merge(DC_ccle_final_dup, BETA_CID_CCLE_SIG, on = ['drug_row_cid','DrugCombCCLE'], how = 'left')# inner 로 해도 마지막 결과는 차이 없음 

BETA_CID_CCLE_SIG.columns=['drug_col_cid', 'DrugCombCCLE', 'BETA_sig_id']
CCLE_DC_BETA_2 = pd.merge(CCLE_DC_BETA_1, BETA_CID_CCLE_SIG, on = ['drug_col_cid','DrugCombCCLE'], how = 'left')

FILTER = [a for a in range(CCLE_DC_BETA_2.shape[0]) if (type(CCLE_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CCLE_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CCLE_DC_BETA = CCLE_DC_BETA_2.loc[FILTER] # 11567
CCLE_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCCLE']].drop_duplicates() # 9161
CCLE_DC_BETA_cids = list(set(list(CCLE_DC_BETA.drug_row_cid) + list(CCLE_DC_BETA.drug_col_cid))) # 158








# Binding DB 로 얼마나 가져올 수 있을지 잠깐 확인 

BD_HEADER = ["V_"+str(a) for a in range(250)] # 이렇게 안해주면 뒤에 무작정 붙은 sequence 들이 문제 생김 
BD_all = pd.read_csv(
	'/st06/jiyeonH/13.DD_SESS/BindingDB.22.04.01/BindingDB_All.tsv', 
	low_memory=False, sep = '\t', names= BD_HEADER, header=None)

BD_summary = BD_all.iloc[:,[a for a in range(37)]+[41]]
BD_summary.columns = list(BD_summary.loc[0])
BD_summary = BD_summary.iloc[1:]
BD_summary.to_csv('/st06/jiyeonH/13.DD_SESS/BindingDB.22.04.01/JY_summary.csv', sep ='\t', index = False)



BD_summary[['BindingDB Ligand Name','Target Name Assigned by Curator or DataSource']]
BD_summary[['Link to Ligand in BindingDB','Link to Target in BindingDB']]
BD_summary[['Link to Ligand-Target Pair in BindingDB','PubChem CID']]

그럼 이제 
EXP 를 어떻게 연결시켜줄건지 지금 고민이잖음 ?
이번 목표 : LINCS 를 matchmaker 랑 연결시켜보기 




list(BD_all.iloc[0,:]).index('UniProt (SwissProt) Primary ID of Target Chain')
BD_all.iloc[:,41]
BD_NAMES = list(set(BD_all.iloc[:,41]))[1:]

list(BD_all.iloc[0,:]).index('Target Name Assigned by Curator or DataSource')
BD_all.iloc[:,6]
BD_NAMES2 = [a for a in list(set(BD_all.iloc[:,6])) if type(a) == str]



# (1) entrez 

import urllib.parse
import urllib.request

url = 'https://www.uniprot.org/uploadlists/'

params = {
'from': 'ACC+ID',
'to': 'P_ENTREZGENEID', # GENENAME 
'format': 'tab',
'query': ' '.join(BD_NAMES)
}

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
with urllib.request.urlopen(req) as f:
	response = f.read()

TMP = response.decode('utf-8')
TMP2= TMP.split('\n') # 5483

ACCID = [a.split('\t')[0] for a in TMP2[1:-1] ] # 5481
ENTREZ = [a.split('\t')[1] if len(a.split('\t')[1])>0 else '' for a in TMP2[1:-1] ] # 5481

NAME_DF = pd.DataFrame({'ACCID':ACCID , 'ENTREZ':ENTREZ})
NAME_DF.to_csv('BD_ENTREZ_uniprot.csv',sep ='\t', index = False)


BD_summary2 = pd.merge(BD_summary, NAME_DF , left_on ='UniProt (SwissProt) Primary ID of Target Chain', right_on = 'ACCID', how = 'left')




# (2) gene name 

params = {
'from': 'ACC+ID',
'to': 'GENENAME', # 
'format': 'tab',
'query': ' '.join(BD_NAMES)
}

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
with urllib.request.urlopen(req) as f:
	response = f.read()

TMP = response.decode('utf-8')
TMP2= TMP.split('\n') # 5856
ACCID2 = [a.split('\t')[0] for a in TMP2[1:-1] ] # 5854
GENENAME = [a.split('\t')[1] if len(a.split('\t')[1])>0 else '' for a in TMP2[1:-1] ] # 5854

NAME_DF2 = pd.DataFrame({'ACCID':ACCID2 , 'GENENAME':GENENAME})
NAME_DF2.to_csv('BD_GENENAME_uniprot.csv',sep ='\t', index = False)



# (3) org

params = {
'from': 'ACC+ID',
'to': 'STRING_ID', # GENENAME P_ENTREZGENEID
'format': 'tab',
'query': ' '.join(BD_NAMES)
}

data = urllib.parse.urlencode(params)
data = data.encode('utf-8')
req = urllib.request.Request(url, data)
with urllib.request.urlopen(req) as f:
	response = f.read()

TMP = response.decode('utf-8')
TMP3= TMP.split('\n') # 
ACCID3 = [a.split('\t')[0] for a in TMP3[1:-1] ] # 5854
ORG = [a.split('\t')[1] if len(a.split('\t')[1])>0 else '' for a in TMP3[1:-1] ] # 5854
ORG2 = [int(a.split('.')[0]) for a in ORG]
NAME_DF2_2 = pd.DataFrame({'ACCID':ACCID3 , 'ORG':ORG2})






NAME_DF3 = pd.merge(NAME_DF, NAME_DF2, on = 'ACCID', how = 'outer')
NAME_DF3.ENTREZ = [int(a) if type(a)==str else 0 for a in NAME_DF3.ENTREZ ]

NAME_DF4 = pd.merge(NAME_DF3, NAME_DF2_2, on = 'ACCID', how = 'outer')
NAME_DF5 = NAME_DF4[NAME_DF4.ORG==9606]

# 근데 이게 다 human 이 아니야. 그리고 ID 에 따른 결과를 다 주는것도 아님 
# ID 중에서 유전자 이름은 있는데 entrez 는 없는 애들도 있음 

BD_summary2 = pd.merge(BD_summary, NAME_DF5 , left_on ='UniProt (SwissProt) Primary ID of Target Chain', right_on = 'ACCID', how = 'left')
BD_summary3 = BD_summary2[['PubChem CID','ENTREZ','GENENAME','ORG']]
BD_summary4 = BD_summary3.drop_duplicates()

BD_summary5 = BD_summary4[BD_summary4.ORG==9606]
BD_summary5 = BD_summary5[BD_summary5.ENTREZ!=0]
tmptmp = [int(a) if type(a)==str else 0 for a in BD_summary5['PubChem CID']]
BD_summary5['PUBCHEM_CID']=tmptmp
BD_summary6 = BD_summary5[BD_summary5.PUBCHEM_CID>0]
BD_summary7 = BD_summary6[['PUBCHEM_CID','ENTREZ']].drop_duplicates() # 1315958



# BD 랑 LINCS 확인 

BD_in_BETA_EXAM = BD_summary7[BD_summary7.PUBCHEM_CID.isin(BETA_SELEC_SIG.pubchem_cid)] # 37805
set(BD_in_BETA_EXAM[BD_in_BETA_EXAM.ENTREZ.isin(BETA_lm_genes.gene_id)]['ENTREZ']) # 208

BD_in_CELLO_DC_BETA = BD_summary7[BD_summary7.PUBCHEM_CID.isin(CELLO_DC_BETA_cids)]
set(BD_in_CELLO_DC_BETA[BD_in_CELLO_DC_BETA.ENTREZ.isin(BETA_lm_genes.gene_id)]['ENTREZ']) # 109 




################## GRAPH examples 
# (1) SRING_PPI - has score 

PPI_N_PATH = '/st06/jiyeonH/00.STRING_v.11.5/'
PPI_11_5_raw = pd.read_csv(PPI_N_PATH+'9606.protein.links.v11.5.txt', sep = ' ')
PPI_11_5_info = pd.read_csv(PPI_N_PATH+'9606.protein.info.v11.5.txt', sep = '\t')
PPI_alias_info = pd.read_csv(PPI_N_PATH+'9606.protein.aliases.v11.5.txt', sep = '\t')


PPI_11_5_info_filter_1 = PPI_11_5_info[PPI_11_5_info.preferred_name.isin(BETA_lm_genes.gene_symbol)]
check_done = BETA_lm_genes[BETA_lm_genes.gene_symbol.isin(PPI_11_5_info.preferred_name)==True]
check_alias = BETA_lm_genes[BETA_lm_genes.gene_symbol.isin(PPI_11_5_info.preferred_name)==False]

# alias check
mini_check = []

for GG in list(check_alias.gene_symbol) :
	tmp_ali = PPI_alias_info[PPI_alias_info.alias == GG]
	tmp_ali_PID = list(set(tmp_ali['#string_protein_id']))
	if len(tmp_ali_PID) == 1 :
		PID = tmp_ali_PID[0]
		if (PID in PPI_11_5_info_filter_1['#string_protein_id']) == False :
			mini_check.append((GG, PID))
		else : 
			print("already in it : {}".fotmat(GG))
	else:
		print("multiple PID : {}".fotmat(GG))

PPI_11_5_info_filter_2 = copy.deepcopy(PPI_11_5_info_filter_1)

for mini in mini_check : 
	mm = pd.DataFrame({'#string_protein_id' : [mini[1]] ,'preferred_name' : [mini[0]] })
	PPI_11_5_info_filter_2 = pd.concat([PPI_11_5_info_filter_2, mm], axis = 0)


check_done2 = pd.merge(check_done, PPI_11_5_info[['#string_protein_id','preferred_name']], left_on='gene_symbol', right_on='preferred_name', how= 'left')
check_alias2 = copy.deepcopy(check_alias)
check_alias2['#string_protein_id'] = [a[1] for a in mini_check]
check_alias2['preferred_name'] = [a[0] for a in mini_check]

PPI_11_5_info_filter_3 = pd.concat([check_done2,check_alias2])

PPI_11_5_raw_1 = PPI_11_5_raw[PPI_11_5_raw.protein1.isin(PPI_11_5_info_filter_3['#string_protein_id'])]
PPI_11_5_raw_2 = PPI_11_5_raw_1[PPI_11_5_raw_1.protein2.isin(PPI_11_5_info_filter_3['#string_protein_id'])] # 103758

PPI_result_1 = pd.merge(PPI_11_5_raw_2, PPI_11_5_info_filter_3[['#string_protein_id', 'preferred_name']] , left_on = 'protein1', right_on = '#string_protein_id', how = 'left')
PPI_result_2 = pd.merge(PPI_result_1, PPI_11_5_info_filter_3[['#string_protein_id', 'preferred_name']] , left_on = 'protein2', right_on = '#string_protein_id', how = 'left')

PPI_result_3 = PPI_result_2[['preferred_name_x','preferred_name_y','combined_score']]
PPI_result_35 = PPI_result_3[PPI_result_3.combined_score>=400]
PPI_result_4 = PPI_result_35[['preferred_name_x','preferred_name_y']].drop_duplicates()

G_mini_115 = nx.from_pandas_edgelist(PPI_result_4, 'preferred_name_x', 'preferred_name_y')
A_mini_115 = nx.adjacency_matrix(G_mini_115) 
A_tmp_mini_115 = A_mini_115.toarray()
A_tmp_mini_115 = A_tmp_mini_115.astype(np.float32) ###################################### ADJ ! 

G_115_mini_ORDER = list(G_mini_115.nodes) # 여기 순서로 앞으로 진행하기 # 400으로 자르면 952 개 



# (2) SNU PPI - no score ()

SNU_PATH = '/st06/jiyeonH/11.TOX/SNU_PPI/'
SNU_FILE = 'Human_TGN_f4_0.9_Source_20200625.csv'
ENTREZ = 'gene_info'

SNU_PPI = pd.read_csv(SNU_PATH+SNU_FILE) # 892,741
ENTREZ_ID = pd.read_csv(SNU_PATH+ENTREZ, sep = '\t')
SNU_PPI_IDS = set(list(SNU_PPI.X1) + list(SNU_PPI.X2)) 


# 앙뜨레즈 테이블은 landmark 를 잘 보필하는가 
ENTREZ_LINCS = ENTREZ_ID[ENTREZ_ID.GeneID.isin(list(BETA_lm_genes.gene_id))]

ENTREZ_ID_FILTER = ENTREZ_LINCS[['GeneID','Symbol','Synonyms']] 

SNU_PPI_1 = SNU_PPI[SNU_PPI.X1.isin(list(ENTREZ_ID_FILTER.GeneID))]
SNU_PPI_2 = SNU_PPI_1[SNU_PPI_1.X2.isin(list(ENTREZ_ID_FILTER.GeneID))] # 이미 landmark filter

SNU_PPI_3 = pd.merge(SNU_PPI_2, ENTREZ_ID_FILTER, left_on ='X1', right_on = 'GeneID', how ='left')
SNU_PPI_4 = pd.merge(SNU_PPI_3, ENTREZ_ID_FILTER, left_on ='X2', right_on = 'GeneID', how ='left')

SNU_PPI_5 = SNU_PPI_4[[ 'GeneID_x','GeneID_y', 'Symbol_x','Symbol_y' ]]
SNU_PPI_5 = SNU_PPI_5.reset_index(drop=True)

SNU_PPI_6 = SNU_PPI_5.drop_duplicates()
SNU_PPI_6 = SNU_PPI_6.reset_index(drop=True)

SNU_G_mini = nx.from_pandas_edgelist(SNU_PPI_6, 'GeneID_x', 'GeneID_y')

MSSNG = ENTREZ_LINCS[ENTREZ_LINCS['GeneID'].isin(list(SNU_G_mini.nodes))==False]['GeneID']
for nn in list(MSSNG):
	SNU_G_mini.add_node(nn)

SNU_G_A_mini = nx.adjacency_matrix(SNU_G_mini) 
SNU_G_A_tmp_mini = SNU_G_A_mini.toarray()
SNU_G_A_tmp_mini = SNU_G_A_tmp_mini.astype(np.float32) ######################### ADJ ! 

SNU_GENE_ORDER_mini = list(SNU_G_mini.nodes) # 여기 순서로 앞으로 진행하기 




# (3) IDEKER PPI 

# check IDEKER 

IDEKER_IAS = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/IAS_score.tsv',sep = '\t')
IDEKER_TOT_GS = list(set(list(IDEKER_IAS['Protein 1'])+list(IDEKER_IAS['Protein 2']))) # 16840

L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

IDEKER_IAS_L1 = IDEKER_IAS[IDEKER_IAS['Protein 1'].isin(L_matching_list.PPI_name)]
IDEKER_IAS_L2 = IDEKER_IAS_L1[IDEKER_IAS_L1['Protein 2'].isin(L_matching_list.PPI_name)] # 20232

len(set(list(IDEKER_IAS_L2['Protein 1']) + list(IDEKER_IAS_L2['Protein 2'])))
ID_G = nx.from_pandas_edgelist(IDEKER_IAS_L2, 'Protein 1', 'Protein 2')

ESSEN_NAMES = list(set(L_matching_list['PPI_name']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)


ID_GENE_ORDER_mini = list(ID_G.nodes()) # 20232
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40464]
ID_WEIGHT = [] # len : 20232 -> 40464


# weight 효과적으로 뽑아내는 방법? 
그래야 GAT 랑 GCN 에 넣어줄 수 있는데 고민이네 


ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['Protein 1','Protein 2']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['Protein 1'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['Protein 2'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = IDEKER_IAS[['Protein 1', 'Protein 2', 'Integrated score']]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['Protein 1']+'__'+IAS_FILTER['Protein 2']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['Protein 2']+'__'+IAS_FILTER['Protein 1']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','Integrated score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'Integrated score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'Integrated score']]
IAS_FILTER1.columns = ['NAMESUM', 'Integrated score']
IAS_FILTER2.columns = ['NAMESUM', 'Integrated score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0) 

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' )


ID_WEIGHT_SCORE = list(ID_WEIGHT['Integrated score'])














