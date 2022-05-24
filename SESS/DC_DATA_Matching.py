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

# 다시 필터링
DC_DATA_filter = DC_tmp_DF2[['drug_row_id','drug_col_id','cell_line_id','synergy_loewe']]
DC_DATA_filter2 = DC_DATA_filter.drop_duplicates()
DC_DATA_filter3 = DC_DATA_filter2[DC_DATA_filter2.drug_col_id>0]
DC_DATA_filter4 = DC_DATA_filter3[DC_DATA_filter3.drug_row_id>0]
DC_DATA_filter4.cell_line_id # unique 295





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

DC_CELL_DF.to_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/cell_lines.csv')

DC_CELL_DF2 = DC_CELL_DF[['id','name','cellosaurus_accession', 'ccle_name']] # 751450
DC_CELL_DF2.columns = ['cell_line_id', 'DC_cellname','DrugCombCello', 'DrugCombCCLE']




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
BETA_BIND = pd.read_csv("/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_SS_df.978.csv")
BETA_SELEC_SIG = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/SIG_INFO.220405') # cell 58가지 
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
BETA_CID_CELLO_SIG = BETA_CID_CELLO_SIG[beta_cid_cello_sig_tf]

ccle_tt=[True if type(a)==str else False for a in list(BETA_SELEC_SIG_wCell2.ccle_name)]
BETA_CID_CCLE_SIG = BETA_SELEC_SIG_wCell2[ccle_tt][['pubchem_cid','ccle_name','sig_id']].drop_duplicates() # 110620
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[BETA_CID_CCLE_SIG.ccle_name!='NA'] 
beta_cid_ccle_sig_tf = [ True if a>0 else False for a in list(BETA_CID_CCLE_SIG.pubchem_cid)]
BETA_CID_CCLE_SIG = BETA_CID_CCLE_SIG[beta_cid_ccle_sig_tf]




# DrugComb & LINCS 비교
# 한번만 다시 확인 LINCS A - LINCS B - DC_result - Cell
DC_cello_final_dup
DC_ccle_final_dup
BETA_CID_CCLE_SIG
BETA_CID_CELLO_SIG

# cello
BETA_CID_CELLO_SIG.columns=['drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9271
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 


# ccle 
BETA_CID_CCLE_SIG.columns=['drug_row_cid', 'DrugCombCCLE', 'BETA_sig_id']
CCLE_DC_BETA_1 = pd.merge(DC_ccle_final_dup, BETA_CID_CCLE_SIG, on = ['drug_row_cid','DrugCombCCLE'], how = 'left')

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




##################

GRAPH examples 







