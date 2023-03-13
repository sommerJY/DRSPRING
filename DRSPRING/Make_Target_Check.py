conda activate JY_6
~/.conda/envs/JY_6/bin/python

import pandas as pd
import json
import copy



																																	00. 유전자 ID match 다시 
																																	00. 유전자 ID match 다시 
																																	00. 유전자 ID match 다시 

																																	LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
																																	LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
																																	LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
																																	LINCS_978_names = list(set(LINCS_978.ensembl_id))
																																	# LINCS_978.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/LINCS_978.csv', sep = '\t')
																																	LINCS_978 = LINCS_978[['gene_id', 'gene_symbol', 'ensembl_id']]
																																	LINCS_978.columns = ['entrez_id', 'gene_symbol', 'ensembl_id']

																																	# uniprot change : 
																																	LINCS_978_uniprot = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/Uniprot_978.csv', sep = '\t')
																																	LINCS_978_uniprot_add = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/Uniprot_978_add.csv', sep = '\t')
																																	LINCS_978_uniprot = pd.concat([LINCS_978_uniprot, LINCS_978_uniprot_add])
																																	LINCS_978_uniprot_reviewed = LINCS_978_uniprot[LINCS_978_uniprot.Reviewed=='reviewed']
																																	LINCS_978_uniprot_reviewed = LINCS_978_uniprot_reviewed.reset_index(drop = True)
																																	LINCS_978_uniprot_reviewed.at[980, 'From'] = 'ENSG00000137700'
																																	LINCS_978_uniprot_reviewed2 = LINCS_978_uniprot_reviewed[['From','Entry','Entry Name', 'Gene Names','STRING','BioGRID','BindingDB','DrugBank','ChEMBL','OpenTargets','PharmGKB','KEGG']]
																																	LINCS_978_uniprot_reviewed2.columns = ['ensembl_id','Uniprot','Uniprot Name', 'Uniprot Gene Names','STRING','BioGRID','BindingDB','DrugBank','ChEMBL','OpenTargets','PharmGKB','KEGG']


																																	[a for a in LINCS_978_names if list(LINCS_978_uniprot_reviewed['From']).count(a) !=1]
																																	# ['ENSG00000147889', 'ENSG00000087460', 'ENSG00000137700']
																																	# LINCS_978_uniprot_reviewed2 = LINCS_978_uniprot_reviewed.drop([548, 1135, 1136])


																																	LINCS_978_id = pd.merge(LINCS_978, LINCS_978_uniprot_reviewed2, on = 'ensembl_id', how = 'left' )
																																	LINCS_978_id.STRING = [a.split(';')[0] if type(a) == str else 'NA' for a in list(LINCS_978_id.STRING)]
																																	LINCS_978_id.ChEMBL = [a.split(';')[0] if type(a) == str else 'NA' for a in list(LINCS_978_id.ChEMBL)]
																																	LINCS_978_id.BioGRID = [a.split(';')[0] if type(a) == str else 'NA' for a in list(LINCS_978_id.BioGRID)]
																																	LINCS_978_id.BindingDB = [a.split(';')[0] if type(a) == str else 'NA' for a in list(LINCS_978_id.BindingDB)]
																																	LINCS_978_id.OpenTargets = [a.split(';')[0] if type(a) == str else 'NA' for a in list(LINCS_978_id.OpenTargets)]
																																	LINCS_978_id.PharmGKB = [a.split(';')[0] if type(a) == str else 'NA' for a in list(LINCS_978_id.PharmGKB)]



																																	978 개 말고 나머지는? 
																																	978 개 말고 나머지는? 
																																	978 개 말고 나머지는? 


																																	# awk '$1==9606' NCBI_gene2refseq > NCBI_gene2refseq.cut

																																	ncbi_gene_info = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene_info.cut2', sep='\t', header = None)
																																	ncbi_gene_info.columns = ['#tax_id', 'GeneID', 'Symbol', 'Synonyms', 'type_of_gene', 'Nomenclature_status', 'Feature_type']

																																	ncbi_gene_info2 = ncbi_gene_info[ncbi_gene_info.type_of_gene == 'protein-coding'] # 20605
																																	# ncbi_gene_info2.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene_info.cut3', sep = '\t')
																																	ncbi_gene_info3 = ncbi_gene_info2[['GeneID','Symbol','Synonyms']].drop_duplicates()
																																	# 20605


																																	ncbi_gene_ensem_info = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene2ensembl.cut', sep='\t', header = None)
																																	ncbi_gene_ensem_info.columns = ['#tax_id','GeneID','Ensembl_gene_identifier', 'RNA_nucleotide_accession.version', 'Ensembl_rna_identifier','protein_accession.version','Ensembl_protein_identifier']
																																	ncbi_gene_ensem_info_re = ncbi_gene_ensem_info[['GeneID','Ensembl_gene_identifier']].drop_duplicates()
																																	# 36503


																																	# 20663 -> protein coding genes only 
																																	ncbi_gene_ensem_MERGE = pd.merge(ncbi_gene_info3, ncbi_gene_ensem_info_re,  on = 'GeneID', how = 'left')


																																	# 19105 -> 왜 차이가 나지 (uniprot 에서 인식 안되는 애들이 꽤 있음 )
																																	ncbi_gene_info_uniprot = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/TOTAL_GENE_UNIPROT', sep='\t', low_memory = False)
																																	ncbi_gene_info_uniprot_re = ncbi_gene_info_uniprot[ncbi_gene_info_uniprot.Reviewed == 'reviewed']
																																	ncbi_gene_info_uniprot_re.columns = ['GeneID', 'UniProt', 'Reviewed', 'Entry Name', 'Protein names',
																																		'Gene Names', 'Organism', 'Length', 'Organism (ID)', 'EMBL', 'BioGRID',
																																		'STRING', 'BindingDB', 'ChEMBL', 'DrugBank', 'Ensembl', 'UCSC', 'CTD',
																																		'OpenTargets', 'PharmGKB', 'HGNC']
																																	ncbi_gene_info_uniprot_re2 = ncbi_gene_info_uniprot_re[['GeneID','Entry Name','UniProt','STRING','HGNC']].drop_duplicates()


																																	ncbi_gene_ensem_uniprot_MERGE = pd.merge(ncbi_gene_ensem_MERGE, ncbi_gene_info_uniprot_re2, on = 'GeneID', how = 'left')

																																	ncbi_gene_ensem_uniprot_MERGE.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/TOTAL_GENE_enz_ens_unip.csv', sep ='\t')




																																	# STRING check 
																																	# STRING check 
																																	# STRING check 

																																	PPI_N_PATH = '/st06/jiyeonH/00.STRING_v.11.5/'
																																	PPI_11_5_raw = pd.read_csv(PPI_N_PATH+'9606.protein.links.v11.5.txt', sep = ' ')
																																	PPI_11_5_info = pd.read_csv(PPI_N_PATH+'9606.protein.info.v11.5.txt', sep = '\t')
																																	PPI_alias_info = pd.read_csv(PPI_N_PATH+'9606.protein.aliases.v11.5.txt', sep = '\t')


																																	PPI_11_5_info = PPI_11_5_info[['#string_protein_id','preferred_name']]
																																	PPI_11_5_info.columns = ['#string_protein_id','STRING_gene']

																																	PPI_11_5_info_filter_1 = PPI_11_5_info[PPI_11_5_info['#string_protein_id'].isin(LINCS_978_id.STRING)]

																																	check_done = LINCS_978_id[LINCS_978_id.STRING.isin(PPI_11_5_info['#string_protein_id'])==True]
																																	check_alias = LINCS_978_id[LINCS_978_id.STRING.isin(PPI_11_5_info['#string_protein_id'])==False]


																																	# alias check
																																	mini_check = []

																																	for GG in list(check_alias.ensembl_id) :
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
																																		mm = pd.DataFrame({'#string_protein_id' : [mini[1]] ,'STRING_gene' : [mini[0]] })
																																		PPI_11_5_info_filter_2 = pd.concat([PPI_11_5_info_filter_2, mm], axis = 0)

																																	PPI_11_5_info_filter_2 = PPI_11_5_info_filter_2.drop_duplicates()

																																	check_done2 = pd.merge(check_done, PPI_11_5_info, left_on='STRING', right_on='#string_protein_id', how= 'left')

																																	check_alias2 = copy.deepcopy(check_alias)
																																	check_alias2['#string_protein_id'] = [a[1] for a in mini_check]
																																	check_alias2['STRING_alias'] = [a[0] for a in mini_check]
																																	check_alias3 = pd.merge(check_alias2, PPI_11_5_info, on='#string_protein_id', how='left')
																																	check_alias3 = check_alias3.drop('STRING_alias', axis =1)

																																	PPI_11_5_info_filter_3 = pd.concat([check_done2,check_alias3])

																																	LINCS_978_id_STck = copy.deepcopy(PPI_11_5_info_filter_3)

																																	LINCS_978_id_STck.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/LINCS_978_reck.csv', sep = '\t')
																																	# uniprot 여러개 붙는거 때문에 981 row 임 

















0. Chemical ID match 
0. Chemical ID match 
0. Chemical ID match 

BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 

# NCBI - entrez & ensembl identifier  - not reliable but used once  
# download : https://ftp.ncbi.nih.gov/gene/DATA/gene2ensembl.gz (2022-12-06 ver )
ensem_all = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene2ensembl.cut', sep = '\t', header =None)
ensem_all.columns = ['tax','GeneID','Ensembl_gene_identifier','NMID','ensembl_t','NPID','Ensembl_protein_identifier']
ensem_filter = ensem_all[['GeneID','Ensembl_gene_identifier','Ensembl_protein_identifier']].drop_duplicates()
# 36382


# uniprot 
# download : https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/idmapping/by_organism/HUMAN_9606_idmapping.dat.gz     
uni_data = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/UNIPROT_HUMAN_9606_idmapping.dat', sep = '\t', header = None)
uni_data.columns = ['ID','DB','DATA']
#uni_data_re = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/UNIPROT_HUMAN_9606_idmapping_selected', sep = '\t', header = None, low_memory = False)


# hgnc get 
# downlaod : https://www.genenames.org/download/archive/ & Current tab separated hgnc_complete_set file
hgnc_full = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/hgnc_complete_set.txt', sep ='\t', low_memory = False)
hgnc_full_re = hgnc_full[['hgnc_id','symbol','alias_symbol','prev_symbol','entrez_id','ensembl_gene_id','uniprot_ids',]]


# awk '$1==9606' NCBI_gene2refseq > NCBI_gene2refseq.cut
# download : https://ftp.ncbi.nih.gov/gene/DATA/gene_info.gz 
ncbi_gene_info = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene_info.cut2', sep='\t', header = None)
ncbi_gene_info.columns = ['#tax_id', 'GeneID', 'Symbol', 'Synonyms', 'type_of_gene', 'Nomenclature_status', 'Feature_type']


[a for a in list(hgnc_full_re.uniprot_ids) if a not in  list(uni_data.ID)]


LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978_names = list(set(LINCS_978.ensembl_id))

PPI_N_PATH = '/st06/jiyeonH/00.STRING_v.11.5/'
PPI_11_5_raw = pd.read_csv(PPI_N_PATH+'9606.protein.links.v11.5.txt', sep = ' ')
PPI_11_5_info = pd.read_csv(PPI_N_PATH+'9606.protein.info.v11.5.txt', sep = '\t')
PPI_alias_info = pd.read_csv(PPI_N_PATH+'9606.protein.aliases.v11.5.txt', sep = '\t')













0) CHEMBL
chembl ID - 여러 uniprot - 여러 entrez 가능 


# chembl ID 매칭 시키기 
전체 interaction 에 대해서 확인해야해서 
해당하는 파일을 pubchem 에서 돌려서 확인하기로 함.
확인해보니 LINCS annotation 받는거보다 이게 더 붙음 

# # https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi 에 넣어서 돌림  
chembl_int_total = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ChEMBL.22.11.18/PubChem_idexchange_CHEMBL', sep = '\t', header = None)
chembl_int_total.columns = ['CHEMBL_CHEM_ID', 'CID']
chembl_int_total = chembl_int_total[chembl_int_total.CID>0].drop_duplicates() # 2544 인데, CID 가 여러개 붙는 chembl 도 있다는거. 참고해야함. 
chembl_int_total = chembl_int_total.reset_index(drop=True) # 2544

# set([a for a in check if check.count(a) > 1])
# 'CHEMBL386630', 'CHEMBL2325741', 'CHEMBL902', 'CHEMBL450044', 'CHEMBL577', 'CHEMBL641', 'CHEMBL284483', 'CHEMBL503', 'CHEMBL656'
# 몇개 안되길래 홈페이지에 직접 검색해봄 
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL386630'] # 1622
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL2325741'] # 373
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL902'] # no drop 
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL450044'] # 2451
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL577'] # 1455
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL641'] # 1976
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL284483'] # 940
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL503'] # 1397
chembl_int_total[chembl_int_total.CHEMBL_CHEM_ID=='CHEMBL656'] # 82 

chembl_int_total = chembl_int_total.drop([1622, 373, 2451, 1455, 1976, 940, 1397, 82]) # 2536 
chembl_int_total = chembl_int_total.drop_duplicates()


# target id check 
uniprot 에 있는 ID 는 entrez 하나인지 
노노.
chembl - 여러 uniprot - 해당 entrez들 

uni_entrez = uni_data[uni_data.DB == 'GeneID'].drop_duplicates()
uni_entrez = uni_entrez[['ID','DATA']]

uni_chembl = uni_data[uni_data.DB == 'ChEMBL'].drop_duplicates()
uni_chembl = uni_chembl[['DATA','ID']] 

target_id_ck = pd.merge(uni_chembl, uni_entrez, on ='ID', how = 'left')
target_id_ck.columns=['target_chembl','uniprot_id','entrez_id']
target_id_ck['entrez_id2'] = [float(a) if type(a) == str else 0 for a in list(target_id_ck.entrez_id)]


# 메인 interaction 
DIR = '/st06/jiyeonH/13.DD_SESS/ChEMBL.22.11.18/'
parent_target = pd.read_csv(DIR+'Chembl_Drug_Mechanism.csv')
parent_target = parent_target[['Parent Molecule ChEMBL ID', 'Parent Molecule Name', 'Parent Molecule Type', 'Target ChEMBL ID','Target Name', 'Action Type', 'Target Type', 'Target Organism','ATC Codes']]
parent_target.columns = ['CHEMBL_CHEM_ID', 'CHEMBL_CHEM_NAME', 'CHEMBL_CHEM_TYPE', 'CHEMBL_TARGET_ID','CHEMBL_TARGET_NAME', 'CHEMBL_ACTION', 'CHEMBL_TARGET_TYPE', 'CHEMBL_TARGET_ORG','CHEMBL_CHEM_ATC']
# 3821
parent_target = parent_target[['CHEMBL_CHEM_ID','CHEMBL_TARGET_ID']].drop_duplicates() # 3819


cb_int_1 = pd.merge(parent_target, chembl_int_total, left_on = 'CHEMBL_CHEM_ID', right_on='CHEMBL_CHEM_ID', how = 'left' ) # 3820
# 3412 cid, 

cb_int_2 = pd.merge(cb_int_1, target_id_ck, left_on = 'CHEMBL_TARGET_ID', right_on='target_chembl', how='left' ) # 4616 

# len(set(cb_int_2[cb_int_2.CID>0]['CID'])) # 2536
# len(set(cb_int_2[cb_int_2.entrez_id2>0]['entrez_id2'])) # 713

cb_int_3 = cb_int_2[cb_int_2.CID>0] # 
cb_int_4 = cb_int_3[cb_int_3.entrez_id2>0] # 537

CHEMBL_FINAL = cb_int_4[['CID', 'entrez_id2']].drop_duplicates()

CHEMBL_FINAL.columns = ['CID','EntrezID']
CHEMBL_FINAL['DB'] = 'ChEMBL'












1) IUPHAR 

DIR = '/st06/jiyeonH/13.DD_SESS/IUPHAR.2022.1/'

# approved_drug_primary_target_interactions = pd.read_csv(DIR+'approved_drug_primary_target_interactions.csv', skiprows = 1) # 1158
ligand_id_mapping = pd.read_csv(DIR+'ligand_id_mapping.csv', skiprows = 1)
interactions = pd.read_csv(DIR+'interactions.csv',low_memory=False , skiprows = 1) # 21145
interactions = interactions[interactions['target_species']=='Human']

len(set(interactions.ligand_id)) # 8135

int_check = interactions[interactions['target_id']>0]
int_check2 = int_check[['ligand_id','target_id']].drop_duplicates() # 17377
len(set(int_check2.target_id)) # 1771



# get cid 
iuphar_id = ligand_id_mapping[['Ligand id', 'PubChem CID' ]]
iuphar_int = interactions[['ligand','ligand_id','target','target_id','target_ensembl_gene_id']]
iuphar_int_1 = iuphar_int[iuphar_int.target_id>0].drop_duplicates() 

iuphar_int_2 = pd.merge(iuphar_int_1, iuphar_id, left_on = 'ligand_id', right_on = 'Ligand id', how = 'left') # 19055
iuphar_int_2['CID'] = [int(a) if type(a) == str else 0 for a in list(iuphar_int_2['PubChem CID']) ] # 19055
iuphar_int_2['ENSG'] = [a if type(a) == str else 'NA' for a in list(iuphar_int_2['target_ensembl_gene_id']) ] # 19055


# pubchem cid check 
# DOWNLOAD : https://pubchem.ncbi.nlm.nih.gov/source/IUPHAR-DB#data=Annotations 에서
# Chemical-Target Interactions (Compound) 다운로드 
with open('/st06/jiyeonH/13.DD_SESS/IUPHAR.2022.1/Pubchem_IUPHAR.json', 'r') as tmp_json:
	pc_iuphar = json.load(tmp_json)

iuphar_cid_RE = pd.DataFrame(columns=['IUPHAR_ID', 'CID'])

for ind in range(len(pc_iuphar['Annotations']['Annotation'])):
	anno = pc_iuphar['Annotations']['Annotation'][ind]
	IUPHAR_ID = anno['SourceID']
	if 'LinkedRecords' in list(anno.keys()) :
		if 'CID' in anno['LinkedRecords'].keys():
			CID = anno['LinkedRecords']['CID']
		else : 
			CID=['NA']
	else:
		CID=['NA']
	tmp_df = pd.DataFrame({'IUPHAR_ID' : IUPHAR_ID, 'CID': CID})
	iuphar_cid_RE = pd.concat([iuphar_cid_RE, tmp_df])
# NOPE


# ensembl | check 
iuphar_int_3_1_ck = [a for a in range(iuphar_int_2.shape[0]) if '|' in list(iuphar_int_2.ENSG)[a] ]
iuphar_int_3_2_ck = [a for a in range(iuphar_int_2.shape[0]) if '|' not in list(iuphar_int_2.ENSG)[a] ]

iuphar_int_3_1 = iuphar_int_2.loc[iuphar_int_3_1_ck]
iuphar_int_3_2 = iuphar_int_2.loc[iuphar_int_3_2_ck]
iuphar_int_3_1 = iuphar_int_3_1.reset_index(drop=True)

for a in range(iuphar_int_3_1.shape[0]) :
	tmp = list(iuphar_int_3_1['ENSG'])[a].split('|')
	tmpdf = pd.concat([iuphar_int_3_1.loc[a:a,]]*len(tmp))
	tmpdf['ENSG'] = tmp
	iuphar_int_3_2 = pd.concat([iuphar_int_3_2, tmpdf])



# ensem & entrez
id_check = hgnc_full_re[['entrez_id','ensembl_gene_id']].drop_duplicates()
ch_1 = list(iuphar_int_3_2.ENSG)
ch_2 = list(hgnc_full_re.ensembl_gene_id)

uni_ensem = uni_data[uni_data.DB=='Ensembl']
uni_ensem['ENSG'] = [a.split('.')[0] for a in uni_ensem['DATA']]
ch_3 = list(uni_ensem.ENSG)

[a for a in ch_1 if a not in ch_2] # 22
# 'ENSG00000103522|ENSG00000147168'

[a for a in ch_1 if a not in ch_3]  # 17
# 'ENSG00000087916', 'ENSG0000010538', 'ENSG00000068383;', 'ENSG00000099118', 'ENSG00000150086', 'ENSG00000132142', 'ENSG00000222040', 'ENSG00000143466', 'ENSG00000183473', 'ENSG00000108516', 'ENSG00000168918;', 'ENSG00000183729', 'ENSG00000143140', 'ENSG00000147402', 'NA', 'ENSG00000110347', 'ENSG00000049319'

iuphar_int_4 = pd.merge(iuphar_int_3_2, id_check, left_on ='ENSG' , right_on ='ensembl_gene_id', how='left' )

IUPHAR_FINAL = iuphar_int_4[['CID','entrez_id']].drop_duplicates()
IUPHAR_FINAL = IUPHAR_FINAL[IUPHAR_FINAL['CID']>0] # CID : 6168
IUPHAR_FINAL = IUPHAR_FINAL[IUPHAR_FINAL['entrez_id']>0] # entrez : 1546
IUPHAR_FINAL.columns = ['CID','EntrezID'] # 12398







2) TTD

DIR = '/st06/jiyeonH/13.DD_SESS/TTD.21.11.08/'

ttd_target = pd.read_csv(DIR+'P1-01-TTD_target_download.txt', skiprows = 40, sep = '\t', header = None)
ttd_target.columns = ['TargetID','col','content1', 'content2', 'content3']
ttd_list = list(set(ttd_target['TargetID']))
ttd_list = [a for a in ttd_list if type(a) == str]

ttd_target_id = pd.DataFrame(columns = ['TARGETID', 'GENENAME', 'UNIPROID', 'TARGTYPE'])
for Key_target in ttd_list :
	tmp_pd = ttd_target[ttd_target.TargetID == Key_target ]
	if 'GENENAME' in list(tmp_pd['col']) :
		GENENAME = list(tmp_pd[tmp_pd['col'] == 'GENENAME']['content1'])[0]
	else :
		GENENAME = 'NA'
	#
	if 'UNIPROID' in list(tmp_pd['col']) :
		UNIPROID = list(tmp_pd[tmp_pd['col'] == 'UNIPROID']['content1'])[0]
	else :
		UNIPROID = 'NA'    
	#
	if 'TARGTYPE' in list(tmp_pd['col']) :
		TARGTYPE = list(tmp_pd[tmp_pd['col'] == 'TARGTYPE']['content1'])[0]
	else :
		TARGTYPE = 'NA'    
	#
	tmp_pd_re = pd.DataFrame({'TARGETID':[Key_target], 'GENENAME':[GENENAME], 'UNIPROID':[UNIPROID], 'TARGTYPE':[TARGTYPE]})
	ttd_target_id = pd.concat([ttd_target_id, tmp_pd_re])
# 하나씩만 붙은거 확인했음 



ttd_cross_match = pd.read_csv(DIR+'P1-03-TTD_crossmatching.txt', skiprows = 28, sep = '\t', header = None)
ttd_cross_match.columns = ['DrugID','col','content']
ttd_list = list(set(ttd_cross_match['DrugID']))
ttd_list = [a for a in ttd_list if type(a) == str]

ttd_cid = pd.DataFrame(columns = ['DrugID', 'CID'])
for Key_drug in ttd_list :
	tmp_pd = ttd_cross_match[ttd_cross_match.DrugID == Key_drug ]
	if 'PUBCHCID' in list(tmp_pd['col']) :
		CID = tmp_pd.loc[tmp_pd[tmp_pd['col'] == 'PUBCHCID'].index,'content'].values[0]
	else :
		CID = 'NA'
	tmp_pd_re = pd.DataFrame({'DrugID':[Key_drug], 'CID':[CID]})
	ttd_cid = pd.concat([ttd_cid, tmp_pd_re])
# 여러개 붙는 애들도 있음 

ttd_cid = ttd_cid.reset_index(drop=True)

ttd_cid_1_ck = [True if ';' in a else False for a in list(ttd_cid['CID'])]
ttd_cid_1 = ttd_cid[ttd_cid_1_ck]
ttd_cid_1 = ttd_cid_1.reset_index(drop=True)

ttd_cid_2_ck = [False if ';' in a else True for a in list(ttd_cid['CID'])]
ttd_cid_2 = ttd_cid[ttd_cid_2_ck]

for i in range(ttd_cid_1.shape[0]) :
	tmp = ttd_cid_1.loc[i]
	ID = tmp['DrugID']
	CIDs = [int(a) for a in tmp['CID'].replace(' ', '').split(';') if a != '']
	new_df = pd.DataFrame({'DrugID' : [ID]*len(CIDs) , 'CID' : CIDs})
	ttd_cid_2 = pd.concat([ttd_cid_2, new_df])
# 여러개 붙었으면, 그냥 CID 다 데려오기로 함 



# 44666
ttd_Drug_TargetMapping = pd.read_excel(DIR+'P1-07-Drug-TargetMapping.xlsx')

len(set(ttd_Drug_TargetMapping.DrugID))
len(set(ttd_Drug_TargetMapping.TargetID))

ttd_map_1 = pd.merge(ttd_Drug_TargetMapping, ttd_target_id, left_on = 'TargetID', right_on = 'TARGETID', how = 'left')
ttd_map_2 = pd.merge(ttd_map_1, ttd_cid_2, left_on = 'DrugID', right_on = 'DrugID', how = 'left')

uni_entry = uni_data[uni_data.DB=='UniProtKB-ID']
uni_entrez = uni_data[uni_data.DB=='GeneID']

id_check = uni_entry[['DATA','ID']]
id_check2 = pd.merge(id_check, uni_entrez[['ID','DATA']] , left_on = 'ID', right_on = 'ID' , how = 'left')
id_check2.columns = ['Entry','DrugID','GeneID']

ttd_map_3 = pd.merge(ttd_map_2, id_check2, left_on = 'UNIPROID', right_on = 'Entry', how = 'left')
tf_check = [True if type(a) == str else False for a in ttd_map_3.CID]
ttd_map_4 = ttd_map_3[tf_check]
ttd_map_4['CID2'] = [int(a) if a!= 'NA' else 0 for a in ttd_map_4.CID]
ttd_map_4['GeneID2'] = [a if type(a)== str else 'NA' for a in ttd_map_4.GeneID]
ttd_map_4['GeneID3'] = [int(a) if a!= 'NA' else 0 for a in ttd_map_4.GeneID2]

# len(set(ttd_map_4[ttd_map_4.CID2>0]['CID2'])) # 22842
# len(set(ttd_map_4[ttd_map_4.GeneID3>0]['GeneID3'])) # 1639

ttd_map_5 = ttd_map_4[(ttd_map_4.CID2>0) & (ttd_map_4.GeneID3>0)]


TTD_FINAL = ttd_map_5[['CID2','GeneID3']].drop_duplicates()
TTD_FINAL.columns = ['CID','EntrezID']







3) CTD

ctd = pd.read_csv("/st06/jiyeonH/13.DD_SESS/CTD/CTD_chem_gene_ixns.csv",low_memory=False,skiprows =29, names=["ChemicalName","ChemicalID","CasRN","GeneSymbol","GeneID","GeneForms","Organism","OrganismID","Interaction","InteractionActions","PubMedIDs"])

ctd_drug=pd.read_csv("/st06/jiyeonH/13.DD_SESS/CTD/CTD_chemicals.csv",low_memory=False,skiprows =29, names=["ChemicalName",'ChemicalID','CasRN','Definition','ParentIDs','TreeNumbers','ParentTreeNumbers','Synonyms'])

# https://pubchem.ncbi.nlm.nih.gov/source/Comparative%20Toxicogenomics%20Database#data=Annotations 에서 
# Chemical-Target Interactions (Compound) 다운로드 

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_1.json', 'r') as tmp_json:
	pc_ctd_chem_gene1 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_2.json', 'r') as tmp_json:
	pc_ctd_chem_gene2 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_3.json', 'r') as tmp_json:
	pc_ctd_chem_gene3 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_4.json', 'r') as tmp_json:
	pc_ctd_chem_gene4 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_5.json', 'r') as tmp_json:
	pc_ctd_chem_gene5 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_6.json', 'r') as tmp_json:
	pc_ctd_chem_gene6 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_7.json', 'r') as tmp_json:
	pc_ctd_chem_gene7 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_8.json', 'r') as tmp_json:
	pc_ctd_chem_gene8 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_9.json', 'r') as tmp_json:
	pc_ctd_chem_gene9 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_10.json', 'r') as tmp_json:
	pc_ctd_chem_gene10 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_11.json', 'r') as tmp_json:
	pc_ctd_chem_gene11 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_12.json', 'r') as tmp_json:
	pc_ctd_chem_gene12 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/CTD/PubChem_CTD_re_json_13.json', 'r') as tmp_json:
	pc_ctd_chem_gene13 = json.load(tmp_json)



ctd_cid_RE = pd.DataFrame(columns=['CTD_ID', 'CID'])

ctd_jsons = [pc_ctd_chem_gene1,pc_ctd_chem_gene2, pc_ctd_chem_gene3,
pc_ctd_chem_gene4, pc_ctd_chem_gene5, pc_ctd_chem_gene6,
pc_ctd_chem_gene7, pc_ctd_chem_gene8, pc_ctd_chem_gene9,
pc_ctd_chem_gene10, pc_ctd_chem_gene11, pc_ctd_chem_gene12, pc_ctd_chem_gene13]

for ctd_json in ctd_jsons:
	print(len(ctd_json['Annotations']['Annotation']))
	for ind in range(len(ctd_json['Annotations']['Annotation'])):
		anno = ctd_json['Annotations']['Annotation'][ind]
		CTD_ID = anno['SourceID']
		if 'CID' in anno['LinkedRecords'].keys() :
			CID = anno['LinkedRecords']['CID']
		else:
			CID='NA'
		tmp_df = pd.DataFrame({'CTD_ID' : CTD_ID, 'CID':CID})
		ctd_cid_RE = pd.concat([ctd_cid_RE, tmp_df])


ctd_cid_RE['new_id'] = [a.split(':')[0] for a in list(ctd_cid_RE['CTD_ID'])]
ctd_filter_1 = ctd[['ChemicalName', 'ChemicalID','GeneSymbol', 'GeneID','GeneForms','Organism']]
ctd_filter_2 = ctd_filter_1[ctd_filter_1['Organism']=='Homo sapiens'] # 1052758
ctd_filter_3 = pd.merge(ctd_filter_2, ctd_cid_RE, left_on = 'ChemicalID', right_on='new_id', how = 'left')

ctd_filter_3['CID2'] = [float(a) for a in ctd_filter_3.CID]
# 안붙는 애들은 아예 pubchem 에 연결이 안되는 애들 

ctd_filter_4 = ctd_filter_3[ctd_filter_3.CID2>0] # CID : 8682
ctd_filter_5 = ctd_filter_4[ctd_filter_4.GeneID>0] # entrez : 26354

CTD_FINAL = ctd_filter_5[['CID','GeneID']].drop_duplicates() # 57449


CTD_FINAL.columns = ['CID','EntrezID']
FINAL_TARGET = pd.concat([FINAL_TARGET, CTD_FINAL])







4) OpenTarget - chembl id 만 다시 확인 

optarget = pd.read_csv("/st06/jiyeonH/13.DD_SESS/OpenTargets/target_all.csv",low_memory=False,index_col = 0)
opdrug = pd.read_csv("/st06/jiyeonH/13.DD_SESS/OpenTargets/moelcule_all.csv",low_memory=False,index_col = 0)
# drug id : 12854
opdrug2 = opdrug[['id', 'canonicalSmiles','linkedTargets']] # 다운로드 받은 내용 

opdrug2['linkedTargets2'] = [a.split('[')[1].split(']')[0] if type(a) == str else 'NA' for a in list(opdrug2.linkedTargets)]
opdrug2['linkedTargets3'] = [a.replace('\n', '').replace(' ','') for a in list(opdrug2['linkedTargets2'])]

OT_KEY2 = list(set(opdrug2.id ))

opdrug3 = pd.DataFrame(columns= ['OT_drug_ID', 'OT_target_ID'])
for ot_id in OT_KEY2:
	tmp_df =  opdrug2[opdrug2['id'] == ot_id]
	targets = tmp_df['linkedTargets3'].values[0]
	targets = targets.split(',')
	re_df = pd.DataFrame({'OT_drug_ID':ot_id, 'OT_target_ID':targets})
	opdrug3 = pd.concat([opdrug3, re_df])

opdrug3['OT_target_ID'] = [a.replace("'", '') for a in list(opdrug3['OT_target_ID'])] 


# drug : len(set(opdrug3.OT_drug_ID)) # 12854
# target : len(set(opdrug3.OT_target_ID)) -1 : 1460






# ID 붙이기 (2중으로 확인!)
# https://pubchem.ncbi.nlm.nih.gov/idexchange/idexchange.cgi 
# 그래서 새로붙인게 아래 
PC_opentarget = pd.read_csv('/st06/jiyeonH/13.DD_SESS/OpenTargets/PubChem_idexchange_OpenTarget', sep ='\t', header = None) # 
PC_opentarget.columns = ['OT_ID','CID'] # 총 12930,  12854 : 12890

OT_KEY = list(set(PC_opentarget.OT_ID ))

tmp = [a for a in OT_KEY if list(PC_opentarget.OT_ID).count(a)>1] # 76 개는 multi cid 붙어서 확인 필요
{'CHEMBL386630', 'CHEMBL405110', 'CHEMBL1357', 'CHEMBL1688530', 'CHEMBL1097279', 'CHEMBL537669', 'CHEMBL641', 'CHEMBL48582', 'CHEMBL577', 'CHEMBL501637', 'CHEMBL503', 'CHEMBL393220', 'CHEMBL12089', 'CHEMBL113150', 'CHEMBL1235132', 'CHEMBL1205', 'CHEMBL136560', 'CHEMBL1767408', 'CHEMBL433', 'CHEMBL18136', 'CHEMBL560511', 'CHEMBL1834657', 'CHEMBL904', 'CHEMBL40422', 'CHEMBL1275', 'CHEMBL2106905', 'CHEMBL2325741', 'CHEMBL1083385', 'CHEMBL3233142', 'CHEMBL1339', 'CHEMBL1923502', 'CHEMBL1706', 'CHEMBL500826', 'CHEMBL502182', 'CHEMBL14', 'CHEMBL690', 'CHEMBL262135', 'CHEMBL523299', 'CHEMBL429852', 'CHEMBL541758', 'CHEMBL1354', 'CHEMBL902', 'CHEMBL490665', 'CHEMBL450044', 'CHEMBL1719', 'CHEMBL170', 'CHEMBL377559', 'CHEMBL754', 'CHEMBL1767407', 'CHEMBL1200803', 'CHEMBL1134', 'CHEMBL812', 'CHEMBL181886', 'CHEMBL121663', 'CHEMBL540445', 'CHEMBL656', 'CHEMBL1708', 'CHEMBL168640', 'CHEMBL542103', 'CHEMBL506', 'CHEMBL289351', 'CHEMBL17879', 'CHEMBL331237', 'CHEMBL1668019', 'CHEMBL141', 'CHEMBL284483', 'CHEMBL67166', 'CHEMBL378544', 'CHEMBL539843', 'CHEMBL274826', 'CHEMBL1944785', 'CHEMBL260629', 'CHEMBL62381', 'CHEMBL85164', 'CHEMBL1607', 'CHEMBL551466'}

# 2681 개는 nan 인데, pubchem 찾아도 안나옴 
tmp_nan = PC_opentarget[np.isnan(PC_opentarget.CID)] # 



# 그래서 민지 코드로 한번 더 확인해보기로 함 (multi 먼저)
PC_opentarget_re = opdrug2[opdrug2.id.isin(tmp)][['id','canonicalSmiles']]

PC_opentarget_re2 = PC_opentarget_re[['id',"canonicalSmiles"]].drop_duplicates().reset_index(drop=True,inplace=False) 
PC_opentarget_re2.canonicalSmiles = PC_opentarget_re2.canonicalSmiles.astype('str')
PC_opentarget_re2["id_cid"] = np.nan
s_cid = []

###pubchem canonical smiles 다시 붙이기
for nm in range(0,len(PC_opentarget_re2)):
	#print(nm)
	nm2 = PC_opentarget_re2.canonicalSmiles[nm]
	s_cid =[]
	try:
		for compound in pcp.get_compounds(nm2, 'smiles'):
			cidss= int(compound.cid)
			print(cidss)
			s_cid.append(str(cidss))
		if len(s_cid) == 1:
			su = float(s_cid[0])
		else :
			su = ','.join(s_cid)
		PC_opentarget_re2.id_cid[nm]= su
		print(nm2,nm,su)
	except Exception as e:    # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
		print(nm, nm2, '예외가 발생했습니다.', e)
		#cp_info_cid.cid[nm]="error"
		PC_opentarget_re2.id_cid[nm]="error"


# 내꺼에서 두개씩 붙는 애들에 대해서 smiles id 확인 하고 smiles 도 맞는 애들 위주로 가져가기로 함  
PC_change = PC_opentarget[PC_opentarget.OT_ID.isin(tmp)]
right_index = [] 

for ot_id_chck in tmp : 
	my_ver = PC_change[PC_change.OT_ID==ot_id_chck]
	pcp_ver = PC_opentarget_re2[PC_opentarget_re2.id==ot_id_chck]
	cids = list(pcp_ver['id_cid']) + list(my_ver['CID'])
	final_cid = list(set([a for a in cids if cids.count(a)>1]))
	if len(final_cid)==1:
		right_index = right_index + list((my_ver[my_ver.CID==final_cid[0]]).index)
	else : 
		print(ot_id_chck)


# 중구난방으로 나오는 애들 있음 (그냥 pubchem 기준 best match 로 맞추기로 함) 
# CHEMBL902 -> index 12525
# CHEMBL262135 -> index 10816
# CHEMBL540445 -> index 11977

right_index = right_index+[12525, 10816, 11977]

PC_change2 = PC_change.loc[right_index]

# 깔꼬롬한 애들 확인 
PC_opentarget_clear = PC_opentarget[-PC_opentarget.OT_ID.isin(tmp)]
PC_opentarget_clear2 = pd.concat([PC_opentarget_clear, PC_change2]) # 12781

# len(set(PC_opentarget_clear2[PC_opentarget_clear2.CID>0]['CID']))
PC_opentarget_clear2.columns = ['chembl_chem','CID']

# gene merge 
target_check = hgnc_full_re[['ensembl_gene_id','entrez_id']].drop_duplicates()

# total merge 확인 
opdrug4 = pd.merge(opdrug3, PC_opentarget_clear2, left_on = 'OT_drug_ID', right_on = 'chembl_chem', how = 'left' ) # 18986


opdrug5 = pd.merge(opdrug4, target_check, left_on = 'OT_target_ID', right_on = 'ensembl_gene_id', how = 'left' )
opdrug6 = opdrug5[['CID','entrez_id']].drop_duplicates()

len(set(opdrug6[opdrug6.CID>0]['CID']))# cid : 16138 -> 10162
len(set(opdrug6[opdrug6.entrez_id>0]['entrez_id']))# entrez_id : 16138 -> 1457

opdrug7 = opdrug6[opdrug6.CID>0] 
opdrug8 = opdrug7[opdrug7.entrez_id>0] # gene : 1085

OT_FINAL = opdrug8.drop_duplicates()
OT_FINAL.columns = ['CID','EntrezID']














5) DGI

dgi_inter = pd.read_csv("/st06/jiyeonH/13.DD_SESS/DGI/interactions.tsv",low_memory=False,sep="\t")
dgi_inter_sy = dgi_inter[["drug_name",'entrez_id']]
dgi_tf_check = [True if type(a)==str else False for a in list(dgi_inter_sy.drug_name)]
dgi_inter_sy2 = dgi_inter_sy[dgi_tf_check].drop_duplicates() # 숫자 확인용 

dgi_new1 = [a.split(':')[1] if type(a)==str else 'NA' for a in list(dgi_inter['drug_concept_id'])]
dgi_new2 = [a if type(a) == str else 'NA' for a in list(dgi_inter.drug_name)]

dgi_new3 = [dgi_new1[ind]+'_'+dgi_new2[ind] for ind in range(len(dgi_new1))]
dgi_inter['DGI_ID'] = dgi_new3

# https://pubchem.ncbi.nlm.nih.gov/source/15679#data=Annotations 에서
# Chemical-Target Interactions (Compound) 다운로드 

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_1.json', 'r') as tmp_json:
	pc_dgi_chem_gene1 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_2.json', 'r') as tmp_json:
	pc_dgi_chem_gene2 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_3.json', 'r') as tmp_json:
	pc_dgi_chem_gene3 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_4.json', 'r') as tmp_json:
	pc_dgi_chem_gene4 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_5.json', 'r') as tmp_json:
	pc_dgi_chem_gene5 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_6.json', 'r') as tmp_json:
	pc_dgi_chem_gene6 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_7.json', 'r') as tmp_json:
	pc_dgi_chem_gene7 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_8.json', 'r') as tmp_json:
	pc_dgi_chem_gene8 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_9.json', 'r') as tmp_json:
	pc_dgi_chem_gene9 = json.load(tmp_json)

with open('/st06/jiyeonH/13.DD_SESS/DGI/PubChem_DGI_10.json', 'r') as tmp_json:
	pc_dgi_chem_gene10 = json.load(tmp_json)




dgi_cid_RE = pd.DataFrame(columns=['DGI_ID', 'CID'])

dgi_jsons = [pc_dgi_chem_gene1,pc_dgi_chem_gene2, pc_dgi_chem_gene3,
pc_dgi_chem_gene4, pc_dgi_chem_gene5, pc_dgi_chem_gene6,
pc_dgi_chem_gene7, pc_dgi_chem_gene8, pc_dgi_chem_gene9,
pc_dgi_chem_gene10]


for dgi_json in dgi_jsons:
	print(len(dgi_json['Annotations']['Annotation']))
	for ind in range(len(dgi_json['Annotations']['Annotation'])):
		anno = dgi_json['Annotations']['Annotation'][ind]
		DGI_ID = anno['SourceID']
		if 'CID' in anno['LinkedRecords'].keys() :
			CID = anno['LinkedRecords']['CID']
		else:
			CID='NA'
		tmp_df = pd.DataFrame({'DGI_ID' : DGI_ID, 'CID':CID})
		dgi_cid_RE = pd.concat([dgi_cid_RE, tmp_df])

# 
dgi_inter2 = dgi_inter[dgi_inter.DGI_ID != "NA_NA"] # 59958
dgi_inter3 = pd.merge(dgi_inter2, dgi_cid_RE, on = 'DGI_ID', how = 'left' ) # 59958
dgi_inter4 = dgi_inter3[['CID','entrez_id']].drop_duplicates() # 45705

len(set(dgi_inter4[dgi_inter4.CID>0]['CID'])) # 9691
len(set(dgi_inter4[dgi_inter4.entrez_id>0]['entrez_id'])) # 9691

cid_check = [True if type(a)==int else False for a in list(dgi_inter4.CID)]
dgi_inter5 = dgi_inter4[cid_check] # 44415
# cid : len(set(dgi_inter5.CID))

DGI_FINAL = dgi_inter5[dgi_inter5.entrez_id>0].drop_duplicates() # 43457
# len(set(DGI_FINAL['entrez_id'])) 2898


DGI_FINAL.columns = ['CID','EntrezID']









7) LINCS target 

BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/compoundinfo_beta.txt')
BETA_CP_info2 = BETA_CP_info[['pert_id','target']].drop_duplicates() # 37158
check = [True if type(a) == str else False for a in list(BETA_CP_info2.target)]
BETA_CP_info3 = BETA_CP_info2[check]
# 34419 & 890

# BETA_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+"Lincs_pubchem_mj.csv") # SID 필터까지 그렇게 완성한 무엇 
# BETA_MJ2 = BETA_MJ[['pert_id','SMILES_cid']].drop_duplicates() # 25903
Pert_JY = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep = '\t')

BETA_GENE = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/geneinfo_beta.txt') # 12328

L_merge_1 = pd.merge(BETA_CP_info2, Pert_JY[['pert_id','CID']], on = 'pert_id', how = 'left' ) # 37158 


# 유전자 이름 없는거 찾기 

ncbi_gene_info2 = ncbi_gene_info[ncbi_gene_info.type_of_gene=='protein-coding']
ncbi_gene_info2 = ncbi_gene_info2.reset_index(drop= True)
target_id_ck = ncbi_gene_info2[['GeneID','Symbol','Synonyms']].drop_duplicates()

# 다 있는지 확인 
set(L_merge_1[-L_merge_1.target.isin(target_id_ck.Symbol)]['target']) # 37158
# GBA

for a in list(target_id_ck.Synonyms) :
	tmp = a.split('|')
	if 'GBA' in tmp:
		print(a)


list(ncbi_gene_info2.Synonyms).index('GBA|GCB|GLUC')
# index 1747 
# ncbi_gene_info2.loc[1747] # GBA1

target_id_ck.at[1747, 'Symbol'] = 'GBA'


L_merge_2 = pd.merge(L_merge_1, target_id_ck, left_on = 'target', right_on = 'Symbol', how = 'left') # 37158
L_merge_2_ck = [True if type(a)==str else False for a in list(L_merge_2.target)]

L_merge_3 = L_merge_2[L_merge_2_ck]

len(set(L_merge_3[L_merge_3.CID>0]['CID'])) # 3009
len(set(L_merge_3[L_merge_3.GeneID>0]['GeneID'])) # 890


L_merge_4 = L_merge_3[L_merge_3.CID > 0]
L_merge_5 = L_merge_4[L_merge_4.GeneID > 0]

L_FINAL = L_merge_5[['CID','GeneID']].drop_duplicates()


L_FINAL.columns = ['CID','EntrezID']





8) DrugBank

																																			# https://pubchem.ncbi.nlm.nih.gov/source/DrugBank#data=Annotations 에서 
																																			# Chemical-Target Interactions (Compound) 다운로드 

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_1.json', 'r') as tmp_json:
																															pc_db_chem_gene1 = json.load(tmp_json)

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_2.json', 'r') as tmp_json:
																															pc_db_chem_gene2 = json.load(tmp_json)

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_3.json', 'r') as tmp_json:
																															pc_db_chem_gene3 = json.load(tmp_json)

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_4.json', 'r') as tmp_json:
																															pc_db_chem_gene4 = json.load(tmp_json)

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_5.json', 'r') as tmp_json:
																															pc_db_chem_gene5 = json.load(tmp_json)

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_6.json', 'r') as tmp_json:
																															pc_db_chem_gene6 = json.load(tmp_json)

																														with open('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/PubChem_DB_7.json', 'r') as tmp_json:
																															pc_db_chem_gene7 = json.load(tmp_json)



																																db_cid_RE = pd.DataFrame(columns=['DB_ID', 'CID']) # 6580

																																db_jsons = [pc_db_chem_gene1, pc_db_chem_gene2, pc_db_chem_gene3,
																																pc_db_chem_gene4, pc_db_chem_gene5, pc_db_chem_gene6, pc_db_chem_gene7]

																																for db_json in db_jsons:
																																	print(len(db_json['Annotations']['Annotation']))
																																	for ind in range(len(db_json['Annotations']['Annotation'])):
																																		anno = db_json['Annotations']['Annotation'][ind]
																																		DB_ID = anno['SourceID']
																																		if 'LinkedRecords' in anno.keys() : 
																																			if 'CID' in anno['LinkedRecords'].keys() :
																																				CID = anno['LinkedRecords']['CID']
																																			else:
																																				CID='NA'
																																			tmp_df = pd.DataFrame({'DB_ID' : DB_ID, 'CID':CID})
																																			db_cid_RE = pd.concat([db_cid_RE, tmp_df])



# CID 그냥 파싱으로 확인 
import xml.etree.ElementTree as ET

doc = ET.parse('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/full_database.xml')
root = doc.getroot()

tot_tags = []
for i in root.iter():
	tot_tags.append(i.tag)

tot_tags = list(set(tot_tags))


DB_name_df = pd.DataFrame(columns = ["Name","Resource","Identifier"])

for a in root :
	NAMEs = a.findall("./{http://www.drugbank.ca}name")
	NAME = NAMEs[0].text # 하나인거 확인함 
	EXT = a.findall('./{http://www.drugbank.ca}external-identifiers/')
	resources = sum([RR.findall('./{http://www.drugbank.ca}resource') for RR in EXT],[])
	ids = sum([II.findall('./{http://www.drugbank.ca}identifier') for II in EXT],[])
	if len(resources) == len(ids) :
		resources_t = [a.text for a in resources]
		ids_t = [a.text for a in ids]
	else :
		print(NAME)
	tmp_df = pd.DataFrame(columns = ["Name","Resource","Identifier"])
	tmp_df['Resource'] = resources_t
	tmp_df['Identifier'] = ids_t
	tmp_df['Name'] = NAME
	DB_name_df = pd.concat([DB_name_df, tmp_df])

DB_CID = DB_name_df[DB_name_df.Resource=='PubChem Compound']



DB_id_df = pd.DataFrame(columns = ["DrugBankID","Name"])

for a in root :
	NAMEs = a.findall("./{http://www.drugbank.ca}name")
	NAME = NAMEs[0].text # 하나인거 확인함 
	DB_IDS = a.findall("./{http://www.drugbank.ca}drugbank-id")
	DB_ID_NAMES = [a.text for a in DB_IDS ]
	tmp_df = pd.DataFrame(columns = ["DrugBankID","Name"])
	tmp_df['DrugBankID'] = DB_ID_NAMES
	tmp_df['Name'] = NAME
	DB_id_df = pd.concat([DB_id_df, tmp_df])






# drug - target 뽑기 
import xml.etree.ElementTree as ET

# /st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9
doc = ET.parse('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/full_database.xml')
root = doc.getroot()

tot_tags = []
for i in root.iter():
	tot_tags.append(i.tag)

tot_tags = list(set(tot_tags))

db_df = pd.DataFrame(columns = ['DrugName', 'DrugID', 'Source', 'TargetName'])

for drug in root.findall("./{http://www.drugbank.ca}drug"):
	name = drug.findtext("./{http://www.drugbank.ca}name")
	db_id = drug.findtext("./{http://www.drugbank.ca}drugbank-id")
	for targets in drug.findall("{http://www.drugbank.ca}targets/{http://www.drugbank.ca}target"):
		target_id =targets.findtext("./{http://www.drugbank.ca}id")
		for hugo_id in targets.iter("{http://www.drugbank.ca}external-identifier"):
			tmp_list = [a for a in hugo_id.iter()]
			tmp_tag = [a.tag for a in hugo_id.iter()]
			print(tmp_tag)
			tmp_text = [a.text for a in hugo_id.iter()]
			tag_ind = tmp_tag.index("{http://www.drugbank.ca}resource")
			text_ind = tmp_tag.index("{http://www.drugbank.ca}identifier")
			tmp_df = pd.DataFrame({
				'DrugName' : [name],
				'DrugID' : [db_id],
				'Source' : [tmp_text[tag_ind]],
				'TargetName' : [tmp_text[text_ind]],
			})
			db_df = pd.concat([db_df,tmp_df ])


# drug id : 7627
db_df_HUGO = db_df[db_df.Source=='HUGO Gene Nomenclature Committee (HGNC)'] # 17233 
# len(set(hgnc_full_re[hgnc_full_re.hgnc_id.isin(db_df_HUGO.TargetName)]['entrez_id'])) # 2894

db_df_unip = db_df[db_df.Source=='UniProtKB'] # 21626 
db_df_unip_ac = db_df[db_df.Source=='UniProt Accession'] # 21626 중에 17137 entrez 겹침 
# tt = list(db_df_unip.TargetName)
# len(set(uni_entrez[uni_entrez.ID.isin(tt)]['DATA'])) # 2881 


db_merged = pd.merge(db_df_HUGO, hgnc_full_re, left_on='TargetName', right_on ='hgnc_id', how = 'left')
# 17233
db_merged2 = pd.merge(db_merged, DB_CID[['Name','Identifier']], left_on = 'DrugName', right_on = 'Name', how = 'left')
db_merged2['CID'] = [int(a) if type(a) == str else 0 for a in list(db_merged2.Identifier)]



len(set(db_merged2[db_merged2.CID>0]['CID'])) # 4818
len(set(db_merged2[db_merged2.entrez_id>0]['entrez_id'])) # 2894

db_merged3 = db_merged2[db_merged2.CID > 0]
db_merged4 = db_merged3[db_merged3.entrez_id>0]

DB_FINAL = db_merged4[['CID','entrez_id']].drop_duplicates() # 14439

DB_FINAL.columns = ['CID','EntrezID']











6) STITCH

문제는 STITCH 에서 쓰는 몇몇 protein ID 는 죽은 ID 라는거임. 
아예 retired gene 을 없애는 방향으로 가자 그냥. entrez 매칭이 안되는걸 어떡해요 

stit = pd.read_csv("/st06/jiyeonH/13.DD_SESS/STITCH.v5.0/9606.protein_chemical.links.transfer.v5.0.tsv", sep="\t") #15473939

# CIDs / CID0... - 
# this is a stereo-specific compound, and the suffix is the PubChem compound id.

# CIDm / CID1... - 
# this is a "flat" compound, 
# i.e. with merged stereo-isomers The suffix (without the leading "1") is the PubChem compound id.

stit['CID'] = [int(a[4:]) for a in list(stit['chemical'])]
stit_info = stit[['CID','protein','experimental_direct','database_direct']]
stit_info2 = stit_info[(stit_info["experimental_direct"]!=0 )|(stit_info["database_direct"]!=0)]
# ss = list(set(stit_info2.protein)) # 17083


# for entrez id match 

ensem_all = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene2ensembl.cut', sep = '\t', header =None)
ensem_all.columns = ['#tax_id', 'GeneID', 'Ensembl_gene_identifier', 'RNA_nucleotide_accession.version', 'Ensembl_rna_identifier', 'protein_accession.version', 'Ensembl_protein_identifier']
ensem_all_re = ensem_all[['GeneID', 'Ensembl_gene_identifier', 'Ensembl_protein_identifier']]
ensem_all_re['TO_STRING'] = ["9606."+a.split('.')[0] if a != '-' else 'NA'for a in ensem_all_re.Ensembl_protein_identifier]
ensem_all_re2 = ensem_all_re[ensem_all_re['TO_STRING'] !='NA']


stit_info3 = pd.merge(stit_info2, ensem_all_re2, left_on = 'protein', right_on = 'TO_STRING', how = 'left' )
stit_info4 = stit_info3[['CID','GeneID']].drop_duplicates()

# len(set(stit_info4[stit_info4.CID>0]['CID'])) # 267325
# len(set(stit_info4[stit_info4.GeneID>0]['GeneID'])) # 15174

stit_info5 = stit_info4[stit_info4.CID>0]
stit_info6 = stit_info5[stit_info5.GeneID>0]


STITCH_FINAL = stit_info6.drop_duplicates()
STITCH_FINAL.columns = ['CID','EntrezID']












9) BindingDB

bdb_jy = pd.read_csv("/st06/jiyeonH/13.DD_SESS/BindingDB.22.04.01/JY_summary.csv",low_memory=False, sep="\t")

bdb_tar = bdb_jy[['PubChem CID','UniProt (SwissProt) Primary ID of Target Chain']]
bdb_tar.rename(columns = {'PubChem CID':'cid','UniProt (SwissProt) Primary ID of Target Chain':'uniprot'}, inplace = True)
bdb_tar = bdb_tar[bdb_tar.cid>0] # cid : 1009306

check = [True if type(a) == str else False for a in bdb_tar.uniprot ]
bdb_tar2 = bdb_tar[check].drop_duplicates()
unip = list(set(bdb_tar2.uniprot)) # 5909
 

bdb_tar3 = pd.merge(bdb_tar2, uni_entrez, left_on = 'uniprot', right_on = 'ID', how = 'left' )

bdb_tar4 = bdb_tar3[['cid','DATA']].drop_duplicates()

len(set(bdb_tar4[bdb_tar4.cid>0]['cid'])) # 931950


check2 = [True if type(a) == str else False for a in bdb_tar4.DATA ]
bdb_tar5 = bdb_tar4[check2]
bdb_tar5['entrez_id'] = [int(a) for a in list(bdb_tar5.DATA)]


len(set(bdb_tar5[bdb_tar5.entrez_id>0 ]['entrez_id'])) # 931950

bdb_tar6 = bdb_tar5[['cid','entrez_id']].drop_duplicates()

BINDING_FINAL = copy.deepcopy(bdb_tar6)
BINDING_FINAL.columns = ['CID','EntrezID']












10) SNAP - 확인해보니 Interdecagon 이 모든걸 가지고 있음 
# ChG-InterDecagon_targets.csv
# ChG-TargetDecagon_targets.csv
	Drug-target protein associations 
	(drugs are given by STITCH chemical IDs and 
	proteins are given by NCBI Entrez Gene IDs)


snap_I = pd.read_csv("/st06/jiyeonH/13.DD_SESS/SNAP/ChG-InterDecagon_targets.csv",low_memory=False,sep=",",skiprows =1, names=["drug","gene"])
snap_I['CID'] = [int(a[3:]) for a in list(snap_I.drug)]
len(set(snap_I.drug)) # 1774 
len(set(snap_I.gene)) # 7795

snap_T = pd.read_csv("/st06/jiyeonH/13.DD_SESS/SNAP/ChG-TargetDecagon_targets.csv",low_memory=False,sep=",",skiprows =1, names=["drug","gene"])
snap_T['CID'] = [int(a[3:]) for a in list(snap_T.drug)] 
len(set(snap_T.drug)) # 284 
len(set(snap_T.gene)) # 3648


SNAP_I_FINAL = snap_I[['CID','gene']]
SNAP_I_FINAL.columns = ['CID','EntrezID']
SNAP_T_FINAL = snap_T[['CID','gene']]
SNAP_T_FINAL.columns = ['CID','EntrezID']



11) another SNAP - 필요없음. 

snap_TOT = pd.read_csv("/st06/jiyeonH/13.DD_SESS/SNAP/bio-decagon-targets-all.csv",low_memory=False,sep=",",skiprows =1, names=["drug","gene"])
# nope same 


SNAP_I_FINAL












# FINAL TABLE

# CID , TARGET GENE, REFERENCE


CHEMBL_FINAL['DB'] = 'CHEMBL'
IUPHAR_FINAL['DB'] = 'IUPHAR'
TTD_FINAL['DB'] = 'TTD'
CTD_FINAL['DB'] = 'CTD'
OT_FINAL['DB'] = 'OPENTARGET'
DGI_FINAL['DB'] = 'DGI'
L_FINAL['DB'] = 'LINCS'
DB_FINAL['DB'] = 'DRUGBANK'
STITCH_FINAL['DB'] = 'STITCH'
BINDING_FINAL['DB'] = 'BINDINGDB'
SNAP_I_FINAL['DB'] = 'SNAP'




FINAL_TARGET = pd.DataFrame(columns=['CID','EntrezID'])

FINAL_TARGET = pd.concat([CHEMBL_FINAL,IUPHAR_FINAL,TTD_FINAL,CTD_FINAL,
							OT_FINAL, DGI_FINAL, L_FINAL, DB_FINAL, STITCH_FINAL, 
							BINDING_FINAL, SNAP_I_FINAL])


FINAL_TARGET['CID_RE'] = [int(a) for a in FINAL_TARGET.CID]
FINAL_TARGET['ENTREZ_RE'] = [int(a) for a in FINAL_TARGET.EntrezID]

F_cids = list(FINAL_TARGET['CID_RE'])
F_ents = list(FINAL_TARGET['ENTREZ_RE'])

FINAL_TARGET['CHECK'] = [str(F_cids[i])+"C_T"+str(F_ents[i]) for i in range(FINAL_TARGET.shape[0])]
# row : 2802983 -> 2802996
# set : 2345322 -> 2345237
# cids : 867139 -> 867077
# entrezs : 26406

FINAL_TARGET.to_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/TARGET_CID_ENTREZ.csv', sep ='\t')
# FINAL_TARGET = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/TARGET_CID_ENTREZ.csv')

request_mj = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/EXP_request.csv', sep ='\t')

sum(request_mj.CID.isin(FINAL_TARGET.CID_RE))
# 19598 -> 19591 (drugbank 를 누구기준으로 하느냐 차이 )





# [ MJ_files ]

TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/merged_target/'

TARGET_DB_1 = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)
TARGET_DB_2 = pd.read_csv(TARGET_PATH+'combined_target_b_woprediction.csv', low_memory=False)
TARGET_DB_3 = pd.read_csv(TARGET_PATH+'combined_target_wcbsi.csv', low_memory=False)


sum(request_mj.CID.isin(TARGET_DB_1.cid)) # 18220
sum(request_mj.CID.isin(TARGET_DB_2.cid)) # 19617
sum(request_mj.CID.isin(TARGET_DB_3.cid)) # 17808









#혹시 request 에 있는 CID 들에 대한 target 많이 다른지? #
예를 들면 

DC_CIDs = list(set(request_mj.CID))
CID = 16220172
CID = 57363


set(TARGET_DB_1[TARGET_DB_1.cid == CID]['target'])
{'SLC26A9', 'CYP2E1', 'CTSB', 'CYP3A4', 'CYP2C9', 'CYP2C19', 'CFTR', 'CYP2D6'}

set(FINAL_TARGET[FINAL_TARGET.CID_RE== CID]['ENTREZ_RE'])
{115019, 1571, , 1576, 1557, 1559, 1080, 1565}

아 좀 차이가 있네










######### 막간 확인 


check = list(set(ncbi_gene_ensem_uniprot_MERGE['UniProt']))[1:]
bdb_check = list(set(bdb_tar.uniprot))

len(set(bdb_check) - set(check)) #mouse done 

uni_data.columns = ['ID', 'DB', 'DATA']
uni_data_ids = set(uni_data.ID)



GeneID HGNC Gene_Name STRING, OpenTargets Ensembl EMBL

  
P31946
ENSG00000166913
9606.ENSP00000361930



bdb_uni = bdb_jy[['UniProt (SwissProt) Primary ID of Target Chain']].drop_duplicates().reset_index(drop=True,inplace=False)

bdb_ug = pd.read_csv("/home/minK/ssse/target_check/bdb_uniprot_gene.txt", low_memory=False, sep="\t")
bdb_ug.rename(columns = {'From' : 'uniprot', 'To' : 'target'}, inplace = True)


bdb_target = pd.merge(bdb_tar, bdb_ug,on='uniprot')
bdb_target2 = bdb_target[bdb_target['cid'].notna()].drop_duplicates().dropna().reset_index(drop=True,inplace=False)
bdb_target2.cid = bdb_target2.cid.astype("int")
bdb_target2["db_name"] = "bindingdb"
bdb_target2.to_csv("/home/minK/ssse/bindingdb_target_filtered.csv")
#prebind를 어디서 본것 같은데 왜 없냐.....이 db가 아닌가...





대충만 비교해보자면

FINAL_TARGET.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/target_all_tmp.csv', sep ='\t')
FINAL_TARGET = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/target_all_tmp.csv', sep ='\t', low_memory=False)

DC_PATH = '/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/' 

ORI_TARGET_DB = pd.read_csv(TARGET_PATH+'combined_target.csv', low_memory=False)

# 9230 
A_B_C_S_SET = A_B_C_S[['drug_row_cid','drug_col_cid','BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates()


# 들어가지 못한 애들 

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_row_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_1 = pd.merge(DC_cello_final_dup, BETA_CID_CELLO_SIG, on = ['drug_row_cid','DrugCombCello'], how = 'left') # 731051

BETA_CID_CELLO_SIG.columns=['pert_id', 'drug_col_cid', 'DrugCombCello', 'BETA_sig_id']
CELLO_DC_BETA_2 = pd.merge(CELLO_DC_BETA_1, BETA_CID_CELLO_SIG, on = ['drug_col_cid','DrugCombCello'], how = 'left') # 731644

BETA_CID_CELLO_SIG.columns=['pert_id', 'pubchem_cid', 'cellosaurus_id', 'sig_id']

FILTER = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
CELLO_DC_BETA = CELLO_DC_BETA_2.loc[FILTER] # 11742
FILTER2 = [True if type(a)==float else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER2] # 11742 ??? 
FILTER3 = [True if np.isnan(a)==False else False for a in CELLO_DC_BETA.synergy_loewe]
CELLO_DC_BETA = CELLO_DC_BETA.loc[FILTER3] # 11701 
CELLO_DC_BETA[['BETA_sig_id_x','BETA_sig_id_y','DrugCombCello']].drop_duplicates() # 9230
CELLO_DC_BETA_cids = list(set(list(CELLO_DC_BETA.drug_row_cid) + list(CELLO_DC_BETA.drug_col_cid))) # 176 

FILTER_OX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == str) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == float)]
FILTER_XO = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == float) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == str)]
FILTER_XX = [a for a in range(CELLO_DC_BETA_2.shape[0]) if (type(CELLO_DC_BETA_2.BETA_sig_id_x[a]) == float) & (type(CELLO_DC_BETA_2.BETA_sig_id_y[a]) == float)]

missing_OX = CELLO_DC_BETA_2.loc[FILTER_OX] # set(missing_OX.drug_col_cid)
missing_XO = CELLO_DC_BETA_2.loc[FILTER_XO] # set(missing_XO.drug_row_cid)
missing_XX = CELLO_DC_BETA_2.loc[FILTER_XX] # set(list(missing_XO.drug_row_cid) + list(missing_OX.drug_col_cid))


missing_cids = list(missing_OX.drug_col_cid) + list(missing_XO.drug_row_cid)+ list(missing_XO.drug_row_cid) + list(missing_OX.drug_col_cid)

true_missing = set(missing_cids) - set(CELLO_DC_BETA_cids) 
아예 시도조차 못한 519 개 중에서 


ori_target_db_filt = ORI_TARGET_DB[ORI_TARGET_DB.db_name.isin(['DGI', 'opentargets', 'TTD', 'IUPHAR', 'CTD', 'lincs'])] # DrugBank
ori_target_db_filt_cids = set(ori_target_db_filt.cid) # 38786 / 34789

target_re_cids = list(FINAL_TARGET.CID)
target_re_cids2 = [a for a in target_re_cids if type(a)!=float]
[a for a in target_re_cids2 if ';' in a]

target_re_cids3 = [float(a) for a in target_re_cids2 if ';' not in a] # 35057



len([a for a in true_missing if a in ori_target_db_filt_cids]) # 345 
len([a for a in true_missing if a in target_re_cids3]) # 343 





























######## TRIALS ###############3



# trial 2 for chemical ID ( annotation 다운받은거 )

with open('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/chembl_1.json', 'r') as tmp_json:
	pc_chembl_gene1 = json.load(tmp_json)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/chembl_2.json', 'r') as tmp_json:
	pc_chembl_gene2 = json.load(tmp_json)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/chembl_3.json', 'r') as tmp_json:
	pc_chembl_gene3 = json.load(tmp_json)

with open('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/chembl_4.json', 'r') as tmp_json:
	pc_chembl_gene4 = json.load(tmp_json)


cb_cid_RE = pd.DataFrame(columns=['CB_ID', 'CID']) # 6580

cb_jsons = [pc_chembl_gene1, pc_chembl_gene2, pc_chembl_gene3,pc_chembl_gene4]

for cb_json in cb_jsons:
	print(len(cb_json['Annotations']['Annotation']))
	for ind in range(len(cb_json['Annotations']['Annotation'])):
		anno = cb_json['Annotations']['Annotation'][ind]
		CB_ID = anno['SourceID']
		if 'LinkedRecords' in anno.keys() : 
			if 'CID' in anno['LinkedRecords'].keys() :
				CID = anno['LinkedRecords']['CID']
			else:
				CID='NA'
			tmp_df = pd.DataFrame({'CB_ID' : CB_ID, 'CID':CID})
			cb_cid_RE = pd.concat([cb_cid_RE, tmp_df])


cb_cid_RE['CB_ID2'] = [a.split("::")[0] for a in list(cb_cid_RE.CB_ID)] # 3641 

cb_cid_RE2 = cb_cid_RE[cb_cid_RE.CID>0] # 3641 -> 이걸로 가기 




# trial 2 for target id 

with open('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/cb_protein.json', 'r') as tmp_json:
	pc_chembl_gene = json.load(tmp_json)

cb_gene_RE = pd.DataFrame(columns=['CB_GID', 'PROT']) # 6580
for ind in range(len(pc_chembl_gene['Annotations']['Annotation'])):
	anno = pc_chembl_gene['Annotations']['Annotation'][ind]
	CB_GID = anno['SourceID']
	if 'LinkedRecords' in anno.keys() : 
		if 'ProteinAccession' in anno['LinkedRecords'].keys() :
			PROT = anno['LinkedRecords']['ProteinAccession']
		else:
			PROT='NA'
		tmp_df = pd.DataFrame({'CB_GID' : CB_GID, 'PROT':PROT})
		cb_gene_RE = pd.concat([cb_gene_RE, tmp_df]) # 588 

# 너무 적은디 



# 스트링에서 가져온거 4213391
proal = pd.read_csv("/st06/jiyeonH/00.STRING_v.11.5/9606.protein.aliases.v11.5.txt", sep="\t") #4213391
proal_set = list(set(proal['#string_protein_id'])) # 19566 개 protein 
# proal_hugo = proal[proal.source=='Ensembl_HGNC_HGNC_ID'] #18566 less... 
# proal_sources = list(set(proal.source))
# 얘가 의미가 있는건지 한번 확인 
proal_pros = set(proal['#string_protein_id'])
stit_pros = set(stit['protein'])
아 여기서 갈려서 그러는거네 
뀨
아... STITCH 이럴거니 retired 때문에 STRING 에서 아예 인식도 안돼  

ensem_all = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/NCBI_gene2ensembl.cut', sep = '\t', header =None)
ensem_all.columns = ['#tax_id', 'GeneID', 'Ensembl_gene_identifier', 'RNA_nucleotide_accession.version', 'Ensembl_rna_identifier', 'protein_accession.version', 'Ensembl_protein_identifier']
ensem_all_re = ensem_all[['GeneID', 'Ensembl_gene_identifier', 'Ensembl_protein_identifier']]
ensem_all_re['TO_STRING'] = ["9606."+a.split('.')[0] if a != '-' else 'NA'for a in ensem_all_re.Ensembl_protein_identifier]


proal_ensem = proal[proal.source=='Ensembl_gene'] # 19566 less... 

tf_check = [True if type(a) == str else False for a in list(proal_ensem.alias)]

hgnc_full = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/hgnc_complete_set.txt', sep ='\t', low_memory = False)
hgnc_full2 = hgnc_full[['hgnc_id','symbol','entrez_id','ensembl_gene_id','uniprot_ids']] 
# row 43517 / ensembl_gene_id 40832

proal_ensem[proal_ensem.alias.isin(list(hgnc_full2.ensembl_gene_id))]

stip_ensem = pd.merge(proal_ensem, hgnc_full2, left_on = 'alias', right_on="ensembl_gene_id", how="left")



# 15473939
stit = pd.read_csv("/st06/jiyeonH/13.DD_SESS/STITCH.v5.0/9606.protein_chemical.links.transfer.v5.0.tsv", sep="\t") #15473939





# CIDs / CID0... - 
# this is a stereo-specific compound, and the suffix is the PubChem compound id.

# CIDm / CID1... - 
# this is a "flat" compound, 
# i.e. with merged stereo-isomers The suffix (without the leading "1") is the PubChem compound id.


stit['CID'] = [int(a[4:]) for a in list(stit['chemical'])]
stit_info = stit[['CID','protein','experimental_direct','database_direct']]
stit_info2 = stit_info[(stit_info["experimental_direct"]!=0 )|(stit_info["database_direct"]!=0)]
stit_info3 = pd.merge(stit_info2, stip_hugo[['#string_protein_id','entrez_id','hgnc_id']] , left_on = 'protein', right_on = '#string_protein_id', how = 'left')

stit_info3[(stit_info3['entrez_id'] > 1 )== False] # 3674 개가 제대로 안붙음 



stit_info4 = 



'9606.ENSP00000330138'




#mmm
stitm=stit[stit['chemical'].str.contains("m")]
stitm['cid'] = stitm['chemical'].str.split('m', 1).str[1].astype("int")
stitm_info =stitm[['cid', 'protein', 'experimental_direct', 'database_direct']]
stitm_info2 = stitm_info[(stitm_info["experimental_direct"]!=0 )| (stitm_info["database_direct"]!=0)]
stitm_info2_sym = pd.merge(stitm_info2,stip_hugo,on='protein').drop_duplicates()
stitm_info2_sym.rename(columns = {'symbol' : 'target'}, inplace = True)
len(stitm_info2_sym.cid.unique()) #202628
len(stitm_info2_sym.target.unique()) # 13298
stitm_info2_sym["db_name"] = "STITCH_m_wt"


#ssss
stits=stit[stit['chemical'].str.contains("s")]
stits['cid'] = stits['chemical'].str.split('s', 1).str[1].astype("int")
stits_info =stits[['cid', 'protein', 'experimental_direct', 'database_direct']]
stits_info2 = stits_info[(stits_info["experimental_direct"]!=0 )| (stits_info["database_direct"]!=0)]
stits_info2_sym = pd.merge(stits_info2,stip_hugo,on='protein').drop_duplicates()
stits_info2_sym.rename(columns = {'symbol' : 'target'}, inplace = True)
len(stits_info2_sym.cid.unique()) #213516
len(stits_info2_sym.target.unique()) # 13298
stits_info2_sym["db_name"] = "STITCH_s_wt"
stits_info2_sym.to_csv("/home/minK/ssse/target_check/STITCHs_woprediction_target_filtered.csv")














CID_SYN = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/tmp/LINCS_CHEM_PCSYN.txt', sep = '\t', header = None)
CID_SYN.columns = ['CID', 'SYN']

TF_check_1 = [ 'NA' if type(a)==float else a for a in list(CID_SYN['SYN']) ]
TF_check_2 = [True if a.startswith('CHEMBL') else False for a in TF_check_1  ]

SYN_filter = CID_SYN[TF_check_2]
SYN_filter_set = list(set(SYN_filter.CID))

[a for a in SYN_filter_set if list(SYN_filter.CID).count(a) >1]
# 근데 ( CID : chembl ) 세트가 두번이상 나오는게 꽤 됨. 



chembl_genes = LINCS_978_id_STck[['entrez_id','ChEMBL']].drop_duplicates()
chembl_genes = chembl_genes[chembl_genes.ChEMBL.isin(['NA'])==False]

check = list(set(chembl_genes['entrez_id']))
[a for a in check if list(chembl_genes.entrez_id).count(a)>1]
[2778, 1029]


check = list(set(chembl_genes['ChEMBL']))
[a for a in check if list(chembl_genes.ChEMBL).count(a)>1]
['CHEMBL2364701', 'CHEMBL2363042', 'CHEMBL2111324', 'CHEMBL2364188']

# 아 1:1 이 아니라는 이 충격적인...? 충격은 또 아닌것 같기도 함.
# PSMD 같은 경우에는 PSMD2 랑 PSMD4 가 각각 entrez 5708, 5710 인데, 각각 Q13200 , P55036 이라서 같은 묶음이긴 함. 그걸 타겟으로 한다는 얘긴것 같음 
# 으응 늘어날수밖에 없네 





DIR = '/st06/jiyeonH/13.DD_SESS/ChEMBL.22.11.18/'

chembl_targets = pd.read_csv(DIR+'TARGET.csv', sep = ';')

# 결국 수기로 다운받음 시불 (DB 기준으로 ID 나열한거)
parent_target = pd.read_csv(DIR+'Chembl_Drug_Mechanism.csv')
parent_target = parent_target[['Parent Molecule ChEMBL ID', 'Parent Molecule Name', 'Parent Molecule Type', 'Target ChEMBL ID','Target Name', 'Action Type', 'Target Type', 'Target Organism','ATC Codes']]


아 근데 LINCS 가 중요한게 아닌것 같고,,,,
CID 전체 필요할것 같은데 




chembl_in_lincs = parent_target[parent_target['Parent Molecule ChEMBL ID'].isin(list(SYN_filter['SYN']))]
chembl_in_lincs_target = chembl_in_lincs[chembl_in_lincs['Target ChEMBL ID'].isin(list(LINCS_978_id_STck['ChEMBL']))]
chembl_in_lincs_target = chembl_in_lincs_target[['Parent Molecule ChEMBL ID','Target ChEMBL ID']].drop_duplicates()
chembl_in_lincs_target.columns = ['CHEMBL_CHEM_ID', 'CHEMBL_TARGET_ID']
# chem chembl 214 , target chembl 58 , chem-target 256

chembl_in_lincs_target_add1 = pd.merge(chembl_in_lincs_target, SYN_filter, left_on = 'CHEMBL_CHEM_ID', right_on='SYN', how='left' )
chembl_in_lincs_target_add2 = pd.merge(chembl_in_lincs_target_add1, chembl_genes, left_on = 'CHEMBL_TARGET_ID', right_on='ChEMBL', how='left' )

CHEMBL_FINAL = chembl_in_lincs_target_add2[['CID','entrez_id']].drop_duplicates()
CHEMBL_FINAL['DB'] = 'ChEMBL'




for a in 
