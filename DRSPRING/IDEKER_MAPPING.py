
# LINCS landmark 

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()

LANDMARK_symbol = list(BETA_lm_genes.gene_symbol)
LANDMARK_entrez = list(BETA_lm_genes.gene_id)
LANDMARK_ensemble = list(BETA_lm_genes.ensembl_id)


# STRING alternatives 
PPI_N_PATH = '/st06/jiyeonH/00.STRING_v.11.5/'
PPI_alias_info = pd.read_csv(PPI_N_PATH+'9606.protein.aliases.v11.5.txt', sep = '\t')
PPI_info = pd.read_csv(PPI_N_PATH+'9606.protein.info.v11.5.txt', sep = '\t')

PPI_info2 = PPI_info[['#string_protein_id','preferred_name']]
PPI_info2.columns= ['#string_protein_id','ORI_NAME']

PPI_ORI_ALI = pd.merge(PPI_alias_info, PPI_info2, on = '#string_protein_id', how = 'left')

PPI_ORI_ALI_LI = PPI_ORI_ALI[PPI_ORI_ALI.ORI_NAME.isin(LANDMARK_symbol)]
dont_alis = list(PPI_ORI_ALI_LI.alias)

PPI_ORI_ALI_LNI = PPI_ORI_ALI[-PPI_ORI_ALI.ORI_NAME.isin(LANDMARK_symbol)]
check_alis = list(PPI_ORI_ALI_LNI.alias)

# check IDEKER 

IDEKER_IAS = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/IAS_score.tsv',sep = '\t')
IDEKER_MJ = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/ideker_hugo_220516.tsv', sep = '\t')
IDEKER_TOT_GS = list(set(list(IDEKER_IAS['Protein 1'])+list(IDEKER_IAS['Protein 2']))) # 16840


(1) LINCS 에 있는 애들 제대로 확인 

L_in_I_symbol = [a for a in LANDMARK_symbol if a in IDEKER_TOT_GS] # 966
L_notin_I_symbol = [a for a in LANDMARK_symbol if a not in IDEKER_TOT_GS] # 12

L_I_IAS = IDEKER_IAS[ (IDEKER_IAS['Protein 1'].isin(L_in_I_symbol)) & (IDEKER_IAS['Protein 2'].isin(L_in_I_symbol))  ]
L_N_I_IAS = IDEKER_IAS[ (IDEKER_IAS['Protein 1'].isin(L_in_I_symbol)==False) & (IDEKER_IAS['Protein 2'].isin(L_in_I_symbol)==False)  ]

CHECK_L_in_alias = [a for a in L_notin_I_symbol if (a in check_alis) & (a not in dont_alis)]
# -> 일단 LINCS 에 있는 IDE 안맞는 12개 애들을 string 기준 alias 에서 찾을 수 있다는 점 ! 

LINCS_ALIAS = PPI_ORI_ALI_LNI[PPI_ORI_ALI_LNI.alias.isin(L_notin_I_symbol) ]
LINCS_ALIAS.columns = 
check_this = list(set(LINCS_ALIAS.ORI_NAME))


(2) LINCS 의 ALIAS 들은 IAS 에 있을까 

IDEKER_NOT_LINCS_GS = list(set(list(L_N_I_IAS['Protein 1'])+list(L_N_I_IAS['Protein 2']))) 

LINCS_ALIAS_IN_IAS = [a for a in check_this if a in IDEKER_NOT_LINCS_GS]
LINCS_ALIAS_NOT_IN_IAS = [a for a in check_this if a not in IDEKER_NOT_LINCS_GS]


(3) 다시 정리 

BETA_lm_genes2 = BETA_lm_genes[['gene_symbol','gene_id','ensembl_id']]
BETA_lm_genes2.columns = ['L_gene_symbol','entrez','ensembl_id']

LINCS_ALIAS2 = PPI_ORI_ALI_LNI[PPI_ORI_ALI_LNI.alias.isin(L_notin_I_symbol) ]
LINCS_ALIAS2.columns = ['#string_protein_id','L_gene_symbol','PPI_source','PPI_name']

BETA_ALI = pd.merge(BETA_lm_genes2, LINCS_ALIAS2, on = 'L_gene_symbol', how ='left' )


for i in range(BETA_ALI.shape[0]):
    ppi = BETA_ALI.PPI_name[i]
    if type(ppi) == float :
        BETA_ALI.PPI_name[i] = BETA_ALI.loc[i]['L_gene_symbol']




[a for a in BETA_ALI.PPI_name if a not in IDEKER_TOT_GS]
'HDGFRP3' # 유일하게 못찾겠는 애 

######################################



BETA_ALI = pd.to_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

BETA_ALI = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')



