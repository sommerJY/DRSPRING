
matchmaker 제대로 비교하기 위함 


NETWORK_PATH = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
DC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'





NETWORK_PATH = '/home01/k040a01/01.Data/HumanNet/'
LINCS_PATH = '/home01/k040a01/01.Data/LINCS/'
DC_PATH = '/home01/k040a01/01.Data/DrugComb/'


DC_DRUG_DF_FULL = pd.read_csv(DC_PATH+'DC_DRUG_DF_PC.csv', sep ='\t')


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

# edge 
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

# 원래는 edge score 있지만 일단은...
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


SAVE_PATH = '/home01/k040a01/02.M3V6/M3V6_349_DATA/'
SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'

file_name = 'M3V6_349_MISS2_FULL' # my total 

A_B_C_S_SET_ADD = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET_ADD.csv'.format(file_name), low_memory=False)

MY_syn = torch.load(SAVE_PATH+'{}.MY_syn.pt'.format(file_name))


A_B_C_S_SET_ADD2 = copy.deepcopy(A_B_C_S_SET_ADD)

cid_a = list(A_B_C_S_SET_ADD2['CID_A'])
cid_b = list(A_B_C_S_SET_ADD2['CID_B'])
sm_a = list(A_B_C_S_SET_ADD2['ROW_CAN_SMILES'])
sm_b = list(A_B_C_S_SET_ADD2['COL_CAN_SMILES'])
ccle = list(A_B_C_S_SET_ADD2['CELL'])

A_B_C_S_SET_ADD2['CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_ADD2.shape[0])]
A_B_C_S_SET_ADD2['SM_C_CHECK'] = [sm_a[i] + '___' + sm_b[i]+ '___' + ccle[i] if sm_a[i] < sm_b[i] else sm_b[i] + '___' + sm_a[i]+ '___' + ccle[i] for i in range(A_B_C_S_SET_ADD2.shape[0])]

A_B_C_S_SET_ADD2['ori_index'] = list(A_B_C_S_SET_ADD2.index)

MISS_filter = ['AOBO','AXBX','AXBO','AOBX'] # 

A_B_C_S_SET = A_B_C_S_SET_ADD2[A_B_C_S_SET_ADD2.Basal_Exp == 'O']

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.SYN_OX == 'O'] # 11639

# A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.T1OX == 'O'] # 8086 -> 이걸 빼야하나 말아야하나 #################

A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.type.isin(MISS_filter)]





# basal node exclude -> CCLE match 만 사용
CCLE_PATH = '/home01/k040a01/01.Data/CCLE/'
# CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']

ccle_cell_info_filt = ccle_cell_info[ccle_cell_info.DepMap_ID.isin(ccle_exp['Unnamed: 0'])]
ccle_names = [a for a in ccle_cell_info_filt.DrugCombCCLE if type(a) == str]


A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.CELL.isin(ccle_names)]

data_ind = list(A_B_C_S_SET.index)
MY_syn_RE = MY_syn[data_ind]

A_B_C_S_SET = A_B_C_S_SET.reset_index(drop = True)


# cell line vector 

DC_CELL_DF2 = pd.read_csv(DC_PATH+'DC_CELL_INFO.csv', sep = '\t')
DC_CELL_DF2 = pd.concat([
	DC_CELL_DF2, 
	pd.DataFrame({'cell_line_id' : [1],'DC_cellname' : ['786O'],'DrugCombCello' : ['CVCL_1051'],'DrugCombCCLE':['786O_KIDNEY']})])

DC_CELL_info_filt = DC_CELL_DF2[DC_CELL_DF2.DrugCombCCLE.isin(A_B_C_S_SET.CELL)] # 38

DC_CELL_info_filt = DC_CELL_info_filt.drop(['Unnamed: 0'], axis = 1)
DC_CELL_info_filt.columns = ['cell_line_id', 'DC_cellname', 'DrugCombCello', 'CELL']
DC_CELL_info_filt = DC_CELL_info_filt[['CELL','DC_cellname']]

A_B_C_S_SET_COH = pd.merge(A_B_C_S_SET, DC_CELL_info_filt, on = 'CELL', how = 'left' )


C_names = list(set(A_B_C_S_SET_COH.DC_cellname))
C_names.sort()

C_freq = [list(A_B_C_S_SET_COH.DC_cellname).count(a) for a in C_names]
C_cclename = [list(A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname==a]['CELL'])[0] for a in C_names]

C_df = pd.DataFrame({'cell' : C_names, 'freq' : C_freq, 'ccle' : C_cclename})
C_df = C_df.sort_values('freq')


C_freq_filter = C_df
# C_freq_filter = C_df[C_df.freq>=200]


A_B_C_S_SET_COH = A_B_C_S_SET_COH[A_B_C_S_SET_COH.DC_cellname.isin(C_freq_filter.cell)]

DC_CELL_info_filt_re = DC_CELL_info_filt[DC_CELL_info_filt.DC_cellname.isin(C_freq_filter.cell)]
DC_CELL_info_filt_re['cell_onehot'] = [a for a in range(len(set(DC_CELL_info_filt_re.CELL)))]

DC_CELL_info_filt_re = DC_CELL_info_filt_re.reset_index(drop = True)

data_ind = list(A_B_C_S_SET_COH.index)
MY_syn_RE2 = MY_syn_RE[data_ind]

A_B_C_S_SET_COH2 = pd.merge(A_B_C_S_SET_COH, DC_CELL_info_filt_re[['DC_cellname','cell_onehot']], on = 'DC_cellname', how='left')
cell_one_hot = torch.nn.functional.one_hot(torch.Tensor(A_B_C_S_SET_COH2['cell_onehot']).long())







# MM_comb_data_RE

cid_a = list(MM_comb_data_RE['drug_row_cid'])
cid_b = list(MM_comb_data_RE['drug_col_cid'])
cells = list(MM_comb_data_RE['cell_line_name'])


MM_comb_data_RE['ON_CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(MM_comb_data_RE.shape[0])]
MM_comb_data_RE['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(MM_comb_data_RE.shape[0])]

MM_CID = set(list(MM_comb_data_RE.drug_row_cid) + list(MM_comb_data_RE.drug_col_cid))
MM_CEL = set(MM_comb_data_RE.cell_line_name)





# no target filtering ver 

cid_a = list(A_B_C_S_SET_COH2['CID_A'])
cid_b = list(A_B_C_S_SET_COH2['CID_B'])
cells = list(A_B_C_S_SET_COH2['DC_cellname'])


A_B_C_S_SET_COH2['ON_CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_COH2.shape[0])]
A_B_C_S_SET_COH2['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(A_B_C_S_SET_COH2.shape[0])]



mine_ccc = A_B_C_S_SET_COH2[['ON_CID_CID_CELL']]
mine_ccc['syn'] = MY_syn_RE2.squeeze().tolist()




my_CID = set(list(A_B_C_S_SET_COH2.CID_A) + list(A_B_C_S_SET_COH2.CID_B))
my_CELL = set(A_B_C_S_SET_COH2.DC_cellname)




all_cid = my_CID  & MM_CID ; len(all_cid) #
all_cell = my_CELL  & MM_CEL ; len(all_cell) # 







(1) XXXXX 





filt_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W215_349_MIS2/'

with open(filt_path+'filter_X_cid.pickle', 'wb') as f:
    pickle.dump(all_cid, f)

with open(filt_path+'filter_X_cell.pickle', 'wb') as f:
    pickle.dump(all_cell, f)



# load
with open(filt_path+'filter_X_cid.pickle', 'rb') as f:
    no_TF_CID = pickle.load(f)


# load
with open(filt_path+'filter_X_cell.pickle', 'rb') as f:
    no_TF_CELL = pickle.load(f)



# MY 
# filter X 

filtt_1 = A_B_C_S_SET_COH2[A_B_C_S_SET_COH2.CID_A.isin(no_TF_CID)]
filtt_2 = filtt_1[filtt_1.CID_B.isin(no_TF_CID)]
filtt_3 = filtt_2[filtt_2.DC_cellname.isin(no_TF_CELL)]  # 177576



# MM 
# filter X 

mm_filtt_1 = MM_comb_data_RE[MM_comb_data_RE.drug_row_cid.isin(no_TF_CID)]
mm_filtt_2 = mm_filtt_1[mm_filtt_1.drug_col_cid.isin(no_TF_CID)]
mm_filtt_3 = mm_filtt_2[mm_filtt_2.cell_line_name.isin(no_TF_CELL)]  # 172527


filter_X_common = list(set(mm_filtt_3.ON_CID_CID_CELL) & set(filtt_3.ON_CID_CID_CELL))

with open(filt_path+'filter_X_common.pickle', 'wb') as f:
    pickle.dump(filter_X_common, f)











cid_a = list(A_B_C_S_SET_COH['CID_A'])
cid_b = list(A_B_C_S_SET_COH['CID_B'])
cells = list(A_B_C_S_SET_COH['DC_cellname'])

A_B_C_S_SET_COH['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(len(cid_a))]



# load
filt_path = '/home01/k040a01/02.M3V6/M3V6_W215_349_MIS2/'
with open(filt_path+'filter_X_common.pickle', 'rb') as f:
    filter_X_common = pickle.load(f)


ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(no_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(no_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(no_TF_CELL)]
ON_filt_4 = ON_filt_3[ON_filt_3.ON_CID_CID_CELL.isin(filter_X_common)] # 148510


A_B_C_S_SET_COH = copy.deepcopy(ON_filt_4)


















(2) OOOOO 





filt_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W215_349_MIS2/'

with open(filt_path+'filter_O_cid.pickle', 'wb') as f:
    pickle.dump(all_cid, f)

with open(filt_path+'filter_O_cell.pickle', 'wb') as f:
    pickle.dump(all_cell, f)



# load
with open(filt_path+'filter_O_cid.pickle', 'rb') as f:
    yes_TF_CID = pickle.load(f)


# load
with open(filt_path+'filter_O_cell.pickle', 'rb') as f:
    yes_TF_CELL = pickle.load(f)



# MY 
# filter X 

filtt_1 = A_B_C_S_SET_COH2[A_B_C_S_SET_COH2.CID_A.isin(yes_TF_CID)]
filtt_2 = filtt_1[filtt_1.CID_B.isin(yes_TF_CID)]
filtt_3 = filtt_2[filtt_2.DC_cellname.isin(yes_TF_CELL)]  # 177576



# MM 
# filter X 

mm_filtt_1 = MM_comb_data_RE[MM_comb_data_RE.drug_row_cid.isin(yes_TF_CID)]
mm_filtt_2 = mm_filtt_1[mm_filtt_1.drug_col_cid.isin(yes_TF_CID)]
mm_filtt_3 = mm_filtt_2[mm_filtt_2.cell_line_name.isin(yes_TF_CELL)]  # 172527


filter_O_common = list(set(mm_filtt_3.ON_CID_CID_CELL) & set(filtt_3.ON_CID_CID_CELL))

with open(filt_path+'filter_O_common.pickle', 'wb') as f:
    pickle.dump(filter_O_common, f)











cid_a = list(A_B_C_S_SET_COH['CID_A'])
cid_b = list(A_B_C_S_SET_COH['CID_B'])
cells = list(A_B_C_S_SET_COH['DC_cellname'])

A_B_C_S_SET_COH['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(len(cid_a))]



# load
filt_path = '/home01/k040a01/02.M3V6/M3V6_W215_349_MIS2/'
with open(filt_path+'filter_O_common.pickle', 'rb') as f:
    filter_O_common = pickle.load(f)


ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(yes_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(yes_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(yes_TF_CELL)]
ON_filt_4 = ON_filt_3[ON_filt_3.ON_CID_CID_CELL.isin(filter_O_common)]

A_B_C_S_SET_COH = copy.deepcopy(ON_filt_4)













########################################3

148510


cell_list = list(ON_filt_4.CELL)
cell_set = list(set(cell_list))
cell_count= [cell_list.count(cell) for cell in cell_set]
cell_df = pd.DataFrame({'cell' : cell_set, 'count' : cell_count})

cell_df = cell_df.sort_values('count')
ordered_cells = list(cell_df['cell'])


cv_0 = [a*5 for a in range(14)]
cv_1 = [a*5 +1 for a in range(14)]
cv_2 = [a*5 +2 for a in range(14)]
cv_3 = [a*5 +3 for a in range(14)]
cv_4 = [a*5 +4 for a in range(13)]

cv_0_cells = [ordered_cells[cv_ind] for cv_ind in cv_0] # 28768
cv_1_cells = [ordered_cells[cv_ind] for cv_ind in cv_1] # 29307
cv_2_cells = [ordered_cells[cv_ind] for cv_ind in cv_2] # 30219
cv_3_cells = [ordered_cells[cv_ind] for cv_ind in cv_3] # 31700
cv_4_cells = [ordered_cells[cv_ind] for cv_ind in cv_4] # 28516






# filter O
'LARGE_INTESTINE',
'LUNG', 
'BREAST', 
'PROSTATE',
'SKIN',
'OVARY',

'SOFT_TISSUE',   
'CENTRAL_NERVOUS_SYSTEM', 
'BONE'
'HAEMATOPOIETIC_AND_LYMPHOID_TISSUE'
'KIDNEY'








################################
MM_comb_data_RE 에서 그래서 CID_CID_CELL 같은데 갈리는거 확인하기 

n1 = list(MM_comb_data_RE.drug_row)
n2 = list(MM_comb_data_RE.drug_col)
n3 = list(MM_comb_data_RE.cell_line_name)
MM_comb_data_RE['n_n_cell'] = [str(n1[i]) + '___' + str(n2[i]) + '___' + str(n3[i]) if n1[i] < n2[i] else str(n2[i]) + '___' + str(n1[i]) + '___' + str(n3[i]) for i in range(len(n1))]
사실 이럴 필요는 없었음 
그냥 다 CIDCID 기준으로 진행한듯 




aaa = list(MM_comb_data_RE['drug_row_cid'])
bbb = list(MM_comb_data_RE['drug_col_cid'])
ccc = list(MM_comb_data_RE['cell_line_name'])


# 306
MM_comb_data_RE['CID_CID'] = [str(int(aaa[i])) + '___' + str(int(bbb[i])) if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i])) for i in range(MM_comb_data_RE.shape[0])]

# 10404 -- duplicated 가 이상한게 아님 
MM_comb_data_RE['CID_CID_CELL'] = [str(int(aaa[i])) + '___' + str(int(bbb[i]))+ '___' + ccc[i] if aaa[i] < bbb[i] else str(int(bbb[i])) + '___' + str(int(aaa[i]))+ '___' + ccc[i] for i in range(MM_comb_data_RE.shape[0])]

set_data = list(set(MM_comb_data_RE['CID_CID_CELL']))
MM_comb_data_RE['ori_index'] = list(MM_comb_data_RE.index)


MM_comb_data_R2 = MM_comb_data_RE[['CID_CID_CELL','ori_index']]
rep_index = list(MM_comb_data_R2.CID_CID_CELL.drop_duplicates().index)
MM_comb_data_R2 = MM_comb_data_R2.loc[rep_index]





