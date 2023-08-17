



checking the data overlap 



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
# SAVE_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_349_FULL/'

file_name = 'M3V6_349_MISS2_FULL' # my total 
file_name = 'M3V6_349_MISS2_ONEIL'

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

# A_B_C_S_SET = A_B_C_S_SET[A_B_C_S_SET.ONEIL == 'O'] # 16422

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




print('CIDs', flush = True)
tmp = list(set(A_B_C_S_SET_COH2.CID_CID))
tmp2 = sum([a.split('___') for a in tmp],[])
print(len(set(tmp2)) , flush = True)


print('CID_CID', flush = True)
print(len(set(A_B_C_S_SET_COH2.CID_CID)), flush = True)



print('CID_CID_CCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.cid_cid_cell)), flush = True)

print('DrugCombCCLE', flush = True)
print(len(set(A_B_C_S_SET_COH2.CELL)), flush = True)





###########################################################################################
###########################################################################################
###########################################################################################
Deep synergy 

TOOL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/01.DeepSynergy'

labels = pd.read_csv(os.path.join(TOOL_PATH,'DeepSynergy-master/labels.csv'), index_col=0) 
labels = pd.concat([labels, labels]) 
DeepS_label = list(set(list(labels.drug_a_name) + list(labels.drug_b_name))) # 38개 


DS_labels_RE = pd.read_csv(os.path.join(TOOL_PATH,'RELABEL.csv'))


def change_cellname(old , new) :
	index_num = list(DS_labels_RE[DS_labels_RE.cell_line==old].index)
	for ind in index_num :
		DS_labels_RE.at[ind, 'cell_line'] = new


change_cellname('UWB1289BRCA1','UWB1289+BRCA1')
change_cellname('SKOV3','SK-OV-3')
change_cellname('SW620','SW-620')
change_cellname('NCIH460','NCI-H460')
change_cellname('T47D','T-47D')




###########################################################################################
###########################################################################################
###########################################################################################
Matchmaker 


TOOL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/other_tools/03.MatchMaker/data_ori/'
CEL_DATA =pd.read_csv(TOOL_PATH + 'E-MTAB-3610.sdrf.txt', sep = '\t')



def data_loader(drug1_chemicals, drug2_chemicals, cell_line_gex, comb_data_name):
    print("File reading ...")
    comb_data = pd.read_csv(comb_data_name, sep="\t")
    cell_line = pd.read_csv(cell_line_gex,header=None)
    chem1 = pd.read_csv(drug1_chemicals,header=None)
    chem2 = pd.read_csv(drug2_chemicals,header=None)
    synergies = np.array(comb_data["synergy_loewe"])
    #
    cell_line = np.array(cell_line.values)
    chem1 = np.array(chem1.values)
    chem2 = np.array(chem2.values)
    return chem1, chem2, cell_line, synergies


with open(TOOL_PATH + 'drugs_info.json') as json_file :
	MM_drug_info =json.load(json_file)


MM_DATA = TOOL_PATH + 'DrugCombinationData.tsv'
MM_comb_data = pd.read_csv(MM_DATA, sep="\t")

drug1 = TOOL_PATH + 'drug1_chem.csv'
drug2 = TOOL_PATH + 'drug2_chem.csv'
CEL_gex = TOOL_PATH + 'cell_line_gex.csv'

chem1, chem2, cell_line, synergies = data_loader(drug1, drug2, CEL_gex, MM_DATA )

chem1.shape # (286421, 541)
chem2.shape # (286421, 541)
cell_line.shape # (286421, 972)
synergies.shape # (286421,)



##### match 확인 

set(MM_comb_data.cell_line_name) - set(DC_CELL_DF2.DC_cellname) # 786-O
set(MM_comb_data.cell_line_name) - set(DRSPRING_CELL)
set(DRSPRING_CELL) - set(MM_comb_data.cell_line_name)
# {'SKMEL30', 'ZR751', 'MSTO', 'CAOV3', 'DLD1', 'NCI-H460', 'KPL1'}


MM_comb_drug = list(set(list(MM_comb_data['drug_row']) + list(MM_comb_data['drug_col'])))
set(MM_comb_drug) - set(DC_DRUG_DF_FULL.dname)

set(DRSPRING_CELL) - set(DC_CELL_DF2.DC_cellname) # 0 




MM_comb_drug_match_name = list(set(list(MM_comb_data.drug_row) + list(MM_comb_data.drug_col))) # 3040
MM_drug_match = pd.DataFrame({'MM_drug' : MM_comb_drug_match_name  })


MM_drug_CID_name = {key : MM_drug_info[key]['name'] for key in MM_drug_info.keys()} # 3952
missing = [a for a in MM_comb_drug_match_name if a not in MM_drug_CID_name.values()] # 119?


MM_drug_CID_name_DF = pd.DataFrame.from_dict(MM_drug_CID_name, orient = 'index')
MM_drug_CID_name_DF['MM_CID'] = list(MM_drug_CID_name_DF.index)
MM_drug_CID_name_DF.columns = ['MM_drug', 'MM_CID']

MM_drug_match_2 = pd.merge(MM_drug_match, MM_drug_CID_name_DF, on='MM_drug', how='left')


for indind in range(MM_drug_match_2.shape[0]) :
    if type(list(MM_drug_match_2.MM_CID)[indind]) != str :
        MM_drug_match_2.at[indind,'MM_CID'] = '0'

MM_drug_match_2['MM_CID'] = [int(a) for a in MM_drug_match_2['MM_CID']]

MM_drug_match_ok = MM_drug_match_2[MM_drug_match_2.MM_CID>0] # 2921
MM_drug_match_miss = MM_drug_match_2[MM_drug_match_2.MM_CID==0] # 119



chem_feat_dict = {a : np.round(MM_drug_info[a]['chemicals'],4) for a in MM_drug_info.keys()}

drug_ind = MM_comb_data[MM_comb_data.drug_row == 'COSTUNOLIDE'].index[0].item()
chem1[drug_ind] # 541,




for indind in list(MM_drug_match_miss.index) : 
    drugname = MM_drug_match_miss.at[indind, 'MM_drug']
    drug_ind = MM_comb_data[MM_comb_data.drug_row == drugname].index[0].item()
    ans_feat = np.round(chem1[drug_ind].tolist(),4)
    #
    for cid in chem_feat_dict.keys() :
        tmp= []
        if all(ans_feat == chem_feat_dict[cid]) == True:
            MM_drug_match_miss.at[indind, 'MM_CID'] = int(cid )
            tmp.append(cid)
        if len(set(tmp)) > 1 :
            print(cid) # 겹치는 feat 없는걸로 
       


MM_drug_match_miss[MM_drug_match_miss.MM_CID==0] # 전부 매치 


MM_drug_match_final = pd.concat([MM_drug_match_ok, MM_drug_match_miss])

check = MM_drug_match_final[MM_drug_match_final.MM_CID.duplicated(False)] # 다 보겠다는거 

check_cid = list(check.MM_CID)

for cid in check_cid :
    names = list(MM_drug_match_final[MM_drug_match_final.MM_CID == cid]['MM_drug'])
    if len(names) == 2 :
        drug_ind_1 = MM_comb_data[MM_comb_data.drug_row == names[0]].index[0].item()
        drug_ind_2 = MM_comb_data[MM_comb_data.drug_row == names[1]].index[0].item()
        if all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_2],4)) == False :
            print(cid)
    else :
        print(cid)


names = list(MM_drug_match_final[MM_drug_match_final.MM_CID == cid]['MM_drug'])
#drug_ind_1 = MM_comb_data[MM_comb_data.drug_row == names[0]].index[0].item()
#drug_ind_2 = MM_comb_data[MM_comb_data.drug_row == names[1]].index[0].item()
#drug_ind_3 = MM_comb_data[MM_comb_data.drug_row == names[2]].index[0].item()


#all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_2],4))
#all(np.round(chem1[drug_ind_1],4) == np.round(chem1[drug_ind_3],4))
#all(np.round(chem1[drug_ind_2],4) == np.round(chem1[drug_ind_3],4))



MM_drug_match_final.columns = ['drug_row','drug_row_cid']
MM_comb_data_RE = pd.merge(MM_comb_data, MM_drug_match_final, on ='drug_row', how = 'left')

MM_drug_match_final.columns = ['drug_col','drug_col_cid']
MM_comb_data_RE = pd.merge(MM_comb_data_RE, MM_drug_match_final, on ='drug_col', how = 'left')












###########################################################
###########################################################
###########################################################
###########################################################
###########################################################


# 진짜 서로꺼 일단 없는건지 확인 

DS_drug_cid_all = list(DS_labels_RE.drug_a_CID) + list(DS_labels_RE.drug_b_CID)
DS_cell_all = list(DS_labels_RE.cell_line)


MM_drug_cid_all = list(MM_comb_data_RE.drug_row_cid) + list(MM_comb_data_RE.drug_col_cid)
MM_cell_all = list(MM_comb_data_RE.cell_line_name)


JY_drug_cid_all = list(A_B_C_S_SET_COH2.CID_A) + list(A_B_C_S_SET_COH2.CID_B)
JY_cell_all = list(A_B_C_S_SET_COH2.DC_cellname)


set(DS_cell_all) - set(JY_cell_all) # 'COLO320DM', 'UWB1289+BRCA1', 'EFM192B', 'LNCAP', 'OCUBM'
set(DS_cell_all) - set(DC_CELL_DF2.DC_cellname) # 없음. 위의 애들은 어느 필터링 과정에서 떨어져 나간거임 . 

set(MM_cell_all) - set(JY_cell_all) # 'TC-71', 'OVCAR-5', 'CAKI-1', 'SN12C', 'UO-31', 'PC-3', 'SF-295', 'ACHN', 'U251', 'MDA-MB-231', 'IGROV1', 'MOLT-4', 'CCRF-CEM', 'SK-MEL-2', 'A549', 'HS 578T', 'HCT-15', 'SF-268', 'SK-MEL-5', 'A498', 'RD', 'OVCAR-4', 'DU-145', 'HOP-62', 'K-562', 'BT-549', 'NCI-H522', 'SF-539', 'HOP-92', 'KM12', 'SK-MEL-28', 'RXF 393', '786-0', 'OVCAR-8', 'UACC-257', 'NCI-H322M', 'LOX IMVI', 'SNB-75', 'OCUBM', 'COLO 205', 'L-428', 'A-673', 'HCC-2998', 'T98G', 'EKVX', 'HDLM-2', 'MDA-MB-468', 'NCI-H226', 'RPMI-8226', 'SR', 'TK-10', 'L-1236', 'M14', 'MCF7'
set(MM_cell_all) - set(DC_CELL_DF2.DC_cellname) # 786-0만 문제. 

set(JY_cell_all) - set(DS_cell_all) # 없음 
set(JY_cell_all) - set(MM_cell_all) # 'ZR751', 'DLD1', 'KPL1', 'CAOV3', 'SKMEL30', 'NCI-H460', 'MSTO' ->??

# 그래 그래서 보니까 이건 matchmaker 의 문제임 
set(DS_cell_all) -set(MM_cell_all) # 'ZR751', 'DLD1', 'KPL1',  'CAOV3', 'SKMEL30', 'NCI-H460', 'MSTO' 'COLO320DM', 'UWB1289+BRCA1', 'EFM192B', 'LNCAP'
set(MM_cell_all)-set(DS_cell_all) # 'TC-71', 'OVCAR-5', 'CAKI-1', 'SN12C', 'UO-31', 'PC-3', 'SF-295', 'ACHN', 'U251', 'MDA-MB-231', 'IGROV1', 'MOLT-4', 'CCRF-CEM', 'SK-MEL-2', 'A549', 'HS 578T', 'HCT-15', 'SF-268', 'SK-MEL-5', 'A498', 'RD', 'OVCAR-4', 'DU-145', 'HOP-62', 'K-562', 'BT-549', 'NCI-H522', 'SF-539', 'HOP-92', 'KM12', 'SK-MEL-28', 'RXF 393', '786-0', 'OVCAR-8', 'UACC-257', 'NCI-H322M', 'LOX IMVI', 'SNB-75', 'COLO 205', 'L-428', 'A-673', 'HCC-2998', 'T98G', 'EKVX', 'HDLM-2', 'MDA-MB-468', 'NCI-H226', 'RPMI-8226', 'SR', 'TK-10', 'L-1236', 'M14', 'MCF7'


(set(JY_cell_all) & set(DS_cell_all)) - set(MM_cell_all) # 'ZR751', 'DLD1', 'KPL1', 'CAOV3', 'SKMEL30', 'NCI-H460', 'MSTO'




# DS_labels_RE

cid_a = list(DS_labels_RE['drug_a_CID'])
cid_b = list(DS_labels_RE['drug_b_CID'])
cells = list(DS_labels_RE['cell_line'])


DS_labels_RE['ON_CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(DS_labels_RE.shape[0])]
DS_labels_RE['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(DS_labels_RE.shape[0])]


DS_CID = set(list(DS_labels_RE.drug_a_CID) + list(DS_labels_RE.drug_b_CID))
DS_CELL = set(DS_labels_RE.cell_line)



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

# in case.. 
ccle_info_name_re = ccle_info[['cell_line_name','stripped_cell_line_name','CCLE_Name']]
ccle_info_name_re.columns = ['ccle_cell_name','STRIP','CELL']
A_B_C_S_SET_COH3 = pd.merge(A_B_C_S_SET_COH2, ccle_info_name_re, on = 'CELL', how ='left')

my_CELL2 = set(A_B_C_S_SET_COH3.ccle_cell_name)
my_CELL3 = set(A_B_C_S_SET_COH3.STRIP)





# target filtering ver 

# A_B_C_S_SET_COH2
my_CID = set(list(A_B_C_S_SET_COH2.CID_A) + list(A_B_C_S_SET_COH2.CID_B)) # 28 
my_CELL = set(A_B_C_S_SET_COH2.DC_cellname) # 34

# in case.. 
ccle_info_name_re = ccle_info[['cell_line_name','stripped_cell_line_name','CCLE_Name']]
ccle_info_name_re.columns = ['ccle_cell_name','STRIP','CELL']
A_B_C_S_SET_COH3 = pd.merge(A_B_C_S_SET_COH2, ccle_info_name_re, on = 'CELL', how ='left')

my_CELL2 = set(A_B_C_S_SET_COH3.ccle_cell_name)
my_CELL3 = set(A_B_C_S_SET_COH3.STRIP)




all_cid = my_CID & DS_CID & MM_CID ; len(all_cid) # 29
all_cell = my_CELL & DS_CELL & MM_CEL ; len(all_cell) # 27 
#all_cell = my_CELL2 & DS_CELL & MM_CEL ; len(all_cell) # 7
#all_cell = my_CELL3 & DS_CELL & MM_CEL ; len(all_cell) # 23



yes_TF_CID = [208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 3062316, 216239, 3385, 5288382, 5311, 60750, 5329102, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
yes_TF_CELL = ['LOVO', 'A375', 'HT29', 'OVCAR3', 'SW-620', 'SK-OV-3', 'MDAMB436', 'NCIH23', 'RKO', 'UACC62', 'A2780', 'VCAP', 'A427', 'T-47D', 'ES2', 'PA1', 'RPMI7951', 'SKMES1', 'NCIH2122', 'HT144', 'NCIH1650', 'SW837', 'OV90', 'UWB1289', 'HCT116', 'A2058', 'NCIH520']

no_TF_CID = [104842, 208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 24748204, 3062316, 216239, 3385, 5288382, 5311, 59691338, 60750, 5329102, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
no_TF_CELL = ['LOVO', 'A375', 'HT29', 'OVCAR3', 'SW-620', 'SK-OV-3', 'MDAMB436', 'NCIH23', 'RKO', 'UACC62', 'A2780', 'VCAP', 'A427', 'T-47D', 'ES2', 'PA1', 'RPMI7951', 'SKMES1', 'NCIH2122', 'HT144', 'NCIH1650', 'SW837', 'OV90', 'UWB1289', 'HCT116', 'A2058', 'NCIH520']





# filter check 
(1) 205_1

my_filter_1 = A_B_C_S_SET_COH2[A_B_C_S_SET_COH2.CID_A.isin(no_TF_CID)]
my_filter_2 = my_filter_1[my_filter_1.CID_B.isin(no_TF_CID)]
my_filter_3 = my_filter_2[my_filter_2.DC_cellname.isin(no_TF_CELL)] # 6778
len(set(my_filter_3.ON_CID_CID_CELL)) # 6778
my_on_c_c_c = list(set(my_filter_3.ON_CID_CID_CELL))


DS_labels_RE_filter_1 = DS_labels_RE[DS_labels_RE.drug_a_CID.isin(no_TF_CID)]
DS_labels_RE_filter_2 = DS_labels_RE_filter_1[DS_labels_RE_filter_1.drug_b_CID.isin(no_TF_CID)]
DS_labels_RE_filter_3 = DS_labels_RE_filter_2[DS_labels_RE_filter_2.cell_line.isin(no_TF_CELL)]
len(set(DS_labels_RE_filter_3.ON_CID_CID_CELL)) # 9477



MM_comb_data_RE_filter_1 = MM_comb_data_RE[MM_comb_data_RE.drug_row_cid.isin(no_TF_CID)]
MM_comb_data_RE_filter_2 = MM_comb_data_RE_filter_1[MM_comb_data_RE_filter_1.drug_col_cid.isin(no_TF_CID)]
MM_comb_data_RE_filter_3 = MM_comb_data_RE_filter_2[MM_comb_data_RE_filter_2.cell_line_name.isin(no_TF_CELL)]
len(set(MM_comb_data_RE_filter_3.ON_CID_CID_CELL)) # 8849

# why different? 
# match with mine 
# why different result? zzzzzzzzzzzzzzz
# match with my synergy answer then. 

DS_labels_RE_filter_4 = DS_labels_RE_filter_3[DS_labels_RE_filter_3.ON_CID_CID_CELL.isin(my_on_c_c_c)]

MM_comb_data_RE_filter_4 = MM_comb_data_RE_filter_3[MM_comb_data_RE_filter_3.ON_CID_CID_CELL.isin(my_on_c_c_c)]


DS_labels_RE_filter_4[DS_labels_RE_filter_4.ON_CID_CID_CELL=='5329102___15953832___A375']
MM_comb_data_RE_filter_4[MM_comb_data_RE_filter_4.ON_CID_CID_CELL=='5329102___15953832___A375']
my_ind = A_B_C_S_SET_COH3[A_B_C_S_SET_COH3.ON_CID_CID_CELL=='5329102___15953832___A375'].index.item()
MY_syn_RE2[my_ind]


DS_labels_RE_filter_4[DS_labels_RE_filter_4.ON_CID_CID_CELL=='5311___60700___VCAP']
MM_comb_data_RE_filter_4[MM_comb_data_RE_filter_4.ON_CID_CID_CELL=='5311___60700___VCAP']
my_ind = A_B_C_S_SET_COH3[A_B_C_S_SET_COH3.ON_CID_CID_CELL=='5311___60700___VCAP'].index.item()
MY_syn_RE2[my_ind]

DS_labels_RE_filter_4[DS_labels_RE_filter_4.ON_CID_CID_CELL=='3062316___24856436___NCIH2122']
MM_comb_data_RE_filter_4[MM_comb_data_RE_filter_4.ON_CID_CID_CELL=='3062316___24856436___NCIH2122']
my_ind = A_B_C_S_SET_COH3[A_B_C_S_SET_COH3.ON_CID_CID_CELL=='3062316___24856436___NCIH2122'].index.item()
MY_syn_RE2[my_ind]






# 일단 생 5CV orderr 



A_B_C_S_SET_SM = copy.deepcopy(A_B_C_S_SET_COH2) # 

# get unique values, remove duplicates, but keep original counts
data_no_dup, counts = np.unique(list(A_B_C_S_SET_SM['SM_C_CHECK']), return_counts=True)
data_no_dup_cells = [setset.split('___')[2] for setset in data_no_dup]
data_no_dup_sm_sm = [setset.split('___')[0]+'___'+setset.split('___')[1] for setset in data_no_dup]
data_nodup_df = pd.DataFrame({
	'setset' : data_no_dup.tolist(), 
	'cell' : data_no_dup_cells,
	'SM_SM' : data_no_dup_sm_sm
	 })


SM_SM_list = list(set(data_nodup_df.SM_SM))
SM_SM_list.sort()
sm_sm_list_1 = sklearn.utils.shuffle(SM_SM_list, random_state=42)

bins = [a for a in range(0, len(sm_sm_list_1), round(len(sm_sm_list_1)*0.2) )]
bins = bins[1:]
res = np.split(sm_sm_list_1, bins)

CV_1_smsm = list(res[0])
CV_2_smsm = list(res[1])
CV_3_smsm = list(res[2])
CV_4_smsm = list(res[3])
CV_5_smsm = list(res[4])
if len(res) > 5 :
	CV_5_smsm = list(res[4]) + list(res[5])

len(sm_sm_list_1)
len(CV_1_smsm) + len(CV_2_smsm) + len(CV_3_smsm) + len(CV_4_smsm) + len(CV_5_smsm)

CV_1_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_1_smsm)]['setset'])
CV_2_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_2_smsm)]['setset'])
CV_3_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_3_smsm)]['setset'])
CV_4_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_4_smsm)]['setset'])
CV_5_setset = list(data_nodup_df[data_nodup_df.SM_SM.isin(CV_5_smsm)]['setset'])



CV_ND_INDS = {
	'CV0_train' : CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset, 
	'CV0_test' : CV_5_setset,
	'CV1_train' : CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset, 
	'CV1_test' : CV_1_setset,
	'CV2_train' : CV_3_setset + CV_4_setset + CV_5_setset + CV_1_setset,
	'CV2_test' : CV_2_setset,
	'CV3_train' : CV_4_setset + CV_5_setset + CV_1_setset + CV_2_setset,
	'CV3_test' : CV_3_setset,
	'CV4_train' : CV_5_setset + CV_1_setset + CV_2_setset + CV_3_setset,
	'CV4_test' : CV_4_setset 
}

print(data_nodup_df.shape)
len( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset)
len(set( CV_1_setset + CV_2_setset + CV_3_setset + CV_4_setset + CV_5_setset ))



A_B_C_S_SET_SM['CID_CID_CELL'] = A_B_C_S_SET_SM.CID_CID +"___"+ A_B_C_S_SET_SM.DC_cellname

ABCS_tv_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_train'])]
ABCS_test_0 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV0_test'])]

ABCS_tv_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_train'])]
ABCS_test_1 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV1_test'])]

ABCS_tv_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_train'])]
ABCS_test_2 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV2_test'])]

ABCS_tv_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_train'])]
ABCS_test_3 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV3_test'])]

ABCS_tv_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_train'])]
ABCS_test_4 = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SM_C_CHECK.isin(CV_ND_INDS['CV4_test'])]


DRSPRING_CID = list(set(list(A_B_C_S_SET_SM.CID_A) + list(A_B_C_S_SET_SM.CID_B))) # 1342
DRSPRING_CELL = list(set(A_B_C_S_SET_SM.DC_cellname))

#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
# only mine & deepsynergy 

# 이번엔 synergy 갈리는것도 안나누기로 함 


# DS_labels_RE

cid_a = list(DS_labels_RE['drug_a_CID'])
cid_b = list(DS_labels_RE['drug_b_CID'])
cells = list(DS_labels_RE['cell_line'])


DS_labels_RE['ON_CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(DS_labels_RE.shape[0])]
DS_labels_RE['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(DS_labels_RE.shape[0])]


DS_CID = set(list(DS_labels_RE.drug_a_CID) + list(DS_labels_RE.drug_b_CID))
DS_CELL = set(DS_labels_RE.cell_line)





# mine 
cid_a = list(A_B_C_S_SET_COH2['CID_A'])
cid_b = list(A_B_C_S_SET_COH2['CID_B'])
cells = list(A_B_C_S_SET_COH2['DC_cellname'])


A_B_C_S_SET_COH2['ON_CID_CID'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(A_B_C_S_SET_COH2.shape[0])]
A_B_C_S_SET_COH2['ON_CID_CID_CELL'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) + '___' + str(cells[i]) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) + '___' + str(cells[i]) for i in range(A_B_C_S_SET_COH2.shape[0])]



mine_ccc = A_B_C_S_SET_COH2[['ON_CID_CID_CELL']]
mine_ccc['syn'] = MY_syn_RE2.squeeze().tolist()


my_CID = set(list(A_B_C_S_SET_COH2.CID_A) + list(A_B_C_S_SET_COH2.CID_B))
my_CELL = set(A_B_C_S_SET_COH2.DC_cellname)



all_cid = my_CID & DS_CID ; len(all_cid) # 30
all_cell = my_CELL & DS_CELL ; len(all_cell) # 34
#all_cell = my_CELL2 & DS_CELL & MM_CEL ; len(all_cell) # 7
#all_cell = my_CELL3 & DS_CELL & MM_CEL ; len(all_cell) # 23


# 30 & 34 
no_TF_CID = [104842, 208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 3062316, 24748204, 216239, 3385, 5288382, 5311, 59691338, 60750, 5329102, 11960529, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
no_TF_CELL = ['T-47D', 'RKO', 'ES2', 'RPMI7951', 'NCIH520', 'MSTO', 'NCIH2122', 'MDAMB436', 'OV90', 'KPL1', 'HT144', 'A375', 'PA1', 'CAOV3', 'OVCAR3', 'LOVO', 'NCIH1650', 'A427', 'VCAP', 'NCI-H460', 'SK-OV-3', 'DLD1', 'A2058', 'SW837', 'SKMES1', 'UWB1289', 'HCT116', 'A2780', 'ZR751', 'UACC62', 'SW-620', 'NCIH23', 'SKMEL30', 'HT29']

# 27 & 34
yes_TF_CID = [208908, 46926350, 24964624, 5394, 11977753, 60700, 15953832, 3062316, 216239, 3385, 5288382, 5311, 60750, 5329102, 11960529, 31703, 2907, 126941, 9826528, 176870, 36462, 5743, 5746, 24856436, 387447, 24958200, 4091]
yes_TF_CELL = ['T-47D', 'RKO', 'ES2', 'RPMI7951', 'NCIH520', 'MSTO', 'NCIH2122', 'MDAMB436', 'OV90', 'KPL1', 'HT144', 'A375', 'PA1', 'CAOV3', 'OVCAR3', 'LOVO', 'NCIH1650', 'A427', 'VCAP', 'NCI-H460', 'SK-OV-3', 'DLD1', 'A2058', 'SW837', 'SKMES1', 'UWB1289', 'HCT116', 'A2780', 'ZR751', 'UACC62', 'SW-620', 'NCIH23', 'SKMEL30', 'HT29']

'T-47D', 
'RKO', 
'ES2', 
'RPMI7951', 
'NCIH520', 
'MSTO', 
'NCIH2122', 
'MDAMB436',
'OV90', 
'KPL1', 

'HT144', 
'A375', 
'PA1', 
'CAOV3', 
'OVCAR3', 
'LOVO', 
'NCIH1650', 
'A427', 

'VCAP', 
'NCI-H460', 
'SK-OV-3', 
'DLD1', 
'A2058', 
'SW837', 
'SKMES1',
'UWB1289',

'HCT116', 
'A2780', 
'ZR751', 
'UACC62', 
'SW-620', 
'NCIH23', 
'SKMEL30', 
'HT29'


ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(yes_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(yes_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(yes_TF_CELL)]

'BREAST', 'OVARY', 'PROSTATE', 'LUNG', 'LARGE_INTESTINE', 'SKIN', 'PLEURA'








cp W801.py W811.py
cp W802.py W812.py
cp W803.py W813.py
cp W804.py W814.py
cp W805.py W815.py
cp W806.py W816.py
cp W807.py W817.py
cp W808.py W818.py
















#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
#############################################
# only mine & matchmaker  



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
#all_cell = my_CELL2 & DS_CELL & MM_CEL ; len(all_cell) # 
#all_cell = my_CELL3 & DS_CELL & MM_CEL ; len(all_cell) # 



yes_TF_CID = [57363, 16678941, 24764449, 442424, 8249, 442428, 52912189, 65628, 11214940, 24739943, 73777259, 57469, 
442514, 16533, 25125014, 41114, 114850, 3825829, 442534, 9863367, 65752, 11493598, 16007391, 82146, 114917, 243, 16097523, 42598643, 24772860, 4210951, 1048845, 10027278, 10240275, 8478, 2777391, 11338033, 44187953, 11673921, 323, 10461508, 8515, 54575456, 115067, 1794427, 46907787, 8593, 9806229, 25182616, 10437018, 46940575, 9929127, 425, 197033, 8617, 444, 65999, 45375953, 2761171, 11313622, 11485656, 25050, 8691, 16886, 16048654, 66062, 5702160, 66065, 24748573, 8742, 672296, 1548887, 5284441, 44483163, 131682, 46244454, 50905713, 16654980, 1204893, 5284513, 60138149, 681, 24756910, 123596, 1549008, 148177, 16736978, 41684, 123608, 148195, 11707110, 156391, 123631, 8955, 56599293, 156414, 156422, 774, 8980, 41774, 1549120, 164676, 33613, 9888590, 312145, 53396311, 9051, 10158940, 10126189, 66414, 53388144, 2950007, 16720766, 10109823, 896, 904, 25207689, 41867, 689043, 9560989, 936, 57336745, 938, 967, 42623951, 10077147, 57336812, 53322732, 1046, 11977753, 9270, 123964, 50248, 1102, 123985, 451668, 135398492, 135398495, 17506, 1123, 135398501, 17513, 135398510, 135398513, 25093235, 135398516, 656516, 9351, 3081361, 1174, 1183, 11314340, 11715767, 22291652, 656583, 5113032, 71271629, 9433, 1271002, 3392731, 11175137, 9444, 107751, 135398635, 74989, 7251185, 9458, 51037431, 135398661, 11617559, 656665, 107807, 1893668, 57394468, 91430, 1345, 25232708, 10200390, 107848, 91466, 6505803, 214348, 91469, 11511120, 135398737, 11248978, 135398739, 135398740, 135398744, 135398745, 11625818, 15983966, 1893730, 5924208, 23635314, 1400, 1401, 2827646, 25150857, 107935, 132519, 9642, 24905142, 11249084, 24905149, 107969, 9840076, 135423438, 16041424, 107985, 91610, 23668193, 24856041, 11609586, 91649, 34312, 1546, 443939, 443943, 4253236, 9782, 34359, 23725625, 108092, 108107, 9823820, 42574, 44516953, 149096, 1645, 108143, 108144, 509554, 444020, 444025, 25183872, 444036, 26248, 42640, 9880, 34458, 15951529, 9903, 100016, 476861, 24995523, 24995524, 1744, 56649450, 1775, 460612, 23643975, 657237, 51039, 51040, 18283, 11167602, 24856436, 5318517, 44607360, 10114, 51082, 25126798, 1935, 657298, 10133, 49850262, 10430360, 25167777, 18343, 10168, 1983, 10177, 452548, 10184653, 10206, 10207, 10208, 10219, 10228, 2044, 2049, 2051, 2052, 2082, 2083, 11634725, 11978790, 44607530, 9914412, 2092, 2828334, 24889392, 24864821, 135415867, 11626560, 11364421, 2125, 2130, 135440466, 125017, 2141, 174174, 2145, 2148, 2153, 2157, 2159, 2161, 2162, 4302963, 49784945, 2836600, 2170, 26757, 2187, 2214, 24963252, 2230, 2244, 379077, 2265, 2267, 135473382, 10494, 26879, 10316032, 10127622, 2333, 2337, 9865515, 2353, 2355, 46180666, 444732, 54675776, 2369, 54675779, 5318980, 135399748, 17754438, 2375, 54675783, 2381, 26964, 5318997, 24775005, 2404, 2405, 56846693, 10607, 387447, 444795, 10621, 135539077, 44599690, 10639, 2453, 92577, 51358113, 2467, 2471, 11561383, 2478, 11626927, 11667893, 6605258, 2518, 2520, 71748056, 11749858, 2531, 2540, 2541, 68077, 54684141, 2548, 2554, 53340666, 25102847, 46926350, 2577, 2578, 5310993, 57379345, 2585, 35370, 92727, 56650312, 5311068, 2662, 5352062, 2703, 2708, 2719, 2722, 10917, 2727, 2733, 1067700, 2741, 5278396, 2749, 2754, 2756, 20916937, 11561674, 2763, 2764, 2779853, 2771, 25242324, 3001055, 16681698, 445154, 24857323, 2796, 2747117, 2800, 2801, 2803, 135400182, 2812, 59239165, 54676228, 11234052, 60164, 27400, 9956119, 10341154, 92965, 60198, 150311, 5311272, 46213929, 969516, 11057, 56953649, 5081913, 11065, 2895, 2907, 11102, 2913, 5311340, 10070893, 5311345, 51039095, 16747388, 9825149, 5360515, 16722836, 9907093, 5311382, 3034010, 17755052, 3378093, 3009, 16755649, 68546, 68551, 11398092, 10275789, 35802, 68589, 3054, 3071, 5311501, 3100, 3108, 3025961, 5327916, 3117, 56970298, 158781, 44093, 11488320, 3025986, 60490, 3151, 3157, 3168, 568416, 3191, 9907323, 46926973, 68740, 60560, 3218, 216210, 3220, 101526, 49867929, 49867930, 16747683, 11201705, 53398697, 216237, 216239, 9931953, 9931954, 216248, 60606, 60613, 46943432, 42642645, 42642648, 68827, 3308, 68861, 11406590, 216326, 57339144, 68873, 3339, 216345, 77082, 11152667, 60700, 60699, 3356, 135564570, 56962337, 3366, 3367, 3371, 3372, 3373, 11676971, 3385, 3394, 3397, 60749, 60750, 9915743, 44383, 5459308, 25062766, 6450551, 54685047, 9883002, 60795, 3456, 3463, 3468, 3475, 3476, 216467, 57519510, 60825, 24882589, 3488, 60834, 11683, 445858, 60837, 60838, 11455910, 57519523, 3499, 3501, 60846, 3503, 60854, 68210102, 60871, 134601, 60877, 3541, 3547, 3549, 9874913, 3559, 24858111, 24964624, 10202642, 3606, 60953, 60961, 159269, 46911017, 159278, 462382, 3637, 3639, 11226684, 3652, 6852167, 3657, 3672, 59174488, 159324, 3676, 159325, 3685, 3690, 36462, 11570805, 134780, 151166, 3712, 3715, 151171, 11964036, 5312137, 11955855, 3747, 3749, 11595431, 59174579, 3767, 135564985, 175804, 3779, 3795, 16666332, 6917864, 3822, 11210478, 3825, 3826, 3827, 151289, 24776445, 12035, 2723601, 72716071, 3883, 3885, 53235510, 20279, 3899, 3902, 3922, 72200024, 667490, 36708, 667493, 3950, 3955, 3957, 5459840, 25022340, 3973, 5328779, 9949093, 15953832, 4008, 446378, 4011, 5287855, 4021, 4030, 4033, 36811, 4044, 15953870, 44240850, 4054, 4055, 5353431, 126941, 54677470, 4075, 53232, 667639, 49836027, 4091, 208898, 9826308, 208908, 4114, 4121, 4122, 176155, 53276, 176158, 176167, 4139, 5328940, 4170, 446541, 4184, 4189, 11317348, 4205, 5329006, 4211, 4212, 24825971, 11751549, 6918296, 4261, 12456, 6918313, 25227436, 77999, 4272, 610479, 11858109, 25227462, 5329098, 5329102, 9826528, 135565545, 135411, 119034, 11178236, 479503, 176406, 135565596, 5321010, 12597, 6918454, 25145656, 4409, 4418, 135565635, 4421, 4463, 4477, 4485, 4488, 6918540, 4493, 119182, 4495, 4497, 4499, 135401907, 54677946, 4539, 5288382, 5362123, 5362124, 5362129, 4566, 4578, 4583, 6918638, 4594, 4595, 4599, 160254, 676352, 4609, 6435335, 4636, 11620908, 9908783, 42611257, 9818682, 4680, 5280343, 4705, 160355, 4708, 447077, 5280360, 4713, 5280373, 16118392, 5280378, 4735, 24785538, 4746, 29327, 5485201, 4760, 4768, 9933475, 21157, 37542, 44462760, 4781, 4784, 4788, 5280443, 5280445, 6918852, 5280453, 4813, 11612883, 22024915, 25260757, 4823, 4829, 176870, 176871, 5280489, 4842, 176873, 54670067, 5362422, 4858, 3003141, 44143370, 4878, 4888, 4893, 25195294, 4911, 4912, 4915, 13109, 5280567, 119607, 5280569, 72201027, 398148, 4932, 4943, 46191454, 10474335, 24965990, 71521142, 11219835, 4993, 5337997, 521106, 4068248, 5035, 24826799, 5042, 10113978, 5056, 5070, 439246, 5073, 13266, 5280723, 5078, 54678486, 439260, 5090, 5092, 766949, 10310632, 10302451, 54259, 57390074, 9884685, 5280793, 5280794, 5280795, 406563, 5280805, 5163, 5166, 5169, 135418940, 5186, 54360, 51000408, 5210, 5280863, 16086114, 644213, 644215, 46224516, 5253, 5258, 644241, 9819282, 5291, 3003565, 54454, 54686904, 5307, 9843900, 135410875, 5311, 5280961, 21700, 16118986, 22049997, 439501, 439503, 5335, 5336, 5339, 5340, 25015515, 3052775, 5355, 5281004, 439533, 5358, 42636535, 24958200, 25023738, 5376, 5379, 5381, 6419718, 5386, 5281034, 5394, 5401, 5405, 5426, 62770, 24794418, 5430, 5281078, 5281081, 5453, 439647, 5472, 5483, 5330286, 5487, 135402864, 23663996, 9549184, 16659841, 62857, 5526, 210332, 5533, 62878, 54687, 447905, 5538, 447912, 5546, 9860529, 300471, 5566, 5281222, 5578, 1324494, 2733525, 5591, 62935, 447966, 13791, 53401057, 9549284, 9549289, 5614, 9549298, 636402, 9549305, 62978, 9934347, 448014, 456214, 5656, 44135961, 63002, 5289501, 63009, 448042, 54840, 5694, 71234, 11556427, 5709, 5712, 5717, 46216796, 5730, 54891, 5743, 5746, 54900, 9934458, 5755, 5754, 5757, 5756, 9868928, 71301, 5281417, 5770, 5775, 5780, 6444692, 5494425, 9811611, 5790, 6420138, 5803, 5494449, 5815, 5816, 5819, 71360, 9893571, 5833, 5852, 16725726, 521951, 71398, 5865, 5870, 5881, 374536, 161557, 5909, 5935, 5281605, 5281607, 5959, 5281612, 5281613, 5281614, 54671203, 5330790, 5991, 178024, 5994, 6009, 6010, 5281672, 5281673, 6029, 6041, 6047, 11671467, 5281708, 5281718, 6072, 38853, 11327430, 55245, 11655119, 219100, 24180719, 10090485, 5281787, 5281807, 9902100, 6167, 448537, 6169, 6175, 30751, 11524144, 6197, 2807869, 5281855, 71764, 6230, 6234, 702558, 23582824, 6251, 6252, 6253, 16095342, 6256, 52918385, 135403646, 39042, 6279, 9926791, 11712649, 25229450, 6291, 6293, 6301, 25254071, 9803963, 25229526, 16038120, 59472121, 1349907, 11630874, 448799, 6436, 104741, 14973220, 10271028, 104762, 104769, 3086685, 3086686, 104799, 22880, 6503, 285033, 9820526, 4659569, 46209401, 16759173, 11647372, 104850, 56973724, 448949, 5282230, 80311, 72139, 11565518, 9869779, 6613, 6445533, 448991, 5478883, 6445540, 9796068, 31211, 72172, 6445562, 31239, 10172943, 10074640, 449054, 105000, 3062316, 5282379, 31307, 6741, 72281, 6758, 440936, 72300, 14957, 72301, 72303, 11270783, 637568, 11713159, 47751, 39562, 14989, 154256, 72344, 5388961, 449193, 31401, 11393719, 47812, 25262792, 66558664, 11442891, 6872, 6890, 16718576, 72435, 11213558, 47866, 15103, 6912, 11238147, 252682, 137994, 56949517, 49855250, 11598628, 441140, 47936, 31553, 25033539, 9927531, 7020, 441203, 25262965, 56207, 39836, 7108, 97226, 31703, 441314, 12000240, 441328, 441336, 11754511, 17751063, 441401, 10288191, 638024, 54680660, 49831008, 9829523, 11476171, 44137675, 64715, 220401, 515328, 16219401, 24771867, 15547703, 3038522, 3038525, 10427712, 204100, 638278, 11656518, 46398810, 44219749, 24821094, 7533, 73078, 9969021, 7550, 49806720, 400769, 392622, 4521392, 64945, 2809273, 23666110, 
163263, 179651, 64971, 7638, 64982, 44137946, 5447130, 65015, 65016, 24066, 135421442, 9952773, 9911830, 65064, 
10296883, 135659062, 11640390, 57335384, 442021, 9903786, 40632, 11984591, 10297043, 11427553, 9871074, 44187362, 
2817763, 11271909, 442088, 9813758, 24788740, 17751819, 16760588, 135413536, 5480230, 24360, 5717801, 59604787, 
65335, 65340, 65348, 16760658, 122724, 16230, 16231, 73581, 180081, 65399, 5283731, 65495, 327653]
yes_TF_CELL = ['TC-71', 'T-47D', 'OVCAR-5', 'CAKI-1', 'UO-31', 'RKO', 'PC-3', 'SF-295', 'ES2', 'ACHN', 'RPMI7951', 'NCIH520', 'U251', 'MDA-MB-231', 'IGROV1', 'SK-MEL-2', 'NCIH2122', 'A549', 'HS 578T', 'MDAMB436', 'OV90', 'HCT-15', 'SF-268', 'SK-MEL-5', 'A498', 'HT144', 'RD', 'A375', 'PA1', 'OVCAR-4', 'OVCAR3', 'LOVO', 'NCIH1650', 'A427', 'VCAP', 'HOP-62', 'K-562', 'BT-549', 'NCI-H522', 'SF-539', 'HOP-92', 'SK-OV-3', 'KM12', 'SK-MEL-28', 'A2058', 'OVCAR-8', 'UACC-257', 'SW837', 'SKMES1', 'LOX IMVI', 'SNB-75', 'UWB1289', 'A-673', 'L-428', 'T98G', 'HCT116', 'EKVX', 'HDLM-2', 'UACC62', 'MDA-MB-468', 'SW-620', 'A2780', 'NCI-H226', 'RPMI-8226', 'SR', 'L-1236', 'NCIH23', 'HT29', 'MCF7']

no_TF_CID = [
16220172, 15581198, 57363, 11501591, 5464092, 16678941, 32798, 24764449, 23625762, 8228, 753704, 10174505, 53239854, 213039, 442424, 8249, 29933626, 9871419, 442428, 52912189, 11591741, 11345983, 11960382, 17571905, 442439, 229455, 1269845, 65625, 65628, 11214940, 11477084, 65632, 49766501, 24739943, 32874, 16490, 73777259, 23642227, 9887867, 114811, 57469, 311434, 144, 24199313, 442514, 688272, 16533, 25125014, 41114, 114850, 3825829, 442534, 155815, 24748204, 65709, 135430323, 114869, 23666879, 10141893, 9863367, 16007367, 15573192, 11313361, 4194514, 65752, 11493598, 16007391, 82146, 82148, 73957, 114917, 54690031, 5742832, 65777, 16097523, 243, 42598643, 442614, 10322165, 248, 24772860, 10256643, 4210951, 24879368, 1048845, 10027278, 16736529, 10240275, 8478, 2777391, 11338033, 44187953, 10182969, 11673921, 323, 65859, 8515, 10461508, 131411, 54575456, 6603108, 196968, 65906, 65909, 8567, 1794427, 115067, 8574, 5702024, 46907787, 729483, 115087, 65935, 8593, 45375887, 60195218, 9806229, 25182616, 65944, 10437018, 65947, 65948, 46940575, 136257951, 136257953, 836002, 65958, 9929127, 8617, 425, 197033, 20849066, 5702062, 54682040, 444, 65981, 16097729, 115150, 65999, 45375953, 2761171, 11313622, 3654103, 11485656, 25050, 10158562, 238053, 5284329, 5284330, 73265276, 188914, 8691, 16886, 8695, 54706679, 24822275, 5284360, 16048654, 66062, 5702160, 66065, 66064, 66070, 8732, 24748573, 8742, 672296, 5284419, 115269, 5702220, 1548887, 66136, 5284441, 44483163, 5284443, 8798, 5284447, 66144, 4022878, 131682, 5284452, 46244454, 66577006, 50905713, 9953906, 2155128, 16654980, 54698642, 148121, 115355, 1204893, 5284513, 60138149, 164521, 681, 24756910, 1901244, 5284544, 57336514, 5284549, 123596, 9839311, 46883536, 148177, 44483281, 16736978, 41684, 66259, 123606, 123607, 123608, 1549008, 3080922, 148195, 148198, 5284583, 156391, 148201, 11707110, 5284587, 17134, 123631, 8955, 56599293, 156414, 156418, 156419, 156422, 774, 5284627, 8980, 9937686, 5284631, 6398764, 41774, 41781, 71729974, 9015, 71729980, 44573504, 1549120, 164676, 51348293, 123723, 33613, 9888590, 312145, 53396311, 11256664, 9051, 10158940, 23667548, 33630, 23675743, 71295844, 71295845, 57394021, 53396327, 53396328, 9066, 10126189, 66414, 53388144, 2950007, 9082, 17275, 16720766, 10109823, 896, 904, 25207689, 41867, 5702541, 11256720, 25256849, 689043, 42623900, 9560989, 1328033, 936, 57336745, 938, 23667628, 23667630, 23667631, 23667642, 15057856, 12944326, 967, 6325199, 44155856, 42623951, 443354, 10077147, 123879, 53322732, 57336812, 443382, 42066955, 1046, 11977753, 23725083, 135742497, 13018151, 9270, 9872438, 123964, 9913405, 50248, 222285, 1102, 123985, 2729042, 451668, 135398491, 135398492, 135398495, 17506, 1123, 135398501, 17513, 984170, 135398510, 135398513, 25093235, 135398516, 45270144, 656516, 9351, 12387471, 3081361, 1174, 1183, 25265312, 45139106, 11314340, 124087, 44205240, 11715767, 107706, 44139710, 656574, 72959169, 50922691, 22291652, 656583, 5113032, 443593, 517321, 1549517, 71271629, 181458, 1236, 9433, 1271002, 3392731, 3503, 56845533, 11175137, 9444, 107751, 135398635, 9954540, 74989, 7251185, 135398641, 9458, 656628, 6440181, 25134326, 656631, 51037431, 107771, 135398661, 11617559, 656665, 656667, 9938203, 107807, 451875, 57394468, 1893668, 91430, 58221879, 72664381, 1345, 25232708, 10200390, 107848, 91466, 6505803, 214347, 214348, 91469, 4908365, 11511120, 135398737, 11248978, 50515, 135398739, 135398740, 6710614, 135398741, 135398744, 135398745, 11625818, 135398742, 135398748, 15983966, 656734, 9848160, 132449, 1893730, 5924208, 91505, 23635314, 1400, 91513, 1401, 189821, 53495165, 2827646, 57345410, 66774402, 6710658, 25150857, 107935, 1615267, 132519, 9642, 24905142, 1361334, 25134521, 26041, 11249084, 24905149, 107969, 107970, 10339779, 18924996, 9840076, 135423438, 5277135, 16041424, 107985, 91610, 443867, 23668193, 443873, 44246499, 493027, 656867, 443878, 9703, 24856041, 108013, 13559281, 11609586, 60196343, 108031, 91649, 72295940, 34312, 1546, 9578005, 11462174, 443936, 443939, 25019940, 443943, 24798764, 443955, 4253236, 9782, 34359, 23725625, 108092, 9807431, 108107, 9823820, 11454028, 42574, 57468496, 165457, 25257557, 44516953, 11249248, 444000, 149096, 1645, 108143, 24872560, 108144, 71304818, 509554, 444020, 21014128, 444025, 444030, 2844288, 157313, 25183872, 444036, 26248, 42640, 9880, 34458, 108187, 44451493, 165542, 15951529, 9903, 100016, 476861, 25749183, 3032771, 1732, 24995523, 24995524, 1744, 56649450, 1775, 15558393, 11519741, 50942, 12150530, 1074953, 6604561, 9955116, 616236, 657201, 3008319, 460612, 44828485, 44828487, 23643975, 44828492, 657237, 51039, 51040, 11609955, 6915944, 132971, 11282283, 53348204, 18283, 83823, 11167602, 24856436, 5318517, 46843772, 44607360, 10114, 51081, 51082, 5351307, 25126798, 1935, 45942672, 16222097, 11626384, 657298, 10133, 657302, 49850262, 10430360, 2819993, 657308, 133021, 25167777, 18343, 1973, 10168, 1981, 1983, 1985, 10177, 135972801, 452548, 10184653, 67533, 1550286, 44632017, 71550931, 6449107, 2006, 14010333, 10206, 10207, 10208, 26596, 46933992, 10219, 44206063, 10228, 23693301, 92151, 10235, 2044, 10237, 83969, 46843906, 2051, 2052, 135440386, 2049, 2082, 2083, 11634725, 11978790, 44607530, 44607531, 9914412, 2092, 2828334, 24889392, 24864821, 23685176, 23685177, 135415867, 46925884, 51038269, 11626560, 11364421, 11159621, 125001, 2123, 2125, 2130, 135440466, 135743573, 125017, 2141, 174174, 2145, 44460130, 23668834, 2148, 2153, 2155, 2157, 10160238, 2159, 49784945, 2162, 4302963, 2161, 2836600, 2170, 2176, 22833280, 26757, 2187, 2196, 2206, 2214, 2926768, 24963252, 2230, 2244, 379077, 9849040, 2264, 2265, 2267, 43231, 157920, 135473382, 45279469, 46926062, 92409, 10494, 26879, 10316032, 2310, 444679, 53356806, 10127622, 2315, 5458190, 46934289, 2327, 2333, 2337, 9865515, 16124208, 2353, 9824562, 2355, 54708532, 9865528, 46180666, 444732, 354624, 2369, 54675776, 54675779, 135399748, 5318980, 17754438, 2375, 2296132, 54675783, 2378, 2370, 2381, 26964, 5318997, 2391, 24775005, 6433118, 6433119, 56846691, 2404, 2405, 56846693, 2415, 10607, 9865587, 387447, 444795, 59772, 10621, 135539077, 2438, 16738693, 11717001, 44599690, 2442, 10639, 2449, 2453, 10648, 92577, 2466, 2467, 51358113, 2471, 11561383, 44247466, 70789547, 2478, 11626927, 11667893, 76219, 6605258, 9890250, 10701, 2518, 71748056, 2520, 5310939, 44591583, 11749858, 2531, 2536, 2540, 54684141, 2541, 68077, 2548, 2554, 53340666, 25102847, 51712, 2563, 11872781, 46926350, 2576, 5310993, 46893585, 2577, 2578, 2581, 57379345, 45042199, 2585, 6335004, 24906275, 3033637, 51754, 51755, 35370, 92722, 92727, 5311037, 27200, 135465539, 5335621, 56650312, 10830, 5483090, 5311066, 5311068, 346721, 2662, 3246697, 10866, 2678, 2682, 49769085, 5352062, 14240392, 2703, 2708, 2717, 166558, 2719, 2720, 2722, 10917, 2727, 10921, 2733, 10930, 46926514, 1067700, 2741, 2746, 5278396, 2749, 2754, 2756, 24857286, 11381449, 11561674, 2762, 2763, 2779853, 5311180, 10152654, 68304, 2764, 20916937, 2771, 25242324, 11676373, 27350, 68313, 2777, 45279963, 3001055, 3033825, 16681698, 2787, 445154, 3033832, 25021162, 24857323, 2794, 2747117, 2796, 2797, 25021165, 5311217, 2800, 2801, 135449332, 2803, 135400182, 46885626, 2812, 59239165, 9906942, 135400189, 969472, 60164, 11234052, 54676228, 27400, 9956119, 44182295, 5311257, 44329754, 84759, 3549980, 10341154, 92965, 60198, 150311, 5311272, 46213929, 969516, 56953649, 11057, 666418, 27447, 11065, 3246906, 5081913, 11963194, 2883, 11079, 2889, 53316426, 2893, 76621, 5311309, 2895, 11092, 2907, 764764, 11102, 2913, 5327721, 5311339, 24800108, 10070893, 46173038, 5311340, 3033968, 2929, 10226546, 5311345, 51039095, 16747388, 9825149, 9956222, 134018, 5360515, 25226117, 16722832, 68234129, 16722834, 16722836, 9907093, 5311382, 3034010, 453548, 3378093, 17755052, 11987888, 68539, 3009, 68546, 16755649, 68551, 68553, 68554, 9808844, 10275789, 11398092, 3023, 9849808, 4369359, 71601111, 11224, 35802, 42609626, 3038, 207841, 68589, 3054, 68591, 68601, 51031035, 3071, 9841667, 3076, 24996872, 11273, 11717641, 5311501, 68624, 68626, 54766613, 5311510, 3025944, 3100, 248862, 68643, 3108, 158758, 68647, 3025961, 5327916, 3117, 68654, 60464, 15068211, 56970298, 44093, 158781, 11488320, 3025986, 60490, 158797, 3151, 3157, 3168, 3169, 568416, 16723044, 71576678, 68712, 3182, 11676786, 3191, 68727, 9907323, 72199292, 46926973, 3198, 240767, 3203, 68740, 60560, 216210, 3218, 3220, 101526, 49867926, 49867928, 49867929, 49867930, 44187, 6851740, 3229, 49867936, 49867937, 72551586, 3235, 16747683, 72551585, 5459110, 53398697, 46918825, 11201705, 9882793, 216237, 216239, 9931953, 9931954, 10366136, 216248, 29985980, 60606, 60613, 46943432, 24988881, 57748689, 42642645, 3288, 42642648, 68827, 3291, 150762, 60651, 68844, 3308, 60656, 25226483, 3324, 68861, 11406590, 3326, 216322, 3331, 3333, 3334, 68871, 216326, 68873, 57339144, 3339, 3341, 46189838, 11373846, 216345, 135564570, 11152667, 60700, 3356, 3357, 77082, 11548, 60699, 60703, 56962337, 3362, 3365, 3366, 3367, 46239014, 11676971, 3372, 3371, 3373, 68911, 3374, 11963697, 3385, 3394, 3397, 3099980, 60749, 60750, 3406, 716121, 9915743, 44383, 470375, 5459308, 25062766, 101744, 3440, 3446, 3447, 6450551, 54685047, 9883002, 3450, 44191096, 789882, 60795, 3456, 60803, 3463, 216457, 60810, 3468, 59428238, 57519505, 57519506, 216467, 3475, 9883029, 44199317, 3476, 57519512, 60825, 57519513, 3478, 42552731, 57519511, 28061, 24882589, 57519517, 25210273, 57519522, 3488, 60835, 445858, 60834, 11683, 60837, 60838, 57519525, 57519528, 57519529, 3499, 60846, 57519535, 3501, 57519537, 57519538, 57519539, 57519540, 57519541, 57519542, 57519543, 57519544, 57519536, 16108977, 60854, 60855, 60857, 68210102, 57519546, 3516, 57519549, 57519550, 57519551, 60865, 57519554, 57519555, 60871, 134601, 60877, 3538, 3541, 25161177, 3547, 3549, 54611422, 9874913, 25169382, 3559, 44551660, 929262, 24858111, 57339395, 11955716, 24866313, 24964624, 
44561, 10202642, 405012, 3606, 60953, 11292191, 54685215, 60961, 151075, 159269, 46911017, 462382, 54726191, 159278, 3637, 3639, 11226684, 3652, 52948550, 6852167, 3657, 36431, 6917719, 59174488, 3672, 3675, 3676, 159324, 159325, 6917733, 3686, 3685, 3690, 675434, 36462, 44543605, 3702, 11570805, 134780, 6450813, 151166, 5312125, 3712, 151170, 151171, 3715, 11964036, 10309252, 44224135, 6450819, 5312137, 151183, 11955855, 5312149, 21081761, 3747, 3749, 11595431, 3752, 3760, 59174579, 3767, 135564985, 175804, 11972288, 3779, 71315139, 3784, 20179, 3795, 3796, 10096344, 16666332, 4787937, 6917864, 421610, 11431660, 3822, 16051951, 11210478, 3825, 3826, 3827, 3830, 135409400, 151289, 24776445, 10317566, 12035, 24768261, 7048968, 2723601, 23662354, 102175, 126758, 72716071, 6852391, 3878, 3883, 3885, 53235510, 20279, 3899, 3902, 49852229, 16052038, 667467, 3922, 72200024, 3932, 3936, 667490, 36708, 667493, 3948, 3950, 5287792, 6918003, 3955, 3957, 454519, 9858940, 11358077, 135565181, 5459840, 11431811, 25022340, 3973, 2723716, 5328779, 9826188, 3994, 9949093, 4006, 4008, 15953832, 2723754, 4011, 73068460, 446378, 5287855, 1150897, 4021, 4030, 6918078, 4033, 4034, 9875401, 36811, 4044, 15953870, 44240850, 4054, 5353431, 4055, 6451164, 126941, 54677470, 135409642, 4075, 53232, 667639, 4091, 49836027, 208898, 9826308, 4100, 4101, 208902, 2797577, 4107, 208908, 46927888, 4114, 11538455, 135191, 4121, 49836058, 4122, 53276, 176155, 73265182, 73265183, 176158, 73265185, 6918178, 73265186, 73265187, 176167, 73265193, 4139, 167980, 5328940, 20525, 73265198, 73265199, 73265201, 73265200, 53325875, 73265204, 73265205, 73265206, 73265208, 73265209, 73265210, 73265211, 73265212, 6918203, 73265214, 73265215, 73265213, 73265218, 73265219, 73265220, 4165, 73265221, 73265223, 73265224, 73265222, 73265226, 73265227, 4170, 24883277, 73265229, 73265231, 73265230, 70701134, 73265232, 4173, 446541, 16666708, 73265237, 102484, 73265240, 73265239, 73265236, 4184, 4189, 9834591, 11317348, 4197, 4201, 4205, 5329006, 24825971, 4211, 4212, 73265271, 73265272, 73265273, 73265274, 73265275, 10047612, 73265277, 11751549, 73265278, 73265279, 73265281, 73265282, 73265283, 73265284, 73265285, 446598, 73265287, 73265289, 73265290, 4235, 73265292, 73265293, 73265294, 73265295, 73265296, 73265301, 73265302, 73265304, 6918296, 73265308, 73265309, 73265310, 73265312, 73265315, 73265316, 73265317, 4261, 77991, 12456, 6918313, 25227436, 77997, 127150, 73265326, 4272, 78000, 73265330, 73265329, 73265332, 77999, 73265334, 73265335, 73265336, 73265340, 11858109, 73265342, 73265343, 73265344, 73265341, 73265347, 73265348, 73265350, 73265351, 73265352, 73265353, 25227462, 73265355, 25022668, 73265357, 73265358, 5329098, 73265360, 73265361, 73265362, 73265363, 73265364, 73265365, 5329102, 73265367, 73265368, 73265369, 73265370, 11530459, 73265372, 53317853, 73265374, 73265371, 9826528, 73265375, 73265376, 73265377, 54685920, 73265379, 73265380, 159968, 73265381, 73265383, 73265382, 135565545, 73265386, 9892071, 73265389, 73265391, 73265392, 9801969, 73265393, 135411, 73265396, 56963315, 73265397, 73265399, 73265401, 119034, 73265403, 73265404, 11178236, 73265402, 73265407, 73265408, 73265409, 3010818, 73265411, 73265412, 73265413, 73265415, 71299339, 479503, 176406, 135565596, 11383075, 6918446, 5321010, 12597, 6918454, 25145656, 4409, 4410, 6918456, 4413, 6918462, 4418, 135565635, 4421, 59691338, 12620, 4440, 57519510, 6918493, 4463, 9818479, 46911863, 4472, 4477, 23630211, 57519518, 4485, 4486, 4487, 4488, 6918537, 57519519, 6918540, 4493, 119182, 57519520, 4494, 4495, 4497, 4499, 119192, 28094875, 4507, 57519523, 6918558, 46199207, 57519526, 135401907, 54677946, 4539, 5288382, 56832449, 11243969, 20933, 57519531, 86471, 5362123, 5362124, 57519533, 5362129, 24211921, 54677971, 57519534, 54677973, 4566, 11719130, 67310049, 59437537, 4578, 4583, 6918638, 4593, 4594, 4595, 4599, 25154041, 4601, 160254, 676352, 4609, 4614, 6435335, 11440648, 23654923, 49852941, 4621, 4622, 2724368, 37393, 23663126, 4634, 4636, 4646, 2765355, 11620908, 9908783, 42611257, 9818682, 4076092, 3035714, 4679, 4680, 119369, 119373, 11539025, 3011155, 5280343, 4705, 3002977, 1274465, 4708, 447077, 160355, 5280360, 44143209, 4713, 12309103, 5280373, 1069686, 16118392, 5280378, 9933439, 4735, 24785538, 4746, 29327, 5485201, 2331284, 4760, 24867488, 4768, 9933475, 21157, 37542, 44462760, 4781, 10367662, 4784, 4788, 160436, 5280443, 5280445, 6918848, 5018304, 5493444, 5280453, 6918852, 4810, 4813, 22024915, 11612883, 25260757, 9958103, 44315352, 4823, 4829, 119525, 176870, 176871, 5280489, 176873, 4842, 4843, 66646767, 54670067, 5362422, 4855, 4858, 3003141, 9868037, 44143370, 4878, 119570, 4887, 4888, 4891, 4893, 25195294, 119583, 119584, 16667431, 4906, 4909, 4911, 4912, 4915, 13109, 5280567, 119607, 5280569, 72201027, 398148, 4932, 4935, 12358477, 6878030, 4943, 60060499, 10253143, 46191454, 12047199, 10474335, 46199646, 3101542, 6746983, 24965990, 11711344, 71521142, 11219835, 16741245, 4993, 4996, 4248455, 447371, 56603532, 5337997, 23671691, 521106, 16307093, 4068248, 5034, 5035, 24826799, 5042, 10113978, 5056, 9810884, 2888648, 9966538, 439246, 5070, 5071, 5073, 13266, 5280723, 23622608, 5078, 54678486, 135705561, 54678490, 21467, 439260, 9950176, 5090, 5092, 766949, 5094, 10310632, 11351021, 439280, 54259, 54260, 10302451, 57390074, 9884685, 37907, 2733079, 16757783, 5280793, 25113626, 5280794, 5280795, 46232606, 168993, 406562, 7566371, 406563, 5280805, 71496742, 9843750, 5161, 5163, 5166, 5169, 135418940, 44250175, 5186, 10286159, 16725073, 791637, 16757846, 54360, 51000408, 5210, 5280863, 69923936, 5215, 16086114, 1029232, 644213, 185462, 644215, 9909368, 16757877, 46224516, 54405, 5253, 1791111, 5258, 23696523, 644241, 9819282, 5291, 3003565, 54454, 2725048, 177336, 54686904, 5307, 9843900, 135410875, 5311, 5280961, 21700, 71300295, 21704, 16118986, 5323, 5324, 439501, 177358, 22049997, 13520, 439503, 5325, 5326, 5327, 5328, 21718, 5335, 38103, 5329, 5330, 5333, 3052762, 5336, 5339, 5340, 439518, 25015515, 5344, 12358875, 3052775, 11654378, 5355, 5281004, 439533, 5358, 5359, 9819382, 42636535, 24958200, 25023738, 11326715, 210172, 5376, 5379, 5381, 6419718, 5281032, 8582409, 5386, 5281034, 9827599, 5394, 5281046, 5401, 25105690, 5405, 5281056, 21800, 6419753, 5281066, 11064619, 5281067, 5281068, 5281069, 5418, 24794418, 62770, 25154867, 5426, 5281078, 5429, 5430, 54585, 5281081, 23663939, 40629571, 23663941, 464205, 5453, 56956240, 23663953, 5281107, 23663956, 439647, 5472, 10138980, 23663973, 5479, 23663976, 5482, 23663979, 5483, 5330286, 11998575, 135402864, 5487, 7271796, 11548023, 23663996, 9549184, 16659841, 5505, 5510, 62857, 62859, 5526, 5530, 42628507, 11957660, 24180125, 62878, 210332, 5533, 447905, 5538, 62882, 11957668, 54687, 447912, 193962, 5546, 9860529, 11957684, 300471, 42628535, 5560, 67089852, 5566, 701891, 5281222, 62920, 13770, 5578, 1324494, 5583, 5585, 2733525, 62935, 5591, 636377, 5593, 11957721, 5597, 447966, 13791, 53401057, 9549284, 6419941, 9549289, 11154925, 636397, 52934127, 9549295, 5614, 636402, 9549298, 71157, 9549305, 62969, 439804, 11957756, 23836158, 62978, 456199, 456201, 9934347, 448013, 448014, 71183, 71188, 456214, 5656, 44135961, 63002, 5289501, 63009, 448042, 6420013, 9901617, 5684, 448055, 54840, 24753719, 5694, 71234, 46945860, 5748293, 6420040, 11556427, 24720972, 5709, 11949646, 5712, 16004692, 5717, 5720, 46216796, 5730, 5734, 54889, 54891, 3036780, 5742, 5743, 71279, 5745, 5746, 54900, 9934458, 5754, 5756, 5757, 5755, 9827968, 9868928, 1013376, 5763, 71301, 5281417, 5770, 5775, 6444692, 5780, 5494425, 9811611, 5281437, 5790, 54671008, 71329, 5795, 5799, 194216, 9803433, 71317162, 6420138, 251562, 5803, 11384493, 5807, 5494449, 51885747, 5815, 71352, 71353, 71354, 5816, 54679224, 5818, 5819, 71360, 25171648, 9893571, 53245636, 456389, 5833, 9795278, 51066577, 3938007, 71384, 5852, 16725726, 521951, 14051, 71398, 71399, 5865, 56645356, 10041070, 5870, 5881, 71420, 374536, 71481097, 12310282, 5905, 161557, 5909, 1554208, 22386467, 5935, 56948527, 71478, 24762166, 71485, 5281605, 5959, 5281607, 5281612, 5281613, 5281614, 5970, 5974, 54671203, 5330790, 11646823, 5991, 178024, 5994, 9820008, 128878, 6009, 6010, 6018, 6019, 5281672, 5281673, 6029, 55182, 49837968, 219025, 4282258, 2987927, 6041, 6047, 6048, 10434468, 11671467, 6060, 5281708, 5281718, 6072, 24795070, 71616, 24278976, 38853, 219078, 11327430, 219077, 55245, 11655119, 9910224, 24770514, 6103, 219099, 219100, 219104, 71651, 71655, 6126, 24180719, 71668, 10090485, 6135, 38904, 5281787, 38911, 3045381, 44341259, 9803788, 5281807, 9902100, 6167, 448537, 6169, 6172, 6175, 30751, 22571, 11524144, 6194, 6197, 67811385, 2807869, 5281855, 6209, 6215, 24901704, 71764, 1005654, 6230, 6234, 71773, 702558, 9918559, 6240, 6241, 71774, 135403620, 50878566, 23582824, 6251, 6252, 6253, 16095342, 6256, 52918385, 16078973, 135403646, 39042, 6279, 71815, 11712649, 16078986, 9926791, 25229450, 6291, 6293, 6301, 6445226, 7059633, 25254071, 11188409, 9803963, 4659392, 68712644, 129228, 135526609, 9910486, 25229526, 16038120, 71916, 11745519, 907504, 59472121, 899323, 11450633, 1349907, 2742550, 4331799, 11630874, 448799, 14973220, 104741, 6436, 54704426, 39212, 10271028, 71992, 104762, 6461, 104769, 153921, 31043, 9967941, 56645961, 66558287, 5462355, 260439, 24197464, 5282136, 5282138, 5282139, 3086685, 3086686, 104799, 31072, 22880, 11417954, 162147, 6503, 285033, 9820526, 47471, 47472, 4659569, 49830258, 49830260, 16759159, 46209401, 170361, 5486971, 170367, 5282181, 16759173, 153994, 104842, 11647372, 153997, 104850, 46209426, 56973724, 72092, 72093, 11155874, 16202152, 47528, 10459564, 448949, 5282230, 80311, 11983295, 72139, 11565518, 9869779, 6613, 44251605, 104920, 6445533, 448991, 5478883, 9796068, 6445540, 16038374, 23615975, 31211, 72172, 11581936, 6445562, 31239, 449035, 10172943, 10074640, 24951314, 449054, 23116322, 105000, 3062316, 60160561, 5282379, 31307, 
5282386, 6741, 9796181, 72281, 5282402, 6758, 5282407, 440936, 9869929, 5282408, 2882155, 72300, 14957, 72301, 72303, 72307, 11270783, 637568, 22624897, 5282435, 47751, 72327, 5282440, 53484170, 39562, 11713159, 14989, 154256, 9804433, 64150, 72344, 5003929, 5282458, 105115, 5388961, 10435235, 449193, 5282474, 31401, 11455910, 5282481, 31411, 11393719, 9894584, 9870009, 47812, 25262792, 66558664, 9910986, 11442891, 51346120, 6872, 56031, 6890, 46832368, 16718576, 72435, 11213558, 47866, 15103, 6912, 11844351, 11238147, 23665411, 56069, 252682, 137994, 56965901, 56949517, 49855250, 11598628, 59652905, 441130, 400169, 16218924, 9829162, 154417, 56654642, 441140, 23360, 47936, 31553, 25033539, 56965963, 39764, 10173277, 50985821, 9927531, 7020, 441203, 25262965, 11074431, 46898058, 56205, 56207, 56208, 6519698, 441242, 39836, 25099184, 1088438, 7108, 97226, 24779724, 441302, 31703, 441307, 441314, 25066467, 15331, 121829, 441325, 72686, 441328, 12000240, 31728, 6437877, 441336, 441344, 441345, 7172, 2800647, 441351, 56329, 5495818, 5495819, 23565, 11754511, 24853523, 17751063, 121892, 46930984, 46930998, 1367095, 441401, 46931003, 3570748, 46931005, 10288191, 52943938, 441411, 638024, 15443, 54680660, 310360, 44473434, 49831008, 3005572, 64648, 146571, 9936012, 56463, 10026128, 35028115, 9829523, 253078, 44309670, 200103, 16759993, 16219326, 9804991, 64710, 64713, 44137675, 64715, 11476171, 9968854, 24812758, 64737, 4979942, 220401, 6438130, 12934390, 400633, 515328, 64769, 16219401, 130313, 122125, 24771867, 57703712, 58539301, 3038502, 51371303, 46931242, 9927978, 941361, 15547703, 3038522, 3038525, 10427712, 204100, 638278, 11656518, 16211283, 49831257, 46398810, 44219749, 24821094, 15723, 4259181, 7533, 73078, 9969021, 7550, 49806720, 400769, 56704, 122262, 73115, 68853159, 32169, 25206185, 392622, 4521392, 64945, 2809273, 23666110, 163263, 179651, 64971, 1383884, 11869649, 64982, 7638, 44137946, 5447130, 16219612, 610479, 65015, 65016, 11673085, 135421442, 24066, 65028, 9952773, 57662985, 49864204, 9911830, 908828, 10231331, 65064, 10296883, 135659062, 5324346, 130621, 11640390, 56843850, 57335384, 110635, 6422124, 11525740, 73339, 56959, 57441923, 73356, 45489809, 24211, 9821849, 155290, 442021, 9903786, 40632, 5316290, 1048267, 11984591, 10297043, 57990869, 65243, 11427553, 73442, 44187362, 2817763, 11271909, 9871074, 442088, 16760554, 65264, 44195571, 135413494, 42647289, 9576185, 9813758, 135413505, 24788740, 49848070, 17751819, 16760588, 135413523, 135413534, 10280735, 135413536, 5480230, 24360, 5717801, 135413545, 11222830, 59604787, 65335, 65340, 65341, 130881, 65348, 16760658, 9936727, 65373, 122724, 16230, 16231, 73581, 180081, 73211763, 163701, 44195701, 65399, 16760696, 16760703, 20635522, 11976582, 5283731, 40854, 11689883, 16220066, 163751, 65464, 50905018, 90045, 65483, 2826191, 65495, 704473, 9887712, 327653, 5283820, 16367, 46931953, 23711819 ]
no_TF_CELL = ['TC-71', 'T-47D', 'OVCAR-5', 'CAKI-1', 'UO-31', 'RKO', 'PC-3', 'SF-295', 'ES2', 'ACHN', 'RPMI7951', 'NCIH520', 'U251', 'MDA-MB-231', 'IGROV1', 'SK-MEL-2', 'NCIH2122', 'A549', 'HS 578T', 'MDAMB436', 'OV90', 'HCT-15', 'SF-268', 'SK-MEL-5', 'A498', 'HT144', 'RD', 'A375', 'PA1', 'OVCAR-4', 'OVCAR3', 'LOVO', 'NCIH1650', 'A427', 'VCAP', 'HOP-62', 'K-562', 'BT-549', 'NCI-H522', 'SF-539', 'HOP-92', 'SK-OV-3', 'KM12', 'SK-MEL-28', 'A2058', 'OVCAR-8', 'UACC-257', 'SW837', 'SKMES1', 'LOX IMVI', 'SNB-75', 'UWB1289', 'A-673', 'L-428', 'T98G', 'HCT116', 'EKVX', 'HDLM-2', 'UACC62', 'MDA-MB-468', 'SW-620', 'A2780', 'NCI-H226', 'RPMI-8226', 'SR', 'L-1236', 'NCIH23', 'HT29', 'MCF7']



# MY 
# filter X 

filtt_1 = A_B_C_S_SET_COH2[A_B_C_S_SET_COH2.CID_A.isin(no_TF_CID)]
filtt_2 = filtt_1[filtt_1.CID_B.isin(no_TF_CID)]
filtt_3 = filtt_2[filtt_2.DC_cellname.isin(no_TF_CELL)]  # 53985


# filter O 

filtt_1 = A_B_C_S_SET_COH2[A_B_C_S_SET_COH2.CID_A.isin(yes_TF_CID)]
filtt_2 = filtt_1[filtt_1.CID_B.isin(yes_TF_CID)]
filtt_3 = filtt_2[filtt_2.DC_cellname.isin(yes_TF_CELL)]  # 31003



# MM 
# filter X 

mm_filtt_1 = MM_comb_data_RE[MM_comb_data_RE.drug_row_cid.isin(no_TF_CID)]
mm_filtt_2 = mm_filtt_1[mm_filtt_1.drug_col_cid.isin(no_TF_CID)]
mm_filtt_3 = mm_filtt_2[mm_filtt_2.cell_line_name.isin(no_TF_CELL)]  # 38697


# filter O 

mm_filtt_1 = MM_comb_data_RE[MM_comb_data_RE.drug_row_cid.isin(yes_TF_CID)]
mm_filtt_2 = mm_filtt_1[mm_filtt_1.drug_col_cid.isin(yes_TF_CID)]
mm_filtt_3 = mm_filtt_2[mm_filtt_2.cell_line_name.isin(yes_TF_CELL)]  # 24653



len(set(mm_filtt_3.ON_CID_CID_CELL) & set(filtt_3.ON_CID_CID_CELL)) # 32347
len(set(mm_filtt_3.ON_CID_CID_CELL) & set(filtt_3.ON_CID_CID_CELL)) # 20891



filter_X_common = list(set(mm_filtt_3.ON_CID_CID_CELL) & set(filtt_3.ON_CID_CID_CELL))
filter_O_common = list(set(mm_filtt_3.ON_CID_CID_CELL) & set(filtt_3.ON_CID_CID_CELL))

filt_path = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W215_349_MIS2/'

filt_path = '/home01/k040a01/02.M3V6/M3V6_W215_349_MIS2/'



with open(filt_path+'filter_X_common.pickle', 'wb') as f:
    pickle.dump(filter_X_common, f)

# load
with open(filt_path+'filter_X_common.pickle', 'rb') as f:
    filter_X_common = pickle.load(f)




with open(filt_path+'filter_O_common.pickle', 'wb') as f:
    pickle.dump(filter_O_common, f)

# load
with open(filt_path+'filter_O_common.pickle', 'rb') as f:
    filter_O_common = pickle.load(f)










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







# load
filt_path = '/home01/k040a01/02.M3V6/M3V6_W215_349_MIS2/'
with open(filt_path+'filter_X_common.pickle', 'rb') as f:
    filter_X_common = pickle.load(f)


ON_filt_1 = A_B_C_S_SET_COH[A_B_C_S_SET_COH.CID_A.isin(no_TF_CID)]
ON_filt_2 = ON_filt_1[ON_filt_1.CID_B.isin(no_TF_CID)]
ON_filt_3 = ON_filt_2[ON_filt_2.DC_cellname.isin(no_TF_CELL)]
ON_filt_4 = ON_filt_3[ON_filt_3.ON_CID_CID_CELL.isin(filter_X_common)]




A_B_C_S_SET_COH = copy.deepcopy(ON_filt_4)










