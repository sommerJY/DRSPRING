
import pytorch 



굳이 나눌 필요가 있었나 싶은것...
그러면 그냥 
모든 CID_CID_CELL 에 대해서 일단 만들고,
순서가 근데
흐으으으으음 
농도가...........
일단 CID_CID_CELL 순서에 맞게 내꺼 데이터는 만들고,
tissue 구분없이 그냥 하는게 나을듯 
그러고 나서 
결과만 correlation 으로 확인해주면 되지 않을까? 
순서에 맞게? 같은 pair 라도,
1) low emax
2) high max 
3) low IC
4) high IC
이렇게 tissue 별로 나타내면 어떨지? 
집에가서 돌려놓고 자자 





gdsc_c_path = '/st06/jiyeonH/11.TOX/DR_SPRING/val_data/'


gdsc_breast = pd.read_csv(gdsc_c_path + 'breast_anchor_combo.csv') # 163470
gdsc_colon = pd.read_csv(gdsc_c_path + 'colon_anchor_combo.csv') # 75400
gdsc_pancreas = pd.read_csv(gdsc_c_path + 'pancreas_anchor_combo.csv') # 66300

g_breast_names = list(set(list(gdsc_breast['Anchor Name']) + list(gdsc_breast['Library Name']))) # 52 
g_colon_names = list(set(list(gdsc_colon['Anchor Name']) + list(gdsc_colon['Library Name']))) # 52 
g_pancreas_names = list(set(list(gdsc_pancreas['Anchor Name']) + list(gdsc_pancreas['Library Name']))) # 26

g_breast_cell = list(set(gdsc_breast['Cell Line name']))
g_colon_cell = list(set(gdsc_colon['Cell Line name']))
g_pancreas_cell = list(set(gdsc_pancreas['Cell Line name']))


gdsc_all = pd.concat([gdsc_breast, gdsc_colon ,gdsc_pancreas ])

gdsc_id_path = '/st06/jiyeonH/13.DD_SESS/GDSC/'
# Drug_listWed_Jul_6_11_12_48_2022.csv → GDSC_LIST_0706.tsv
gdsc_chem_list = pd.read_csv(gdsc_id_path + 'GDSC_LIST_0706.tsv', sep = '\t')
gdsc_chem_list2 = gdsc_chem_list[['Name','PubCHEM']].drop_duplicates() # 472
len(set(gdsc_chem_list2.Name)) # 449

gdsc_dups = gdsc_chem_list2[gdsc_chem_list2.Name.duplicated()]

gdsc_chem_list_dups = gdsc_chem_list2[gdsc_chem_list2.Name.isin(gdsc_dups.Name)]
gdsc_chem_list_nodups = gdsc_chem_list2[gdsc_chem_list2.Name.isin(gdsc_dups.Name)==False]

dup_names = list(set(gdsc_chem_list_dups.Name))

str_inds = []
for nana in dup_names : 
	tmp = gdsc_chem_list_dups[gdsc_chem_list_dups.Name==nana]
	tmp2 = tmp[tmp.PubCHEM.apply(lambda x : type(x)==str)]
	str_inds = str_inds + list(tmp2.index)


gdsc_chem_list = pd.concat([gdsc_chem_list_nodups, gdsc_chem_list_dups.loc[str_inds]])


gdsc_chem_list = pd.concat([gdsc_chem_list, pd.DataFrame({'Name' : ['Galunisertib'], "PubCHEM" : ['10090485']})])

gdsc_chem_list_re = gdsc_chem_list[gdsc_chem_list.Name.isin(g_breast_names + g_colon_names + g_pancreas_names)]
gdsc_chem_list_re['CID'] = gdsc_chem_list_re.PubCHEM.apply(lambda x : int(x))

gdsc_chem_list_re = gdsc_chem_list_re[['Name','CID']]


# 305170
gdsc_chem_list_re.columns = ['Anchor Name','Anchor CID']
gdsc_all1 = pd.merge(gdsc_all, gdsc_chem_list_re, on ='Anchor Name', how ='left')

gdsc_chem_list_re.columns = ['Library Name','Library CID']
gdsc_all2 = pd.merge(gdsc_all1, gdsc_chem_list_re, on ='Library Name', how ='left')



CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)

gdsc_cell = pd.read_csv(gdsc_id_path +'GDSC_CELL_0707.tsv', sep = '\t')

gdsc_cell_ccle = pd.merge(gdsc_cell, ccle_info, left_on = 'Passport', right_on = 'Sanger_Model_ID', how = 'left')
gdsc_cell_ccle2 = gdsc_cell_ccle[['Name','stripped_cell_line_name','CCLE_Name']]
gdsc_cell_ccle2 = pd.concat([gdsc_cell_ccle2 , pd.DataFrame({'Name' : ['ZR-75-1'],'stripped_cell_line_name':['ZR751'], 'CCLE_Name':['ZR751_BREAST']})])
gdsc_cell_ccle2 = gdsc_cell_ccle2.drop_duplicates()

gdsc_cell_ccle2.columns = ['Cell Line name','strip_name','ccle_name']

# 305170
gdsc_all3 = pd.merge(gdsc_all2, gdsc_cell_ccle2, on = 'Cell Line name', how = 'left')

gdsc_all4 = gdsc_all3[gdsc_all3['Anchor CID']>0]
gdsc_all5 = gdsc_all4[gdsc_all4['Library CID']>0]



cid_a = list(gdsc_all5['Anchor CID'])
cid_b = list(gdsc_all5['Library CID'])
cell = list(gdsc_all5['ccle_name'])

gdsc_all5['cid_cid_cell'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i]))+ '___' + cell[i] if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i]))+ '___' + cell[i] for i in range(gdsc_all5.shape[0])]
gdsc_all5['CID_A_CELL'] = gdsc_all5['Anchor CID'].apply(lambda a : str(int(a))) + '__' + gdsc_all5['ccle_name']
gdsc_all5['CID_B_CELL'] = gdsc_all5['Library CID'].apply(lambda b : str(int(b))) + '__' + gdsc_all5['ccle_name']
gdsc_all5['cid_cid'] = [str(int(cid_a[i])) + '___' + str(int(cid_b[i])) if cid_a[i] < cid_b[i] else str(int(cid_b[i])) + '___' + str(int(cid_a[i])) for i in range(gdsc_all5.shape[0])]
# 299610


len(gdsc_all5.cid_cid_cell) # 299610
len(set(gdsc_all5.cid_cid_cell)) # 71600




# lincs 10um 24h
filter2 = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_FILTER.20230614.csv')
LINCS_PERT_MATCH = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')

filter3 = filter2[['pert_id','sig_id','cell_iname']]
filter4 = pd.merge(filter3, LINCS_PERT_MATCH[['pert_id','CID']] , on = 'pert_id', how = 'left')
filter5 = pd.merge(filter4, ccle_info[['stripped_cell_line_name','CCLE_Name']].drop_duplicates(), left_on = 'cell_iname', right_on = 'stripped_cell_line_name', how = 'left' )



filter6 = filter5[filter5.CID>0]
filter7 = filter6[filter6.CCLE_Name.apply(lambda x : type(x) == str)]
filter7['CID_CELL'] = filter7.CID.apply(lambda x : str(int(x))) + '__' +filter7.CCLE_Name
filter7['long_id'] = filter7.CID.apply(lambda x : str(int(x))) + '___' +filter7.cell_iname

filter8 = filter7[['long_id','CID_CELL']].drop_duplicates() # 물론 CID 여러개 붙어서 문제는 있음. 만약에 붙는거 보고 sig dup 일어나면 평균 취해줘야함 

filter8.columns = ['long_id_A' , 'CID_A_CELL']
gdsc_all6 = pd.merge(gdsc_all5, filter8, on = 'CID_A_CELL', how = 'left') # row 61804, ccc : 47706

filter8.columns = ['long_id_B' , 'CID_B_CELL']
gdsc_all7 = pd.merge(gdsc_all6, filter8, on = 'CID_B_CELL', how = 'left') # 63800, ccc : 47706

g_long_A = list(gdsc_all7.long_id_A)
g_long_B = list(gdsc_all7.long_id_B)

ttype = [] 
for a in range(gdsc_all7.shape[0]) :
	type_a = type(g_long_A[a])
	type_b = type(g_long_B[a])
	if (type_a == str) & (type_b == str) : 
		ttype.append('AOBO')
	elif (type_a != str) & (type_b != str) : 
		ttype.append('AXBX')
	else : 
		ttype.append('AXBO')



gdsc_all7['type'] = ttype

gdsc_all7[gdsc_all7['type'] =='AOBO'] # 7894
gdsc_all7[gdsc_all7['type'] =='AXBO'] # 16248
gdsc_all7[gdsc_all7['type'] =='AXBX'] # 275468





# 0909 이후 
MJ_gdsc = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/GDSC_EXP_ccle_cellall_fugcn_hhhdttf3.csv')


entrez_id = list(MJ_gdsc['entrez_id'])
MJ_gdsc_re = MJ_gdsc.drop(['entrez_id','Unnamed: 0','CID__CELL'], axis =1)
MJ_gdsc_re['entrez_id'] = entrez_id

ord = [list(MJ_gdsc_re.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_gdsc_re = MJ_gdsc_re.loc[ord] 


# fu (M3 & M33)
def get_MJ_data( CHECK ): 
	if CHECK in list(MJ_gdsc_re.columns) :
		RES = MJ_gdsc_re[CHECK]
		OX = 'O'
	else : 
		RES = [0]*349        ##############
		OX = 'X'
	return list(RES), OX





LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

BETA_BIND_MEAN = torch.load( LINCS_ALL_PATH + "10_24_sig_cell_mean.0620.pt")
BETA_BIND_M_SIG_df_CID = pd.read_csv(LINCS_ALL_PATH + '10_24_sig_cell_mean.0620.csv')

BETA_BIND_M_SIG_df_CID['CID__CELL'] = BETA_BIND_M_SIG_df_CID.CID.apply(lambda x : str(x)) + "__" + BETA_BIND_M_SIG_df_CID.CCLE_Name

# 여기에 새로 줍줍한거 들어가야함 

def get_LINCS_data(CID__CELL):
	bb_ind = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID_CELL == CID__CELL ].index.item()
	sig_ts = BETA_BIND_MEAN[bb_ind]
	#
	return sig_ts



DC_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'

all_chem_DF = pd.read_csv(DC_ALL_PATH+'DC_ALL_7555_ORDER.csv')
all_chem_feat_TS = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_feat.pt')
all_chem_feat_adj = torch.load(DC_ALL_PATH+'DC_ALL.MY_chem_adj.pt')
all_chem_cids = list(all_chem_DF.CID)

additional_cids = {
	44450571 : 'C1=CC2=C(C=CC(=C2)C=C3C(=O)N=C(S3)NCC4=CC=CS4)N=C1',
	44224160 : 'COC1=CC=CC2=CC(=C3C4=C(N=CNN4C(=N3)C5CCC(CC5)C(=O)O)N)N=C21',
	10096043 : 'CCNC(=O)C1=C(C(=C2C=C(C(=CC2=O)O)C(C)C)ON1)C3=CC=C(C=C3)CN4CCOCC4',
    5310940 : 'C1CCC(C(C1)N)N.C(=O)(C(=O)O)O.[Pt]'
}


pres = pcp.get_compounds(148124,'cid')
pp = pres[0]
len(pp.canonical_smiles)
pres = pcp.get_compounds(10384072,'cid')
pp = pres[0]
len(pp.canonical_smiles)
pres = pcp.get_compounds(5311497,'cid')
pp = pres[0]
len(pp.canonical_smiles)
pres = pcp.get_compounds(24978538,'cid')
pp = pres[0]
len(pp.canonical_smiles)
pres = pcp.get_compounds(49846579,'cid')
pp = pres[0]
len(pp.canonical_smiles)
pres = pcp.get_compounds(36314,'cid')
pp = pres[0]
len(pp.canonical_smiles)
pres = pcp.get_compounds(5310940,'cid')
pp = pres[0]
len(pp.canonical_smiles)




def check_drug_f_ts(CID) :
	if CID in all_chem_cids :  
		INDEX = all_chem_DF[all_chem_DF.CID == CID].index.item()
		adj_pre = all_chem_feat_adj[INDEX]
		feat = all_chem_feat_TS[INDEX]
	#
	elif CID in pc_sm :
		feat, adj_pre = get_CHEM(CID)
	else  :
		feat, adj_pre = get_CHEM2(additional_cids[CID])
	return feat, adj_pre



PC_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
for_CAN_smiles = pd.read_csv(PC_PATH+'CID_SMILES.csv', low_memory = False)
pc_sm = list(for_CAN_smiles.CID)

def get_CHEM(cid, k=1):
	maxNumAtoms = max_len
	smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
	iMol = Chem.MolFromSmiles(smiles.strip())
	iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) 
	iFeature = np.zeros((maxNumAtoms, 64)) ################## feature 크기 고정 
	iFeatureTmp = []
	for atom in iMol.GetAtoms():
		iFeatureTmp.append( atom_feature(atom) )### atom features only
	iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
	iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
	iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
	ADJ = adj_k(np.asarray(iAdj), k)
	return iFeature, ADJ

def get_CHEM2(smiles, k=1):
	maxNumAtoms = max_len
	#smiles = for_CAN_smiles[for_CAN_smiles.CID == cid]['CAN_SMILES'].item()
	iMol = Chem.MolFromSmiles(smiles.strip())
	iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) 
	iFeature = np.zeros((maxNumAtoms, 64)) ################## feature 크기 고정 
	iFeatureTmp = []
	for atom in iMol.GetAtoms():
		iFeatureTmp.append( atom_feature(atom) )### atom features only
	iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
	iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
	iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
	ADJ = adj_k(np.asarray(iAdj), k)
	return iFeature, ADJ



def atom_feature(atom):
	ar = np.array(one_of_k_encoding_unk(atom.GetSymbol(),
									  ['C', 'N', 'O', 'S', 'F', 'Cl', 'P', 'F', 'Br', 'I',
									   'Na', 'Fe', 'B', 'Mg', 'Al', 'Si', 'K', 'H', 'Se', 'Ca',
									   'Zn', 'As', 'Mo', 'V', 'Cu', 'Hg', 'Cr', 'Co', 'Bi','Tc',
									   'Sb', 'Gd', 'Li', 'Ag', 'Au', 'Unknown']) +
					one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
					one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
					one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5]) +
					[atom.GetIsAromatic()])    # (36, 8, 5, 5, 9, 1) -> 64 
	return ar



def one_of_k_encoding(x, allowable_set):
	if x not in allowable_set:
		raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
	#print list((map(lambda s: x == s, allowable_set)))
	return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding_unk(x, allowable_set):
	"""Maps inputs not in the allowable set to the last element."""
	if x not in allowable_set:
		x = allowable_set[-1]
	return list(map(lambda s: x == s, allowable_set))

def convertAdj(adj):
	dim = len(adj)
	a = adj.flatten()
	b = np.zeros(dim*dim)
	c = (np.ones(dim*dim)-np.equal(a,b)).astype('float64')
	d = c.reshape((dim, dim))
	return d



def adj_k(adj, k): # 근데 k 가 왜 필요한거지 -> 민지 말에 의하면 
	ret = adj
	for i in range(0, k-1):
		ret = np.dot(ret, adj)  
	return convertAdj(ret)



gdsc_c_path = '/st06/jiyeonH/11.TOX/DR_SPRING/val_data/'
SAVE_PATH = gdsc_c_path 

PRJ_NAME = 'GDSC_BREAST' # save the original ver 

pre_made_df1 = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
pre_1_MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
pre_1_MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
pre_1_MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
pre_1_MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
pre_1_MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
pre_1_MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))

PRJ_NAME = 'GDSC_COLON' # save the original ver 

pre_made_df2 = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
pre_2_MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
pre_2_MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
pre_2_MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
pre_2_MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
pre_2_MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
pre_2_MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))

PRJ_NAME = 'GDSC_PANCRE' # save the original ver 

pre_made_df3 = pd.read_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
pre_3_MY_chem_A_feat = torch.load(SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
pre_3_MY_chem_B_feat = torch.load(SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
pre_3_MY_chem_A_adj = torch.load(SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
pre_3_MY_chem_B_adj = torch.load(SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
pre_3_MY_g_EXP_A = torch.load(SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
pre_3_MY_g_EXP_B = torch.load(SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))


pre_made = pd.concat([pre_made_df1, pre_made_df2, pre_made_df3])
pre_MY_chem_A_feat = torch.concat((pre_1_MY_chem_A_feat, pre_2_MY_chem_A_feat, pre_3_MY_chem_A_feat))
pre_MY_chem_B_feat = torch.concat((pre_1_MY_chem_B_feat, pre_2_MY_chem_B_feat, pre_3_MY_chem_B_feat))
pre_MY_chem_A_adj = torch.concat((pre_1_MY_chem_A_adj, pre_2_MY_chem_A_adj, pre_3_MY_chem_A_adj))
pre_MY_chem_B_adj = torch.concat((pre_1_MY_chem_B_adj, pre_2_MY_chem_B_adj, pre_3_MY_chem_B_adj))
pre_MY_g_EXP_A = torch.concat((pre_1_MY_g_EXP_A, pre_2_MY_g_EXP_A, pre_3_MY_g_EXP_A))
pre_MY_g_EXP_B = torch.concat((pre_1_MY_g_EXP_B, pre_2_MY_g_EXP_B, pre_3_MY_g_EXP_B))



gdsc_all_in = gdsc_all7[gdsc_all7.cid_cid_cell.isin(pre_made.cid_cid_cell)]
gdsc_all_out = gdsc_all7[-gdsc_all7.cid_cid_cell.isin(pre_made.cid_cid_cell)]

rms = [148124.0, 10384072.0, 5311497.0, 24978538.0, 49846579.0, 36314.0]
gdsc_all_50 = gdsc_all_out[-gdsc_all_out['Anchor CID'].isin(rms)]
gdsc_all_50 = gdsc_all_50[-gdsc_all_50['Library CID'].isin(rms)]

mjs = list(MJ_gdsc_re.columns)
gdsc_all_mj = gdsc_all_50[gdsc_all_50.CID_A_CELL.isin(mjs)]
gdsc_all_mj = gdsc_all_mj[gdsc_all_mj.CID_B_CELL.isin(mjs)]


gdsc_all_out_filt = gdsc_all_mj[['Anchor CID', 'Library CID', 'cid_cid_cell', 'CID_A_CELL', 'CID_B_CELL', 'ccle_name', 'type']].drop_duplicates()
gdsc_all_out_filt = gdsc_all_out_filt.reset_index(drop = True)



max_len = 50

MY_chem_A_feat = torch.empty(size=(gdsc_all_out_filt.shape[0], max_len, 64))
MY_chem_B_feat= torch.empty(size=(gdsc_all_out_filt.shape[0], max_len, 64))
MY_chem_A_adj = torch.empty(size=(gdsc_all_out_filt.shape[0], max_len, max_len))
MY_chem_B_adj= torch.empty(size=(gdsc_all_out_filt.shape[0], max_len, max_len))

MY_g_EXP_A = torch.empty(size=(gdsc_all_out_filt.shape[0], 349, 1))##############
MY_g_EXP_B = torch.empty(size=(gdsc_all_out_filt.shape[0], 349, 1))##############


Fail_ind = []
from datetime import datetime

for IND in range(0, gdsc_all_out_filt.shape[0]): #  100
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(gdsc_all_out_filt.shape[0]) )
		Fail_ind
		datetime.now()
	#
	cid_cid_cell = gdsc_all_out_filt.cid_cid_cell[IND]
	DrugA_CID = gdsc_all_out_filt['Anchor CID'][IND]
	DrugB_CID = gdsc_all_out_filt['Library CID'][IND]
	CELL = gdsc_all_out_filt['ccle_name'][IND]
	dat_type = gdsc_all_out_filt.type[IND]
	DrugA_CID_CELL = gdsc_all_out_filt.CID_A_CELL[IND]
	DrugB_CID_CELL = gdsc_all_out_filt.CID_B_CELL[IND]
	#
	k=1
	DrugA_Feat, DrugA_ADJ = check_drug_f_ts(DrugA_CID)
	DrugB_Feat, DrugB_ADJ = check_drug_f_ts(DrugB_CID)
	# 
	if dat_type == 'AOBO' :
		mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
		EXP_A = BETA_BIND_MEAN[mean_ind_A]
		mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
		EXP_B = BETA_BIND_MEAN[mean_ind_B]
	#
	else :
		DrugA_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL]
		DrugB_check = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL]
	#
		if len(DrugA_check) == 0 :
			EXP_A, OX = get_MJ_data(DrugA_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_A = torch.Tensor(EXP_A).unsqueeze(1)
		else : 
			mean_ind_A = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugA_CID_CELL].index.item()
			EXP_A = BETA_BIND_MEAN[mean_ind_A]
	#
		if len(DrugB_check) == 0 :
			EXP_B, OX = get_MJ_data(DrugB_CID_CELL)
			if 'X' in OX :
				Fail_ind.append(IND)
			EXP_B = torch.Tensor(EXP_B).unsqueeze(1)
		else : 
			mean_ind_B = BETA_BIND_M_SIG_df_CID[BETA_BIND_M_SIG_df_CID.CID__CELL == DrugB_CID_CELL].index.item()
			EXP_B = BETA_BIND_MEAN[mean_ind_B]
	# 
	MY_chem_A_feat[IND] = torch.Tensor(DrugA_Feat)
	MY_chem_B_feat[IND] = torch.Tensor(DrugB_Feat)
	MY_chem_A_adj[IND] = torch.Tensor(DrugA_ADJ)
	MY_chem_B_adj[IND] = torch.Tensor(DrugB_ADJ)
	MY_g_EXP_A[IND] = torch.Tensor(EXP_A)
	MY_g_EXP_B[IND] = torch.Tensor(EXP_B)



PRJ_NAME = 'GDSC_out' # save the original ver 

SAVE_PATH = gdsc_c_path

torch.save(MY_chem_A_feat, SAVE_PATH+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_feat, SAVE_PATH+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
torch.save(MY_chem_A_adj, SAVE_PATH+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
torch.save(MY_chem_B_adj, SAVE_PATH+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_A, SAVE_PATH+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
torch.save(MY_g_EXP_B, SAVE_PATH+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))
torch.save(MY_syn, SAVE_PATH+'{}.MY_syn.pt'.format(PRJ_NAME))

gdsc_all_out_filt.to_csv(SAVE_PATH+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))

gdsc_all_out_filt[['cid_cid_cell','type']].drop_duplicates() #  



gdsc_c_path = '/st06/jiyeonH/11.TOX/DR_SPRING/val_data/'

1) OUT

PRJ_NAME = 'GDSC_out'
g_out = pd.read_csv(gdsc_c_path+'{}.A_B_C_S_SET.csv'.format(PRJ_NAME))
g_out[['cid_cid_cell','type']].drop_duplicates() # 32522 

MY_chem_A_feat = torch.load(gdsc_c_path+'{}.MY_chem_A_feat.pt'.format(PRJ_NAME))
MY_chem_B_feat = torch.load(  gdsc_c_path+'{}.MY_chem_B_feat.pt'.format(PRJ_NAME))
MY_chem_A_adj = torch.load( gdsc_c_path+'{}.MY_chem_A_adj.pt'.format(PRJ_NAME))
MY_chem_B_adj = torch.load( gdsc_c_path+'{}.MY_chem_B_adj.pt'.format(PRJ_NAME))
MY_g_EXP_A = torch.load( gdsc_c_path+'{}.MY_g_EXP_A.pt'.format(PRJ_NAME))
MY_g_EXP_B = torch.load( gdsc_c_path+'{}.MY_g_EXP_B.pt'.format(PRJ_NAME))

g_out_tuple = [(g_out['Anchor CID'][a], g_out['Library CID'][a], g_out['ccle_name'][a]) for a in range(g_out.shape[0])]



# LINCS 값을 우선시 하는 버전 (마치 MISS 2)
def check_exp_f_ts(A, CID, CELLO) :
	if A == 'A' : 
		indexx = g_out[ (g_out['Anchor CID'] == CID) & (g_out['ccle_name'] == CELLO)].index[0]
		EXP_vector = MY_g_EXP_A[indexx]
	else :
		indexx = g_out[ (g_out['Library CID'] == CID) & (g_out['ccle_name'] == CELLO)].index[0]
		EXP_vector = MY_g_EXP_B[indexx]
	#
	# TARGET 
	TG_vector = get_targets(CID)
	#
	# BASAL EXP 
	B_vector = get_ccle_data(CELLO)
	#
	#
	FEAT = torch.Tensor(np.array([ EXP_vector.squeeze().tolist() , TG_vector, B_vector]).T)
	return FEAT.view(-1,3)



chem_A_pre = []
for i in range(MY_chem_A_adj.shape[0]):
	adj_pre = MY_chem_A_adj[i]
	adj_proc = adj_pre.long().to_sparse().indices()
	chem_A_pre.append(adj_proc)

chem_B_pre = []
for i in range(MY_chem_B_adj.shape[0]):
	adj_pre = MY_chem_B_adj[i]
	adj_proc = adj_pre.long().to_sparse().indices()
	chem_B_pre.append(adj_proc)


dataset = GDSC_Dataset(g_out_tuple)

dataloader = torch.utils.data.DataLoader(dataset, batch_size = 64 , collate_fn = graph_collate_fn, shuffle = False , num_workers = 8)# , num_workers=my_config['config/n_workers'].item()


CELL_PRED_DF = pd.DataFrame(columns = ['PRED','ROW_CID','COL_CID','CCLE'])
CELL_PRED_DF.to_csv(gdsc_c_path+'PRED_{}.FINAL_ing2.csv'.format(PRJ_NAME), index=False)

with torch.no_grad():
	for batch_idx_t, (tup_list, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(dataloader):
		print("{} / {}".format(batch_idx_t, len(dataloader)) , flush = True)
		print(datetime.now(), flush = True)
		list_ROW_CID = [a[0] for a in tup_list]
		list_COL_CID = [a[1] for a in tup_list]
		list_CELLO = [a[2] for a in tup_list]
		#
		output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y) 
		outputs = output.squeeze().tolist() # [output.squeeze().item()]
		#
		tmp_df = pd.DataFrame({
		'PRED': outputs,
		'ROW_CID' : list_ROW_CID,
		'COL_CID' : list_COL_CID,
		'CCLE' : list_CELLO,
		})
		CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])
		tmp_df.to_csv(gdsc_c_path+'PRED_{}.FINAL_ing2.csv'.format(PRJ_NAME), mode='a', index=False, header = False)
		


g_out['PRED'] = list(CELL_PRED_DF['PRED'])
g_out.to_csv(gdsc_c_path+'FINAL_{}.csv'.format(PRJ_NAME))



##########################
그래서 일단 다 만든 무엇. 

gdsc_all7 # 이게 전체 

gdsc_c_path = '/st06/jiyeonH/11.TOX/DR_SPRING/val_data/'
SAVE_PATH = gdsc_c_path 

g_breast8 = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format('GDSC_BREAST'))
g_colon8 = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format('GDSC_COLON'))
g_pancreas8 = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format('GDSC_PANCRE'))
g_out = pd.read_csv(gdsc_c_path+'FINAL_{}.csv'.format('GDSC_out'))

pred_all = pd.concat([g_breast8, g_colon8, g_pancreas8, g_out]) # 

pred_all_re = pred_all[['cid_cid_cell','PRED','type']].drop_duplicates()
pred_all_re2 = pd.DataFrame(pred_all_re.groupby('cid_cid_cell').mean())
pred_all_re2['cid_cid_cell'] = list(pred_all_re2.index)
pred_all_re2.index.name = ''

gdsc_all7_with_p = pd.merge(gdsc_all7, pred_all_re2, on = 'cid_cid_cell', how = 'left')

ac = list(gdsc_all7_with_p['Anchor Name'])
lb = list(gdsc_all7_with_p['Library Name'])
cc = list(gdsc_all7_with_p['Cell Line name'])

gdsc_all7_with_p['A_C_C'] = [(ac[i],lb[i],cc[i]) for i in range(gdsc_all7_with_p.shape[0])]

gdsc_all7_with_p_f = gdsc_all7_with_p[['A_C_C','cid_cid_cell','PRED']].drop_duplicates()

NS_raw = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/Original_screen_All_tissues_fitted_data.csv', low_memory=False)
NS_val = pd.read_csv('/st06/jiyeonH/13.DD_SESS/GDSC^2/Validation_screen_All_tissues_fitted_data.csv', low_memory=False)

ac = list(NS_raw['ANCHOR_NAME'])
lb = list(NS_raw['LIBRARY_NAME'])
cc = list(NS_raw['CELL_LINE_NAME'])

NS_raw['A_C_C'] = [(ac[i],lb[i],cc[i]) for i in range(NS_raw.shape[0])]

NS_raw2 = pd.merge(NS_raw, gdsc_all7_with_p_f, on = 'A_C_C', how = 'left')

NS_raw3 = NS_raw2[NS_raw2.PRED.apply(lambda x : np.isnan(x) == False)]
NS_raw3['ANCHOR_CONC'] = [float(a) for a in NS_raw3.ANCHOR_CONC]


NS_raw3_breast = NS_raw3[NS_raw3.Tissue == 'Breast']
NS_raw3_colon = NS_raw3[NS_raw3.Tissue == 'Colon']
NS_raw3_pancreas = NS_raw3[NS_raw3.Tissue == 'Pancreas']



# 이쁘게 4개쌍 있는거만 써보기 


NS_raw3_breast2 = NS_raw3_breast.sort_values('cid_cid_cell')

count_acc = NS_raw3_breast2.groupby('A_C_C').count()
count_acc_2 = list(count_acc[count_acc.BARCODE==2].index) # 35662
count_acc_ov2 = list(count_acc[count_acc.BARCODE>2].index) # 4700
count_acc_lo2 = list(count_acc[count_acc.BARCODE<2].index) # 220



NS_raw3_breast2['combi_cell'] = NS_raw3_breast2.COMBI_ID + '__' + NS_raw3_breast2.CELL_LINE_NAME
NS_raw3_breast2_combcell_id = list(set(NS_raw3_breast2.combi_cell)) # 40582 


combi = '1022:1032__AU565'

breast_mean = pd.DataFrame(columns = ['combcell','LowHigh','DELTA_EMAX', 'DELTA_XMID', 'num'] )

missing = []

for combi in NS_raw3_breast2_combcell_id : 
    if NS_raw3_breast2_combcell_id.index(combi) % 100 == 0 :
        print(NS_raw3_breast2_combcell_id.index(combi))
        print(datetime.now(), flush = True)
    combi_df = NS_raw3_breast2[NS_raw3_breast2.combi_cell == combi]
    anchor_conc = list(set(combi_df.ANCHOR_CONC ))
    if len(anchor_conc) == 2 : 
        low = np.min(anchor_conc)
        high = np.max(anchor_conc)
        low_emax = np.mean(combi_df[combi_df.ANCHOR_CONC == low]['SYNERGY_DELTA_EMAX'])
        low_xmid = np.mean(combi_df[combi_df.ANCHOR_CONC == low]['SYNERGY_DELTA_XMID'])
        high_emax = np.mean(combi_df[combi_df.ANCHOR_CONC == high]['SYNERGY_DELTA_EMAX'])
        high_xmid = np.mean(combi_df[combi_df.ANCHOR_CONC == high]['SYNERGY_DELTA_XMID'])
        mini_df = pd.DataFrame({
            'LowHigh' : ['low', 'high'], 'combcell' : combi, 
            'DELTA_EMAX' : [low_emax, high_emax], 'DELTA_XMID' : [low_xmid, high_xmid], 'num' : 2 })
        #
        breast_mean = pd.concat([breast_mean, mini_df])
    elif len(anchor_conc) == 1 :
        emax = np.mean(combi_df['SYNERGY_DELTA_EMAX'])
        xmid = np.mean(combi_df['SYNERGY_DELTA_XMID'])
        mini_df = pd.DataFrame({
            'LowHigh' : ['one'], 'combcell' : combi, 
            'DELTA_EMAX' : [emax], 'DELTA_XMID' : [xmid], 'num' : 1 })
    else : 
        missing.append(combi)
        print(combi)

breast_mean.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/val_data/LH_breast.csv')
        

# 일단 이거 잘 되는지 확인하고... 
mean 따라서 plot 그려봐야함.. 색깔도 low one high 따라서 그려보기 
논문 낼수 있을까 하... 이렇게 시간이 걸린다니 진짜 빡치구여 
서버...아악 
집에 가서 혹시 GPU 로 pred 돌리면 더 잘 될지 확인해보기 



one more check 
gdsc_all7_with_p


gdsc_all7_with_p_ACC = list(set(gdsc_all7_with_p.A_C_C)) # 40582 

all_mean2 = pd.DataFrame(columns = ['combcell','LowHigh','DELTA_EMAX', 'DELTA_XMID', 'num'] )

missing = []

for combi in gdsc_all7_with_p_ACC : 
    if gdsc_all7_with_p_ACC.index(combi) % 100 == 0 :
        print(gdsc_all7_with_p_ACC.index(combi))
        print(datetime.now(), flush = True)
    combi_df = gdsc_all7_with_p[gdsc_all7_with_p.A_C_C == combi]
    anchor_conc = list(set(combi_df['Anchor Conc'] ))
    if len(anchor_conc) == 2 : 
        low = np.min(anchor_conc)
        high = np.max(anchor_conc)
        low_emax = np.mean(combi_df[combi_df['Anchor Conc'] == low]['Delta Emax'])
        low_xmid = np.mean(combi_df[combi_df['Anchor Conc'] == low]['Delta Xmid'])
        high_emax = np.mean(combi_df[combi_df['Anchor Conc'] == high]['Delta Emax'])
        high_xmid = np.mean(combi_df[combi_df['Anchor Conc'] == high]['Delta Xmid'])
        mini_df = pd.DataFrame({
            'LowHigh' : ['low', 'high'], 'combcell' : [combi]*2, 
            'DELTA_EMAX' : [low_emax, high_emax], 'DELTA_XMID' : [low_xmid, high_xmid], 'num' : 2 })
        #
        all_mean2 = pd.concat([all_mean2, mini_df])
    elif len(anchor_conc) == 1 :
        emax = np.mean(combi_df['Delta Emax'])
        xmid = np.mean(combi_df['Delta Xmid'])
        mini_df = pd.DataFrame({
            'LowHigh' : ['one'], 'combcell' : [combi]*1, 
            'DELTA_EMAX' : [emax], 'DELTA_XMID' : [xmid], 'num' : 1 })
        all_mean2 = pd.concat([all_mean2, mini_df])
    else : 
        missing.append(combi)
        print(combi)

all_mean2.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/val_data/LH_all.csv')
        


jyjy = gdsc_all7_with_p[['Cell Line name', 'Tissue', 'Anchor Name',
       'Anchor Target', 'Anchor Pathway', 'Library Name',
       'library Target', ' Library Pathway', 'Synergy?', 
       'Anchor CID', 'Library CID', 'strip_name',
       'ccle_name', 'cid_cid_cell', 'CID_A_CELL', 'CID_B_CELL', 'cid_cid',
       'long_id_A', 'long_id_B', 'type', 'PRED']].drop_duplicates()

ac = list(jyjy['Anchor Name'])
lb = list(jyjy['Library Name'])
cc = list(jyjy['Cell Line name'])

jyjy['A_C_C'] = [(ac[i],lb[i],cc[i]) for i in range(jyjy.shape[0])]

all_mean2_p = pd.merge(all_mean2, jyjy, left_on ='combcell', right_on ='A_C_C')






NS_raw3_breast2[(NS_raw3_breast2.COMBI_ID=='1022:1032') & (NS_raw3_breast2.CELL_LINE_NAME=='AU565')]

ovov = NS_raw3_breast2[NS_raw3_breast2.A_C_C.isin(count_acc_ov2)]

tmp = NS_raw3_breast2[(NS_raw3_breast2.COMBI_ID=='1022:1032') & (NS_raw3_breast2.CELL_LINE_NAME=='AU565')]

SYNERGY_DELTA_EMAX
SYNERGY_DELTA_XMID

for combi in count_acc_ov2 : 
    combi_df = NS_raw3_breast2[NS_raw3_breast2.A_C_C == combi]
    anchor_conc = list(set(combi_df.ANCHOR_CONC ))
    if len(anchor_conc) != 2 : 
        print(combi)
        print(len(anchor_conc))







tmp_breast = NS_raw3_breast2[NS_raw3_breast2.A_C_C.isin(count_acc_2)]

f, ax = plt.subplots()

sns.scatterplot(ax = ax, data=tmp_breast, x="PRED", y="SYNERGY_DELTA_EMAX", hue="CELL_LINE_NAME", size = 'ANCHOR_CONC', alpha = 0.5)


plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/BREAST.EMAX.pdf", bbox_inches='tight')
plt.close()












all_mean2_p_f = all_mean2_p[all_mean2_p.PRED.apply(lambda x : np.isnan(x)==False)]


import scipy.stats as stats

def draw_gdsc(Cell_line, Tissue) :
    tmp_df = all_mean2_p_f[all_mean2_p_f.ccle_name == Cell_line]
    #
    fig, ax = plt.subplots(6 , 2 ,figsize=(12, 20))
    ax0 = ax[0][0]
    ax1 = ax[0][1]
    ax2 = ax[1][0]
    ax3 = ax[1][1]
    ax4 = ax[2][0]
    ax5 = ax[2][1]
    ax6 = ax[3][0]
    ax7 = ax[3][1]
    ax8 = ax[4][0]
    ax9 = ax[4][1]
    ax10 = ax[5][0]
    ax11 = ax[5][1]
    #    
    color_dict = dict({'low':'blue',
                    'high':'orange'})
    #
    g0 = sns.scatterplot(ax = ax0, data=tmp_df, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax0.legend().set_visible(False)
    pr,pp = stats.pearsonr(tmp_df['PRED'], tmp_df['DELTA_EMAX'])
    ax0.set_title('{} - {} delta Emax, PC:{}'.format( Tissue, Cell_line, np.round(pr,4)) )
    #
    g1 = sns.scatterplot(ax = ax1, data=tmp_df, x="PRED", y="DELTA_XMID", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    pr,pp = stats.pearsonr(tmp_df['PRED'], tmp_df['DELTA_XMID'])
    ax1.set_title ('{} - {} delta IC50, PC:{}'.format( Tissue, Cell_line, np.round(pr,4)))
    #
    ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #
    #
    #
    tmp__AOBO = tmp_df[tmp_df.type=='AOBO']
    g2 = sns.scatterplot(ax = ax2, data=tmp__AOBO, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax2.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__AOBO['PRED'], tmp__AOBO['DELTA_EMAX'])
    except : 
        pr = 0
    ax2.set_title('{} AOBO - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
    #
    g3 = sns.scatterplot(ax = ax3, data=tmp__AOBO, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax3.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__AOBO['PRED'], tmp__AOBO['DELTA_EMAX'])
    except : 
        pr = 0
    ax3.set_title('{} AOBO - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
    #    
    #
    #
    tmp__AOBX = tmp_df[tmp_df.type.isin(['AOBX', 'AXBO'])]
    g4 = sns.scatterplot(ax = ax4, data=tmp__AOBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax4.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__AOBX['PRED'], tmp__AOBX['DELTA_EMAX'])
    except : 
        pr = 0
    ax4.set_title('{} AOBX - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
    #
    g5 = sns.scatterplot(ax = ax5, data=tmp__AOBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax5.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__AOBX['PRED'], tmp__AOBX['DELTA_EMAX'])
    except : 
        pr = 0
    ax5.set_title('{} AOBX - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
    #    
    #
    #
    tmp__AXBX = tmp_df[tmp_df.type=='AXBX']
    g6 = sns.scatterplot(ax = ax6, data=tmp__AXBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax6.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__AXBX['PRED'], tmp__AXBX['DELTA_EMAX'])
    except : 
        pr = 0
    ax6.set_title('{} AXBX - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
    #
    g7 = sns.scatterplot(ax = ax7, data=tmp__AXBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax7.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__AXBX['PRED'], tmp__AXBX['DELTA_EMAX'])
    except : 
        pr = 0
    ax7.set_title('{} AXBX - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
    #    
    #
    #
    tmp__low = tmp_df[tmp_df.LowHigh=='low']
    g8 = sns.scatterplot(ax = ax8, data=tmp__low, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax8.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__low['PRED'], tmp__low['DELTA_EMAX'])
    except : 
        pr = 0
    ax8.set_title('{} LOW - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
    #
    g9 = sns.scatterplot(ax = ax9, data=tmp__low, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax9.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__low['PRED'], tmp__low['DELTA_EMAX'])
    except : 
        pr = 0
    ax9.set_title('{} LOW - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
    #    
    #
    #
    tmp__hgh = tmp_df[tmp_df.LowHigh=='high']
    g10 = sns.scatterplot(ax = ax10, data=tmp__hgh, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax10.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__hgh['PRED'], tmp__hgh['DELTA_EMAX'])
    except : 
        pr = 0
    ax10.set_title('{} HIGH - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
    #
    g11 = sns.scatterplot(ax = ax11, data=tmp__hgh, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
    ax11.legend().set_visible(False)
    try :
        pr,pp = stats.pearsonr(tmp__hgh['PRED'], tmp__hgh['DELTA_EMAX'])
    except : 
        pr = 0
    ax11.set_title('{} HIGH - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
    #    
    #
    #
    plt.tight_layout()
    plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/{}_RE.{}.pdf".format(Tissue,Cell_line), bbox_inches='tight')
    plt.close()


breast_cell_line = ['T47D_BREAST', 'HCC1599_BREAST', 'UACC812_BREAST', 'HCC1428_BREAST', 'BT483_BREAST', 'CAL148_BREAST', 'MDAMB231_BREAST', 'BT20_BREAST', 'HCC1569_BREAST', 'ZR7530_BREAST', 'HCC1419_BREAST', 'EFM192A_BREAST', 'COLO824_BREAST', 'CAMA1_BREAST', 'CAL851_BREAST', 'MDAMB175VII_BREAST', 'HCC1500_BREAST', 'HCC1395_BREAST', 'HCC1806_BREAST', 'UACC893_BREAST', 'HCC1143_BREAST', 'MDAMB436_BREAST', 'HCC1937_BREAST', 'CAL51_BREAST', 'HS578T_BREAST', 'HCC1187_BREAST', 'MDAMB157_BREAST', 'HCC2218_BREAST', 'MFM223_BREAST', 'HCC202_BREAST', 'MDAMB361_BREAST', 'JIMT1_BREAST', 'HCC38_BREAST', 'MDAMB468_BREAST', 'HCC2157_BREAST', 'HDQP1_BREAST', 'MDAMB415_BREAST', 'ZR751_BREAST', 'BT549_BREAST', 'CAL120_BREAST', 'EFM19_BREAST', 'HCC70_BREAST', 'HCC1954_BREAST', 'DU4475_BREAST', 'MDAMB453_BREAST', 'AU565_BREAST', 'MCF7_BREAST', 'BT474_BREAST']

for bbc in breast_cell_line : 
    print(bbc)
    draw_gdsc(bbc, 'BREAST')



pancreas_cell_line = list(set(all_mean2_p_f[all_mean2_p_f.Tissue=='Pancreas']['ccle_name']))

for ccc in pancreas_cell_line : 
    print(ccc)
    draw_gdsc(ccc, 'PANCREAS')



colon_cell_line = list(set(all_mean2_p_f[all_mean2_p_f.Tissue=='Large Intestine']['ccle_name']))

for ccc in colon_cell_line : 
    print(ccc)
    draw_gdsc(ccc, 'COLON')











9월 말 한번 더 시도

import scipy.stats as stats

fig, ax = plt.subplots(1 , 1 ,figsize=(12, 20))

ax = ax[0]

g0 = sns.scatterplot(ax = ax0, data=tmp_df, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)





#    
color_dict = dict({'low':'blue',
                'high':'orange'})
#
ax0.legend().set_visible(False)
pr,pp = stats.pearsonr(tmp_df['PRED'], tmp_df['DELTA_EMAX'])
ax0.set_title('{} - {} delta Emax, PC:{}'.format( Tissue, Cell_line, np.round(pr,4)) )
#
g1 = sns.scatterplot(ax = ax1, data=tmp_df, x="PRED", y="DELTA_XMID", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
pr,pp = stats.pearsonr(tmp_df['PRED'], tmp_df['DELTA_XMID'])
ax1.set_title ('{} - {} delta IC50, PC:{}'.format( Tissue, Cell_line, np.round(pr,4)))
#
ax1.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
#
#
#
tmp__AOBO = tmp_df[tmp_df.type=='AOBO']
g2 = sns.scatterplot(ax = ax2, data=tmp__AOBO, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax2.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__AOBO['PRED'], tmp__AOBO['DELTA_EMAX'])
except : 
    pr = 0
ax2.set_title('{} AOBO - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
#
g3 = sns.scatterplot(ax = ax3, data=tmp__AOBO, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax3.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__AOBO['PRED'], tmp__AOBO['DELTA_EMAX'])
except : 
    pr = 0
ax3.set_title('{} AOBO - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
#    
#
#
tmp__AOBX = tmp_df[tmp_df.type.isin(['AOBX', 'AXBO'])]
g4 = sns.scatterplot(ax = ax4, data=tmp__AOBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax4.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__AOBX['PRED'], tmp__AOBX['DELTA_EMAX'])
except : 
    pr = 0
ax4.set_title('{} AOBX - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
#
g5 = sns.scatterplot(ax = ax5, data=tmp__AOBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax5.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__AOBX['PRED'], tmp__AOBX['DELTA_EMAX'])
except : 
    pr = 0
ax5.set_title('{} AOBX - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
#    
#
#
tmp__AXBX = tmp_df[tmp_df.type=='AXBX']
g6 = sns.scatterplot(ax = ax6, data=tmp__AXBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax6.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__AXBX['PRED'], tmp__AXBX['DELTA_EMAX'])
except : 
    pr = 0
ax6.set_title('{} AXBX - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
#
g7 = sns.scatterplot(ax = ax7, data=tmp__AXBX, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax7.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__AXBX['PRED'], tmp__AXBX['DELTA_EMAX'])
except : 
    pr = 0
ax7.set_title('{} AXBX - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
#    
#
#
tmp__low = tmp_df[tmp_df.LowHigh=='low']
g8 = sns.scatterplot(ax = ax8, data=tmp__low, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax8.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__low['PRED'], tmp__low['DELTA_EMAX'])
except : 
    pr = 0
ax8.set_title('{} LOW - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
#
g9 = sns.scatterplot(ax = ax9, data=tmp__low, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax9.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__low['PRED'], tmp__low['DELTA_EMAX'])
except : 
    pr = 0
ax9.set_title('{} LOW - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
#    
#
#
tmp__hgh = tmp_df[tmp_df.LowHigh=='high']
g10 = sns.scatterplot(ax = ax10, data=tmp__hgh, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax10.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__hgh['PRED'], tmp__hgh['DELTA_EMAX'])
except : 
    pr = 0
ax10.set_title('{} HIGH - {} delta Emax, PC:{}'.format( Tissue,Cell_line, np.round(pr,4)) )
#
g11 = sns.scatterplot(ax = ax11, data=tmp__hgh, x="PRED", y="DELTA_EMAX", hue="LowHigh", size = 'type', alpha = 0.5, palette=color_dict)
ax11.legend().set_visible(False)
try :
    pr,pp = stats.pearsonr(tmp__hgh['PRED'], tmp__hgh['DELTA_EMAX'])
except : 
    pr = 0
ax11.set_title('{} HIGH - {} delta Emax, PC:{}'.format(Tissue, Cell_line, np.round(pr,4)) )
#    
#
#
plt.tight_layout()
plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/{}_RE.{}.pdf".format(Tissue,Cell_line), bbox_inches='tight')
plt.close()


breast_cell_line = ['T47D_BREAST', 'HCC1599_BREAST', 'UACC812_BREAST', 'HCC1428_BREAST', 'BT483_BREAST', 'CAL148_BREAST', 'MDAMB231_BREAST', 'BT20_BREAST', 'HCC1569_BREAST', 'ZR7530_BREAST', 'HCC1419_BREAST', 'EFM192A_BREAST', 'COLO824_BREAST', 'CAMA1_BREAST', 'CAL851_BREAST', 'MDAMB175VII_BREAST', 'HCC1500_BREAST', 'HCC1395_BREAST', 'HCC1806_BREAST', 'UACC893_BREAST', 'HCC1143_BREAST', 'MDAMB436_BREAST', 'HCC1937_BREAST', 'CAL51_BREAST', 'HS578T_BREAST', 'HCC1187_BREAST', 'MDAMB157_BREAST', 'HCC2218_BREAST', 'MFM223_BREAST', 'HCC202_BREAST', 'MDAMB361_BREAST', 'JIMT1_BREAST', 'HCC38_BREAST', 'MDAMB468_BREAST', 'HCC2157_BREAST', 'HDQP1_BREAST', 'MDAMB415_BREAST', 'ZR751_BREAST', 'BT549_BREAST', 'CAL120_BREAST', 'EFM19_BREAST', 'HCC70_BREAST', 'HCC1954_BREAST', 'DU4475_BREAST', 'MDAMB453_BREAST', 'AU565_BREAST', 'MCF7_BREAST', 'BT474_BREAST']

for bbc in breast_cell_line : 
    print(bbc)
    draw_gdsc(bbc, 'BREAST')




































NS_raw3_colon2 = NS_raw3_colon.sort_values('cid_cid_cell')

count_acc = NS_raw3_colon2.groupby('A_C_C').count()
count_acc_2 = list(count_acc[count_acc.BARCODE==2].index) # 16749
count_acc_ov2 = list(count_acc[count_acc.BARCODE>2].index) # 2727
count_acc_lo2 = list(count_acc[count_acc.BARCODE<2].index) # 78


tmp_colon = NS_raw3_colon2[NS_raw3_colon2.A_C_C.isin(count_acc_2)]


import scipy.stats as stats

def draw_gdsc(Cell_line) :
    # Cell_line = 'BT-20'
    tmp = NS_raw3_colon2[NS_raw3_colon2.A_C_C.isin(count_acc_2)]
    tmp = tmp[tmp.CELL_LINE_NAME == Cell_line]
    #
    fig, ax = plt.subplots(1,2 ,figsize=(12, 4))
    ax1 = ax[0]
    ax2 = ax[1]
    #
    g1 = sns.scatterplot(ax = ax1, data=tmp, x="PRED", y="SYNERGY_DELTA_EMAX", hue="cid_cid_cell", size = 'ANCHOR_CONC', alpha = 0.5)
    ax1.legend().set_visible(False)
    pr,pp = stats.pearsonr(tmp['PRED'], tmp['SYNERGY_DELTA_EMAX'])
    ax1.set_title('COLON - {} delta Emax, PC:{}'.format( Cell_line, np.round(pr,4)) )
    #
    g2 = sns.scatterplot(ax = ax2, data=tmp, x="PRED", y="SYNERGY_DELTA_XMID", hue="cid_cid_cell", size = 'ANCHOR_CONC', alpha = 0.5)
    pr,pp = stats.pearsonr(tmp['PRED'], tmp['SYNERGY_DELTA_EMAX'])
    ax2.set_title ('COLON - {} delta IC50, PC:{}'.format( Cell_line, np.round(pr,4)))
    #
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #fig.suptitle('BREAST - {}'.format(Cell_line))
    #fig.subplots_adjust( top = 0.85 )
    plt.tight_layout()
    #plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/COLON.{}.png".format(Cell_line), bbox_inches='tight')
    plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/COLON.{}.pdf".format(Cell_line), bbox_inches='tight')
    plt.close()


colon_cell_line = ['C2BBe1', 'CCK-81', 'CL-11', 'COLO-678', 'CW-2', 'GP5d', 'HCT-116',
       'HT-115', 'HT55', 'KM12', 'LS-1034', 'LS-123', 'LS-180', 'LS-411N',
       'LS-513', 'LoVo', 'MDST8', 'NCI-H508', 'NCI-H716', 'NCI-H747',
       'RCM-1', 'RKO', 'SNU-1040', 'SNU-175', 'SNU-407', 'SNU-81',
       'SNU-C1', 'SNU-C5', 'SW1116', 'SW1417', 'SW1463', 'SW48', 'SW620',
       'SW837', 'SW948', 'T84']

for ccc in colon_cell_line : 
    print(ccc)
    draw_gdsc(ccc)










NS_raw3_pancreas2 = NS_raw3_pancreas.sort_values('cid_cid_cell')

count_acc = NS_raw3_pancreas2.groupby('A_C_C').count()
count_acc_2 = list(count_acc[count_acc.BARCODE==2].index) # 16749
count_acc_ov2 = list(count_acc[count_acc.BARCODE>2].index) # 2727
count_acc_lo2 = list(count_acc[count_acc.BARCODE<2].index) # 78


tmp_colon = NS_raw3_pancreas2[NS_raw3_pancreas2.A_C_C.isin(count_acc_2)]


import scipy.stats as stats

def draw_gdsc(Cell_line) :
    # Cell_line = 'BT-20'
    tmp = NS_raw3_pancreas2[NS_raw3_pancreas2.A_C_C.isin(count_acc_2)]
    tmp = tmp[tmp.CELL_LINE_NAME == Cell_line]
    #
    fig, ax = plt.subplots(1,2 ,figsize=(12, 4))
    ax1 = ax[0]
    ax2 = ax[1]
    #
    g1 = sns.scatterplot(ax = ax1, data=tmp, x="PRED", y="SYNERGY_DELTA_EMAX", hue="cid_cid_cell", size = 'ANCHOR_CONC', alpha = 0.5)
    ax1.legend().set_visible(False)
    pr,pp = stats.pearsonr(tmp['PRED'], tmp['SYNERGY_DELTA_EMAX'])
    ax1.set_title('COLON - {} delta Emax, PC:{}'.format( Cell_line, np.round(pr,4)) )
    #
    g2 = sns.scatterplot(ax = ax2, data=tmp, x="PRED", y="SYNERGY_DELTA_XMID", hue="cid_cid_cell", size = 'ANCHOR_CONC', alpha = 0.5)
    pr,pp = stats.pearsonr(tmp['PRED'], tmp['SYNERGY_DELTA_EMAX'])
    ax2.set_title ('COLON - {} delta IC50, PC:{}'.format( Cell_line, np.round(pr,4)))
    #
    ax2.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    #fig.suptitle('BREAST - {}'.format(Cell_line))
    #fig.subplots_adjust( top = 0.85 )
    plt.tight_layout()
    #plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/PANCRE.{}.png".format(Cell_line), bbox_inches='tight')
    plt.savefig("/st06/jiyeonH/11.TOX/DR_SPRING/val_data/PANCRE.{}.pdf".format(Cell_line), bbox_inches='tight')
    plt.close()


pancre_cell_line = ['AsPC-1', 'BxPC-3', 'CAPAN-1', 'CAPAN-2', 'CFPAC-1', 'DAN-G',
       'HPAC', 'Hs-766T', 'HuP-T3', 'HuP-T4', 'KP-2', 'KP-3',
       'MIA-PaCa-2', 'PA-TU-8902', 'PANC-02-03', 'PANC-03-27',
       'PANC-04-03', 'PANC-08-13', 'PANC-10-05', 'PSN1', 'SU8686',
       'SW1990', 'YAPC']

for pcc in pancre_cell_line : 
    print(pcc)
    draw_gdsc(pcc)




























tmptmp = gdsc_all7_with_p[gdsc_all7_with_p['Cell Line name']=='UACC-893']
tmptmp1 = tmptmp[tmptmp['Anchor Name']=='Gemcitabine'] # Gemcitabine
tmptmp11 = tmptmp1[tmptmp1['Library Name']=='AZD7762']



tmptmp = gdsc_all7_with_p[gdsc_all7_with_p['Cell Line name']=='UACC-893']
tmptmp2 = tmptmp[tmptmp['Anchor Name']=='AZD7762'] # Gemcitabine
tmptmp22 = tmptmp2[tmptmp2['Library Name']=='Gemcitabine']


# 일단 다들 low, high 있는지, 반대방향 있는지 확인 
# 
A_L_C = gdsc_all7_with_p[['Anchor Name','Library Name', 'Cell Line name']].drop_duplicates()
A_L_C = A_L_C.reset_index(drop = True)


multis = []

for i in range(A_L_C.shape[0]) :
    tt = gdsc_all7_with_p[  (gdsc_all7_with_p['Anchor Name'] == A_L_C.at[i,'Anchor Name']) & (gdsc_all7_with_p['Library Name'] == A_L_C.at[i,'Library Name']) &  (gdsc_all7_with_p['Cell Line name'] == A_L_C.at[i,'Cell Line name']) ]
    if tt.shape[0] > 2 :
        multis.append(i)
    elif tt.shape[0] <2 :
        print(i)
        A_L_C.loc[i]
  



