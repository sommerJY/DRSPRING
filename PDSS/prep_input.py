
import pandas as pd
import numpy as np
import torch 

import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.Chem.QED import qed

import torch_geometric.nn as pyg_nn
from torch.utils.data import Dataset
import networkx as nx
import random

import copy

# read pre - required files 

def read_all_data() :
	MY_G = nx.read_gpickle("./data/myGraph.gpickle")
	LINCS_DATA = pd.read_csv('./data/10_24_sig_cell_mean.0620.csv')
	LINCS_EXP = torch.load('./data/10_24_sig_cell_mean.0620.pt')
	TARGET_DATA = pd.read_csv('./data/TARGET_CID_ENTREZ.csv', '\t')
	BASAL_DATA = pd.read_csv('./data/CCLE_BASAL.csv')
	return MY_G, LINCS_DATA, LINCS_EXP, TARGET_DATA, BASAL_DATA



MY_G, LINCS_DATA, LINCS_EXP, TARGET_DATA, BASAL_DATA = read_all_data()
ENTREZ_ORDER = [1676, 1677, 1738, 501, 5019, 622, 3157, 39, 1019, 1021, 51495, 79071, 230, 5211, 1022, 902, 891, 983, 55256, 58478, 5925, 3098, 22934, 51071, 64080, 226, 5255, 5257, 5261, 3988, 4864, 2184, 2954, 51382, 533, 9133, 30, 6342, 1633, 4860, 1111, 11200, 4088, 7048, 1026, 1647, 8312, 8321, 1950, 1956, 7043, 1029, 10298, 56924, 3628, 8821, 178, 2548, 836, 840, 4067, 695, 581, 4331, 5836, 2131, 9653, 1017, 16, 2058, 1027, 5898, 5900, 595, 1978, 6009, 5223, 6597, 6599, 5747, 5829, 22926, 7494, 3251, 7157, 2673, 51005, 4850, 9924, 2778, 5290, 2353, 3725, 2222, 50814, 1846, 1848, 3108, 3122, 6772, 6774, 54205, 4616, 6850, 6117, 6118, 6119, 6500, 7027, 2770, 2771, 10007, 3930, 3978, 57804, 835, 843, 3033, 211, 5498, 1643, 5366, 637, 5287, 1870, 2956, 355, 5982, 5985, 6194, 3280, 4851, 5106, 6499, 80349, 427, 64781, 8837, 5427, 5111, 1398, 1399, 5583, 5588, 3551, 5566, 2597, 3303, 3312, 4792, 4200, 2810, 6777, 6616, 6804, 8900, 54512, 1605, 6443, 6812, 890, 148022, 9641, 207, 993, 5058, 5096, 8324, 4482, 5110, 1385, 998, 1212, 1213, 1891, 55825, 6251, 7016, 55748, 2690, 22908, 5289, 5373, 7264, 5525, 5529, 10525, 9695, 4780, 9817, 79094, 9134, 3553, 4893, 7099, 10775, 60528, 2064, 3315, 7867, 3162, 644, 5603, 5331, 5236, 572, 8503, 2042, 2048, 4891, 5899, 4846, 51070, 8884, 1282, 6696, 10797, 2356, 4690, 8440, 5720, 5721, 2065, 26520, 29928, 10245, 10559, 23443, 51024, 5827, 142, 9261, 3329, 4605, 55012, 332, 6390, 23530, 25805, 5889, 670, 874, 3454, 672, 291, 5580, 5770, 5792, 665, 1845, 10434, 6697, 5347, 958, 10051, 9918, 4638, 6709, 1445, 5788, 7020, 1277, 10398, 6810, 2582, 4791, 5971, 6908, 9519, 8349, 85236, 1635, 2745, 4609, 2946, 4775, 4776, 51021, 9801, 51116, 6182, 1981, 873, 7159, 23368, 5601, 1759, 4793, 2037, 9455, 124583, 5440, 9533, 387, 23463, 7852, 10221, 8061, 3398, 7416, 10682, 30849, 8678, 23659, 5321, 8720, 10165, 466, 4836, 823, 1123, 392, 8869, 1786, 1788, 596, 11157, 23658, 10206, 4313, 7077, 79947, 10329, 11041, 329, 5708, 5710, 128, 55620, 200081, 10174, 9517, 3611, 3028, 3909, 3480, 4208, 5607, 8573, 11284, 5423, 102, 351, 5092, 10695, 5641, 10557, 2958, 2961]


#  0 ) graph generation 


git_data = '/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/'

def define_graph ():
	MY_G = nx.read_gpickle("./data/myGraph.gpickle")
	G_ADJ = nx.adjacency_matrix(MY_G)
	G_ADJ_tmp = torch.LongTensor(G_ADJ.toarray())
	G_ADJ_IDX = G_ADJ_tmp.to_sparse().indices()
	G_IDX_WEIGHT = torch.Tensor([1] * G_ADJ_IDX.shape[1]).view(1,-1)
	ENTREZ_ORDER = [1676, 1677, 1738, 501, 5019, 622, 3157, 39, 1019, 1021, 51495, 79071, 230, 5211, 1022, 902, 891, 983, 55256, 58478, 5925, 3098, 22934, 51071, 64080, 226, 5255, 5257, 5261, 3988, 4864, 2184, 2954, 51382, 533, 9133, 30, 6342, 1633, 4860, 1111, 11200, 4088, 7048, 1026, 1647, 8312, 8321, 1950, 1956, 7043, 1029, 10298, 56924, 3628, 8821, 178, 2548, 836, 840, 4067, 695, 581, 4331, 5836, 2131, 9653, 1017, 16, 2058, 1027, 5898, 5900, 595, 1978, 6009, 5223, 6597, 6599, 5747, 5829, 22926, 7494, 3251, 7157, 2673, 51005, 4850, 9924, 2778, 5290, 2353, 3725, 2222, 50814, 1846, 1848, 3108, 3122, 6772, 6774, 54205, 4616, 6850, 6117, 6118, 6119, 6500, 7027, 2770, 2771, 10007, 3930, 3978, 57804, 835, 843, 3033, 211, 5498, 1643, 5366, 637, 5287, 1870, 2956, 355, 5982, 5985, 6194, 3280, 4851, 5106, 6499, 80349, 427, 64781, 8837, 5427, 5111, 1398, 1399, 5583, 5588, 3551, 5566, 2597, 3303, 3312, 4792, 4200, 2810, 6777, 6616, 6804, 8900, 54512, 1605, 6443, 6812, 890, 148022, 9641, 207, 993, 5058, 5096, 8324, 4482, 5110, 1385, 998, 1212, 1213, 1891, 55825, 6251, 7016, 55748, 2690, 22908, 5289, 5373, 7264, 5525, 5529, 10525, 9695, 4780, 9817, 79094, 9134, 3553, 4893, 7099, 10775, 60528, 2064, 3315, 7867, 3162, 644, 5603, 5331, 5236, 572, 8503, 2042, 2048, 4891, 5899, 4846, 51070, 8884, 1282, 6696, 10797, 2356, 4690, 8440, 5720, 5721, 2065, 26520, 29928, 10245, 10559, 23443, 51024, 5827, 142, 9261, 3329, 4605, 55012, 332, 6390, 23530, 25805, 5889, 670, 874, 3454, 672, 291, 5580, 5770, 5792, 665, 1845, 10434, 6697, 5347, 958, 10051, 9918, 4638, 6709, 1445, 5788, 7020, 1277, 10398, 6810, 2582, 4791, 5971, 6908, 9519, 8349, 85236, 1635, 2745, 4609, 2946, 4775, 4776, 51021, 9801, 51116, 6182, 1981, 873, 7159, 23368, 5601, 1759, 4793, 2037, 9455, 124583, 5440, 9533, 387, 23463, 7852, 10221, 8061, 3398, 7416, 10682, 30849, 8678, 23659, 5321, 8720, 10165, 466, 4836, 823, 1123, 392, 8869, 1786, 1788, 596, 11157, 23658, 10206, 4313, 7077, 79947, 10329, 11041, 329, 5708, 5710, 128, 55620, 200081, 10174, 9517, 3611, 3028, 3909, 3480, 4208, 5607, 8573, 11284, 5423, 102, 351, 5092, 10695, 5641, 10557, 2958, 2961]
	return MY_G, G_ADJ_IDX, G_IDX_WEIGHT, ENTREZ_ORDER





#####################################################################
# 1 ) drug feature 

def check_length(SMILES) :
	maxNumAtoms = 50
	iMol = Chem.MolFromSmiles(SMILES.strip())
	NUM = iMol.GetNumAtoms()
	if NUM > maxNumAtoms :
		return 0
	else : 
		return 1


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
					[atom.GetIsAromatic()])    
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

def adj_k(adj, k): 
	ret = adj
	for i in range(0, k-1):
		ret = np.dot(ret, adj)  
	return convertAdj(ret)


def make_rdkit_var(SMILES) :
	maxNumAtoms = 50
	try : 
		if check_length(SMILES) == 1 :
			iMol = Chem.MolFromSmiles(SMILES.strip())
			iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol) 
			iFeature = np.zeros((maxNumAtoms, 64)) ################## feature 크기 고정 
			iFeatureTmp = []
			for atom in iMol.GetAtoms():
				iFeatureTmp.append( atom_feature(atom) )### atom features only
			iFeature[0:len(iFeatureTmp), 0:64] = iFeatureTmp ### 0 padding for feature-set
			iAdj = np.zeros((maxNumAtoms, maxNumAtoms))
			iAdj[0:len(iFeatureTmp), 0:len(iFeatureTmp)] = iAdjTmp + np.eye(len(iFeatureTmp))
			k = 1
			ADJ = adj_k(np.asarray(iAdj), k)
			ADJ_RE = torch.Tensor(ADJ).long().to_sparse().indices()
			return torch.Tensor(iFeature), ADJ_RE
	except : 
		print('length error')


#####################################################


# 2) get exp data 


def get_EXP_DATA(CID, CELL, M1_file) :
	LINCS_DATA = globals()['LINCS_DATA']
	LINCS_EXP = globals()['LINCS_EXP']
	try : 	
		M1_DATA = pd.read_csv(M1_file)
		ord = [list(M1_DATA.entrez_id).index(a) for a in ENTREZ_ORDER]
		M1_DATA_re = M1_DATA.iloc[ord]
		lincs_tmp = LINCS_DATA[(LINCS_DATA.CID == CID) & (LINCS_DATA.CCLE_Name == CELL)]
		long_id = str(CID) + '__' + str(CELL)
		#
		if lincs_tmp.shape[0] == 1 : # 데이터 있을때의 얘기 
			lincs_index = lincs_tmp.index.item()
			exp_result = LINCS_EXP[lincs_index]
			print('exist in LINCS')
		elif long_id in M1_DATA_re.columns :
			exp_result = torch.Tensor(M1_DATA_re[long_id]).view(349,1)
			print('made with module 1')
		else :
			exp_result = torch.Tensor([0]*349).view(349,1)
			print('no exp data')
		#
		return exp_result
	except :
		print("Please make M1 Data!")


#####################################################


# 3) TARGET data

def get_targets(CID): # 
	TARGET_DATA = globals()['TARGET_DATA']
	target_cids = list(set(TARGET_DATA.CID))
	target_cids.sort()
	gene_ids = ENTREZ_ORDER
	#
	if CID in target_cids:
		tmp_df2 = TARGET_DATA[TARGET_DATA.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		vec = [0] * 349
	return torch.Tensor(vec)


#####################################################



# 4) 이미 있는 CCLE 사용 원할 시 
def get_CCLE(CELL, BASAL_DATA) : 
	BASAL_DATA = globals()['BASAL_DATA']
	ENTREZ_ORDER_str = [str(a) for a in ENTREZ_ORDER]
	if CELL in list(BASAL_DATA.DrugCombCCLE) : 
		ccle_exp_df = BASAL_DATA[BASAL_DATA.DrugCombCCLE==CELL][ENTREZ_ORDER_str]
		ccle_result = torch.Tensor(list(ccle_exp_df.T.iloc[:,0])).view(349,1)
	else : 
		print("no data in our ccle")
		ccle_result = [0]*349
	#
	return ccle_result




# 5) 새 CCLE 값 받는걸로 진행하는 경우
def get_New_CCLE(new_ccle_df, new_ccle_name) : 
	ENTREZ_ORDER_str = [str(a) for a in ENTREZ_ORDER]
	try : 
		new_ccle_df.columns = [str(int(a)) for a in new_ccle_df.columns]
		new_ccle_df = new_ccle_df[ENTREZ_ORDER_str]
		ccle_result = torch.Tensor(list(new_ccle_df.T.iloc[:,0])).view(349,1)
		return ccle_result
	except:
		print('New CCLE data value error')
	






##########

# make dataset 

def check_CID(SMILES) :
	try : 
		pcp_res = pcp.get_compounds(SMILES, namespace = 'smiles')
		print('checking your smiles...')
		if len(pcp_res) == 1:
			comp = pcp_res[0]
			lincs_cid = comp.cid
			if lincs_cid > 0 :
				return lincs_cid
		else : 
			print('multiple cid')
	except : 
		print('Cannot get your CID for input SMILES')
		return 0





def make_simple_input_data(SM_A, SM_B, M1_A, M1_B, CELL, new_ccle_df = None):
	#
	print('Make data for {}'.format(CELL))
	MY_G, G_ADJ_IDX, G_IDX_WEIGHT, _ = define_graph()
	#
	print(SM_A)
	drug1_f, drug1_a = make_rdkit_var(SM_A)
	print(SM_B)
	drug2_f, drug2_a = make_rdkit_var(SM_B)
	#
	CID_A = check_CID(SM_A)
	CID_B = check_CID(SM_B)
	#CID_A = 135403648
	#CID_B = 3385 # 9887053
	#
	print('Drug A EXP')
	expA = get_EXP_DATA(CID_A, CELL, M1_A)
	print('Drug B EXP')
	expB = get_EXP_DATA(CID_B, CELL, M1_B)
	#
	targetA = get_targets(CID_A)
	targetB = get_targets(CID_B)
	#
	BASAL_DATA = globals()['BASAL_DATA']
	if CELL in list(BASAL_DATA.DrugCombCCLE) :
		basal = get_CCLE(CELL, BASAL_DATA)
	else :
		basal = get_New_CCLE(new_ccle_df, CELL)
	#
	FEAT_A = torch.Tensor(np.array([ expA.squeeze().tolist() , targetA.squeeze().tolist(), basal.squeeze().tolist()]).T)
	FEAT_A = FEAT_A.view(-1,3)
	FEAT_B = torch.Tensor(np.array([ expB.squeeze().tolist() , targetB.squeeze().tolist(), basal.squeeze().tolist()]).T)
	FEAT_B = FEAT_B.view(-1,3)
	#
	adj = copy.deepcopy(G_ADJ_IDX).long()
	adj_w = torch.Tensor(G_IDX_WEIGHT).squeeze()
	#
	return drug1_f, drug2_f, drug1_a, drug2_a, FEAT_A, FEAT_B, adj, adj_w



def make_input_by_cell (SM_A, SM_B, M1_A , M1_B, new_ccle = None):
	pre_c_dict = {}
	#
	if new_ccle == None :
		cell_list = ['HCC1500_BREAST', 'G361_SKIN', 'UACC62_SKIN', 'SKOV3_OVARY', 'T47D_BREAST', 'SKMEL30_SKIN', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'KPL1_BREAST', 'MDAMB468_BREAST', 'A498_KIDNEY', 'EKVX_LUNG', 'RPMI7951_SKIN', 'MDAMB175VII_BREAST', 'UO31_KIDNEY', 'MDAMB436_BREAST', 'NCIH522_LUNG', 'OV90_OVARY', 'SKMEL2_SKIN', 'LOXIMVI_SKIN', 'WM115_SKIN', 'NCIH460_LUNG', '786O_KIDNEY', 'NCIH1650_LUNG', 'NIHOVCAR3_OVARY', 'A2780_OVARY', 'UWB1289_OVARY', 'A673_BONE', 'NCIH226_LUNG', 'COLO829_SKIN', 'HCT15_LARGE_INTESTINE', 'BT474_BREAST', 'PA1_OVARY', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'OVCAR5_OVARY', 'SKMES1_LUNG', 'CAOV3_OVARY', 'SW620_LARGE_INTESTINE', 'KM12_LARGE_INTESTINE', 'COLO800_SKIN', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1419_BREAST', 'HOP92_LUNG', 'A101D_SKIN', 'HS578T_BREAST', 'CAMA1_BREAST', 'UACC257_SKIN', 'LOVO_LARGE_INTESTINE', 'A549_LUNG', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'SW837_LARGE_INTESTINE', 'MCF7_BREAST', 'A427_LUNG', 'IGROV1_OVARY', 'NCIH520_LUNG', 'IPC298_SKIN', 'MEWO_SKIN', 'RKO_LARGE_INTESTINE', 'OVCAR4_OVARY', 'ZR751_BREAST', 'OVCAR8_OVARY', 'HCT116_LARGE_INTESTINE', 'COLO792_SKIN', 'MSTO211H_PLEURA', 'CAKI1_KIDNEY', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'VCAP_PROSTATE', 'HT144_SKIN', 'MDAMB231_BREAST', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'ES2_OVARY', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'RVH421_SKIN', 'UACC812_BREAST', 'HT29_LARGE_INTESTINE', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'BT549_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH2122_LUNG', 'A375_SKIN', 'SKMEL28_SKIN', 'MALME3M_SKIN', 'NCIH23_LUNG', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'PC3_PROSTATE', 'SKMEL5_SKIN', 'MDAMB361_BREAST', 'ACHN_KIDNEY', 'MELHO_SKIN', 'DLD1_LARGE_INTESTINE', 'A2058_SKIN', 'HOP62_LUNG']	
		for precell in cell_list :
			pre_c_dict[precell] = make_simple_input_data(SM_A, SM_B, M1_A, M1_B, precell)
			print('\n')
	else :
		new_ccle_df = pd.read_csv(new_ccle, index_col = 0)
		cell_list = list(new_ccle_df.index)
		check_name = [cell for cell in cell_list if type(cell) ==str]
		new_ccle_df = new_ccle_df.loc[check_name]
		for precell in cell_list :
			pre_c_dict[precell] = make_simple_input_data(SM_A, SM_B, M1_A, M1_B, precell, new_ccle_df)
			print('\n')
	#
	return pre_c_dict


	
	






def prepare_data_GCN(A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2 ) : 
	#
	#
	ABCS_train = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SPLIT=='train']
	ABCS_val = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SPLIT=='val']
	ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.SPLIT=='test']
	#
	train_ind = list(ABCS_train.index)
	random.shuffle(train_ind)
	val_ind = list(ABCS_val.index)
	test_ind = list(ABCS_test.index)
	# 
	chem_feat_A_train = MY_chem_A_feat_RE2[train_ind]; chem_feat_A_val = MY_chem_A_feat_RE2[val_ind]; chem_feat_A_test = MY_chem_A_feat_RE2[test_ind]
	chem_feat_B_train = MY_chem_B_feat_RE2[train_ind]; chem_feat_B_val = MY_chem_B_feat_RE2[val_ind]; chem_feat_B_test = MY_chem_B_feat_RE2[test_ind]
	chem_adj_A_train = MY_chem_A_adj_RE2[train_ind]; chem_adj_A_val = MY_chem_A_adj_RE2[val_ind]; chem_adj_A_test = MY_chem_A_adj_RE2[test_ind]
	chem_adj_B_train = MY_chem_B_adj_RE2[train_ind]; chem_adj_B_val = MY_chem_B_adj_RE2[val_ind]; chem_adj_B_test = MY_chem_B_adj_RE2[test_ind]
	gene_A_train = MY_g_EXP_A_RE2[train_ind]; gene_A_val = MY_g_EXP_A_RE2[val_ind]; gene_A_test = MY_g_EXP_A_RE2[test_ind]
	gene_B_train = MY_g_EXP_B_RE2[train_ind]; gene_B_val = MY_g_EXP_B_RE2[val_ind]; gene_B_test = MY_g_EXP_B_RE2[test_ind]
	target_A_train = MY_Target_A2[train_ind]; target_A_val = MY_Target_A2[val_ind]; target_A_test = MY_Target_A2[test_ind]
	target_B_train = MY_Target_B2[train_ind]; target_B_val = MY_Target_B2[val_ind]; target_B_test = MY_Target_B2[test_ind]
	cell_basal_train = MY_CellBase_RE2[train_ind]; cell_basal_val = MY_CellBase_RE2[val_ind]; cell_basal_test = MY_CellBase_RE2[test_ind]
	syn_train = MY_syn_RE2[train_ind]; syn_val = MY_syn_RE2[val_ind]; syn_test = MY_syn_RE2[test_ind]
	#
	train_data = {}
	val_data = {}
	test_data = {}
	#
	train_data['drug1_feat'] = torch.concat([chem_feat_A_train, chem_feat_B_train], axis = 0)
	val_data['drug1_feat'] = chem_feat_A_val
	test_data['drug1_feat'] = chem_feat_A_test
	#
	train_data['drug2_feat'] = torch.concat([chem_feat_B_train, chem_feat_A_train], axis = 0)
	val_data['drug2_feat'] = chem_feat_B_val
	test_data['drug2_feat'] = chem_feat_B_test
	#
	train_data['drug1_adj'] = torch.concat([chem_adj_A_train, chem_adj_B_train], axis = 0)
	val_data['drug1_adj'] = chem_adj_A_val
	test_data['drug1_adj'] = chem_adj_A_test
	#
	train_data['drug2_adj'] = torch.concat([chem_adj_B_train, chem_adj_A_train], axis = 0)
	val_data['drug2_adj'] = chem_adj_B_val
	test_data['drug2_adj'] = chem_adj_B_test
	#
	train_data['GENE_A'] = torch.concat([gene_A_train, gene_B_train], axis = 0)
	val_data['GENE_A'] = gene_A_val
	test_data['GENE_A'] = gene_A_test
	#
	train_data['GENE_B'] = torch.concat([gene_B_train, gene_A_train], axis = 0)
	val_data['GENE_B'] = gene_B_val
	test_data['GENE_B'] = gene_B_test
	#
	train_data['TARGET_A'] = torch.concat([target_A_train, target_B_train], axis = 0)
	val_data['TARGET_A'] = target_A_val
	test_data['TARGET_A'] = target_A_test
	#
	train_data['TARGET_B'] = torch.concat([target_B_train, target_A_train], axis = 0)
	val_data['TARGET_B'] = target_B_val
	test_data['TARGET_B'] = target_B_test
	#   #
	train_data['cell_BASAL'] = torch.concat((cell_basal_train, cell_basal_train), axis=0)
	val_data['cell_BASAL'] = cell_basal_val
	test_data['cell_BASAL'] = cell_basal_test
	##
	train_data['y'] = torch.concat((syn_train, syn_train), axis=0)
	val_data['y'] = syn_val
	test_data['y'] = syn_test
	#
	print('train rows : ')
	print(train_data['drug1_feat'].shape[0], flush=True)
	print('validation rows : ')
	print(val_data['drug1_feat'].shape[0], flush=True)
	print('test rows : ')
	print(test_data['drug1_feat'].shape[0], flush=True)
	return train_data, val_data, test_data





class DATASET_GCN_W_FT(Dataset):
	def __init__(self, gcn_drug1_F, gcn_drug2_F, gcn_drug1_ADJ, gcn_drug2_ADJ, 
	gcn_gene_A, gcn_gene_B, target_A, target_B, cell_basal, gcn_adj, gcn_adj_weight, syn_ans ):
		self.gcn_drug1_F = gcn_drug1_F
		self.gcn_drug2_F = gcn_drug2_F
		self.gcn_drug1_ADJ = gcn_drug1_ADJ
		self.gcn_drug2_ADJ = gcn_drug2_ADJ
		self.gcn_gene_A = gcn_gene_A
		self.gcn_gene_B = gcn_gene_B
		self.target_A = target_A
		self.target_B = target_B
		self.cell_basal = cell_basal
		self.gcn_adj = gcn_adj
		self.gcn_adj_weight = gcn_adj_weight
		self.syn_ans = syn_ans
	#
	def __len__(self):
		return len(self.gcn_drug1_F)
	#
	def __getitem__(self, index):
		adj_re_A = self.gcn_drug1_ADJ[index].long().to_sparse().indices()
		adj_re_B = self.gcn_drug2_ADJ[index].long().to_sparse().indices()
		#
		FEAT_A = torch.Tensor(np.array([ self.gcn_gene_A[index].squeeze().tolist() , self.target_A[index].tolist(), self.cell_basal[index].tolist()]).T)
		FEAT_B = torch.Tensor(np.array([ self.gcn_gene_B[index].squeeze().tolist() , self.target_B[index].tolist(), self.cell_basal[index].tolist()]).T)
		#
		return self.gcn_drug1_F[index], self.gcn_drug2_F[index],adj_re_A, adj_re_B, FEAT_A, FEAT_B, self.gcn_adj, self.gcn_adj_weight , self.syn_ans[index]


def graph_collate_fn(batch):
	drug1_f_list = []
	drug2_f_list = []
	drug1_adj_list = []
	drug2_adj_list = []
	expA_list = []
	expB_list = []
	exp_adj_list = []
	exp_adj_w_list = []
	y_list = []
	EXP_num_nodes_seen = 0
	DRUG_1_num_nodes_seen = 0
	DRUG_2_num_nodes_seen = 0
	#
	for drug1_f, drug2_f, drug1_adj, drug2_adj, expA, expB, exp_adj, exp_adj_w, y in batch :
		drug1_f_list.append(drug1_f)
		drug2_f_list.append(drug2_f)
		drug1_adj_list.append(drug1_adj+DRUG_1_num_nodes_seen)
		drug2_adj_list.append(drug2_adj+DRUG_2_num_nodes_seen)
		expA_list.append(expA)
		expB_list.append(expB)
		exp_adj_list.append(exp_adj+EXP_num_nodes_seen)
		exp_adj_w_list.append(exp_adj_w)
		y_list.append(torch.Tensor(y))
		EXP_num_nodes_seen += expA.shape[0]
		DRUG_1_num_nodes_seen += drug1_f.shape[0]
		DRUG_2_num_nodes_seen += drug2_f.shape[0]
	#
	drug1_f_new = torch.cat(drug1_f_list, 0)
	drug2_f_new = torch.cat(drug2_f_list, 0)
	drug1_adj_new = torch.cat(drug1_adj_list, 1)
	drug2_adj_new = torch.cat(drug2_adj_list, 1)
	expA_new = torch.cat(expA_list, 0)
	expB_new = torch.cat(expB_list, 0)
	exp_adj_new = torch.cat(exp_adj_list, 1)
	exp_adj_w_new = torch.cat(exp_adj_w_list, 1)
	y_new = torch.stack(y_list, 0)
	return drug1_f_new, drug2_f_new, drug1_adj_new, drug2_adj_new, expA_new, expB_new, exp_adj_new, exp_adj_w_new, y_new




def get_loss_weight(T_train) :
	ys = T_train.syn_ans.squeeze().tolist()
	min_s = np.amin(ys)
	loss_weight = np.log(ys - min_s + np.e)
	return list(loss_weight)



# DATA check  
def make_merged_data() :
	MY_G, G_ADJ_IDX, G_IDX_WEIGHT, _ = define_graph()
	#
	A_B_C_S_SET_SM = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/' + 'final_dataset.csv')
	MY_chem_A_feat_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'Chem_A_feat.pt')
	MY_chem_B_feat_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'Chem_B_feat.pt')
	MY_chem_A_adj_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'Chem_A_adj.pt')
	MY_chem_B_adj_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'Chem_B_adj.pt')
	MY_g_EXP_A_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'EXP_A.pt')
	MY_g_EXP_B_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'EXP_B.pt')
	MY_Target_A2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'TARGET_A.pt')
	MY_Target_B2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'TARGET_B.pt')
	MY_CellBase_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'BASAL.pt')
	MY_syn_RE2 = torch.load('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/PDSS/data/'+'SYN.pt')
	#
	train_data, val_data, test_data = prepare_data_GCN(
		A_B_C_S_SET_SM, MY_chem_A_feat_RE2, MY_chem_B_feat_RE2, MY_chem_A_adj_RE2, MY_chem_B_adj_RE2, 
		MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2
	)
	#
	#
	T_train = DATASET_GCN_W_FT(
		torch.Tensor(train_data['drug1_feat']), torch.Tensor(train_data['drug2_feat']), 
		torch.Tensor(train_data['drug1_adj']), torch.Tensor(train_data['drug2_adj']),
		torch.Tensor(train_data['GENE_A']), torch.Tensor(train_data['GENE_B']), 
		torch.Tensor(train_data['TARGET_A']), torch.Tensor(train_data['TARGET_B']), torch.Tensor(train_data['cell_BASAL']), 
		G_ADJ_IDX, G_IDX_WEIGHT, 
		torch.Tensor(train_data['y'])
		)
	#
	T_val = DATASET_GCN_W_FT(
		torch.Tensor(val_data['drug1_feat']), torch.Tensor(val_data['drug2_feat']), 
		torch.Tensor(val_data['drug1_adj']), torch.Tensor(val_data['drug2_adj']),
		torch.Tensor(val_data['GENE_A']), torch.Tensor(val_data['GENE_B']), 
		torch.Tensor(val_data['TARGET_A']), torch.Tensor(val_data['TARGET_B']), torch.Tensor(val_data['cell_BASAL']), 
		G_ADJ_IDX, G_IDX_WEIGHT, 
		torch.Tensor(val_data['y'])
		)
	#	
	T_test = DATASET_GCN_W_FT(
		torch.Tensor(test_data['drug1_feat']), torch.Tensor(test_data['drug2_feat']), 
		torch.Tensor(test_data['drug1_adj']), torch.Tensor(test_data['drug2_adj']),
		torch.Tensor(test_data['GENE_A']), torch.Tensor(test_data['GENE_B']), 
		torch.Tensor(test_data['TARGET_A']), torch.Tensor(test_data['TARGET_B']), torch.Tensor(test_data['cell_BASAL']), 
		G_ADJ_IDX, G_IDX_WEIGHT, 
		torch.Tensor(test_data['y'])
		)
	#
	return T_train, T_val, T_test






