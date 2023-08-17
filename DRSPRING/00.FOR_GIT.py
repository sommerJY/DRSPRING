
어차피 정리해야할 코드인데... 
해보자 그냥 

민지꺼에 있는 값인지 아닌지에 대한 확인 과정이 main 에 필요할듯 





INPUT Chem

CID_A = 135402009
CID_B = 3385
CELL = 'DLD1_LARGE_INTESTINE'

135403648 : C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O
135402009 : C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O

SM_A = 'C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O'
SM_B = 'C1=C(C(=O)NC(=O)N1)F'
CELL = 'DLD1_LARGE_INTESTINE'

# LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
LINCS_ALL_PATH = 
LINCS_DATA = pd.read_csv(LINCS_ALL_PATH + '10_24_sig_cell_mean.0620.csv')
LINCS_EXP = torch.load( LINCS_ALL_PATH + "10_24_sig_cell_mean.0620.pt")

# BASAL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W213_349_MIS2/'
BASAL_PATH = 
BASAL_DATA = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/CCLE_BASAL.csv')

# TARGET_PATH = '/st06/jiyeonH/13.DD_SESS/01.PRJ2/'
TARGET_PATH = 
TARGET_DATA = pd.read_csv(TARGET_PATH+'TARGET_CID_ENTREZ.csv', sep ='\t', index_col = 0)

maxNumAtoms = 50



0) graph generation 
# nx.write_gpickle(JY_GRAPH,'/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/myGraph.gpickle')
# 

G = nx.read_gpickle('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/myGraph.gpickle')

G_ADJ = nx.adjacency_matrix(G)
G_ADJ_tmp = torch.LongTensor(G_ADJ.toarray())
G_ADJ_IDX = G_ADJ_tmp.to_sparse().indices()
G_IDX_WEIGHT = [1] * G_ADJ_IDX.shape[1]





1) drug feature generation 

def check_length(SMILES) :
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


def make_rdkit_var(SMILES) :
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
    




# 2) 그거에 맞게 EXP 가져오기 
안되면 민지꺼에서 다시 만들어서 가져오기 

지금 당장 필요한건 민지꺼에 있음. 
내일 92개 해보기 

근데 일단 LINCS 에 있는지 없는지 확인하고 
민지꺼에서 만들어오는 방향으로 해야할듯 


MJ_COLON = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/fugcn_hhhdt3/PRJ2_EXP_ccle_cellall_fugcn_hhhdt3_tvt_LARGE_INTESTINE.csv')
MJ_COLON_135403648 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cid135403648_fugcn_hhhdt3.csv')
MJ_COLON_135402009 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cid135403648_fugcn_hhhdt3.csv')
MJ_COLON_135402009 = MJ_COLON_135402009.iloc[:,3:]
MJ_COLON_135402009.columns = [a.replace('135403648', '135402009') for a in list(MJ_COLON_135402009.columns)]

all(MJ_COLON.entrez_id == MJ_COLON_135403648.entrez_id)

MJ_COLONS = pd.concat([MJ_COLON, MJ_COLON_135403648.iloc[:,3:], MJ_COLON_135402009], axis =1 )

BETA_ENTREZ_ORDER = [1676, 1677, 1738, 501, 5019, 622, 3157, 39, 1019, 1021, 51495, 79071, 230, 5211, 1022, 902, 891, 983, 55256, 58478, 5925, 3098, 22934, 51071, 64080, 226, 5255, 5257, 5261, 3988, 4864, 2184, 2954, 51382, 533, 9133, 30, 6342, 1633, 4860, 1111, 11200, 4088, 7048, 1026, 1647, 8312, 8321, 1950, 1956, 7043, 1029, 10298, 56924, 3628, 8821, 178, 2548, 836, 840, 4067, 695, 581, 4331, 5836, 2131, 9653, 1017, 16, 2058, 1027, 5898, 5900, 595, 1978, 6009, 5223, 6597, 6599, 5747, 5829, 22926, 7494, 3251, 7157, 2673, 51005, 4850, 9924, 2778, 5290, 2353, 3725, 2222, 50814, 1846, 1848, 3108, 3122, 6772, 6774, 54205, 4616, 6850, 6117, 6118, 6119, 6500, 7027, 2770, 2771, 10007, 3930, 3978, 57804, 835, 843, 3033, 211, 5498, 1643, 5366, 637, 5287, 1870, 2956, 355, 5982, 5985, 6194, 3280, 4851, 5106, 6499, 80349, 427, 64781, 8837, 5427, 5111, 1398, 1399, 5583, 5588, 3551, 5566, 2597, 3303, 3312, 4792, 4200, 2810, 6777, 6616, 6804, 8900, 54512, 1605, 6443, 6812, 890, 148022, 9641, 207, 993, 5058, 5096, 8324, 4482, 5110, 1385, 998, 1212, 1213, 1891, 55825, 6251, 7016, 55748, 2690, 22908, 5289, 5373, 7264, 5525, 5529, 10525, 9695, 4780, 9817, 79094, 9134, 3553, 4893, 7099, 10775, 60528, 2064, 3315, 7867, 3162, 644, 5603, 5331, 5236, 572, 8503, 2042, 2048, 4891, 5899, 4846, 51070, 8884, 1282, 6696, 10797, 2356, 4690, 8440, 5720, 5721, 2065, 26520, 29928, 10245, 10559, 23443, 51024, 5827, 142, 9261, 3329, 4605, 55012, 332, 6390, 23530, 25805, 5889, 670, 874, 3454, 672, 291, 5580, 5770, 5792, 665, 1845, 10434, 6697, 5347, 958, 10051, 9918, 4638, 6709, 1445, 5788, 7020, 1277, 10398, 6810, 2582, 4791, 5971, 6908, 9519, 8349, 85236, 1635, 2745, 4609, 2946, 4775, 4776, 51021, 9801, 51116, 6182, 1981, 873, 7159, 23368, 5601, 1759, 4793, 2037, 9455, 124583, 5440, 9533, 387, 23463, 7852, 10221, 8061, 3398, 7416, 10682, 30849, 8678, 23659, 5321, 8720, 10165, 466, 4836, 823, 1123, 392, 8869, 1786, 1788, 596, 11157, 23658, 10206, 4313, 7077, 79947, 10329, 11041, 329, 5708, 5710, 128, 55620, 200081, 10174, 9517, 3611, 3028, 3909, 3480, 4208, 5607, 8573, 11284, 5423, 102, 351, 5092, 10695, 5641, 10557, 2958, 2961]

ORD = [list(MJ_COLONS.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_COLONS2 = MJ_COLONS.iloc[ORD]


MJ_tissue = os.listdir('/st06/jiyeonH/13.DD_SESS/01.PRJ2/fugcn_hhhdt3')
MJ_COLON_135403648 = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/cid135403648_fugcn_hhhdt3.csv')
BETA_ENTREZ_ORDER = [1676, 1677, 1738, 501, 5019, 622, 3157, 39, 1019, 1021, 51495, 79071, 230, 5211, 1022, 902, 891, 983, 55256, 58478, 5925, 3098, 22934, 51071, 64080, 226, 5255, 5257, 5261, 3988, 4864, 2184, 2954, 51382, 533, 9133, 30, 6342, 1633, 4860, 1111, 11200, 4088, 7048, 1026, 1647, 8312, 8321, 1950, 1956, 7043, 1029, 10298, 56924, 3628, 8821, 178, 2548, 836, 840, 4067, 695, 581, 4331, 5836, 2131, 9653, 1017, 16, 2058, 1027, 5898, 5900, 595, 1978, 6009, 5223, 6597, 6599, 5747, 5829, 22926, 7494, 3251, 7157, 2673, 51005, 4850, 9924, 2778, 5290, 2353, 3725, 2222, 50814, 1846, 1848, 3108, 3122, 6772, 6774, 54205, 4616, 6850, 6117, 6118, 6119, 6500, 7027, 2770, 2771, 10007, 3930, 3978, 57804, 835, 843, 3033, 211, 5498, 1643, 5366, 637, 5287, 1870, 2956, 355, 5982, 5985, 6194, 3280, 4851, 5106, 6499, 80349, 427, 64781, 8837, 5427, 5111, 1398, 1399, 5583, 5588, 3551, 5566, 2597, 3303, 3312, 4792, 4200, 2810, 6777, 6616, 6804, 8900, 54512, 1605, 6443, 6812, 890, 148022, 9641, 207, 993, 5058, 5096, 8324, 4482, 5110, 1385, 998, 1212, 1213, 1891, 55825, 6251, 7016, 55748, 2690, 22908, 5289, 5373, 7264, 5525, 5529, 10525, 9695, 4780, 9817, 79094, 9134, 3553, 4893, 7099, 10775, 60528, 2064, 3315, 7867, 3162, 644, 5603, 5331, 5236, 572, 8503, 2042, 2048, 4891, 5899, 4846, 51070, 8884, 1282, 6696, 10797, 2356, 4690, 8440, 5720, 5721, 2065, 26520, 29928, 10245, 10559, 23443, 51024, 5827, 142, 9261, 3329, 4605, 55012, 332, 6390, 23530, 25805, 5889, 670, 874, 3454, 672, 291, 5580, 5770, 5792, 665, 1845, 10434, 6697, 5347, 958, 10051, 9918, 4638, 6709, 1445, 5788, 7020, 1277, 10398, 6810, 2582, 4791, 5971, 6908, 9519, 8349, 85236, 1635, 2745, 4609, 2946, 4775, 4776, 51021, 9801, 51116, 6182, 1981, 873, 7159, 23368, 5601, 1759, 4793, 2037, 9455, 124583, 5440, 9533, 387, 23463, 7852, 10221, 8061, 3398, 7416, 10682, 30849, 8678, 23659, 5321, 8720, 10165, 466, 4836, 823, 1123, 392, 8869, 1786, 1788, 596, 11157, 23658, 10206, 4313, 7077, 79947, 10329, 11041, 329, 5708, 5710, 128, 55620, 200081, 10174, 9517, 3611, 3028, 3909, 3480, 4208, 5607, 8573, 11284, 5423, 102, 351, 5092, 10695, 5641, 10557, 2958, 2961]
ORD = [list(MJ_COLON_135403648.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
MJ_COLON_135403648 = MJ_COLON_135403648.iloc[ORD]

MJ_tissue_all = MJ_COLON_135403648

for MJ_t in MJ_tissue :
    print (MJ_t)
    tmp_csv = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/fugcn_hhhdt3/' + MJ_t)
    BETA_ENTREZ_ORDER = [1676, 1677, 1738, 501, 5019, 622, 3157, 39, 1019, 1021, 51495, 79071, 230, 5211, 1022, 902, 891, 983, 55256, 58478, 5925, 3098, 22934, 51071, 64080, 226, 5255, 5257, 5261, 3988, 4864, 2184, 2954, 51382, 533, 9133, 30, 6342, 1633, 4860, 1111, 11200, 4088, 7048, 1026, 1647, 8312, 8321, 1950, 1956, 7043, 1029, 10298, 56924, 3628, 8821, 178, 2548, 836, 840, 4067, 695, 581, 4331, 5836, 2131, 9653, 1017, 16, 2058, 1027, 5898, 5900, 595, 1978, 6009, 5223, 6597, 6599, 5747, 5829, 22926, 7494, 3251, 7157, 2673, 51005, 4850, 9924, 2778, 5290, 2353, 3725, 2222, 50814, 1846, 1848, 3108, 3122, 6772, 6774, 54205, 4616, 6850, 6117, 6118, 6119, 6500, 7027, 2770, 2771, 10007, 3930, 3978, 57804, 835, 843, 3033, 211, 5498, 1643, 5366, 637, 5287, 1870, 2956, 355, 5982, 5985, 6194, 3280, 4851, 5106, 6499, 80349, 427, 64781, 8837, 5427, 5111, 1398, 1399, 5583, 5588, 3551, 5566, 2597, 3303, 3312, 4792, 4200, 2810, 6777, 6616, 6804, 8900, 54512, 1605, 6443, 6812, 890, 148022, 9641, 207, 993, 5058, 5096, 8324, 4482, 5110, 1385, 998, 1212, 1213, 1891, 55825, 6251, 7016, 55748, 2690, 22908, 5289, 5373, 7264, 5525, 5529, 10525, 9695, 4780, 9817, 79094, 9134, 3553, 4893, 7099, 10775, 60528, 2064, 3315, 7867, 3162, 644, 5603, 5331, 5236, 572, 8503, 2042, 2048, 4891, 5899, 4846, 51070, 8884, 1282, 6696, 10797, 2356, 4690, 8440, 5720, 5721, 2065, 26520, 29928, 10245, 10559, 23443, 51024, 5827, 142, 9261, 3329, 4605, 55012, 332, 6390, 23530, 25805, 5889, 670, 874, 3454, 672, 291, 5580, 5770, 5792, 665, 1845, 10434, 6697, 5347, 958, 10051, 9918, 4638, 6709, 1445, 5788, 7020, 1277, 10398, 6810, 2582, 4791, 5971, 6908, 9519, 8349, 85236, 1635, 2745, 4609, 2946, 4775, 4776, 51021, 9801, 51116, 6182, 1981, 873, 7159, 23368, 5601, 1759, 4793, 2037, 9455, 124583, 5440, 9533, 387, 23463, 7852, 10221, 8061, 3398, 7416, 10682, 30849, 8678, 23659, 5321, 8720, 10165, 466, 4836, 823, 1123, 392, 8869, 1786, 1788, 596, 11157, 23658, 10206, 4313, 7077, 79947, 10329, 11041, 329, 5708, 5710, 128, 55620, 200081, 10174, 9517, 3611, 3028, 3909, 3480, 4208, 5607, 8573, 11284, 5423, 102, 351, 5092, 10695, 5641, 10557, 2958, 2961]
    ORD = [list(tmp_csv.entrez_id).index(a) for a in BETA_ENTREZ_ORDER]
    tmp_csv2 = tmp_csv.iloc[ORD]
    tmp_csv3 = tmp_csv2.iloc[:,3:]
    MJ_tissue_all = pd.concat([MJ_tissue_all, tmp_csv3], axis = 1)


MJ_tissue_all = pd.concat([MJ_tissue_all, MJ_COLON_135402009], axis =1 )







def get_EXP_DATA(CID, CELL) :
    lincs_tmp = LINCS_DATA[(LINCS_DATA.CID == CID) & (LINCS_DATA.CCLE_Name == CELL)]
    long_id = str(CID) + '__' + str(CELL)
    #
    if lincs_tmp.shape[0] == 1 : 
        lincs_index = lincs_tmp.index.item()
        exp_result = LINCS_EXP[lincs_index]
        print('exist in LINCS')
    elif long_id in MJ_tissue_all.columns :
        exp_result = torch.Tensor(MJ_tissue_all[long_id]).view(349,1)
        print('made with module 1')
    else :
        exp_result = torch.Tensor([0]*349).view(349,1)
        print('no exp data')
    #
    return exp_result






# 3) TARGET 

def get_targets(CID): # 
	target_cids = list(set(TARGET_DATA.CID))
	target_cids.sort()
	gene_ids = BETA_ENTREZ_ORDER
	#
	if CID in target_cids:
		tmp_df2 = TARGET_DATA[TARGET_DATA.CID == CID]
		targets = list(set(tmp_df2.EntrezID))
		vec = [1 if a in targets else 0 for a in gene_ids ]
	else :
		vec = [0] * 349
	return torch.Tensor(vec)



# 4) BASAL 가져오기

#CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'
#ccle_exp = pd.read_csv(CCLE_PATH+'CCLE_expression.csv', low_memory=False)
#ccle_info= pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)
#ori_col = list( ccle_exp.columns ) # entrez!
#for_gene = ori_col[1:]
#for_gene2 = [int(a.split('(')[1].split(')')[0]) for a in for_gene]
#new_col = ['DepMap_ID']+for_gene2 
#ccle_exp.columns = new_col
#ccle_cell_info = ccle_info[['DepMap_ID','CCLE_Name']]
#ccle_cell_info.columns = ['DepMap_ID','DrugCombCCLE']
#ccle_exp2 = pd.merge(ccle_exp, ccle_cell_info, on = 'DepMap_ID' , how='left')
#ccle_exp3 = ccle_exp2[['DepMap_ID','DrugCombCCLE']+BETA_ENTREZ_ORDER]
#ccle_cello_names = [a for a in ccle_exp3.DrugCombCCLE if type(a) == str]
# ccle_exp3.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/TO_GIT/CCLE_BASAL.csv', index = False)


def get_CCLE(CELL) : # 근데 어차피 새 CCLE 값 받는걸로 고쳐야함 
    BETA_ENTREZ_ORDER_str = [str(a) for a in BETA_ENTREZ_ORDER]
    if CELL in list(BASAL_DATA.DrugCombCCLE) : 
        ccle_exp_df = BASAL_DATA[BASAL_DATA.DrugCombCCLE==CELL][BETA_ENTREZ_ORDER_str]
        ccle_result = torch.Tensor(list(ccle_exp_df.T.iloc[:,0])).view(349,1)
    else : 
        print("no data in our ccle")
        ccle_result = [0]*349
    #
    return ccle_result



################################################
# Make input for eval :


class MY_expGCN_parallel_model(torch.nn.Module):
	def __init__(self, G_layer_chem, G_indim_chem, G_hiddim_chem, 
	G_layer_exp, G_indim_exp, G_hiddim_exp, 
	layers_1, layers_2, layers_3, 
	out_dim, inDrop, drop):
		super(MY_expGCN_parallel_model, self).__init__()
		self.G_layer_chem = G_layer_chem
		self.G_indim_chem = G_indim_chem
		self.G_hiddim_chem = G_hiddim_chem
		self.G_layer_exp = G_layer_exp
		self.G_indim_exp = G_indim_exp
		self.G_hiddim_exp = G_hiddim_exp
		self.G_Common_dim = min([G_hiddim_chem,G_hiddim_exp])
		self.layers_1 = [int(a) for a in layers_1]
		self.layers_2 = [int(a) for a in layers_2]
		self.layers_3 = [int(a) for a in layers_3]
		self.out_dim = out_dim
		self.inDrop = inDrop
		self.drop = drop
		self.tanh = torch.nn.Tanh()
		self.pool = pyg_nn.global_mean_pool
		#
		self.G_convs_1_chem = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_chem, self.G_hiddim_chem)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_hiddim_chem) for i in range(self.G_layer_chem-2)])
		self.G_convs_1_chem.extend([pyg_nn.GCNConv(self.G_hiddim_chem, self.G_Common_dim)])
		self.G_bns_1_chem = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_chem) for i in range(self.G_layer_chem-1)])
		##
		self.G_convs_1_exp = torch.nn.ModuleList([pyg_nn.GCNConv(self.G_indim_exp, self.G_hiddim_exp)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_hiddim_exp) for i in range(self.G_layer_exp-2)])
		self.G_convs_1_exp.extend([pyg_nn.GCNConv(self.G_hiddim_exp, self.G_Common_dim)])
		self.G_bns_1_exp = torch.nn.ModuleList([torch.nn.BatchNorm1d(self.G_hiddim_exp) for i in range(self.G_layer_exp-1)])
		##
		self.Convs_1 = torch.nn.ModuleList([torch.nn.Linear(self.G_Common_dim+self.G_Common_dim, self.layers_1[0] )])
		self.Convs_1.extend([torch.nn.Linear(self.layers_1[a], self.layers_1[a+1]) for a in range(len(self.layers_1)-1)])
		##
		self.SNPs = torch.nn.ModuleList([torch.nn.Linear(self.layers_1[-1]+self.layers_2[-1], self.layers_3[0] )])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[a], self.layers_3[a+1]) for a in range(len(self.layers_3)-1)])
		self.SNPs.extend([torch.nn.Linear(self.layers_3[-1], self.out_dim)])
		#
		self.reset_parameters()
	#
	def reset_parameters(self):
		for conv in self.G_convs_1_chem :
			conv.reset_parameters()
		for bns in self.G_bns_1_chem :
			bns.reset_parameters()
		for conv in self.G_convs_1_exp :
			conv.reset_parameters()
		for bns in self.G_bns_1_exp :
			bns.reset_parameters()
		for conv in self.Convs_1 :
			conv.reset_parameters()
		for conv in self.SNPs:
			conv.reset_parameters()
	#
	def calc_batch_label (self, syn, feat) :
		batchnum = syn.shape[0]
		nodenum = feat.shape[0]/batchnum
		Num = [a for a in range(batchnum)]
		Rep = np.repeat(Num, nodenum)
		batch_labels = torch.Tensor(Rep).long()
		if torch.cuda.is_available():
			batch_labels = batch_labels.cuda()
		return batch_labels
	#
	def forward(self, Drug1_F, Drug2_F, Drug1_ADJ, Drug2_ADJ, EXP1, EXP2, EXP_ADJ, EXP_ADJ_WGT, syn ):
		Drug_batch_label = self.calc_batch_label(syn, Drug1_F)
		Exp_batch_label = self.calc_batch_label(syn, EXP1)
		#
		for G_1_C in range(len(self.G_convs_1_chem)):
			if G_1_C == len(self.G_convs_1_chem)-1 :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = F.dropout(Drug1_F, p=self.inDrop, training=self.training)
				Drug1_F = self.pool(Drug1_F, Drug_batch_label )
				Drug1_F = self.tanh(Drug1_F)
				G_1_C_out = Drug1_F
			else :
				Drug1_F = self.G_convs_1_chem[G_1_C](x=Drug1_F, edge_index=Drug1_ADJ)
				Drug1_F = self.G_bns_1_chem[G_1_C](Drug1_F)
				Drug1_F = F.elu(Drug1_F)
		#
		for G_2_C in range(len(self.G_convs_1_chem)):
			if G_2_C == len(self.G_convs_1_chem)-1 :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = F.dropout(Drug2_F, p=self.inDrop, training=self.training)
				Drug2_F = self.pool(Drug2_F, Drug_batch_label )
				Drug2_F = self.tanh(Drug2_F)
				G_2_C_out = Drug2_F
			else :
				Drug2_F = self.G_convs_1_chem[G_2_C](x=Drug2_F, edge_index=Drug2_ADJ)
				Drug2_F = self.G_bns_1_chem[G_2_C](Drug2_F)
				Drug2_F = F.elu(Drug2_F)
		#
		for G_1_E in range(len(self.G_convs_1_exp)):
			if G_1_E == len(self.G_convs_1_exp)-1 :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = F.dropout(EXP1, p=self.inDrop, training=self.training)
				EXP1 = self.pool(EXP1, Exp_batch_label )
				EXP1 = self.tanh(EXP1)
				G_1_E_out = EXP1
			else :
				EXP1 = self.G_convs_1_exp[G_1_E](x=EXP1, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP1 = self.G_bns_1_exp[G_1_E](EXP1)
				EXP1 = F.elu(EXP1)
		#
		for G_2_E in range(len(self.G_convs_1_exp)):
			if G_2_E == len(self.G_convs_1_exp)-1 :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = F.dropout(EXP2, p=self.inDrop, training=self.training)
				EXP2 = self.pool(EXP2, Exp_batch_label )
				EXP2 = self.tanh(EXP2)
				G_2_E_out = EXP2
			else :
				EXP2 = self.G_convs_1_exp[G_2_E](x=EXP2, edge_index=EXP_ADJ, edge_weight= EXP_ADJ_WGT)
				EXP2 = self.G_bns_1_exp[G_2_E](EXP2)
				EXP2 = F.elu(EXP2)
		#
		input_drug1 = torch.concat( (G_1_C_out, G_1_E_out), 1 ) # normalization 추가 해야할것 같은데 
		input_drug2 = torch.concat( (G_2_C_out, G_2_E_out), 1 )
		#
		for L1 in range(len(self.Convs_1)):
			if L1 != len(self.Convs_1)-1 :
				input_drug1 = self.Convs_1[L1](input_drug1)
				input_drug1 = F.dropout(input_drug1, p=self.inDrop, training = self.training)
				input_drug1 = F.elu(input_drug1)
			else :
				input_drug1 = self.Convs_1[L1](input_drug1)
		#
		for L2 in range(len(self.Convs_1)):
			if L2 != len(self.Convs_1)-1 :
				input_drug2 = self.Convs_1[L2](input_drug2)
				input_drug2 = F.dropout(input_drug2, p=self.inDrop, training = self.training)
				input_drug2 = F.elu(input_drug2)
			else :
				input_drug2 = self.Convs_1[L2](input_drug2)
		#
		X = torch.cat(( input_drug1, input_drug2 ), 1)
		for L3 in range(len(self.SNPs)):
			if L3 != len(self.SNPs)-1 :
				X = self.SNPs[L3](X)
				X = F.dropout(X, p=self.drop, training = self.training)
				X = F.elu(X)
			else :
				X = self.SNPs[L3](X)
		return X






def check_CID(SMILES) :
    try : 
        pcp_res = pcp.get_compounds(SMILES, namespace = 'smiles')
        if len(pcp_res) == 1:
            comp = pcp_res[0]
            lincs_cid = comp.cid
            if lincs_cid > 0 :
                return lincs_cid
        else : 
            print('multiple cid')
    except : 
        return 0



def make_input_data (SM_A, SM_B, CELL):
    #
    drug1_f, drug1_a = make_rdkit_var(SM_A)
    drug2_f, drug2_a = make_rdkit_var(SM_B)
    #
    CID_A = check_CID(SM_A)
    # CID_B = check_CID(SM_B)
    CID_B = 9887053
    #
    print('Drug A EXP')
    expA = get_EXP_DATA(CID_A, CELL)
    print('Drug B EXP')
    expB = get_EXP_DATA(CID_B, CELL)
    targetA = get_targets(CID_A)
    targetB = get_targets(CID_B)
    basal = get_CCLE(CELL)
    FEAT_A = torch.Tensor(np.array([ expA.squeeze().tolist() , targetA.squeeze().tolist(), basal.squeeze().tolist()]).T)
    FEAT_A = FEAT_A.view(-1,3)#### 다른점 
    FEAT_B = torch.Tensor(np.array([ expB.squeeze().tolist() , targetB.squeeze().tolist(), basal.squeeze().tolist()]).T)
    FEAT_B = FEAT_B.view(-1,3)#### 다른점 
    #
    adj = copy.deepcopy(G_ADJ_IDX).long()
    adj_w = torch.Tensor(G_IDX_WEIGHT).squeeze()
    #
    return drug1_f, drug2_f, drug1_a, drug2_a, FEAT_A, FEAT_B, adj, adj_w





OLD_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W204_349_MIS2'
ANA_DF_CSV = pd.read_csv(os.path.join(OLD_PATH,'RAY_ANA_DF.{}.csv'.format('M3V6_W204_349_MIS2')))
my_config = ANA_DF_CSV.loc[0]
KEY_EPC = 963
checkpoint = "checkpoint_"+str(KEY_EPC).zfill(6)
CKP_PATH = os.path.join( ANA_DF_CSV.logdir.item(), checkpoint, 'checkpoint')
CKP_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/M3V6_W204_349_MIS2/checkpoint'
-> 나중에 다시 지정해줘야함 






def pred_synergy_single (my_config, SM_A, SM_B, CELL):
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item()
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#      
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, 64 , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
				inDrop, Drop
				) 
	#
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		state_dict = torch.load(CKP_PATH) #### change ! 
	else:
		state_dict = torch.load(CKP_PATH, map_location=torch.device('cpu'))
	# 
	# 
	#print("state_dict_done", flush = True)
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)
	#
	#print("state_load_done", flush = True)
	#
	best_model.to(device)
	best_model.eval()
	#
	single = torch.Tensor([[0]])
	drug1_f, drug2_f, drug1_a, drug2_a, FEAT_A, FEAT_B, adj, adj_w = make_input_data(SM_A, SM_B, CELL)
	output_1 = best_model(drug1_f, drug2_f, drug1_a, drug2_a, FEAT_A, FEAT_B, adj, adj_w, single) 
	output_2 = best_model(drug2_f, drug1_f, drug2_a, drug1_a, FEAT_B, FEAT_A, adj, adj_w, single) 
	result_value = np.round(np.mean([output_1.item(), output_2.item()]),4)
	print('Expected Loewe Score is : {}'.format(result_value))
	print('\n')


avail_cell_list = ['CAMA1_BREAST','VCAP_PROSTATE', 'NIHOVCAR3_OVARY', 'SW620_LARGE_INTESTINE', 'OVCAR4_OVARY', 'BT549_BREAST', 'A549_LUNG', 'SKMEL5_SKIN', 'A427_LUNG', 'BT474_BREAST', 'HOP92_LUNG', 'T98G_CENTRAL_NERVOUS_SYSTEM', 'NCIH23_LUNG', 'HT144_SKIN', 'RVH421_SKIN', 'MDAMB361_BREAST', 'LOVO_LARGE_INTESTINE', 'MDAMB231_BREAST', 'SKMEL28_SKIN', 'NCIH1650_LUNG', 'RKO_LARGE_INTESTINE', 'OVCAR5_OVARY', 'UACC812_BREAST', 'KPL1_BREAST', 'MSTO211H_PLEURA', 'KM12_LARGE_INTESTINE', 'IGROV1_OVARY', 'UHO1_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'NCIH520_LUNG', 'OVCAR8_OVARY', 'HCT15_LARGE_INTESTINE', 'A375_SKIN', 'CAKI1_KIDNEY', 'MDAMB468_BREAST', 'K562_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'A101D_SKIN', 'PA1_OVARY', 'UO31_KIDNEY', 'HOP62_LUNG', 'SF539_CENTRAL_NERVOUS_SYSTEM', 'MDAMB175VII_BREAST', 'U251MG_CENTRAL_NERVOUS_SYSTEM', 'HCC1500_BREAST', 'L1236_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'HCC1419_BREAST', 'NCIH460_LUNG', 'NCIH2122_LUNG', 'COLO792_SKIN', 'SR786_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'UACC257_SKIN', 'SNB75_CENTRAL_NERVOUS_SYSTEM', 'HCT116_LARGE_INTESTINE', 'PC3_PROSTATE', 'NCIH226_LUNG', 'RPMI8226_HAEMATOPOIETIC_AND_LYMPHOID_TISSUE', 'EKVX_LUNG', 'COLO800_SKIN', 'HT29_LARGE_INTESTINE', 'UWB1289_OVARY', 'MDAMB436_BREAST', 'SKOV3_OVARY', 'ZR751_BREAST', 'MEWO_SKIN', 'MELHO_SKIN', 'A2058_SKIN', 'RPMI7951_SKIN', 'SF268_CENTRAL_NERVOUS_SYSTEM', 'ACHN_KIDNEY', 'IPC298_SKIN', 'MALME3M_SKIN', 'A673_BONE', 'SF295_CENTRAL_NERVOUS_SYSTEM', 'CAOV3_OVARY', 'A498_KIDNEY', 'SKMEL2_SKIN', 'UACC62_SKIN', 'ES2_OVARY', 'LOXIMVI_SKIN', '786O_KIDNEY', 'MCF7_BREAST', 'WM115_SKIN', 'A2780_OVARY', 'DLD1_LARGE_INTESTINE', 'HS578T_BREAST', 'SKMES1_LUNG', 'T47D_BREAST', 'OV90_OVARY', 'G361_SKIN', 'SKMEL30_SKIN', 'COLO829_SKIN', 'SW837_LARGE_INTESTINE', 'NCIH522_LUNG']

for CELL in avail_cell_list:
    print(CELL)
    SM_A = 'C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O' # 135402009 & 135403648
    SM_B = 'C1=C(C(=O)NC(=O)N1)F' # 3385 
    pred_synergy_single(my_config, SM_A, SM_B, CELL)


for CELL in avail_cell_list:
    print(CELL)
    SM_A = 'C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O' # 135402009 & 135403648
    SM_B = 'C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]' # 9887053
    pred_synergy_single(my_config, SM_A, SM_B, CELL)


for CELL in avail_cell_list:
    print(CELL)
    SM_A = 'C1=C(C(=O)NC(=O)N1)F' # 3385
    SM_B = 'C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]' # 9887053
    pred_synergy_single(my_config, SM_A, SM_B, CELL)


for CELL in avail_cell_list:
    print(CELL)
    SM_A = 'CCCCCOC(=O)NC1=NC(=O)N(C=C1F)[C@H]2[C@@H]([C@@H]([C@H](O2)C)O)O' # 60953
    SM_B = 'C1CCC(C(C1)[NH-])[NH-].C(=O)(C(=O)O)O.[Pt+2]' # 9887053
    pred_synergy_single(my_config, SM_A, SM_B, CELL)


SM_A = 'CCCCCOC(=O)NC1=NC(=O)N(C=C1F)[C@H]2[C@@H]([C@@H]([C@H](O2)C)O)O'

400633







def pred_synergy_list (CELLO, MODEL_NAME, use_cuda = True) :
    print(CELLO)
    tt_df = pd.DataFrame()
    #
	G_chem_layer = my_config['config/G_chem_layer'].item()
	G_chem_hdim = my_config['config/G_chem_hdim'].item()
	G_exp_layer = my_config['config/G_exp_layer'].item()
	G_exp_hdim = my_config['config/G_exp_hdim'].item()
	dsn_layers = [int(a) for a in my_config["config/dsn_layer"].split('-') ]
	snp_layers = [int(a) for a in my_config["config/snp_layer"].split('-') ]
	inDrop = my_config['config/dropout_1'].item()
	Drop = my_config['config/dropout_2'].item()
	#      
	best_model = MY_expGCN_parallel_model(
				G_chem_layer, 64 , G_chem_hdim,
				G_exp_layer, 3, G_exp_hdim,
				dsn_layers, dsn_layers, snp_layers, 
				1,
				inDrop, Drop
				) 
	#
	device = "cuda:0" if torch.cuda.is_available() else "cpu"
	if torch.cuda.is_available():
		state_dict = torch.load(MODEL_NAME) #### change ! 
	else:
		state_dict = torch.load(MODEL_NAME, map_location=torch.device('cpu'))
	# 
	# 
	print("state_dict_done", flush = True)
	if type(state_dict) == tuple:
		best_model.load_state_dict(state_dict[0])
	else : 
		best_model.load_state_dict(state_dict)
	#
	print("state_load_done", flush = True)
	#
	#
	best_model.to(device)
	best_model.eval()
	#
	dataset = CellTest_Dataset(tuple_list)
	dataloader = torch.utils.data.DataLoader(dataset, batch_size = 2048 , collate_fn = graph_collate_fn, shuffle = False, num_workers = 128 )# , num_workers=my_config['config/n_workers'].item()
	# 약간 헷갈리는게, 1gpu 기준으로 2048 도 작음 --> 그리고 자꾸 0 % 로 내려감 -> 4096 -> 8192 //-> 16384 -> 32768
	CELL_PRED_DF = pd.DataFrame(columns = ['PRED','ROW_CID','COL_CID','CCLE','Y'])
	CELL_PRED_DF.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL_ing.csv'.format(CELLO), index=False)
	#
	with torch.no_grad():
		for batch_idx_t, (tup_list, drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) in enumerate(dataloader):
			print("{} / {}".format(batch_idx_t, len(dataloader)) , flush = True)
			print(datetime.now(), flush = True)
			list_ROW_CID = [a[0] for a in tup_list]
			list_COL_CID = [a[1] for a in tup_list]
			list_CELLO = [a[2] for a in tup_list]
			#
			if use_cuda:
				drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w, y, cell = drug1_f.cuda(), drug2_f.cuda(), drug1_a.cuda(), drug2_a.cuda(), expA.cuda(), expB.cuda(), adj.cuda(), adj_w.cuda(), y.cuda(), cell.cuda()
			#
			output = best_model(drug1_f, drug2_f, drug1_a, drug2_a, expA, expB, adj, adj_w.squeeze(), cell.squeeze(), y) 
			outputs = output.squeeze().tolist() # [output.squeeze().item()]
			#print(outputs)
			#
			tmp_df = pd.DataFrame({
			'PRED': outputs,
			'ROW_CID' : list_ROW_CID,
			'COL_CID' : list_COL_CID,
			'CCLE' : list_CELLO,
			'Y' : y.squeeze().tolist()
			})
			CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])
			tmp_df.to_csv(CELVAL_PATH+'/CELL_VAL/'+'PRED_{}.FINAL_ing.csv'.format(CELLO), mode='a', index=False, header = False)
	return CELL_PRED_DF















