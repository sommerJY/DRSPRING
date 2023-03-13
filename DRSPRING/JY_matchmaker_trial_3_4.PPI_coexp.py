
PPI 크기 및 결과 비교 
잘 나왔던 애 하나로 진행해보면 어떨까 (다 돌리기엔 좀 부담스럽다면) 

그러면 꼭 gpu 4 쓰지 않아도 되고 







# 방향 다시 확인하기 



(1) IDEKER FULL 
(1) IDEKER FULL 
(1) IDEKER FULL 
IDK_PATH=

IDEKER_IAS = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/IAS_score.tsv', sep = '\t')

fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=list(IDEKER_IAS['Integrated score']), bins = 100, alpha = 0.7, rwidth=0.8)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(0, 1) # 일정한 scale 
plt.title('IDEKER TOT Histogram') 
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/IDK.tot.hist.png', bbox_inches = 'tight')


IDEKER_TOT_GS = list(set(list(IDEKER_IAS['Protein 1'])+list(IDEKER_IAS['Protein 2']))) # 16840

check_tmp_G = nx.from_pandas_edgelist(IDEKER_IAS, 'Protein 1', 'Protein 2')
len(check_tmp_G.nodes()) # 16840
len(check_tmp_G.edges()) # 1848498 -> 3696996

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


ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40464]
ID_WEIGHT = [] # len : 20232 -> 40464



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
IAS_FILTER = IAS_FILTER.drop_duplicates()


ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' )
ID_WEIGHT_SCORE = list(ID_WEIGHT['Integrated score'])



MY_G = ID_G # GENENAME
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE




(2) IDEKER Physical only 
(2) IDEKER Physical only 
(2) IDEKER Physical only 


IDEKER_IAS = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/IAS_score.tsv', sep = '\t') # 1,848,498
IDEKER_IAS_PPI_0 = IDEKER_IAS[IDEKER_IAS['evidence: Physical']>0] # 1,843,932
IDEKER_IAS_PPI_1 = IDEKER_IAS_PPI_0[['Protein 1','Protein 2','evidence: Physical']]
IDEKER_IAS_PPI_2 = IDEKER_IAS_PPI_0[['Protein 2','Protein 1','evidence: Physical']]
IDEKER_IAS_PPI_2.columns = ['Protein 1','Protein 2','evidence: Physical']
IDEKER_IAS_PPI = pd.concat([IDEKER_IAS_PPI_1, IDEKER_IAS_PPI_2])

fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=list(IDEKER_IAS['evidence: Physical']), bins = 100, alpha = 0.7, rwidth=0.8)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(0, 1) # 일정한 scale
plt.title('IDEKER PPI Histogram')
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/IDK.ppi.hist2.png', bbox_inches = 'tight')


check_tmp_G = nx.from_pandas_edgelist(IDEKER_IAS_PPI, 'Protein 1', 'Protein 2')
len(check_tmp_G.nodes()) # 16839
len(check_tmp_G.edges()) # 1843932

IDEKER_TOT_GS = list(set(list(IDEKER_IAS_PPI['Protein 1'])+list(IDEKER_IAS_PPI['Protein 2']))) # 16839
L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

IDEKER_IAS_L1 = IDEKER_IAS_PPI[IDEKER_IAS_PPI['Protein 1'].isin(L_matching_list.PPI_name)]
IDEKER_IAS_L2 = IDEKER_IAS_L1[IDEKER_IAS_L1['Protein 2'].isin(L_matching_list.PPI_name)] # 20209

len(set(list(IDEKER_IAS_L2['Protein 1']) + list(IDEKER_IAS_L2['Protein 2'])))
ID_G_PPI = nx.from_pandas_edgelist(IDEKER_IAS_L2, 'Protein 1', 'Protein 2')

ESSEN_NAMES = list(set(L_matching_list['PPI_name']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G_PPI.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G_PPI.add_node(nn)


ID_GENE_ORDER_mini = list(ID_G_PPI.nodes()) # 929
ID_ADJ = nx.adjacency_matrix(ID_G_PPI)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40418]
ID_WEIGHT = [] # len : 20209 -> 40418

ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['Protein 1','Protein 2']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['Protein 1'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['Protein 2'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = IDEKER_IAS[['Protein 1', 'Protein 2', 'evidence: Physical']]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['Protein 1']+'__'+IAS_FILTER['Protein 2']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['Protein 2']+'__'+IAS_FILTER['Protein 1']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','evidence: Physical']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'evidence: Physical']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'evidence: Physical']]
IAS_FILTER1.columns = ['NAMESUM', 'evidence: Physical']
IAS_FILTER2.columns = ['NAMESUM', 'evidence: Physical']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0)
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' )

ID_WEIGHT_SCORE = list(ID_WEIGHT['evidence: Physical'])


MY_G = ID_G_PPI # GENENAME
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



(3-1) IDEKER mRNA coexp only 
(3-1) IDEKER mRNA coexp only 
(3-1) IDEKER mRNA coexp only 

IDEKER_IAS = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/IAS_score.tsv', sep = '\t') # 1,848,498
IDEKER_IAS_m_cxp_0 = IDEKER_IAS[IDEKER_IAS['evidence: mRNA co-expression']>0] # 1,848,498
IDEKER_IAS_m_cxp_1 = IDEKER_IAS_m_cxp_0[['Protein 1','Protein 2','evidence: mRNA co-expression']]
IDEKER_IAS_m_cxp_2 = IDEKER_IAS_m_cxp_0[['Protein 2','Protein 1','evidence: mRNA co-expression']]
IDEKER_IAS_m_cxp_2.columns = ['Protein 1','Protein 2','evidence: mRNA co-expression']
IDEKER_IAS_m_cxp = pd.concat([IDEKER_IAS_m_cxp_1, IDEKER_IAS_m_cxp_2])


fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=list(IDEKER_IAS['evidence: Protein co-expression']), bins = 100, alpha = 0.7, rwidth=0.8)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(0, 1) # 일정한 scale
plt.title('IDEKER Protein Coexp Histogram')
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.4.ppi_exp/IDK.p_co.hist.png', bbox_inches = 'tight')


fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=list(IDEKER_IAS['evidence: mRNA co-expression']), bins = 100, alpha = 0.7, rwidth=0.8)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(0, 1) # 일정한 scale
plt.title('IDEKER mRNA Coexp Histogram')
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/IDK.m_co.hist.png', bbox_inches = 'tight')


tmp = list(IDEKER_IAS['evidence: mRNA co-expression']) + list(IDEKER_IAS['evidence: Protein co-expression'])
fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=tmp, bins = 100, alpha = 0.7, rwidth=0.8)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(0, 1) # 일정한 scale
plt.title('IDEKER mRNA&Protein Coexp Histogram')
plt.grid(True)
plt.legend()
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/IDK.pm_co.hist.png', bbox_inches = 'tight')



IDEKER_TOT_GS = list(set(list(IDEKER_IAS_m_cxp['Protein 1'])+list(IDEKER_IAS_m_cxp['Protein 2']))) # 16840

L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

IDEKER_IAS_L1 = IDEKER_IAS_m_cxp[IDEKER_IAS_m_cxp['Protein 1'].isin(L_matching_list.PPI_name)]
IDEKER_IAS_L2 = IDEKER_IAS_L1[IDEKER_IAS_L1['Protein 2'].isin(L_matching_list.PPI_name)] # 20209

len(set(list(IDEKER_IAS_L2['Protein 1']) + list(IDEKER_IAS_L2['Protein 2'])))
ID_G = nx.from_pandas_edgelist(IDEKER_IAS_L2, 'Protein 1', 'Protein 2')

ESSEN_NAMES = list(set(L_matching_list['PPI_name']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)


ID_GENE_ORDER_mini = list(ID_G.nodes()) # 929
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40464]
ID_WEIGHT = [] # len : 20232 -> 40464

ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['Protein 1','Protein 2']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['Protein 1'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['Protein 2'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = IDEKER_IAS_m_cxp[['Protein 1', 'Protein 2', 'evidence: mRNA co-expression']]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['Protein 1']+'__'+IAS_FILTER['Protein 2']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['Protein 2']+'__'+IAS_FILTER['Protein 1']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','evidence: mRNA co-expression']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'evidence: mRNA co-expression']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'evidence: mRNA co-expression']]
IAS_FILTER1.columns = ['NAMESUM', 'evidence: mRNA co-expression']
IAS_FILTER2.columns = ['NAMESUM', 'evidence: mRNA co-expression']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0)
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' )

ID_WEIGHT_SCORE = list(ID_WEIGHT['evidence: mRNA co-expression'])



MY_G = ID_G # GENENAME
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE


(3-2) IDEKER Protein coexp only 
(3-2) IDEKER Protein coexp only 
(3-2) IDEKER Protein coexp only 

IDEKER_IAS = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/IAS_score.tsv', sep = '\t') # 1,848,498
IDEKER_IAS_p_cxp = IDEKER_IAS[IDEKER_IAS['evidence: Protein co-expression']>0] # 1,837,540
IDEKER_IAS_p_cxp_0 = IDEKER_IAS[IDEKER_IAS['evidence: Protein co-expression']>0] # 1,837,540
IDEKER_IAS_p_cxp_1 = IDEKER_IAS_p_cxp_0[['Protein 1','Protein 2','evidence: Protein co-expression']]
IDEKER_IAS_p_cxp_2 = IDEKER_IAS_p_cxp_0[['Protein 2','Protein 1','evidence: Protein co-expression']]
IDEKER_IAS_p_cxp_2.columns = ['Protein 1','Protein 2','evidence: Protein co-expression']
IDEKER_IAS_p_cxp = pd.concat([IDEKER_IAS_p_cxp_1, IDEKER_IAS_p_cxp_2])

check_tmp_G = nx.from_pandas_edgelist(IDEKER_IAS_p_cxp, 'Protein 1', 'Protein 2')
len(check_tmp_G.nodes()) # 16834
len(check_tmp_G.edges()) # 1837540

L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

IDEKER_IAS_L1 = IDEKER_IAS_p_cxp[IDEKER_IAS_p_cxp['Protein 1'].isin(L_matching_list.PPI_name)]
IDEKER_IAS_L2 = IDEKER_IAS_L1[IDEKER_IAS_L1['Protein 2'].isin(L_matching_list.PPI_name)] # 20232

len(set(list(IDEKER_IAS_L2['Protein 1']) + list(IDEKER_IAS_L2['Protein 2'])))
ID_G = nx.from_pandas_edgelist(IDEKER_IAS_L2, 'Protein 1', 'Protein 2')

ESSEN_NAMES = list(set(L_matching_list['PPI_name']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)


ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 40280]
ID_WEIGHT = [] # len : 20140 -> 40280


ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['Protein 1','Protein 2']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['Protein 1'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['Protein 2'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = IDEKER_IAS_p_cxp[['Protein 1', 'Protein 2', 'evidence: Protein co-expression']]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['Protein 1']+'__'+IAS_FILTER['Protein 2']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['Protein 2']+'__'+IAS_FILTER['Protein 1']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','evidence: Protein co-expression']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'evidence: Protein co-expression']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'evidence: Protein co-expression']]
IAS_FILTER1.columns = ['NAMESUM', 'evidence: Protein co-expression']
IAS_FILTER2.columns = ['NAMESUM', 'evidence: Protein co-expression']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0) 
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' ) # 40280
ID_WEIGHT_SCORE = list(ID_WEIGHT['evidence: Protein co-expression'])


MY_G = ID_G # GENENAME
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



(4) HuRI -- no score
(4) HuRI -- no score 
(4) HuRI -- no score

huri_dir = '/home01/k006a01/01.DATA/HuRI/'

huri_dir = '/st06/jiyeonH/00.huRI.20220120/'
huri_union = pd.read_csv(huri_dir+'HI-union.tsv', sep = '\t', header = None) #  edge 64006
huri_union.columns = ['G_A','G_B']
huri_union_1 = huri_union[['G_A','G_B']]
huri_union_2 = huri_union[['G_B','G_A']]
huri_union_2.columns = ['G_A','G_B']
huri_union = pd.concat([huri_union_1, huri_union_2])

huri_only = pd.read_csv(huri_dir+'HuRI.tsv', sep = '\t', header = None) # edge 52548
huri_only.columns = ['G_A','G_B']
huri_only_1 = huri_only[['G_A','G_B']]
huri_only_2 = huri_only[['G_B','G_A']]
huri_only_2.columns = ['G_A','G_B']
huri_only = pd.concat([huri_only_1, huri_only_2])


check_tmp_G = nx.from_pandas_edgelist(huri_union, 'G_A', 'G_B')
len(check_tmp_G.nodes()) # 9094
len(check_tmp_G.edges()) # 64006

check_tmp_G = nx.from_pandas_edgelist(huri_only, 'G_A', 'G_B')
len(check_tmp_G.nodes()) # 8272
len(check_tmp_G.edges()) # 52548

	1) 
	1)
L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

huri_L1 = huri_union[huri_union['G_A'].isin(L_matching_list.ensembl_id)]
huri_L2 = huri_L1[huri_L1['G_B'].isin(L_matching_list.ensembl_id)] # 252

len(set(list(huri_L2['G_A']) + list(huri_L2['G_B']))) # 228
ID_G = nx.from_pandas_edgelist(huri_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['ensembl_id']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

len(ID_G.edges()) # 252
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 447]
ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]

	2) 
	2)
L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

huri_L1 = huri_only[huri_only['G_A'].isin(L_matching_list.ensembl_id)]
huri_L2 = huri_L1[huri_L1['G_B'].isin(L_matching_list.ensembl_id)] # 252

len(set(list(huri_L2['G_A']) + list(huri_L2['G_B']))) # 183
ID_G = nx.from_pandas_edgelist(huri_L2, 'G_A', 'G_B') # edges = 187

ESSEN_NAMES = list(set(L_matching_list['ensembl_id']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 335]
ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.ensembl_id == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)
MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



(5) BioGRID
(5) BioGRID
(5) BioGRID

BG_dir = '/st06/jiyeonH/13.DD_SESS/BioGRID.4.4.211/'
BG_dir = '/home01/k006a01/01.DATA/BioGRID/'

BG_tab_1 = pd.read_csv(BG_dir+'BIOGRID-ALL-4.4.211.tab.txt', sep = '\t', skiprows = 35) #  row 2398426
BG_tab_3 = pd.read_csv(BG_dir+'BIOGRID-ALL-4.4.211.tab3.txt', sep = '\t', low_memory=False) #  row 2400280

check_tmp_G = nx.from_pandas_edgelist(BG_tab_1, 'OFFICIAL_SYMBOL_A', 'OFFICIAL_SYMBOL_B')
len(check_tmp_G.nodes()) # 77737
len(check_tmp_G.edges()) # 1859189

check_tmp_G = nx.from_pandas_edgelist(BG_tab_3, 'Entrez Gene Interactor A', 'Entrez Gene Interactor B')
len(check_tmp_G.nodes()) # 86235
len(check_tmp_G.edges()) # 1860486

tmp_score_1 = list(BG_tab_3['Score'])
tmp_score_2 = [a if a != '-' else '0.0' for a in tmp_score_1]
tmp_score_3 = [float(s) for s in list(tmp_score_1) if s !='-'] # 0 만 1190388 개 
tmp_score_3 = [float(s) for s in list(tmp_score_1) if s !='-'] # 0 만 1190388 개 


fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=tmp_score_3, bins = 100, alpha = 0.7, rwidth=0.8, log=True)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
# plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(min(tmp_score_3), max(tmp_score_3)) # 일정한 scale
plt.title('BioGRID tab3 Histogram')
plt.grid(True)
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/BGD.t3.hist.png', bbox_inches = 'tight')


set(BG_tab_3['Experimental System Type'])
{'genetic', 'physical'}

BG_tab_1_9606 = BG_tab_1[ (BG_tab_1['ORGANISM_A_ID'] == 9606) & (BG_tab_1['ORGANISM_B_ID'] == 9606) ]
# 987573 row
BG_tab_3_9606 = BG_tab_3[ (BG_tab_3['Organism ID Interactor A'] == 9606) & (BG_tab_3['Organism ID Interactor B'] == 9606) ]
# 987573 row


check_tmp_G = nx.from_pandas_edgelist(BG_tab_1_9606, 'OFFICIAL_SYMBOL_A', 'OFFICIAL_SYMBOL_B')
len(check_tmp_G.nodes()) # 19855
len(check_tmp_G.edges()) # 738415

check_tmp_G = nx.from_pandas_edgelist(BG_tab_3_9606, 'Entrez Gene Interactor A', 'Entrez Gene Interactor B')
len(check_tmp_G.nodes()) # 19857
len(check_tmp_G.edges()) # 738415


# 기왕 쓸거 많은걸 쓰자 
L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')

BG_tab_3_9606['entrez_A'] = [int(a) for a in BG_tab_3_9606['Entrez Gene Interactor A']]
BG_tab_3_9606['entrez_B'] = [int(a) for a in BG_tab_3_9606['Entrez Gene Interactor B']]

BG_L1 = BG_tab_3_9606[BG_tab_3_9606['entrez_A'].isin(L_matching_list.entrez)]
BG_L2 = BG_L1[BG_L1['entrez_B'].isin(L_matching_list.entrez)] # row 16531

len(set(list(BG_L2['entrez_A']) + list(BG_L2['entrez_B']))) # 949
ID_G = nx.from_pandas_edgelist(BG_L2, 'entrez_A', 'entrez_B') # edges = 8961

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

len(ID_G.edges()) # 8961
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 17581]

ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['entrez_A','entrez_B']
ID_ADJ_IDX_T['NodeA'] = [list(ID_GENE_ORDER_mini)[A] for A in list(ID_ADJ_IDX_T['entrez_A'])]
ID_ADJ_IDX_T['NodeB'] = [list(ID_GENE_ORDER_mini)[B] for B in list(ID_ADJ_IDX_T['entrez_B'])]
ID_ADJ_IDX_T['str_entrez_A'] = [str(a) for a in list(ID_ADJ_IDX_T['NodeA'])]
ID_ADJ_IDX_T['str_entrez_B'] = [str(a) for a in list(ID_ADJ_IDX_T['NodeB'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['str_entrez_A']+'__'+ID_ADJ_IDX_T['str_entrez_B']

IAS_FILTER = BG_tab_3_9606[['entrez_A', 'entrez_B', 'Score']].drop_duplicates()
IAS_FILTER['str_entrez_A'] = [str(a) for a in list(IAS_FILTER['entrez_A'])]
IAS_FILTER['str_entrez_B'] = [str(a) for a in list(IAS_FILTER['entrez_B'])]
IAS_FILTER['NAMESUM_1']=IAS_FILTER['str_entrez_A']+'__'+IAS_FILTER['str_entrez_B']
IAS_FILTER['NAMESUM_2']=IAS_FILTER['str_entrez_B']+'__'+IAS_FILTER['str_entrez_A']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','Score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'Score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'Score']]
IAS_FILTER1.columns = ['NAMESUM', 'Score']
IAS_FILTER2.columns = ['NAMESUM', 'Score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2], axis = 0) 
IAS_FILTER = IAS_FILTER.drop_duplicates()
IAS_FILTER = IAS_FILTER.drop_duplicates()

tmp1 = list(ID_ADJ_IDX_T.NAMESUM)
tmp2 = list(IAS_FILTER.NAMESUM)
tmp3 = [a for a in tmp1 if tmp2.count(a)>1] # 1278
# 지금 그 뭐야 짝마다 실험 방법에 따라서 점수가 다르게 나오는 경우가 있음. 그래서 뭘로 통일할지가 좀 고민이네 

tt = list(map(lambda x: x.replace('-', '0'), list(IAS_FILTER['Score'])))
ttt = [float(a) for a in tt]

IAS_FILTER['new_score'] = ttt
IAS_FILTER_RE1 = IAS_FILTER[IAS_FILTER.NAMESUM.isin(tmp3)]
IAS_FILTER_RE2 = IAS_FILTER[-IAS_FILTER.NAMESUM.isin(tmp3)]
IAS_FILTER_RE1 = IAS_FILTER_RE1.sort_values('NAMESUM')

for a in range(len(tmp3)): # 0 이 아닌 값을 고르고, 그 안에서 평균 내기로 함 
	namesum = tmp3[a]
	tmpdf = IAS_FILTER_RE1[IAS_FILTER_RE1.NAMESUM==namesum]
	tmpdf2 = tmpdf[tmpdf.new_score >0]
	if tmpdf2.shape[0]==1:
		IAS_FILTER_RE2 = pd.concat([IAS_FILTER_RE2,tmpdf2 ])
	else:
		tmp_med = np.median(tmpdf2.new_score)
		tmp_new = pd.DataFrame({'NAMESUM' : [namesum], 'Score':[str(tmp_med)], 'new_score':[tmp_med]})
		IAS_FILTER_RE2 = pd.concat([IAS_FILTER_RE2,tmp_new ])

IAS_FILTER_RE2= IAS_FILTER_RE2.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER_RE2, on = 'NAMESUM', how = 'left' ) # 22050
ID_WEIGHT_SCORE = list(ID_WEIGHT['new_score'])
# Quantitative Score 

new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)
MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE




(6) SNU -- no score 
(6) SNU # 전에 한번 해둔거라 금방할듯 
(6) SNU -- no score 

SNU_PATH = '/home01/k006a01/01.DATA/SNU/'
SNU_PATH = '/st06/jiyeonH/11.TOX/SNU_PPI/'
SNU_FILE = 'Human_TGN_f4_0.9_Source_20200625.csv'
ENTREZ = 'gene_info'

SNU_PPI = pd.read_csv(SNU_PATH+SNU_FILE) # 892,741
ENTREZ_ID = pd.read_csv(SNU_PATH+ENTREZ, sep = '\t')
SNU_PPI_IDS = set(list(SNU_PPI.X1) + list(SNU_PPI.X2)) 


check_tmp_G = nx.from_pandas_edgelist(SNU_PPI, 'X1', 'X2')
len(check_tmp_G.nodes()) # 19184
len(check_tmp_G.edges()) # 742630



# 앙뜨레즈 테이블은 landmark 를 잘 보필하는가 

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()

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


SNU_GENE_ORDER_mini = list(SNU_G_mini.nodes()) # 978
SNU_ADJ = nx.adjacency_matrix(SNU_G_mini)
SNU_ADJ_tmp = torch.LongTensor(SNU_ADJ.toarray())
SNU_ADJ_IDX = SNU_ADJ_tmp.to_sparse().indices()  # [2, 335]
SNU_WEIGHT_SCORE = [1 for a in range(SNU_ADJ_IDX.shape[1])]


new_node_name = []
for a in SNU_G_mini.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(SNU_G_mini.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(SNU_G_mini, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE




(7) HumanNet # 이거부터 해야함 
(7) HumanNet
(7) HumanNet

hunet_dir = '/home01/k006a01/01.DATA/HumanNet/'
hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'
hunet_gsp = pd.read_csv(hunet_dir+'HumanNet-GSP.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B']

hunet_fn = pd.read_csv(hunet_dir+'HumanNet-FN.tsv', sep = '\t', header = None)
hunet_fn.columns = ['G_A','G_B', 'score']


				fig = plt.figure(figsize=(10,8))
				n , bins, patches = plt.hist(x=list(hunet_fn['score']), bins = 100, alpha = 0.7, rwidth=0.8, log=True)
				maxfreq = n.max()
				plt.xlabel('val')
				plt.ylabel('num')
				# plt.ylim(0, np.ceil(maxfreq)+10000) 
				plt.xlim(min(list(hunet_fn['score'])), max(list(hunet_fn['score']))) # 일정한 scale
				plt.title('HumanNet fn Histogram')
				plt.grid(True)
				plt.tight_layout()
				fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/Hnet.fn.hist.png', bbox_inches = 'tight')


hunet_cx = pd.read_csv(hunet_dir+'HS-CX.tsv', sep = '\t', header = None)
hunet_cx.columns = ['G_A','G_B', 'score']

				fig = plt.figure(figsize=(10,8))
				n , bins, patches = plt.hist(x=list(hunet_cx['score']), bins = 100, alpha = 0.7, rwidth=0.8, log=True)
				maxfreq = n.max()
				plt.xlabel('val')
				plt.ylabel('num')
				# plt.ylim(0, np.ceil(maxfreq)+10000) 
				plt.xlim(min(list(hunet_cx['score'])), max(list(hunet_cx['score']))) # 일정한 scale
				plt.title('HumanNet cx Histogram')
				plt.grid(True)
				plt.tight_layout()
				fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/Hnet.cx.hist.png', bbox_inches = 'tight')



hunet_path = pd.read_csv(hunet_dir+'HS-DB.tsv', sep = '\t', header = None)
hunet_path.columns = ['G_A','G_B', 'score']

				fig = plt.figure(figsize=(10,8))
				n , bins, patches = plt.hist(x=list(hunet_path['score']), bins = 100, alpha = 0.7, rwidth=0.8, log=True)
				maxfreq = n.max()
				plt.xlabel('val')
				plt.ylabel('num')
				# plt.ylim(0, np.ceil(maxfreq)+10000) 
				plt.xlim(min(list(hunet_path['score'])), max(list(hunet_path['score']))) # 일정한 scale
				plt.title('HumanNet path Histogram')
				plt.grid(True)
				plt.tight_layout()
				fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/Hnet.path.hist.png', bbox_inches = 'tight')





check_tmp_G = nx.from_pandas_edgelist(hunet_gsp, 'G_A', 'G_B')
len(check_tmp_G.nodes()) # 8779
len(check_tmp_G.edges()) # 259932

check_tmp_G = nx.from_pandas_edgelist(hunet_fn, 'G_A', 'G_B')
len(check_tmp_G.nodes()) # 18459
len(check_tmp_G.edges()) # 977495

check_tmp_G = nx.from_pandas_edgelist(hunet_cx, 'G_A', 'G_B')
len(check_tmp_G.nodes()) # 12180
len(check_tmp_G.edges()) # 81064

check_tmp_G = nx.from_pandas_edgelist(hunet_path, 'G_A', 'G_B')
len(check_tmp_G.nodes()) # 8540
len(check_tmp_G.edges()) # 135327

L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')

L_matching_list = pd.read_csv('/st06/jiyeonH/13.DD_SESS/ideker/L_12_string.csv', sep = '\t')


	1) gsp 
	1) gsp

hnet_IAS_L1 = hunet_gsp[hunet_gsp['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 611
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 3871
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE


	2) hunet_fn
	2) hunet_fn

hnet_IAS_L1 = hunet_fn[hunet_fn['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 947
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 9300
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 18600]
ID_WEIGHT = [] # len : 9300 -> 18600


ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['G_A','G_B']
ID_ADJ_IDX_T['NodeA'] = [str(list(ID_GENE_ORDER_mini)[A]) for A in list(ID_ADJ_IDX_T['G_A'])]
ID_ADJ_IDX_T['NodeB'] = [str(list(ID_GENE_ORDER_mini)[B]) for B in list(ID_ADJ_IDX_T['G_B'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = copy.deepcopy(hunet_fn)
IAS_FILTER['NodeA'] = [str(a) for a in list(IAS_FILTER['G_A'])]
IAS_FILTER['NodeB'] = [str(a) for a in list(IAS_FILTER['G_B'])]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['NodeA']+'__'+IAS_FILTER['NodeB']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['NodeB']+'__'+IAS_FILTER['NodeA']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'score']]
IAS_FILTER1.columns = ['NAMESUM', 'score']
IAS_FILTER2.columns = ['NAMESUM', 'score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0) 
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' ) # 40280
ID_WEIGHT_SCORE = list(ID_WEIGHT['score'])


new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE



	3) cc
	3) cc

hnet_IAS_L1 = hunet_cx[hunet_cx['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 20232

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 426
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 656
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 1312]
ID_WEIGHT = [] # len : 656 -> 1312


ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['G_A','G_B']
ID_ADJ_IDX_T['NodeA'] = [str(list(ID_GENE_ORDER_mini)[A]) for A in list(ID_ADJ_IDX_T['G_A'])]
ID_ADJ_IDX_T['NodeB'] = [str(list(ID_GENE_ORDER_mini)[B]) for B in list(ID_ADJ_IDX_T['G_B'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = copy.deepcopy(hunet_cx)
IAS_FILTER['NodeA'] = [str(a) for a in list(IAS_FILTER['G_A'])]
IAS_FILTER['NodeB'] = [str(a) for a in list(IAS_FILTER['G_B'])]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['NodeA']+'__'+IAS_FILTER['NodeB']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['NodeB']+'__'+IAS_FILTER['NodeA']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'score']]
IAS_FILTER1.columns = ['NAMESUM', 'score']
IAS_FILTER2.columns = ['NAMESUM', 'score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0) 
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' ) # 40280
ID_WEIGHT_SCORE = list(ID_WEIGHT['score'])



new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE




	4) path
	4) path

hnet_IAS_L1 = hunet_path[hunet_path['G_A'].isin(L_matching_list.entrez)]
hnet_IAS_L2 = hnet_IAS_L1[hnet_IAS_L1['G_B'].isin(L_matching_list.entrez)] # 1278

len(set(list(hnet_IAS_L2['G_A']) + list(hnet_IAS_L2['G_B']))) # 554
ID_G = nx.from_pandas_edgelist(hnet_IAS_L2, 'G_A', 'G_B')

ESSEN_NAMES = list(set(L_matching_list['entrez']))

MSSNG = [a for a in ESSEN_NAMES if a not in list(ID_G.nodes)]
# MSSNG = L_matching_list[L_matching_list['PPI_name'].isin(list(ID_G.nodes))==False]['PPI_name']
for nn in list(MSSNG):
	ID_G.add_node(nn)

# edge 1312
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 2556]
ID_WEIGHT = [] # len : 1312 -> 2556


ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['G_A','G_B']
ID_ADJ_IDX_T['NodeA'] = [str(list(ID_GENE_ORDER_mini)[A]) for A in list(ID_ADJ_IDX_T['G_A'])]
ID_ADJ_IDX_T['NodeB'] = [str(list(ID_GENE_ORDER_mini)[B]) for B in list(ID_ADJ_IDX_T['G_B'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = copy.deepcopy(hunet_cx)
IAS_FILTER['NodeA'] = [str(a) for a in list(IAS_FILTER['G_A'])]
IAS_FILTER['NodeB'] = [str(a) for a in list(IAS_FILTER['G_B'])]
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['NodeA']+'__'+IAS_FILTER['NodeB']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['NodeB']+'__'+IAS_FILTER['NodeA']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'score']]
IAS_FILTER1.columns = ['NAMESUM', 'score']
IAS_FILTER2.columns = ['NAMESUM', 'score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0) 
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' ) # 40280
ID_WEIGHT_SCORE = list(ID_WEIGHT['score'])



new_node_name = []
for a in ID_G.nodes():
	tmp_name = list(set(L_matching_list[L_matching_list.entrez == a ]['L_gene_symbol'])) # 6118
	new_node_name = new_node_name + tmp_name

mapping = {list(ID_G.nodes())[a]:new_node_name[a] for a in range(len(new_node_name))}

ID_G_RE = nx.relabel_nodes(ID_G, mapping)

MY_G = ID_G_RE 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE







(8) string # 얘가 좀 복잡할것 같은데 다른애들 해보고 진행하는게 어때 
(8) string
(8) string

PPI_N_PATH = '/home01/k006a01/01.DATA/STRING/'

PPI_N_PATH = '/st06/jiyeonH/00.STRING_v.11.5/'
PPI_11_5_raw = pd.read_csv(PPI_N_PATH+'9606.protein.links.v11.5.txt', sep = ' ')
PPI_11_5_info = pd.read_csv(PPI_N_PATH+'9606.protein.info.v11.5.txt', sep = '\t')
PPI_alias_info = pd.read_csv(PPI_N_PATH+'9606.protein.aliases.v11.5.txt', sep = '\t')

check_tmp_G = nx.from_pandas_edgelist(PPI_11_5_raw, 'protein1', 'protein2')
len(check_tmp_G.nodes()) # 19385
len(check_tmp_G.edges()) # 5969249


fig = plt.figure(figsize=(10,8))
n , bins, patches = plt.hist(x=list(PPI_11_5_raw.combined_score), bins = 100, alpha = 0.7, rwidth=0.8, log=True)
maxfreq = n.max()
plt.xlabel('val')
plt.ylabel('num')
# plt.ylim(0, np.ceil(maxfreq)+10000) 
plt.xlim(min(list(PPI_11_5_raw.combined_score)), max(list(PPI_11_5_raw.combined_score))) # 일정한 scale
plt.title('string Histogram')
plt.grid(True)
plt.tight_layout()
fig.savefig('/st06/jiyeonH/11.TOX/MY_TRIAL_6/t.3_4.ppi_exp/string.cx.hist.png', bbox_inches = 'tight')





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
PPI_result_4 = PPI_result_3[['preferred_name_x','preferred_name_y']].drop_duplicates()

string_G = nx.from_pandas_edgelist(PPI_result_4, 'preferred_name_x', 'preferred_name_y')

# edge 51879
ID_GENE_ORDER_mini = list(string_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(string_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 103758]
ID_WEIGHT = [] # len : 51879 -> 103758


ID_ADJ_IDX_T = pd.DataFrame(ID_ADJ_IDX.T.detach().tolist())
ID_ADJ_IDX_T.columns = ['G_A','G_B']
ID_ADJ_IDX_T['NodeA'] = [str(list(ID_GENE_ORDER_mini)[A]) for A in list(ID_ADJ_IDX_T['G_A'])]
ID_ADJ_IDX_T['NodeB'] = [str(list(ID_GENE_ORDER_mini)[B]) for B in list(ID_ADJ_IDX_T['G_B'])]
ID_ADJ_IDX_T['NAMESUM'] = ID_ADJ_IDX_T['NodeA'] +'__'+ID_ADJ_IDX_T['NodeB']

IAS_FILTER = copy.deepcopy(PPI_result_3)
IAS_FILTER['NAMESUM_1'] = IAS_FILTER['preferred_name_x']+'__'+IAS_FILTER['preferred_name_y']
IAS_FILTER['NAMESUM_2'] = IAS_FILTER['preferred_name_y']+'__'+IAS_FILTER['preferred_name_x']
IAS_FILTER = IAS_FILTER[['NAMESUM_1','NAMESUM_2','combined_score']]
IAS_FILTER1= IAS_FILTER[['NAMESUM_1', 'combined_score']]
IAS_FILTER2= IAS_FILTER[['NAMESUM_2', 'combined_score']]
IAS_FILTER1.columns = ['NAMESUM', 'combined_score']
IAS_FILTER2.columns = ['NAMESUM', 'combined_score']
IAS_FILTER = pd.concat([IAS_FILTER1, IAS_FILTER2],axis = 0) 
IAS_FILTER = IAS_FILTER.drop_duplicates()

ID_WEIGHT = pd.merge(ID_ADJ_IDX_T, IAS_FILTER, on = 'NAMESUM', how = 'left' ) # 40280
ID_WEIGHT_SCORE = list(ID_WEIGHT['combined_score'])



MY_G = string_G 
MY_WEIGHT_SCORE = ID_WEIGHT_SCORE # SCORE









(7) GeneMania # 얘가 좀 복잡할것 같은데 다른애들 해보고 진행하는게 어때 
(7) GeneMania
(7) GeneMania
