time & dose 매칭 시킨 결과에 대한 애들 가져오려고 
20230614 에 짠거



# R ! 
library(data.table)
GSEPATH = '/st06/jiyeonH/11.TOX/LINCS'

GCT_BIG_BETA.v1 = fread(paste0(GSEPATH,"/L_2020/BETA_1.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  18959 
GCT_BIG_BETA.v2 = fread(paste0(GSEPATH,"/L_2020/BETA_2.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  50272
GCT_BIG_BETA.v3 = fread(paste0(GSEPATH,"/L_2020/BETA_3.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  23107
GCT_BIG_BETA.v4 = fread(paste0(GSEPATH,"/L_2020/BETA_4.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  5038
GCT_BIG_BETA.v5 = fread(paste0(GSEPATH,"/L_2020/BETA_5.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  23898
GCT_BIG_BETA.v6 = fread(paste0(GSEPATH,"/L_2020/BETA_6.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  7137
GCT_BIG_BETA.v7 = fread(paste0(GSEPATH,"/L_2020/BETA_7.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  6820
GCT_BIG_BETA.v8 = fread(paste0(GSEPATH,"/L_2020/BETA_8.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  1237
GCT_BIG_BETA.con = fread(paste0(GSEPATH,"/L_2020/level5_beta_ctl_n58022x12328.gct"), header = FALSE, sep = '\t', skip = 2, nThread=16) 
# [1] 12329  40570




sig_for_DS = read.csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_FILTER.20230614.csv',  stringsAsFactors=FALSE)


for (NUM in c(1:8)){ # 1, 3,4, 5,8
	print(NUM)
	BETA = get(paste0('GCT_BIG_BETA.v',NUM))
	DF_NAME = paste0("GCT_BIG_BETA.v",NUM,".DF")
	FI_NAME = paste0("GCT_BIG_BETA.TOTAL_",NUM)
	DF = data.frame(BETA)
	FI = DF[,c(1, which(DF[1,] %in% sig_for_DS$sig_id))]
	assign(DF_NAME, DF )
	assign(FI_NAME, FI )
	#colnames(FI)[1] = 'pr_gene_id'
	print(dim(DF))
	print(dim(FI))
	#LIST = list(DF,FI )
}



BETA = GCT_BIG_BETA.con
DF_NAME = paste0("GCT_BIG_BETA.v","Con",".DF")
FI_NAME = paste0("GCT_BIG_BETA.TOTAL_","Con")
DF = data.frame(BETA)
FI = DF[,c(1, which(DF[1,] %in% sig_for_DS$sig_id))] # 43953
assign(DF_NAME, DF )
assign(FI_NAME, FI )
colnames(FI)[1] = 'pr_gene_id'
print(FI[1:3,1:3])
print(dim(DF))
print(dim(FI))


BETA_BIND = cbind(GCT_BIG_BETA.TOTAL_1,
	GCT_BIG_BETA.TOTAL_2[,1:ncol(GCT_BIG_BETA.TOTAL_2)],
	GCT_BIG_BETA.TOTAL_3[,1:ncol(GCT_BIG_BETA.TOTAL_3)],
	GCT_BIG_BETA.TOTAL_4[,1:ncol(GCT_BIG_BETA.TOTAL_4)],
	GCT_BIG_BETA.TOTAL_5[,1:ncol(GCT_BIG_BETA.TOTAL_5)],
	GCT_BIG_BETA.TOTAL_6[,1:ncol(GCT_BIG_BETA.TOTAL_6)],
	GCT_BIG_BETA.TOTAL_7[,1:ncol(GCT_BIG_BETA.TOTAL_7)],
	GCT_BIG_BETA.TOTAL_8[,1:ncol(GCT_BIG_BETA.TOTAL_8)],
	GCT_BIG_BETA.TOTAL_Con[,1:ncol(GCT_BIG_BETA.TOTAL_Con)]
	)

BETA_BIND_2 = BETA_BIND[,c(1, which(BETA_BIND[1,] %in% sig_for_DS$sig_id))]
dim(BETA_BIND_2) # 128078


my_sigs = sig_for_DS$sig_id # 181240 # untrt 때문에 차이 남 

new_sigs = BETA_BIND[1,1:180421]
new_sigs2 = as.vector(unlist(new_sigs))
sig_check = my_sigs[which(my_sigs %in% new_sigs2 == FALSE)] 
sig_check = my_sigs[which(my_sigs %in% new_sigs2 == FALSE)]

colnames(BETA_BIND_2)[2:ncol(BETA_BIND_2)] = paste0("COL_",2:ncol(BETA_BIND_2))

# write.csv(BETA_BIND_2, "/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_20230614.csv" )

landmark = read.csv(paste0(GSEPATH,'/L_2020/geneinfo_beta.txt'), sep = '\t' , stringsAsFactors=FALSE)
landmark_lm = landmark[landmark$feature_space == 'landmark' ,]
BETA_BIND_3 = BETA_BIND_2[BETA_BIND_2$V1 %in% landmark_lm$gene_id,]
colnames(BETA_BIND_3) = BETA_BIND_2[1,]

# write.csv(BETA_BIND_3, "/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_20230614_978.csv" )







############################################################


BETA_BIND = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/'+"BETA_DATA_for_20220705_978.csv")
BETA_CP_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'compoundinfo_beta.txt') # pert 34419 , cansmiles : 
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
BETA_SIG_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'siginfo_beta.txt', low_memory = False)



하던거 : 
tb01 에서는 일단 

시간줄이기를 위해 아예 BETA_BIND 를 우선 torch 화 시키기로 함 


문제는 node 순서를 고정시켜놔야한다는거. 


# HS 다른 pathway 사용 
print('NETWORK')
# HUMANNET 사용 


hunet_dir = '/st06/jiyeonH/13.DD_SESS/HumanNetV3/'

hunet_gsp = pd.read_csv(hunet_dir+'HS-DB.tsv', sep = '\t', header = None)
hunet_gsp.columns = ['G_A','G_B','SC']

BETA_GENE = pd.read_table('/st06/jiyeonH/11.TOX/LINCS/L_2020/geneinfo_beta.txt')
BETA_lm_genes = BETA_GENE[BETA_GENE.feature_space=='landmark'] # 978
BETA_lm_genes = BETA_lm_genes.reset_index()
lm_entrezs = list(BETA_lm_genes.gene_id)

hnet_L1 = hunet_gsp[hunet_gsp['G_A'].isin(BETA_lm_genes.gene_id)]
hnet_L2 = hnet_L1[hnet_L1['G_B'].isin(BETA_lm_genes.gene_id)] # 3885
hnet_L3 = hnet_L2[hnet_L2.SC >= 3.5]


len(set(list(hnet_L3['G_A']) + list(hnet_L3['G_B']))) # 611

ID_G = nx.from_pandas_edgelist(hnet_L3, 'G_A', 'G_B')

# MSSNG = [a for a in lm_entrezs if a not in list(ID_G.nodes)]

#for nn in list(MSSNG):
#	ID_G.add_node(nn)

# edge 
ID_GENE_ORDER_mini = list(ID_G.nodes()) # 978
ID_ADJ = nx.adjacency_matrix(ID_G)
ID_ADJ_tmp = torch.LongTensor(ID_ADJ.toarray())
ID_ADJ_IDX = ID_ADJ_tmp.to_sparse().indices()  # [2, 7742]
ID_WEIGHT = [] # len : 3871 -> 7742

# 원래는 edge score 있지만 일단은...
ID_WEIGHT_SCORE = [1 for a in range(ID_ADJ_IDX.shape[1])]


# 유전자 이름으로 붙이기 
LINCS_PATH = '/st06/jiyeonH/11.TOX/MY_TRIAL_5/' 
LINCS_gene_file = pd.read_csv(LINCS_PATH+'geneinfo_beta.txt', sep = '\t')
LINCS_978 = LINCS_gene_file[LINCS_gene_file.feature_space == 'landmark']
LINCS_978 = LINCS_978[['gene_id','gene_symbol']]
LINCS_978['new_node'] = [str(list(LINCS_978.gene_id)[i]) + "__" + list(LINCS_978.gene_symbol)[i] for i in range(978)]
LINCS_978 = LINCS_978.reset_index(drop=True)


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




BETA_ORDER_pre = [list(LINCS_978.new_node).index(a) for a in JY_GRAPH_ORDER]
BETA_ORDER_DF = LINCS_978.iloc[BETA_ORDER_pre] # 어차피 ref 다르고 같은 애들이라 괜춘 
BETA_ENTREZ_ORDER = list(BETA_ORDER_DF.gene_id)
BETA_SYMBOL_ORDER = list(BETA_ORDER_DF.gene_symbol)
BETA_NEWNOD_ORDER = list(BETA_ORDER_DF.new_node)



def get_LINCS_data(DRUG_SIG):
	Drug_EXP = BETA_BIND[['id',DRUG_SIG]]
	BIND_ORDER =[list(BETA_BIND.id).index(a) for a in BETA_ORDER_DF.gene_id]
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	#
	return list(Drug_EXP_ORD[DRUG_SIG])



BETA_BIND_SIG_DF = BETA_SIG_info[BETA_SIG_info.sig_id.isin(BETA_BIND.columns)]
BETA_BIND_SIG_DF = BETA_BIND_SIG_DF.reset_index(drop=True)


BETA_BIND_SIG_RE = torch.empty(size=(BETA_BIND_SIG_DF.shape[0], 349, 1))##############


for IND in range(BETA_BIND_SIG_DF.shape[0]) : 
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(BETA_BIND_SIG_DF.shape[0]) )
		datetime.now()
	#
	sig = BETA_BIND_SIG_DF.at[IND, 'sig_id']
	res = get_LINCS_data(sig)
	res2 = torch.Tensor(res).unsqueeze(1)
	BETA_BIND_SIG_RE[IND] = res2


LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
torch.save(BETA_BIND_SIG_RE, LINCS_ALL_PATH+'BETA_BIND.349.pt')

BETA_BIND_SIG_DF.to_csv(LINCS_ALL_PATH+'BETA_BIND.349.siglist.csv', index = False)


지금 int2 에서는 그래서



보니까 AOBO 에서는 확실히 Gene exp 가 영향이 좋았다는걸로 나오는데, 
진짜 expression 한번만 더 바꿔서 해보고 
만약에 했는데도 별로면
진짜 LINCS 버릴 생각을 잠깐 해봐야할듯
그냥 GCN 으로 네트워크 기반의 뭔가가 좋았다 
이런얘기로 끝내는게 낫지 않을까 

난 진짜 이거 여름 안에 끝낼거임 
진심으로오오옥 



added 


BETA_BIND_ADD = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/BETA_DATA_for_20230614_978.csv')

tmp = list(BETA_BIND_ADD.columns)
re_col = [a for a in tmp if ':' in a]
re_col= list(set(re_col) - set(BETA_BIND.columns))
re_col = ['id'] + re_col

BETA_BIND_ADDre = BETA_BIND_ADD[re_col]



BETA_BIND_SIG_DF = BETA_SIG_info[BETA_SIG_info.sig_id.isin(BETA_BIND_ADDre.columns)]
BETA_BIND_SIG_DF = BETA_BIND_SIG_DF.reset_index(drop=True)


BETA_BIND_SIG_RE = torch.empty(size=(BETA_BIND_SIG_DF.shape[0], 349, 1))##############


def get_LINCS_data(DRUG_SIG):
	Drug_EXP = BETA_BIND_ADDre[['id',DRUG_SIG]]
	BIND_ORDER =[list(BETA_BIND_ADDre.id).index(a) for a in BETA_ORDER_DF.gene_id]
	Drug_EXP_ORD = Drug_EXP.iloc[BIND_ORDER]
	#
	return list(Drug_EXP_ORD[DRUG_SIG])



for IND in range(BETA_BIND_SIG_DF.shape[0]) :
	if IND%100 == 0 : 
		print(str(IND)+'/'+str(BETA_BIND_SIG_DF.shape[0]) )
		datetime.now()
	#
	sig = BETA_BIND_SIG_DF.at[IND, 'sig_id']
	res = get_LINCS_data(sig)
	res2 = torch.Tensor(res).unsqueeze(1)
	BETA_BIND_SIG_RE[IND] = res2


LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
torch.save(BETA_BIND_SIG_RE, LINCS_ALL_PATH+'BETA_BIND2.349.pt')

BETA_BIND_SIG_DF.to_csv(LINCS_ALL_PATH+'BETA_BIND2.349.siglist.csv', index = False)





BETA_BIND_ORI = torch.load(LINCS_ALL_PATH+'BETA_BIND.349.pt')
BETA_BIND_ORI_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND.349.siglist.csv')
BETA_BIND_NEW = torch.load(LINCS_ALL_PATH+'BETA_BIND2.349.pt')
BETA_BIND_NEW_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND2.349.siglist.csv')








생각을 해보니 CID 별로 sig mean 만드는거 그냥 민지처럼 미리 만들어두면 되는거 아님? 
그래서 그냥 mean 만들어둔걸 조합만 하면 되게 하는게 나을것 같음 

BETA_BIND = torch.concat([BETA_BIND_ORI, BETA_BIND_NEW])
BETA_BIND_SIG_df = pd.concat([BETA_BIND_ORI_DF, BETA_BIND_NEW_DF])
BETA_BIND_SIG_df = BETA_BIND_SIG_df.reset_index(drop = True)

LINCS_PERT_MATCH = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')
LINCS_PERT_MATCH = LINCS_PERT_MATCH[['pert_id','CID']]

BETA_BIND_SIG_df_CID = pd.merge(BETA_BIND_SIG_df, LINCS_PERT_MATCH, on = 'pert_id', how = 'left')

BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID[BETA_BIND_SIG_df_CID.CID>0]
BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID2[BETA_BIND_SIG_df_CID2.pert_idose == '10 uM']
BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID2[BETA_BIND_SIG_df_CID2.pert_itime == '24 h']

BETA_BIND_SIG_df_CID3 = BETA_BIND_SIG_df_CID2[['pert_idose','pert_itime','CID','cell_iname','sig_id']]
BETA_BIND_SIG_df_CID3['long_id'] = BETA_BIND_SIG_df_CID3.CID.apply(lambda x : str(int(x))) + '___' + BETA_BIND_SIG_df_CID3.cell_iname




id_list = list(set(BETA_BIND_SIG_df_CID3.long_id))
id_list.sort()

Mean_exp = []

for ind in range(len(id_list)) : 
    if ind % 100 == 0 : 
        print( '{} / {}'.format(ind, len(id_list)) )
        datetime.now()
    #
    lid = id_list[ind]
    sig_ids = list(BETA_BIND_SIG_df_CID3[BETA_BIND_SIG_df_CID3.long_id==lid]['sig_id'])
    sig_ind = list(BETA_BIND_SIG_df[BETA_BIND_SIG_df.sig_id.isin(sig_ids)].index)
    EXP = torch.mean(torch.concat( [BETA_BIND[sig_iii] for sig_iii in sig_ind] , axis = 1), axis = 1).view(-1,1)
    Mean_exp.append(EXP)
    

                    52912189
                    A375


Mean_exp_expand = [a.unsqueeze(0) for a in Mean_exp]

Mean_exp_t = torch.concat(Mean_exp_expand)

torch.save( Mean_exp_t, LINCS_ALL_PATH + "10_24_sig_cell_mean.pt")


BETA_BIND_SIG_df_CID4 = pd.DataFrame({'long_id' : id_list})
BETA_BIND_SIG_df_CID4['stripped_cell_line_name'] = BETA_BIND_SIG_df_CID4.long_id.apply(lambda x : x.split('___')[1])
BETA_BIND_SIG_df_CID4['CID'] = BETA_BIND_SIG_df_CID4.long_id.apply(lambda x : x.split('___')[0])


ccle_info = pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)
ccle_info_filt = ccle_info[['stripped_cell_line_name','CCLE_Name']]
BETA_BIND_SIG_df_CID5 = pd.merge(BETA_BIND_SIG_df_CID4, ccle_info_filt, on = 'stripped_cell_line_name', how='left')


BETA_BIND_SIG_df_CID5.to_csv(LINCS_ALL_PATH + '10_24_sig_cell_mean.csv', index = False)












#################
nonononono
mj 0.3 filtering sig_id removal 


MJ_lv4_cor = pd.read_csv('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/lb_level4_corr.csv', low_memory = False)

MJ_lv4_cor_re = MJ_lv4_cor[['sig_id','corr']]

MJ_lv4_cor_re2 = pd.DataFrame(MJ_lv4_cor_re.groupby('sig_id').mean().reset_index())

MJ_lv4_cor_re3 = MJ_lv4_cor_re2[MJ_lv4_cor_re2['corr']>0.3]
MJ_lv4_cor_re3_not = MJ_lv4_cor_re2[MJ_lv4_cor_re2['corr']<0.3]
MJ_lv4_cor_re3_not_sig = list(MJ_lv4_cor_re3_not['sig_id'])





LINCS_ALL_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/'
BETA_BIND_ORI = torch.load(LINCS_ALL_PATH+'BETA_BIND.349.pt')
BETA_BIND_ORI_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND.349.siglist.csv')
BETA_BIND_NEW = torch.load(LINCS_ALL_PATH+'BETA_BIND2.349.pt')
BETA_BIND_NEW_DF = pd.read_csv(LINCS_ALL_PATH+'BETA_BIND2.349.siglist.csv')




BETA_BIND = torch.concat([BETA_BIND_ORI, BETA_BIND_NEW])
BETA_BIND_SIG_df = pd.concat([BETA_BIND_ORI_DF, BETA_BIND_NEW_DF])
BETA_BIND_SIG_df = BETA_BIND_SIG_df.reset_index(drop = True)

LINCS_PERT_MATCH = pd.read_csv('/st06/jiyeonH/13.DD_SESS/01.PRJ2/LINCS_PERT_CID_MATCH.1207.csv', sep ='\t')
LINCS_PERT_MATCH = LINCS_PERT_MATCH[['pert_id','CID']]

BETA_BIND_SIG_df_CID = pd.merge(BETA_BIND_SIG_df, LINCS_PERT_MATCH, on = 'pert_id', how = 'left')

BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID[BETA_BIND_SIG_df_CID.CID>0]
BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID2[BETA_BIND_SIG_df_CID2.pert_idose == '10 uM']
BETA_BIND_SIG_df_CID2 = BETA_BIND_SIG_df_CID2[BETA_BIND_SIG_df_CID2.pert_itime == '24 h']

BETA_BIND_SIG_df_CID3 = BETA_BIND_SIG_df_CID2[['pert_idose','pert_itime','CID','cell_iname','sig_id']]
BETA_BIND_SIG_df_CID3['long_id'] = BETA_BIND_SIG_df_CID3.CID.apply(lambda x : str(int(x))) + '___' + BETA_BIND_SIG_df_CID3.cell_iname

BETA_BIND_SIG_df_CID3 = BETA_BIND_SIG_df_CID3[BETA_BIND_SIG_df_CID3.sig_id.isin(MJ_lv4_cor_re3_not.sig_id)]




id_list = list(set(BETA_BIND_SIG_df_CID3.long_id))
id_list.sort()

Mean_exp = []

for ind in range(len(id_list)) : 
    if ind % 100 == 0 : 
        print( '{} / {}'.format(ind, len(id_list)) )
        datetime.now()
    #
    lid = id_list[ind]
    sig_ids = list(BETA_BIND_SIG_df_CID3[BETA_BIND_SIG_df_CID3.long_id==lid]['sig_id'])
    sig_ind = list(BETA_BIND_SIG_df[BETA_BIND_SIG_df.sig_id.isin(sig_ids)].index)
    EXP = torch.mean(torch.concat( [BETA_BIND[sig_iii] for sig_iii in sig_ind] , axis = 1), axis = 1).view(-1,1)
    Mean_exp.append(EXP)
    

                    52912189
                    A375


Mean_exp_expand = [a.unsqueeze(0) for a in Mean_exp]

Mean_exp_t = torch.concat(Mean_exp_expand)

torch.save( Mean_exp_t, LINCS_ALL_PATH + "10_24_sig_cell_mean_0.3.pt")


BETA_BIND_SIG_df_CID4 = pd.DataFrame({'long_id' : id_list})
BETA_BIND_SIG_df_CID4['stripped_cell_line_name'] = BETA_BIND_SIG_df_CID4.long_id.apply(lambda x : x.split('___')[1])
BETA_BIND_SIG_df_CID4['CID'] = BETA_BIND_SIG_df_CID4.long_id.apply(lambda x : x.split('___')[0])

CCLE_PATH = '/st06/jiyeonH/13.DD_SESS/CCLE.22Q1/'

ccle_info = pd.read_csv(CCLE_PATH+'sample_info.csv', low_memory=False)
ccle_info_filt = ccle_info[['stripped_cell_line_name','CCLE_Name']]
BETA_BIND_SIG_df_CID5 = pd.merge(BETA_BIND_SIG_df_CID4, ccle_info_filt, on = 'stripped_cell_line_name', how='left')


BETA_BIND_SIG_df_CID5.to_csv(LINCS_ALL_PATH + '10_24_sig_cell_mean_0.3.csv', index = False)




