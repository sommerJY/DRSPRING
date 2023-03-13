
prepare same cid-cid-cell data 


python main.py --comb-data-name 'data/DrugCombinationData.tsv' --cell_line-gex 'data/cell_line_gex.csv' 
--drug1-chemicals 'data/drug1_chem.csv' --drug2-chemicals 'data/drug2_chem.csv'
--test-ind 'data/train_inds.txt'
--saved-model-name matchmaker_saved.h5 --train-test-mode 0


맞게 데이터 가공이 필요함... 그래서 일단 모델 input 에 맞게 우리껄 바꾸려고 
input 모양새를 찾아보고 있었음... 
cell line basal expression 도 같이 넣어주고 
우리 drug 관련도 chemopy 이용해서 넣어줄 수 있을것 같았는데 흠 


cell =pd.read_csv('cell_line_gex.csv', header = None) # cell line expression feature 그러나 순서를 모르겠음 
drug1 = pd.read_csv('drug1_chem.csv', header = None) 
drug2 = pd.read_csv('drug2_chem.csv', header = None)
train_inds = pd.read_csv('train_inds.txt', header = None)
# 
RMA_exp = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/Cell_line_RMA_proc_basalExp.txt', sep = '\t') # 오리지널 RMA 내용 
IDK_PATH = '/st06/jiyeonH/13.DD_SESS/ideker/' 
L_matching_list = pd.read_csv(IDK_PATH+'L_12_string.csv', sep = '\t')
HGNC_ENTREZ = pd.read_csv('/st06/jiyeonH/11.TOX/matchmaker-master/HGNC_ENTREZ.2022.10.06.csv', sep = '\t')
HGNC_ENTREZ['HGNC_ID'] = [int(a.split(':')[1]) for a in list(HGNC_ENTREZ['HGNC ID'])]
FILT = HGNC_ENTREZ[HGNC_ENTREZ['NCBI Gene ID'].isin(L_matching_list.entrez)] # 972
아니 왜 안맞아 돌겠네 



# columns 
CELL_info = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/CELLs.1007.csv', sep = '\t')
BETA_CEL_info = pd.read_table('/st06/jiyeonH/13.DD_SESS/LINCS_BETA/'+'cellinfo_beta.txt')
DEPMAP_info = pd.read_csv('/st06/jiyeonH/11.TOX/MY_TRIAL_5/01.Trial_Matchmaker/'+'DEPMAP_sample_info.csv')

RMA_COL_1 = list(RMA_exp.columns)
RMA_COL_2 = [int(a.split('.')[1].split('.')[0]) for a in RMA_COL_1[2:]]
RMA_COL_3 = RMA_COL_1[0:2] + RMA_COL_2


cos_check = [a.replace('-','_') for a in CELL_info['Sample Name']]
CELL_info['re_name'] = cos_check
CELL_info[CELL_info.re_name.isin(list(BETA_CEL_info.cell_iname))]

DEP_in_RMA = DEPMAP_info[DEPMAP_info.COSMICID.isin(RMA_COL_2)]
DEP_notin_RMA = DEPMAP_info[-DEPMAP_info.COSMICID.isin(RMA_COL_2)]
RMA_notin_DEP = [a for a in set(RMA_COL_2) if a not in list(DEPMAP_info.COSMICID) ]




for cos in RMA_COL_2 : 
    cell_name = CELL_info[CELL_info['COSMIC identifier']==906794]['Sample Name']



RMA_exp['source'] =  [a.split('[')[1].split(']')[0].split(';')[0] if type(a) == str else 0 for a in list(RMA_exp.GENE_title)]
RMA_exp['IDs'] =  [a.split('[')[1].split(']')[0].split(';')[1] if type(a) == str else 0 for a in list(RMA_exp.GENE_title)]


RMA_HGNC = RMA_exp[RMA_exp.source=='Source:HGNC Symbol']
RMA_HGNC['ID'] = [float(a.split(':')[1]) for a in list(RMA_HGNC['IDs'])]
tmp1 = RMA_HGNC[RMA_HGNC.ID.isin(list(HGNC_ENTREZ['HGNC_ID']))] # 17419
tmp2 = RMA_HGNC[RMA_HGNC.ID.isin(list(FILT['HGNC_ID']))] # 921







# cell_line_gex.csv : (286420, 972) -> GDSC 에서 가져온 cell basal exp 라는데. 일단 왜 972 냐고 
chemopy 
# drug1_chem.csv : (286420, 541) -> chemopyu 가지고 만든 drug 구조 
# drug2_chem.csv : (286420, 541) -> 
# train_inds : (171852, 1) -> 그냥 input 데이터 index 좌락 넣어주면 될듯 




train_data, val_data, test_data = MatchMaker.prepare_data(chem1, chem2, cell_line, synergies, norm,
                                            args.train_ind, args.val_ind, args.test_ind)

