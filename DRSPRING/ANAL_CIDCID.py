
# cid-cid wise 가 궁금해짐 




WORK_NAME = 'WORK_403' # 349
W_NAME = 'W403'
PRJ_NAME = 'M3V8'
MJ_NAME = 'M3V8'
MISS_NAME = 'MIS2'
PPI_NAME = '349'

PRJ_PATH = '/st06/jiyeonH/11.TOX/DR_SPRING/trials/{}_{}_{}_{}/'.format(PRJ_NAME, W_NAME, PPI_NAME, MISS_NAME)

ABCS_test_result = pd.read_csv(PRJ_PATH+'ABCS_test_result.csv')


ABCS_test_result['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(ABCS_test_result['CELL'])]
DC_CELL_info_filt['tissue'] = [ '_'.join(a.split('_')[1:]) if type(a) == str else 'NA' for a in list(DC_CELL_info_filt['CELL'])]


test_cidcid_df = pd.DataFrame({'CID_CID' : list(set(ABCS_test_result.CID_CID))})


cc_P = []
cc_S = []
cc_num = []
cc_tiss = []

for cc in list(test_cidcid_df.CID_CID) :
	tmp_test_re = ABCS_test_result[ABCS_test_result.CID_CID == cc]
	if tmp_test_re.shape[0] > 1 : 
		cc_P_corr, _ = stats.pearsonr(tmp_test_re.ANS, tmp_test_re.FULL)
		cc_S_corr, _ = stats.spearmanr(tmp_test_re.ANS, tmp_test_re.FULL)
		cc_nums = tmp_test_re.shape[0]
	else : 
		cc_P_corr = 0
		cc_S_corr = 0
		cc_nums = 1
	cc_P.append(cc_P_corr)
	cc_S.append(cc_S_corr)
	cc_num.append(cc_nums)


for cc in list(test_cidcid_df.CID_CID) :
	tmp_test_re = ABCS_test_result[ABCS_test_result.CID_CID == cc]
	tislen = len(set(tmp_test_re.tissue))
	cc_tiss.append(tislen)



test_cidcid_df['P_COR'] = cc_P
test_cidcid_df['S_COR'] = cc_S
test_cidcid_df['cid_num'] = cc_num
test_cidcid_df['tissue_num'] = cc_tiss

test_cidcid_df.to_csv(PRJ_PATH+'CID_COR.csv') # 16987


test_cidcid_df2 = test_cidcid_df[test_cidcid_df.cid_num >= 5] # 9824
test_cidcid_df2


ABCS_test_result[ABCS_test_result.CID_CID == '5278396___24856436']


test_cidcid_df3 = test_cidcid_df[test_cidcid_df.cid_num >= 30] # 9824
test_cidcid_df3
test_cidcid_df3.sort_values('P_COR')

ABCS_test_result[ABCS_test_result.CID_CID == '5278396___24856436']
ABCS_test_result[ABCS_test_result.CID_CID == '3657___42611257']



test_cidcid_df4 = test_cidcid_df3[test_cidcid_df3.P_COR>0.8]



근데 이걸 뭐라고 해석해야함..?
뭐라고 나타내야할지 약간 의문


test_cidcid_df5 = test_cidcid_df[test_cidcid_df.tissue_num>1] # 4709
test_cidcid_df6 = test_cidcid_df5[test_cidcid_df5.cid_num>=5] # 4611
test_cidcid_df6.sort_values('P_COR')

ABCS_test_result[ABCS_test_result.CID_CID == '25126798___46215815']



