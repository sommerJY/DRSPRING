
# oneil 따로 learning 할거 만들어줘야함 

DC_DRUG_DF2 = DC_DRUG_DF_FULL[['id_re','dname','cid','CAN_SMILES']] # puibchem 공식 smiles 
																	# dnames = list(DC_DRUG_DF2['dname']) # 그래 이부분때문에 다시 다운로드 받은거였음. 이게 문제가 drug 마다 CID 가 여러개 붙는 애들이 있어서



# sumary info 
summ =  pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugComb.1.5/summary_v_1_5.csv', low_memory=False)
summ2 = summ[['block_id','study_name']]
summ3 = summ2[summ2.study_name=='ONEIL']

Oneil_block = DC_tmp_DF1[DC_tmp_DF1.block_id.isin(summ3.block_id)]
Oneil_block_1 = Oneil_block[['drug_row_id', 'drug_col_id', 'cell_line_id','synergy_loewe','quality']]
# -> 92208
Oneil_block_2 = pd.merge(Oneil_block_1, DC_CELL_DF2 , on= 'cell_line_id', how = 'left') # 92208


DC_DRUG_DF2.columns = ['drug_row_id','drug_row','drug_row_CID', 'drug_row_sm']
Oneil_block_3 = pd.merge(Oneil_block_2, DC_DRUG_DF2 , on= 'drug_row_id', how = 'left') # 92208

DC_DRUG_DF2.columns = ['drug_col_id','drug_col','drug_col_CID', 'drug_col_sm']
Oneil_block_4 = pd.merge(Oneil_block_3, DC_DRUG_DF2 , on= 'drug_col_id', how = 'left') # 92208


# CID 체크 
set([int(a) for a in Oneil_block_4.drug_row_CID]) # 문제되는 애가 없음 
set([int(a) for a in Oneil_block_4.drug_col_CID]) # 문제되는 애가 없음 

Oneil_block_5 = Oneil_block_4[Oneil_block_4.DrugCombCCLE!='NA']

'NCIH520_LUNG'
'OCUBM_BREAST'
'NCIH23_LUNG'
'PA1_OVARY'
'SKMES1_LUNG'
'UACC62_SKIN'
'SKOV3_OVARY'
'RPMI7951_SKIN'
'NCIH460_LUNG'
'LOVO_LARGE_INTESTINE'
'HT29_LARGE_INTESTINE'
'MDAMB436_BREAST'
'OV90_OVARY'
'KPL1_BREAST'
'VCAP_PROSTATE'
'HT144_SKIN'
'A427_LUNG'
'CAOV3_OVARY'
'ZR751_BREAST'
'UWB1289_OVARY'
'HCT116_LARGE_INTESTINE'
'SW620_LARGE_INTESTINE'
'NIHOVCAR3_OVARY'
'RKO_LARGE_INTESTINE'
'MSTO211H_PLEURA'
'A2058_SKIN'
'T47D_BREAST'
'ES2_OVARY'
'NCIH2122_LUNG'
'SW837_LARGE_INTESTINE'
'A375_SKIN'
'A2780_OVARY'
'NCIH1650_LUNG'
'SKMEL30_SKIN'
'DLD1_LARGE_INTESTINE'

# 35개 cell line 

Oneil_block_6 = Oneil_block_5[['drug_row_CID','drug_col_CID','DrugCombCCLE','synergy_loewe']]


Oneil_block_6[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates()
# 20405 A_B_C

Oneil_cell = list(set(Oneil_block_6.DrugCombCCLE))
# 35개 cell line 

Oneil_CID = list(set(list(set(Oneil_block_6.drug_row_CID)) + list(set(Oneil_block_6.drug_col_CID))))
# 38개 drugs 


Oneil_block_6.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/DC_ONEIL_ALL.csv', sep= '\t', index = False)



이중에 내 데이터에서 빠지는 애들 : 
long_leng = [5311497.0, 13342.0, 36314.0, 11520894.0]

DC_DRUG_DF_FULL[DC_DRUG_DF_FULL.cid==5311497].T # 57 
DC_DRUG_DF_FULL[DC_DRUG_DF_FULL.cid==13342].T # 59
DC_DRUG_DF_FULL[DC_DRUG_DF_FULL.cid==36314].T # 62
DC_DRUG_DF_FULL[DC_DRUG_DF_FULL.cid==11520894].T # 69


Oneil_block_7 = Oneil_block_6[Oneil_block_6.drug_row_CID.isin(long_leng)==False] # 73332
Oneil_block_8 = Oneil_block_7[Oneil_block_7.drug_col_CID.isin(long_leng)==False] # 68556

Oneil_cell_filt = list(set(Oneil_block_8.DrugCombCCLE)) # 35개 
Oneil_CID_filt = list(set(list(set(Oneil_block_8.drug_row_CID)) + list(set(Oneil_block_8.drug_col_CID))))

Oneil_block_8[['drug_row_CID','drug_col_CID','DrugCombCCLE']].drop_duplicates()
# 16905

이중에서 synergy 갈리는 애들도 빠질거임 아마도..? 

