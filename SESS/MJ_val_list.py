# go # mj's validation list 




clinical_cell = set(["CVCL_0023","CVCL_0479","CVCL_0031","CVCL_2760","CVCL_0023","CVCL_0479","CVCL_0031","CVCL_2760","CVCL_0023","CVCL_0479","CVCL_0031","CVCL_2760","CVCL_0023","CVCL_0479","CVCL_0031","CVCL_2760","CVCL_0023","CVCL_0479","CVCL_0031","CVCL_2760","CVCL_0320","CVCL_B8AQ","CVCL_2478","CVCL_0320","CVCL_0547","CVCL_0014"])

MJ_VAL_DF = pd.read_csv('/st06/jiyeonH/11.TOX/DR_SPRING/trials/FINAL_VALIDATION/val_real_examples_CSV.CSV')


check_clinical = MJ_VAL_DF[MJ_VAL_DF.CELL_CELLO.isin(avail_cell_list)]
check_clinical = check_clinical.reset_index(drop = True)

check_tuple = [(check_clinical['DRUG_A_CID'][i],check_clinical['DRUG_B_CID'][i], check_clinical['CELL_CELLO'][i]) for i in range(check_clinical.shape[0])]

check_clinical['tuple'] = check_tuple


PRED_list = []

# ROW_CID, COL_CID, CELLO = lst_test[0]

for IND in range(len(check_tuple)) :
ROW_CID, COL_CID, CELLO = check_tuple[IND]
ROW_CID, COL_CID, CELLO = check_tuple[-3] # 0.8454
ROW_CID, COL_CID, CELLO = check_tuple[-2] # -6.7915
ROW_CID, COL_CID, CELLO = check_tuple[-1] # -6.7915 


TUP_1 = (str(int(ROW_CID)), CELLO)
TUP_2 = (str(int(COL_CID)), CELLO)
#
if (TUP_1 in TPs_all) & (TUP_2 in TPs_all) : 
k=1
drug1_f, DrugA_ADJ = get_CHEM(ROW_CID, k)
drug1_a = torch.Tensor(DrugA_ADJ).long().to_sparse().indices()
#
drug2_f, DrugB_ADJ = get_CHEM(COL_CID, k)
drug2_a = torch.Tensor(DrugB_ADJ).long().to_sparse().indices()
#
expA = check_exp_f_ts(ROW_CID, CELLO)
expB = check_exp_f_ts(COL_CID, CELLO)
#
adj = copy.deepcopy(JY_ADJ_IDX).long()
adj_w = torch.Tensor(JY_IDX_WEIGHT).squeeze()
#
cell = check_cell_oh(CELLO)
cell = cell.unsqueeze(0)
	#
	#
best_model.eval()
with torch.no_grad():
	y = torch.Tensor([1]).float().unsqueeze(1)
	output= best_model(torch.Tensor(drug1_f), torch.Tensor(drug2_f), drug1_a, drug2_a, expA, expB, adj, adj_w, cell, y) 

outputs = [output.squeeze().item()]
	print(outputs)
	PRED_list = PRED_list+outputs
else :
	PRED_list = PRED_list+ ['NA']
	#


	tmp_df = pd.DataFrame({
		'ROW_CID' : [ROW_CID],
		'COL_CID' : [COL_CID],
		'CELLO' : [CELLO],
		'PRED_RES' : outputs,
		})
	CELL_PRED_DF = pd.concat([CELL_PRED_DF, tmp_df])






'/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/drugbank_interaction5.1.9.csv' 에 저장 
키워드는 cid2_inter3 = cid2_inter2[cid2_inter2.description.str.contains("therapeutic efficacy")]


dbnk_interaction = pd.read_csv('/st06/jiyeonH/13.DD_SESS/DrugBank.5.1.9/drugbank_interaction5.1.9.csv')
# 아 근데 진짜 cell line 이 제약인게 문제가 되긴 하는구나 


