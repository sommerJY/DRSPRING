
# 주어진 환경 맞춰서 5CV만 돌릴거라서 수정버전 
# 다른 ML 방법에 맞춰서 만들어주기 
def prepare_data_GCN(CELL_NAME, A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS,
MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2,
MY_syn_RE2, norm ) :
        # 
        # CV_num = 0
        # train_key = 'CV{}_train'.format(CV_num)
        # test_key = 'CV{}_test'.format(CV_num)
        # 
        #
        ABCS_tv = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname!=CELL_NAME]
        ABCS_test = A_B_C_S_SET_SM[A_B_C_S_SET_SM.DC_cellname==CELL_NAME]
        #
        tv_ind = list(ABCS_tv.index)
        random.shuffle(tv_ind)
        test_ind = list(ABCS_test.index)
        # 
        chem_feat_A_tv = drug_A_arrS[tv_ind]; chem_feat_A_test = drug_A_arrS[test_ind]
        chem_feat_B_tv = drug_B_arrS[tv_ind]; chem_feat_B_test = drug_B_arrS[test_ind]
        gene_A_tv = MY_g_EXP_A_RE2[tv_ind];  gene_A_test = MY_g_EXP_A_RE2[test_ind]
        gene_B_tv = MY_g_EXP_B_RE2[tv_ind];  gene_B_test = MY_g_EXP_B_RE2[test_ind]
        target_A_tv = MY_Target_A2[tv_ind];  target_A_test = MY_Target_A2[test_ind]
        target_B_tv = MY_Target_B2[tv_ind];  target_B_test = MY_Target_B2[test_ind]
        cell_basal_tv = MY_CellBase_RE2[tv_ind];  cell_basal_test = MY_CellBase_RE2[test_ind]
        syn_tv = MY_syn_RE2[tv_ind];  syn_test = MY_syn_RE2[test_ind]
        #
        tv_data = {}
        test_data = {}
        #
        tv_data['drug1_feat'] = torch.concat([chem_feat_A_tv, chem_feat_B_tv], axis = 0)
        test_data['drug1_feat'] = chem_feat_A_test
        #
        tv_data['drug2_feat'] = torch.concat([chem_feat_B_tv, chem_feat_A_tv], axis = 0)
        test_data['drug2_feat'] = chem_feat_B_test
        #
        tv_data['GENE_A'] = torch.concat([gene_A_tv, gene_B_tv], axis = 0)
        test_data['GENE_A'] = gene_A_test
        #
        tv_data['GENE_B'] = torch.concat([gene_B_tv, gene_A_tv], axis = 0)
        test_data['GENE_B'] = gene_B_test
        #
        tv_data['TARGET_A'] = torch.concat([target_A_tv, target_B_tv], axis = 0)
        test_data['TARGET_A'] = target_A_test
        #
        tv_data['TARGET_B'] = torch.concat([target_B_tv, target_A_tv], axis = 0)
        test_data['TARGET_B'] = target_B_test
        #
        tv_data['cell_BASAL'] = torch.concat((cell_basal_tv, cell_basal_tv), axis=0)
        test_data['cell_BASAL'] = cell_basal_test
        #
        tv_data['Merged_features'] = torch.concat(([tv_data['drug1_feat'], tv_data['drug2_feat'], tv_data['GENE_A'], tv_data['GENE_B'], tv_data['TARGET_A'], tv_data['TARGET_B'], tv_data['cell_BASAL'] ]), axis=1)
        test_data['Merged_features'] = torch.concat(([test_data['drug1_feat'], test_data['drug2_feat'], test_data['GENE_A'], test_data['GENE_B'], test_data['TARGET_A'], test_data['TARGET_B'], test_data['cell_BASAL'] ]), axis=1)
        #            
        tv_data['y'] = torch.concat((syn_tv, syn_tv), axis=0)
        test_data['y'] = syn_test
        #
        print(tv_data['drug1_feat'].shape, flush=True)
        print(test_data['drug1_feat'].shape, flush=True)
        return tv_data, test_data




# CV_0
train_data_0, test_data_0 = prepare_data_GCN('T-47D', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_1, test_data_1 = prepare_data_GCN('RKO', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_2, test_data_2 = prepare_data_GCN('ES2', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_3, test_data_3 = prepare_data_GCN('RPMI7951', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_4, test_data_4 = prepare_data_GCN('NCIH520', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_5, test_data_5 = prepare_data_GCN('MSTO', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_6, test_data_6 = prepare_data_GCN('NCIH2122', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_7, test_data_7 = prepare_data_GCN('MDAMB436', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_8, test_data_8 = prepare_data_GCN('OV90', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_9, test_data_9 = prepare_data_GCN('KPL1', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_10, test_data_10 = prepare_data_GCN('HT144', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_11, test_data_11 = prepare_data_GCN('A375', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_12, test_data_12 = prepare_data_GCN('PA1', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_13, test_data_13 = prepare_data_GCN('CAOV3', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_14, test_data_14 = prepare_data_GCN('OVCAR3', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_15, test_data_15 = prepare_data_GCN('LOVO', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_16, test_data_16 = prepare_data_GCN('NCIH1650', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_17, test_data_17 = prepare_data_GCN('A427', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_18, test_data_18 = prepare_data_GCN('VCAP', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_19, test_data_19 = prepare_data_GCN('NCI-H460', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_20, test_data_20 = prepare_data_GCN('SK-OV-3', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_21, test_data_21 = prepare_data_GCN('DLD1', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_22, test_data_22 = prepare_data_GCN('A2058', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_23, test_data_23 = prepare_data_GCN('SW837', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_24, test_data_24 = prepare_data_GCN('SKMES1', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_25, test_data_25 = prepare_data_GCN('UWB1289', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )

train_data_26, test_data_26 = prepare_data_GCN('HCT116', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_27, test_data_27 = prepare_data_GCN('A2780', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_28, test_data_28 = prepare_data_GCN('ZR751', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_29, test_data_29 = prepare_data_GCN('UACC62', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_30, test_data_30 = prepare_data_GCN('SW-620', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_31, test_data_31 = prepare_data_GCN('NCIH23', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_32, test_data_32 = prepare_data_GCN('SKMEL30', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )
train_data_33, test_data_33 = prepare_data_GCN('HT29', A_B_C_S_SET_SM, drug_A_arrS, drug_B_arrS, MY_g_EXP_A_RE2, MY_g_EXP_B_RE2, MY_Target_A2, MY_Target_B2, MY_CellBase_RE2, MY_syn_RE2, norm )






def train_cv (CVnum, train_data_0, test_data_0) :
    X_train, X_test, y_train, y_test = np.array(train_data['Merged_features']), np.array(test_data['Merged_features']), np.array(train_data['y']), np.array(test_data['y'])
    #
    y_train = np.ravel(y_train)
    y_test = np.ravel(y_test)
    #
    param_grid = {
        'n_estimators': [128, 256, 512, 1024],
        'max_features': [round(np.sqrt(X_train.shape[1])), 128, 256], # 
    }
    #
    grid_search = GridSearchCV(RandomForestRegressor(),
                            param_grid=param_grid, verbose = 2, error_score='raise')
    grid_search.fit(X_train, y_train)
    print(grid_search.best_estimator_)
    #
    gs_df = pd.DataFrame(columns = ['n_estimators', 'max_features', 'CV0','CV1','CV2','CV3','CV4', 'test_mean', 'test_std'])
    #
    gs_df['test_mean'] = list(grid_search.cv_results_['mean_test_score'])
    gs_df['test_std'] = list(grid_search.cv_results_['std_test_score'])
    gs_df['n_estimators'] = list(grid_search.cv_results_['param_n_estimators'])
    gs_df['max_features'] = list(grid_search.cv_results_['param_max_features'])
    gs_df['CV0'] = list(grid_search.cv_results_['split0_test_score'])
    gs_df['CV1'] = list(grid_search.cv_results_['split1_test_score'])
    gs_df['CV2'] = list(grid_search.cv_results_['split2_test_score'])
    gs_df['CV3'] = list(grid_search.cv_results_['split3_test_score'])
    gs_df['CV4'] = list(grid_search.cv_results_['split4_test_score'])
    #
    with open('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'+'RF.W2.CV{}.pkl'.format(CVnum),'wb') as f:
        pickle.dump(grid_search,f)
    #
    gs_df.to_csv('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'+'RF.W2.CV{}.csv'.format(CVnum), index = False)
    return grid_search, gs_df


gs_model_0, gs_df_0 = train_cv(0,train_data_0, test_data_0)
gs_model_1, gs_df_1 = train_cv(1,train_data_1, test_data_1)
gs_model_2, gs_df_2 = train_cv(2,train_data_2, test_data_2)
gs_model_3, gs_df_3 = train_cv(3,train_data_3, test_data_3)
gs_model_4, gs_df_4 = train_cv(4,train_data_4, test_data_4)
gs_model_5, gs_df_5 = train_cv(5,train_data_5, test_data_5)
gs_model_6, gs_df_6 = train_cv(6,train_data_6, test_data_6)
gs_model_7, gs_df_7 = train_cv(7,train_data_7, test_data_7)
gs_model_8, gs_df_8 = train_cv(8,train_data_8, test_data_8)
gs_model_9, gs_df_9 = train_cv(9,train_data_9, test_data_9)

gs_model_10, gs_df_10 = train_cv(10, train_data_10, test_data_10)
gs_model_11, gs_df_11 = train_cv(11, train_data_11, test_data_11)
gs_model_12, gs_df_12 = train_cv(12, train_data_12, test_data_12)
gs_model_13, gs_df_13 = train_cv(13, train_data_13, test_data_13)
gs_model_14, gs_df_14 = train_cv(14, train_data_14, test_data_14)
gs_model_15, gs_df_15 = train_cv(15, train_data_15, test_data_15)
gs_model_16, gs_df_16 = train_cv(16, train_data_16, test_data_16)
gs_model_17, gs_df_17 = train_cv(17, train_data_17, test_data_17)
gs_model_18, gs_df_18 = train_cv(18, train_data_18, test_data_18)
gs_model_19, gs_df_19 = train_cv(19, train_data_19, test_data_19)

gs_model_20, gs_df_20 = train_cv(20, train_data_20, test_data_20)
gs_model_21, gs_df_21 = train_cv(21, train_data_21, test_data_21)
gs_model_22, gs_df_22 = train_cv(22, train_data_22, test_data_22)
gs_model_23, gs_df_23 = train_cv(23, train_data_23, test_data_23)
gs_model_24, gs_df_24 = train_cv(24, train_data_24, test_data_24)
gs_model_25, gs_df_25 = train_cv(25, train_data_25, test_data_25)
gs_model_26, gs_df_26 = train_cv(26, train_data_26, test_data_26)
gs_model_27, gs_df_27 = train_cv(27, train_data_27, test_data_27)
gs_model_28, gs_df_28 = train_cv(28, train_data_28, test_data_28)
gs_model_29, gs_df_29 = train_cv(29, train_data_29, test_data_29)

gs_model_30, gs_df_30 = train_cv(30, train_data_30, test_data_30)
gs_model_31, gs_df_31 = train_cv(31, train_data_31, test_data_31)
gs_model_32, gs_df_32 = train_cv(32, train_data_32, test_data_32)
gs_model_33, gs_df_33 = train_cv(33, train_data_33, test_data_33)



with open('/st06/jiyeonH/11.TOX/DR_SPRING/ml_result/RandomForest/'+'RF_W2_CV0.pkl','wb') as f:
    pickle.dump(clf,f)