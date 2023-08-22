import pandas as pd
import numpy as np

#####early stopping can be used
class EarlyStopping:
    def __init__(self, patience=200):
        self.loss = np.inf
        self.patience = 0
        self.patience_limit = patience
    def step(self, loss):
        if self.loss > loss:
            self.loss = loss
            self.patience = 0
        else:
            self.patience += 1
    def is_stop(self):
        return self.patience >= self.patience_limit


def after_process(info_dfpc,predict_df):
    pred_np_withinfo = pd.concat([info_dfpc,predict_df],axis=1)
    pred_np_withinfo['pert_id'] = pred_np_withinfo['pert_id'].astype(int).astype(str)
    pred_np_withinfo['entrez'] = pred_np_withinfo['pert_id'].astype(int).astype(str)+"__"+pred_np_withinfo['ccle_name']
    pred_np_withinfos = pred_np_withinfo.drop(columns = ['pert_id', 'ccle_name'])
    pred_np_info_tp = pred_np_withinfos.set_index('entrez').T
    pred_np_info_tp['CID__CELL'] = 'lv5_exp'
    pred_np_info_tp.index.name = 'entrez_id'
    pred_np_info_tp.reset_index(inplace=True)
    pred_np_processed = pd.concat([pred_np_info_tp.iloc[:,(len(pred_np_info_tp.columns)-1):(len(pred_np_info_tp.columns))],pred_np_info_tp.iloc[:,:(len(pred_np_info_tp.columns)-1)]],axis = 1)
    return pred_np_processed
