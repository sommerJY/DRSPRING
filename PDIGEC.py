import torch
import pandas as pd
from torch import nn
import numpy as np
import random
from datetime import datetime
import argparse
from utils import model_modtf1

# python PDIGEC.py /home/minK/ssse/final/final_data/  --early_stopping 'es'


parser = argparse.ArgumentParser()
parser.add_argument("save_file", type=str, default='./results', help='path to save')          # extra value
parser.add_argument("--mode", type=str, help='select mode', default='train')          # extra value
parser.add_argument("--saved_model", type=str, help='saved model location')          # extra value
parser.add_argument("--early_stopping", type=str, default=None)           #
parser.add_argument("--drug_cell", type=str, default=None, help='put drug-cell file here')          # extra value
parser.add_argument("--smiles", type=str, default=None,help='put smiles file here')          # extra value
parser.add_argument("--basal", type=str, default='M1_lincs_wth_ccle_org_all.csv')           # existence/nonexistence
parser.add_argument("--jobname", type=str, default='M1_result')           # test all

args = parser.parse_args()
# python PDIGEC.py /home/minK/ssse/final/final_data/ \
# --mode 'new_data' --saved_model /home/minK/ssse/final/final_data/model_tvt.pt \
# --drug_cell 'new_drug_cellline.csv' \
# --smiles 'new_drug_0815.csv'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# print(args.drug_cell)

# FuTrainh3dttfHSDB_L3Dataset
if args.mode=='train':
    from utils import modtf1_dataset
    train_data = modtf1_dataset.FuTrainh3dttfHSDB_L3Dataset(root='./')
    print("train_data", len(train_data))
    dev_data = modtf1_dataset.FuDevh3dttfHSDB_L3Dataset(root='./')
    print("dev_data", len(dev_data))
    test_data = modtf1_dataset.FuTesth3dttfHSDB_L3Dataset(root='./')
    print("test_data", len(test_data))
elif args.mode=='new_data':
    if args.drug_cell == None:
        print("DRUG_CELL FILE MISSING")
    if args.smiles == None:
        print("SMILES FILE MISSING")
    from utils import modtf1_dataset
    test_data = modtf1_dataset.FuNewh3dttfHSDB_L3Dataset(root='./', new_drug_cellline = args.drug_cell, new_smiles = args.smiles, new_basal = args.basal)
    print("test_data", len(test_data))
elif args.mode=='new_data_cellline_all':
    if args.smiles == None:
        print("SMILES FILE MISSING")
    from utils import modtf1_dataset
    test_data = modtf1_dataset.FuNewallh3dttfHSDB_L3Dataset(root='./', new_smiles = args.smiles, new_basal = args.basal)
    print("test_data", len(test_data))



start_time = datetime.now()


max_epoch = int(1000)
learning_rate = float(0.004)
drop_pert = float(0.1)
hid_dim = int(32)
ghid_dim = int(16)
b_size = int(64)
layer_count = 2
glayer_count = 3
hidden_layer_size = 64
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



if args.mode=='train':
    model_modtf1.training_model([train_data,dev_data,test_data], args.save_file, device, args.early_stopping, args.jobname,
                   max_epoch,learning_rate,drop_pert,hid_dim,ghid_dim,b_size,layer_count,glayer_count,hidden_layer_size)
    print("finish training and saved model")
elif args.mode=='new_data' or args.mode=='new_data_cellline_all':
    model_modtf1.testing_model(test_data, args.save_file, args.saved_model, args.basal, args.jobname, device,learning_rate,
                  drop_pert,hid_dim,ghid_dim,b_size,layer_count,glayer_count,hidden_layer_size)
    print("finish testing")



end_time = datetime.now()
print(end_time)
print(end_time - start_time)
