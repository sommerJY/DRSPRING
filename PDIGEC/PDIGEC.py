import torch
import pandas as pd
from torch import nn
import numpy as np
import random
from datetime import datetime
import argparse
from model_mod1 import training_model, testing_model

# python PDIGEC.py /home/minK/ssse/final/final_data/  --early_stopping 'es'


parser = argparse.ArgumentParser()
parser.add_argument("save_file", type=str, help='path to save')          # extra value
parser.add_argument("--mode", type=str, help='select mode', default='train')          # extra value
parser.add_argument("--saved_model", type=str, help='saved model location')          # extra value
parser.add_argument("--early_stopping", type=str, default=None)           #
parser.add_argument("--drug_cell", type=str, default=None, help='put drug-cell file here')          # extra value
parser.add_argument("--smiles", type=str, default=None,help='put smiles file here')          # extra value
parser.add_argument("--basal", type=str, default='lincs_wth_ccle_org_all.csv')           # existence/nonexistence

args = parser.parse_args()
# python PDIGEC.py /home/minK/ssse/final/final_data/ \
# --mode 'new_data' --saved_model /home/minK/ssse/final/final_data/model_tvt.pt \
# --drug_cell 'new_drug_cellline.csv' \
# --smiles 'new_drug_0815.csv'


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
# print(args.drug_cell)


if args.mode=='train':
    from mod1_dataset import FuTrainh3dtHSDB_L3Dataset, FuDevh3dtHSDB_L3Dataset,FuTesth3dtHSDB_L3Dataset
    train_data = FuTrainh3dtHSDB_L3Dataset(root='./final_data')
    print("train_data", len(train_data))
    dev_data = FuDevh3dtHSDB_L3Dataset(root='./final_data')
    print("dev_data", len(dev_data))
    test_data = FuTesth3dtHSDB_L3Dataset(root='./final_data')
    print("test_data", len(test_data))
elif args.mode=='new_data':
    from mod1_dataset import FuNewh3dtHSDB_L3Dataset
    test_data = FuNewh3dtHSDB_L3Dataset(root='./final_data', new_drug_cellline = args.drug_cell, new_smiles = args.smiles, new_basal = args.basal)
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
    training_model([train_data,dev_data,test_data], args.save_file, device, args.early_stopping,
                   max_epoch,learning_rate,drop_pert,hid_dim,ghid_dim,b_size,layer_count,glayer_count,hidden_layer_size)
    print("finish training and saved model")
elif args.mode=='new_data':
    testing_model(test_data, args.save_file, args.saved_model, args.basal, device,learning_rate,
                  drop_pert,hid_dim,ghid_dim,b_size,layer_count,glayer_count,hidden_layer_size)
    print("finish testing")



end_time = datetime.now()
print(end_time)
print(end_time - start_time)
