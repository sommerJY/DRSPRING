import pandas as pd
import torch
from torch_geometric.data import Data, InMemoryDataset
from rdkit import Chem, RDLogger
import numpy as np
import uuid

def under_number(data, maxNumAtoms):
    tf = list()
    for i in data['smiles']:
        iMol = Chem.MolFromSmiles(i.strip())
        #Adj
        try:
            iAdjTmp = Chem.rdmolops.GetAdjacencyMatrix(iMol)
            if( iAdjTmp.shape[0] <= maxNumAtoms):
                tf.append("T")
            else:
                tf.append("F")
        except:
            tf.append("error")
    data["tf"] = tf
    data2 = data[data.tf=="T"].reset_index(drop=True).drop('tf', axis=1)
    return data2




def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))



def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))




###################################
# '''with dose time data'''
# ##############################

class PairData_DT(Data):
    def __init__(self, x_s=None, edge_index_s=None, y=None, smiles=None, x_p=None, \
        edge_index_p=None, dose = None, time = None, pert_id=None, cell_id=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.y = y
        self.edge_index_p = edge_index_p
        self.x_p = x_p
        self.smiles = smiles
        self.pert_id = pert_id
        self.cell_id = cell_id
        self.dose = dose
        self.time = time
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_p':
            return self.x_p.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)



def from_gene_to_node_HSDB_L3(df_full_ccle, hu_gene_lm2):
    genes = df_full_ccle.iloc[:,5:354].columns.to_list()
    mapping = {gene: i for i, gene in enumerate(genes)}
    humannet_lm_a = hu_gene_lm2.copy()
    hu_gene_lm2.columns = ['G_B','G_A','score']
    humannet_lm_b = hu_gene_lm2[['G_A','G_B','score']]
    humannet_lm_all = pd.concat([humannet_lm_a,humannet_lm_b]) #bi-directional 
    humannet_lm_all2 = humannet_lm_all.drop_duplicates().reset_index(drop=True,inplace=False)  # unique
    geneid1 = [mapping[geneid] for geneid in humannet_lm_all2['G_A']]
    geneid2 = [mapping[geneid] for geneid in humannet_lm_all2['G_B']]
    gene_edge = torch.tensor([(geneid1), (geneid2)]) 
    return gene_edge





def from_smiles_transfer_HSDB_L3(smiles: str, expressions, maxNumAtoms, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data':
    #
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles('')
    #
    af_list = []
    af_list64 = np.zeros((maxNumAtoms, 64))
    for atom in mol.GetAtoms():
        atom_feature = (one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Cl', 'P',
        'F', 'Br', 'I','Na', 'Fe', 'B', 'Mg', 'Al', 'Si', 'K', 'H', 'Se', 'Ca','Zn', 
        'As', 'Mo', 'V', 'Cu', 'Hg', 'Cr', 'Co', 'Bi','Tc','Sb', 'Gd', 'Li', 'Ag', 'Au', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()]) 
        atom_feature = list(map(int, atom_feature))
        af_list.append(atom_feature)
    #
    af_list64[0:len(af_list), 0:64] = af_list ### 0 padding for feature-set
    x = torch.FloatTensor(af_list64).view(50, 64)#.device('cpu')
    #
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    #
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.int64).view(2, -1)#.device('cpu')
    #
    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort() #정렬
        edge_index = edge_index[:, perm]
    #
    exp_list = expressions.to_list()
    y = torch.FloatTensor(exp_list).view(-1, 349)
    return x,edge_index,y,smiles





################    
class FuTrainh3dttfHSDB_L3Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(FuTrainh3dttfHSDB_L3Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])    #
    #
    @property
    def raw_file_names(self):
        return ['train_tf.csv', 'lb_drug_to_smiles_cid.csv','HSDB_network.csv','lincs_wth_ccle_org_all.csv','target_lm349_all.csv']
    #
    @property
    def processed_file_names(self):
        return ['train_tf_hsdb.pt']
    #
    def process(self):
        data_list = []
        maxNumAtoms = 50
        data_df = pd.read_csv(self.raw_paths[0])
        drug_df = pd.read_csv(self.raw_paths[1],sep = '\t',index_col=0)
        drug_df[['CID']] = drug_df[['CID']].astype('int').astype('str')
        drug_df.columns = ['pert_id', 'smiles', 'CID']
        hu_gene_lm2 = pd.read_csv(self.raw_paths[2], index_col=0, dtype='str')
        lincs_wth_ccle_org_all = pd.read_csv(self.raw_paths[3], index_col=0)
        lincs_wth_ccle_org_all2 = lincs_wth_ccle_org_all.rename(columns={'cell_iname' : 'cell_id'})
        target_lm349_all = pd.read_csv(self.raw_paths[4], sep ='\t')
        target_lm349_all.CID = target_lm349_all.CID.astype("str")
        df_full = pd.merge(drug_df, data_df, how='inner')
        df_full_num = under_number(df_full, maxNumAtoms)
        lincs_wth_ccle_org_all2_filt = lincs_wth_ccle_org_all2[['cell_id']+df_full_num.iloc[:,7:356].columns.to_list()]
        df_full_num_filt = df_full_num[df_full_num.cell_id.isin(lincs_wth_ccle_org_all2_filt.cell_id)].reset_index(drop=True,inplace=False)  # 2977
        df_full_ccle = pd.merge(df_full_num_filt[['sig_id', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']], lincs_wth_ccle_org_all2_filt,how='left', on = "cell_id")
        gene_edge = from_gene_to_node_HSDB_L3(df_full_ccle, hu_gene_lm2)
        df_full_target = pd.merge(df_full_num_filt[['pert_id', 'smiles', 'CID', 'sig_id', 'pert_type', 'cell_id','pert_idose','pert_itime']],target_lm349_all,how='left')
        df_full_target = df_full_target.fillna(0)
        for i in range(0,len(df_full_num_filt)): #len(df_full_num_filt)
            x_data,edge_data,y_data,s_data = from_smiles_transfer_HSDB_L3(df_full_num_filt['smiles'][i], df_full_num_filt.iloc[i,6:355], maxNumAtoms)
            df_full_ccle_lst = pd.DataFrame(df_full_ccle.iloc[i,5:354])
            df_full_ccle_lst.columns = ["exp"]
            # ppprint(df_full_ccle_lst)
            df_full_target_lst = pd.DataFrame(df_full_target.iloc[i,8:357])
            df_full_target_lst.columns = ["targets"]
            # ppprint(df_full_target_lst)
            ccle_target = pd.concat([df_full_ccle_lst,df_full_target_lst],axis=1)
            ccle_target[['exp','targets']] = ccle_target[['exp','targets']].astype('float')
            target_data = torch.FloatTensor(np.array(ccle_target)).view(349, 2) 
            d_data = torch.FloatTensor(np.array(df_full_num_filt['pert_idose'][i])).view(1, 1)
            t_data = torch.FloatTensor(np.array(df_full_num_filt['pert_itime'][i])).view(1, 1)
            data_list.append(PairData_DT(x_s=x_data, edge_index_s=edge_data, y=y_data, \
                smiles=s_data, x_p=target_data, edge_index_p=gene_edge, dose = d_data, time = t_data, \
                pert_id = df_full_num_filt['pert_id'][i], cell_id = df_full_num_filt['cell_id'][i]))
        data, slices = self.collate(data_list)
        self._data_list = None 
        torch.save((data, slices), self.processed_paths[0])






################    
class FuDevh3dttfHSDB_L3Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(FuDevh3dttfHSDB_L3Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])    #
    #
    @property
    def raw_file_names(self):
        return ['dev_tf.csv', 'lb_drug_to_smiles_cid.csv','HSDB_network.csv','lincs_wth_ccle_org_all.csv','target_lm349_all.csv']
    #
    @property
    def processed_file_names(self):
        return ['dev_tf_hsdb.pt']
    #
    def process(self):
        data_list = []
        maxNumAtoms = 50
        data_df = pd.read_csv(self.raw_paths[0])
        drug_df = pd.read_csv(self.raw_paths[1],sep = '\t',index_col=0)
        drug_df[['CID']] = drug_df[['CID']].astype('int').astype('str')
        drug_df.columns = ['pert_id', 'smiles', 'CID']
        hu_gene_lm2 = pd.read_csv(self.raw_paths[2], index_col=0, dtype='str')
        lincs_wth_ccle_org_all = pd.read_csv(self.raw_paths[3], index_col=0)
        lincs_wth_ccle_org_all2 = lincs_wth_ccle_org_all.rename(columns={'cell_iname' : 'cell_id'})
        target_lm349_all = pd.read_csv(self.raw_paths[4], sep ='\t')
        target_lm349_all.CID = target_lm349_all.CID.astype("str")
        df_full = pd.merge(drug_df, data_df, how='inner')
        df_full_num = under_number(df_full, maxNumAtoms)
        lincs_wth_ccle_org_all2_filt = lincs_wth_ccle_org_all2[['cell_id']+df_full_num.iloc[:,7:356].columns.to_list()]
        df_full_num_filt = df_full_num[df_full_num.cell_id.isin(lincs_wth_ccle_org_all2_filt.cell_id)].reset_index(drop=True,inplace=False)  # 2977
        df_full_ccle = pd.merge(df_full_num_filt[['sig_id', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']], lincs_wth_ccle_org_all2_filt,how='left', on = "cell_id")
        gene_edge = from_gene_to_node_HSDB_L3(df_full_ccle, hu_gene_lm2)
        df_full_target = pd.merge(df_full_num_filt[['pert_id', 'smiles', 'CID', 'sig_id', 'pert_type', 'cell_id','pert_idose','pert_itime']],target_lm349_all,how='left')
        df_full_target = df_full_target.fillna(0)
        for i in range(0,len(df_full_num_filt)): #len(df_full_num_filt)
            x_data,edge_data,y_data,s_data = from_smiles_transfer_HSDB_L3(df_full_num_filt['smiles'][i], df_full_num_filt.iloc[i,6:355], maxNumAtoms)
            df_full_ccle_lst = pd.DataFrame(df_full_ccle.iloc[i,5:354])
            df_full_ccle_lst.columns = ["exp"]
            df_full_target_lst = pd.DataFrame(df_full_target.iloc[i,8:357])
            df_full_target_lst.columns = ["targets"]
            ccle_target = pd.concat([df_full_ccle_lst,df_full_target_lst],axis=1)
            ccle_target[['exp','targets']] = ccle_target[['exp','targets']].astype('float')
            target_data = torch.FloatTensor(np.array(ccle_target)).view(349, 2) 
            d_data = torch.FloatTensor(np.array(df_full_num_filt['pert_idose'][i])).view(1, 1)
            t_data = torch.FloatTensor(np.array(df_full_num_filt['pert_itime'][i])).view(1, 1)
            data_list.append(PairData_DT(x_s=x_data, edge_index_s=edge_data, y=y_data, \
                smiles=s_data, x_p=target_data, edge_index_p=gene_edge, dose = d_data, time = t_data, \
                pert_id = df_full_num_filt['pert_id'][i], cell_id = df_full_num_filt['cell_id'][i]))
        data, slices = self.collate(data_list)
        self._data_list = None 
        torch.save((data, slices), self.processed_paths[0])






################    
class FuTesth3dttfHSDB_L3Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(FuTesth3dttfHSDB_L3Dataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])    #
    #
    @property
    def raw_file_names(self):
        return ['test_tf.csv', 'lb_drug_to_smiles_cid.csv','HSDB_network.csv','lincs_wth_ccle_org_all.csv','target_lm349_all.csv']
    #
    @property
    def processed_file_names(self):
        return ['test_tf_hsdb.pt']
    #
    def process(self):
        data_list = []
        maxNumAtoms = 50
        data_df = pd.read_csv(self.raw_paths[0])
        drug_df = pd.read_csv(self.raw_paths[1],sep = '\t',index_col=0)
        drug_df[['CID']] = drug_df[['CID']].astype('int').astype('str')
        drug_df.columns = ['pert_id', 'smiles', 'CID']
        hu_gene_lm2 = pd.read_csv(self.raw_paths[2], index_col=0, dtype='str')
        lincs_wth_ccle_org_all = pd.read_csv(self.raw_paths[3], index_col=0)
        lincs_wth_ccle_org_all2 = lincs_wth_ccle_org_all.rename(columns={'cell_iname' : 'cell_id'})
        target_lm349_all = pd.read_csv(self.raw_paths[4], sep ='\t')
        target_lm349_all.CID = target_lm349_all.CID.astype("str")
        df_full = pd.merge(drug_df, data_df, how='inner')
        df_full_num = under_number(df_full, maxNumAtoms)
        lincs_wth_ccle_org_all2_filt = lincs_wth_ccle_org_all2[['cell_id']+df_full_num.iloc[:,7:356].columns.to_list()]
        df_full_num_filt = df_full_num[df_full_num.cell_id.isin(lincs_wth_ccle_org_all2_filt.cell_id)].reset_index(drop=True,inplace=False)  # 2977
        df_full_ccle = pd.merge(df_full_num_filt[['sig_id', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']], lincs_wth_ccle_org_all2_filt,how='left', on = "cell_id")
        gene_edge = from_gene_to_node_HSDB_L3(df_full_ccle, hu_gene_lm2)
        df_full_target = pd.merge(df_full_num_filt[['pert_id', 'smiles', 'CID', 'sig_id', 'pert_type', 'cell_id','pert_idose','pert_itime']],target_lm349_all,how='left')
        df_full_target = df_full_target.fillna(0)
        for i in range(0,len(df_full_num_filt)): #len(df_full_num_filt)
            x_data,edge_data,y_data,s_data = from_smiles_transfer_HSDB_L3(df_full_num_filt['smiles'][i], df_full_num_filt.iloc[i,6:355], maxNumAtoms)
            df_full_ccle_lst = pd.DataFrame(df_full_ccle.iloc[i,5:354])
            df_full_ccle_lst.columns = ["exp"]
            df_full_target_lst = pd.DataFrame(df_full_target.iloc[i,8:357])
            df_full_target_lst.columns = ["targets"]
            ccle_target = pd.concat([df_full_ccle_lst,df_full_target_lst],axis=1)
            ccle_target[['exp','targets']] = ccle_target[['exp','targets']].astype('float')
            target_data = torch.FloatTensor(np.array(ccle_target)).view(349, 2) 
            d_data = torch.FloatTensor(np.array(df_full_num_filt['pert_idose'][i])).view(1, 1)
            t_data = torch.FloatTensor(np.array(df_full_num_filt['pert_itime'][i])).view(1, 1)
            data_list.append(PairData_DT(x_s=x_data, edge_index_s=edge_data, y=y_data, \
                smiles=s_data, x_p=target_data, edge_index_p=gene_edge, dose = d_data, time = t_data, \
                pert_id = df_full_num_filt['pert_id'][i], cell_id = df_full_num_filt['cell_id'][i]))
        data, slices = self.collate(data_list)
        self._data_list = None 
        torch.save((data, slices), self.processed_paths[0])






def from_newsmiles_transfer_HSDB_L3(smiles: str, maxNumAtoms, with_hydrogen: bool = False, kekulize: bool = False) -> 'torch_geometric.data.Data':
    #
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles('')
    #
    af_list = []
    af_list64 = np.zeros((maxNumAtoms, 64))
    for atom in mol.GetAtoms():
        atom_feature = (one_of_k_encoding_unk(atom.GetSymbol(),['C', 'N', 'O', 'S', 'F', 'Cl', 'P',
        'F', 'Br', 'I','Na', 'Fe', 'B', 'Mg', 'Al', 'Si', 'K', 'H', 'Se', 'Ca','Zn', 
        'As', 'Mo', 'V', 'Cu', 'Hg', 'Cr', 'Co', 'Bi','Tc','Sb', 'Gd', 'Li', 'Ag', 'Au', 'Unknown']) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7]) +
        one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4]) +
        one_of_k_encoding_unk(atom.GetFormalCharge(), [-3, -2, -1, 0, 1, 2, 3, 4, 5]) +
        [atom.GetIsAromatic()]) 
        atom_feature = list(map(int, atom_feature))
        af_list.append(atom_feature)
    #
    af_list64[0:len(af_list), 0:64] = af_list ### 0 padding for feature-set
    x = torch.FloatTensor(af_list64).view(50, 64)#.device('cpu')
    #
    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]
    #
    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.int64).view(2, -1)#.device('cpu')
    #
    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort() #정렬
        edge_index = edge_index[:, perm]
    #
    # exp_list = expressions.to_list()
    # y = torch.FloatTensor(exp_list).view(-1, 349)
    return x,edge_index,smiles






def ppprint(ddata):
    print(ddata)



################    
class FuNewh3dttfHSDB_L3Dataset(InMemoryDataset):
    def __init__(self, root,new_drug_cellline, new_smiles, new_basal): # 
        self.file_name = uuid.uuid4().hex.upper()[0:6]
        self.new_drug_cellline, self.new_smiles2, self.new_basal = new_drug_cellline, new_smiles, new_basal
        super(FuNewh3dttfHSDB_L3Dataset, self).__init__(root, new_drug_cellline, new_smiles, new_basal) #
        self.transform = None
        self.data, self.slices = torch.load(self.processed_paths[0])    #이거 안하면 여러개가 겹챠지지가 않아
    #
    @property
    def raw_file_names(self): 
        return  [self.new_drug_cellline, self.new_smiles2, 'HSDB_network.csv', 'ccle_lincs_convert.csv','target_lm349_all.csv', self.new_basal,'test.csv']
    #
    @property
    def processed_file_names(self):
        return [str(self.file_name+'.pt')] #
    #
    def process(self):
        data_list = []
        maxNumAtoms = 50
        data_df = pd.read_csv(self.raw_paths[0])
        data_df.pert_itime= data_df.pert_itime.astype(float)
        data_df[['sig_id','pert_type']] = np.NaN
        data_df.pert_id = data_df.pert_id.astype(int).astype(str)
        ccle_lincs_df = pd.read_csv(self.raw_paths[3])
        if self.new_basal == 'lincs_wth_ccle_org_all.csv' and len(data_df[data_df.cell_id.isin(ccle_lincs_df['cell_iname'])])>0:
            data_df2 = pd.merge(ccle_lincs_df, data_df,left_on='cell_iname',right_on='cell_id')
            data_df3 = data_df2.drop(columns = ['ccle_name','cell_id'])
            data_df = data_df3.rename(columns={'cell_iname' : 'cell_id'})
            data_df = data_df[['pert_id','cell_id','pert_idose','pert_itime','sig_id','pert_type']]
        if self.new_basal == 'lincs_wth_ccle_org_all.csv' and len(data_df[data_df.cell_id.isin(ccle_lincs_df['ccle_name'])])>0:
            data_df2 = pd.merge(ccle_lincs_df, data_df,left_on='ccle_name',right_on='cell_id')
            data_df3 = data_df2.drop(columns = ['ccle_name','cell_id'])
            data_df = data_df3.rename(columns={'cell_iname' : 'cell_id'})
            data_df = data_df[['pert_id','cell_id','pert_idose','pert_itime','sig_id','pert_type']]
        else:
            data_df = data_df[['pert_id','cell_id','pert_idose','pert_itime','sig_id','pert_type']]
        drug_df = pd.read_csv(self.raw_paths[1], names=['CID','smiles'])
        drug_df.CID = drug_df.CID.astype(str)
        drug_df[['pert_id']] = drug_df[['CID']]
        hu_gene_lm2 = pd.read_csv(self.raw_paths[2], index_col=0, dtype='str')
        lincs_wth_ccle_org_all = pd.read_csv(self.raw_paths[5], index_col=0)
        lincs_wth_ccle_org_all2 = lincs_wth_ccle_org_all.rename(columns={'cell_iname' : 'cell_id'})
        target_lm349_all = pd.read_csv(self.raw_paths[4], sep ='\t')
        target_lm349_all.CID = target_lm349_all.CID.astype("str")
        df_full = pd.merge(drug_df, data_df, how='inner')
        df_full_num = under_number(df_full, maxNumAtoms)
        test_df = pd.read_csv(self.raw_paths[6])
        test_df.pert_id = test_df.pert_id.astype(str)
        df_full_test = pd.merge(drug_df, test_df, how='inner')
        lincs_wth_ccle_org_all2_filt = lincs_wth_ccle_org_all2[['cell_id']+df_full_test.iloc[:,7:356].columns.to_list()]
        df_full_num_filt = df_full_num[df_full_num.cell_id.isin(lincs_wth_ccle_org_all2_filt.cell_id)].reset_index(drop=True,inplace=False)  # 2977
        df_full_ccle = pd.merge(df_full_num_filt[['sig_id', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']], lincs_wth_ccle_org_all2_filt,how='left', on = "cell_id")
        gene_edge = from_gene_to_node_HSDB_L3(df_full_ccle, hu_gene_lm2)
        df_full_target = pd.merge(df_full_num_filt[['pert_id', 'smiles', 'CID', 'sig_id', 'pert_type', 'cell_id','pert_idose','pert_itime']],target_lm349_all,how='left')
        df_full_target = df_full_target.fillna(0)
        for i in range(0,len(df_full_num_filt)):
            x_data,edge_data,s_data = from_newsmiles_transfer_HSDB_L3(df_full_num_filt['smiles'][i], maxNumAtoms)
            df_full_ccle_lst = pd.DataFrame(df_full_ccle.iloc[i,5:354])
            df_full_ccle_lst.columns = ["exp"]
            df_full_target_lst = pd.DataFrame(df_full_target.iloc[i,8:357])
            df_full_target_lst.columns = ["targets"]
            ccle_target = pd.concat([df_full_ccle_lst,df_full_target_lst],axis=1)
            ccle_target[['exp','targets']] = ccle_target[['exp','targets']].astype('float')
            target_data = torch.FloatTensor(np.array(ccle_target)).view(349, 2) 
            d_data = torch.FloatTensor(np.array(df_full_num_filt['pert_idose'][i])).view(1, 1)
            t_data = torch.FloatTensor(np.array(df_full_num_filt['pert_itime'][i])).view(1, 1)
            data_list.append(PairData_DT(x_s=x_data, edge_index_s=edge_data, \
                smiles=s_data, x_p=target_data, edge_index_p=gene_edge, dose = d_data, time = t_data, \
                pert_id = df_full_num_filt['pert_id'][i], cell_id = df_full_num_filt['cell_id'][i]))# y=y_data,
        data, slices = self.collate(data_list)
        self._data_list = None 
        torch.save((data, slices), self.processed_paths[0])





################    
class FuNewallh3dttfHSDB_L3Dataset(InMemoryDataset):
    def __init__(self, root, new_smiles, new_basal): # 
        self.file_name = uuid.uuid4().hex.upper()[0:6]
        self.new_smiles2, self.new_basal = new_smiles, new_basal
        super(FuNewallh3dttfHSDB_L3Dataset, self).__init__(root, new_smiles, new_basal) #
        self.transform = None
        self.data, self.slices = torch.load(self.processed_paths[0])    #이거 안하면 여러개가 겹챠지지가 않아
    #
    @property
    def raw_file_names(self): 
        return  [self.new_smiles2, 'HSDB_network.csv', 'ccle_lincs_convert.csv','target_lm349_all.csv', self.new_basal,'test.csv']
    #
    @property
    def processed_file_names(self):
        return [str(self.file_name+'.pt')] #
    #
    def process(self):
        data_list = []
        maxNumAtoms = 50
        drug_df = pd.read_csv(self.raw_paths[0], names=['CID','smiles'])
        drug_df.CID = drug_df.CID.astype(str)
        drug_df[['pert_id']] = drug_df[['CID']]
        lincs_wth_ccle_org_all = pd.read_csv(self.raw_paths[4], index_col=0)
        lincs_wth_ccle_org_all2 = lincs_wth_ccle_org_all.rename(columns={'cell_iname' : 'cell_id'})
        pre_data_df = drug_df[['pert_id']]
        pre_data_df['cell_id'] = [lincs_wth_ccle_org_all2.cell_id.to_list()]*len(pre_data_df)
        data_df = pre_data_df.explode('cell_id').reset_index(drop=True,inplace=False) 
        data_df[['pert_idose']] = 0.1
        data_df[['pert_itime']] = 24
        data_df.pert_itime= data_df.pert_itime.astype(float)
        data_df[['sig_id','pert_type']] = np.NaN
        data_df.pert_id = data_df.pert_id.astype(int).astype(str)
        ccle_lincs_df = pd.read_csv(self.raw_paths[2])
        data_df2 = pd.merge(ccle_lincs_df, data_df,left_on='cell_iname',right_on='cell_id')
        data_df3 = data_df2.drop(columns = ['ccle_name','cell_id'])
        data_df = data_df3.rename(columns={'cell_iname' : 'cell_id'})
        data_df = data_df[['pert_id','cell_id','pert_idose','pert_itime','sig_id','pert_type']]
        hu_gene_lm2 = pd.read_csv(self.raw_paths[1], index_col=0, dtype='str')
        target_lm349_all = pd.read_csv(self.raw_paths[3], sep ='\t')
        target_lm349_all.CID = target_lm349_all.CID.astype("str")
        df_full = pd.merge(drug_df, data_df, how='inner')
        df_full_num = under_number(df_full, maxNumAtoms)
        test_df = pd.read_csv(self.raw_paths[5])
        test_df.pert_id = test_df.pert_id.astype(str)
        df_full_test = pd.merge(drug_df, test_df, how='inner')
        lincs_wth_ccle_org_all2_filt = lincs_wth_ccle_org_all2[['cell_id']+df_full_test.iloc[:,7:356].columns.to_list()]
        df_full_num_filt = df_full_num[df_full_num.cell_id.isin(lincs_wth_ccle_org_all2_filt.cell_id)].reset_index(drop=True,inplace=False)  # 2977
        df_full_ccle = pd.merge(df_full_num_filt[['sig_id', 'pert_id', 'pert_type', 'cell_id', 'pert_idose']], lincs_wth_ccle_org_all2_filt,how='left', on = "cell_id")
        gene_edge = from_gene_to_node_HSDB_L3(df_full_ccle, hu_gene_lm2)
        df_full_target = pd.merge(df_full_num_filt[['pert_id', 'smiles', 'CID', 'sig_id', 'pert_type', 'cell_id','pert_idose','pert_itime']],target_lm349_all,how='left')
        df_full_target = df_full_target.fillna(0)
        for i in range(0,len(df_full_num_filt)):
            x_data,edge_data,s_data = from_newsmiles_transfer_HSDB_L3(df_full_num_filt['smiles'][i], maxNumAtoms)
            df_full_ccle_lst = pd.DataFrame(df_full_ccle.iloc[i,5:354])
            df_full_ccle_lst.columns = ["exp"]
            df_full_target_lst = pd.DataFrame(df_full_target.iloc[i,8:357])
            df_full_target_lst.columns = ["targets"]
            ccle_target = pd.concat([df_full_ccle_lst,df_full_target_lst],axis=1)
            ccle_target[['exp','targets']] = ccle_target[['exp','targets']].astype('float')
            target_data = torch.FloatTensor(np.array(ccle_target)).view(349, 2) 
            d_data = torch.FloatTensor(np.array(df_full_num_filt['pert_idose'][i])).view(1, 1)
            t_data = torch.FloatTensor(np.array(df_full_num_filt['pert_itime'][i])).view(1, 1)
            data_list.append(PairData_DT(x_s=x_data, edge_index_s=edge_data, \
                smiles=s_data, x_p=target_data, edge_index_p=gene_edge, dose = d_data, time = t_data, \
                pert_id = df_full_num_filt['pert_id'][i], cell_id = df_full_num_filt['cell_id'][i]))# y=y_data,
        data, slices = self.collate(data_list)
        self._data_list = None 
        torch.save((data, slices), self.processed_paths[0])


