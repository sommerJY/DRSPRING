# DRSPRING
DRug Synergy PRediction by INtegrated GCN(DRSRPING) is separated into two modules. 

The module_1, PDIGEC, predicts Drug Induced gene expression of each Drug-Cell line-Dose-Time profile. 

The module_2, PDSS, predicts Drug synergy score(Loewe score) of each drug-drug-cell triads.

![alt text](total_fig.png)



## Prerequisites
cuda 11.6 (in case of using GPU)




## Install required packages
Use the requirement.txt file in directory
둘중에 하나 활용 가능할듯 
```
conda env create --file environment.yml
conda activate py37
```

아니면 

```
conda create -y --name py37 python=3.7
conda install --force-reinstall -y -q --name py37 -c conda-forge --file requirements.txt
conda activate py37
```




## Training model
**(1) Module_1 (PDIGEC)**
```
code
```

**(2) Module_2 (PDSS)**

In case you want to use early stopping in training module, you can use the option `--early_stopping`

The example code is presented below : 
```
python PDSS.py [result directory] --mode 'train' --early_stopping 'es'
python PDSS.py ~/DRSPRING/PDSS/result --mode 'train' --early_stopping 'es'
```


## Use our trained model to predict your data
**(1) Module_1 (PDIGEC)**

```
code
```

**(2) Module_2 (PDSS)**
1) In case you just give just new smiles of two drugs, we automatically present all predicted scores of 92 cells we used in training.
   Also, this mode requires the Module 1 derived files of each input SMILES.

```
python PDSS.py [result directory] --mode 'new_data' --saved_model [pretrained model] --DrugAsmiles [SMILES A] --DrugBsmiles [SMILES B] --M1_DrugA [Module 1 result of SMILES A] --M1_DrugB [Module 1 result of SMILES B]
python PDSS.py ~/DRSPRING/PDSS/result --mode 'new_data' --saved_model ~/DRSPRING/PDSS/result/MODEL.pt --DrugAsmiles 'C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O' --DrugBsmiles 'C1=C(C(=O)NC(=O)N1)F' --M1_DrugA '~/DRSPRING/PDIGEC/M1_expression_A.csv' --M1_DrugB '~/DRSPRING/PDIGEC/M1_expression_B.csv'
```

2) In case you give new smiles of two drugs and new CCLE data, you should additionally provide new data directory.
This also requires the Module 1 derived files of each input SMILES.
```
python PDSS.py [result directory] --mode 'new_data' --saved_model [pretrained model] --DrugAsmiles [SMILES A] --DrugBsmiles [SMILES B] --M1_DrugA [Module 1 result of SMILES A] --M1_DrugB [Module 1 result of SMILES B] --Basal_Cell [User provided new CCLE data]
python PDSS.py ~/DRSPRING/PDSS/result --mode 'new_data' --saved_model ~/DRSPRING/PDSS/result/MODEL.pt --DrugAsmiles 'C1C(N(C2=C(N1)N=C(NC2=O)N)C=O)CNC3=CC=C(C=C3)C(=O)NC(CCC(=O)O)C(=O)O' --DrugBsmiles 'C1=C(C(=O)NC(=O)N1)F' --M1_DrugA '~/DRSPRING/PDIGEC/M1_expression_A.csv' --M1_DrugB '~/DRSPRING/PDIGEC/M1_expression_B.csv' --Basal_Cell '~/DRSPRING/data/new_cell.csv'
```



