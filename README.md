# DRSPRING
DRug Synergy PRediction by INtegrated GCN(DRSRPING) is separated into two modules. 

The module_1, PDIGEC, predicts Drug Induced gene expression of each Drug-Cell line-Dose-Time profile. 

The module_2, PDSS, predicts Drug synergy score(Loewe score) of each drug-drug-cell triads.

![alt text](total_fig.png)



## Prerequisites
cuda 11.6 (in case of using GPU)




## Install required packages
Use the environment.yml file in directory

```
conda create -y --name py37 python=3.7.16
conda activate py37
conda env update --file environment.yml
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cu116.html
pip install torch-geometric==2.0.4
```


## Additional Data
Due to the large file capacity, the necessary input files are provided through Google Drive.

Link : (This will be prepared soon)

Unpack all downloaded data in `raw` directory for further procedure. 



## Training model
**(1) Module_1 (PDIGEC)**
```
python PDIGEC.py [result directory]
python PDIGEC.py ./results
```

**(2) Module_2 (PDSS)**

The example code is presented below : 
```
python PDSS.py [result directory] --mode 'train'
python PDSS.py ./results --mode 'train'
```




## Use our trained model to predict your data
We support cell line specific basal expression data from CCLE 22Q1.(check cell line names in `lincs_wth_ccle_org_all.csv`). If you want to train or test cell lines that are not in CCLE 22Q1, you have to provide your own cell line's basal expression to use our model. Add `--basal` option to use new cell line, and make sure all your input files are all in `raw/` directory.

**(1) Module_1 (PDIGEC)**

We support new drug-gene expression prediction for user input chemicals with ccle cell line, experiment doses, experiment dosage time.
Check the format of `'new_drug_cellline.csv'` and `'new_drug.csv'` files in `raw/` directory. 
```
python PDIGEC.py [saving directory] --mode 'new_data' --saved_model [M1 trained model] \
--drug_cell [user input drug-cell combination] \
--smiles [user input drug canonical smiles] \
--jobname [user input jobname]

(Example)
python PDIGEC.py ./results --mode 'new_data' --saved_model ./results/M1_model.pt \
--drug_cell 'new_drug_cellline.csv' \
--smiles 'new_drug.csv'
--jobname 'M1_result'
```


If you want to get drug induced gene expression predicted value of all 1393 ccle cell lines that we used, try this code!
All you need to provide is PubChem CID and SMILES as csv file.
Check the format of `'new_drug.csv'` file in `raw/` directory. 
```
python PDIGEC.py [saving directory] --mode 'new_data_cellline_all' --saved_model [M1 trained model] \
--smiles [user input drug canonical smiles] \
--jobname [user input jobname]

(Example)
python PDIGEC.py ./results --mode 'new_data_cellline_all' --saved_model ./results/M1_model.pt \
--smiles 'new_drug.csv' \
--jobname 'M1_result'
```

Both methods will make result file under the `'results/'` directory with user named output. 



**(2) Module_2 (PDSS)**
1) In case you just give just new smiles of two drugs, we automatically present all predicted scores of 92 cells we used in training.
   Also, this mode requires the Module 1 resulted files of each input SMILES.

```
python PDSS.py [result file name] --mode 'new_data' --saved_model [pretrained model] \
--InputSM [user input drug canonical smiles] --InputEXP [Module 1 result of Input] \
--ACID [CID of drug A] --BCID [CID of drug B]

(Example)
python PDSS.py 'M2_result.csv' --mode 'new_data' --saved_model ./results/M2_model.pt \
--InputSM ./raw/new_drugAB.csv --InputEXP ./results/M1_resultAB_predicted_expression.csv \
--ACID 0000 --BCID 0000
```


2) In case you give new smiles of two drugs and new CCLE data, you should additionally provide new data directory.
This also requires the Module 1 derived files of each input SMILES.
```
python PDSS.py [result file name] --mode 'new_data' --saved_model [pretrained model] \
--InputSM [user input drug canonical smiles] --InputEXP [Module 1 result of Input] \
--ACID [CID of drug A] --BCID [CID of drug B] \
--Basal_Cell [User provided new CCLE data]


python PDSS.py 'M2_result.csv' --mode 'new_data' --saved_model ./results/M2_model.pt \
--InputSM ./raw/new_drugAB.csv --InputEXP ./results/M1_resultAB_predicted_expression.csv \
--ACID 0000 --BCID 0000
--Basal_Cell ./raw/new_cell.csv
```



