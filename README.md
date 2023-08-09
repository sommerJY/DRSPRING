# DRSPRING
DRug Synergy PRediction by INtegrated GCN(DRSRPING) is separated into two modules. 
The module_1, PDIGEC, predicts Drug Induced gene expression of each Drug-Cell line-Dose-Time profile. 
The module_2, PDSS, predicts Drug synergy score(Loewe score) of each drug-drug-cell triads.


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
```
code
```


## Use our trained model to predict your data

```
code
```








