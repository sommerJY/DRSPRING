#!/bin/bash
#SBATCH --job-name=GPU1_T
#SBATCH --partition=1gpu
#SBATCH --nodelist=bdata5 # 둘중 하나 지워도 ok
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/home01/k006a01/logs/test_GPU1_%j.log
#SBATCH --error=/home01/k006a01/logs/test_GPU1_%j.log
#SBATCH --time=00:15:00


echo "START"
pwd; hostname; date

echo "sourcing"
source /home01/k006a01/.bashrc
module load compilers/cuda/11.4


echo "init conda"
conda init
conda activate MY_4
# conda activate 하면 which python 했을때 내 python 나옴 

echo "which python"
which python


echo "work"
python /home01/k006a01/00.CODE/test_GPU.py

date
