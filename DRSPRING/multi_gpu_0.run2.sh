#!/bin/bash
#SBATCH --job-name=test_multi
#SBATCH --partition=4gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --output=/home01/k020a01/TEST/multi_check_%j.log
#SBATCH --error=/home01/k020a01/TEST/multi_check_%j.log

echo "START"
pwd; hostname; date
# 메일보내기는 의외로 안됨 힝구 


echo "sourcing"
source /home01/k020a01/.bashrc
module load compilers/cuda/11.6
# 우리가 쓰는 TORCH 나 tf 같은 경우에는 cuda 써서 GPU 랑 연결시켜줘야하니까 cuda 모듈 불러와줘야함 

echo "init conda"
conda init
conda activate MY_5


#nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
#nodes_array=($nodes)
#head_node=${nodes_array[0]}
#head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

ip1=`hostname -I | awk '{print $2}'`
echo Node IP: $ip1
export LOGLEVEL=INFO
export NCCL_DEBUG=INFO

export MASTER_ADDR=$(hostname)
echo "r$SLURM_NODEID master: $MASTER_ADDR"

echo "r$SLURM_NODEID work!"


srun torchrun --nnodes 1 --nproc_per_node 1 /home01/k020a01/TEST/multi_gpu_0.singleGPUDDP.py > /home01/k020a01/TEST/multi_check11.txt


date



나의 궁금증
왜 내가 만든 multi_gpu 에서는 되고 
오히려 튜토리얼 sh 는 안될까..? 

