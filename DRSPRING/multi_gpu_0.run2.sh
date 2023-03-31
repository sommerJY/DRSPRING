#!/bin/bash
#SBATCH --job-name=test_multi
#SBATCH --partition=8gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=1
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


# srun torchrun --nnodes 1 --nproc_per_node 1 /home01/k020a01/TEST/multi_gpu_0.singleGPUDDP.py > /home01/k020a01/TEST/multi_check11.txt


# best practice 
# srun torchrun --nnodes 1 --nproc_per_node 4 /home01/k020a01/TEST/multi_gpu_0.multinodeGPUDDP.py > /home01/k020a01/TEST/multi_check11.txt 

# my trial 
## srun torchrun --nnodes 1 --nproc_per_node 8 /home01/k020a01/TEST/multi_gpu_test_7.py > /home01/k020a01/TEST/multi_my.1.txt
# srun torchrun --nnodes 1 --nproc_per_node 8 /home01/k020a01/TEST/multi_gpu_test_7.py --world_size 8 > /home01/k020a01/TEST/multi_my.2.txt # train successful
srun torchrun --nnodes 1 --nproc_per_node 8 /home01/k020a01/TEST/multi_gpu_test_7.py --world_size 8 > /home01/k020a01/TEST/multi_my.3.txt

date








