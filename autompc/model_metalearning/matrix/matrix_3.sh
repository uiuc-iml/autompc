#!/bin/bash

###############################################################################
#
#SBATCH --time=160:00:00                  # Job run time (hh:mm:ss)
#SBATCH --nodes=1                        # Number of nodes
#SBATCH --ntasks-per-node=16             # Number of task (cores/ppn) per node
#SBATCH --job-name=matrix_3      # Name of batch job
#SBATCH --partition=cs                   # Partition (queue)
#SBATCH --output=/home/baoyul2/scratch/sbatch_out_3/matrix_3.o%j           # Name of batch job output file
##SBATCH --error=/home/baoyul2/scratch/sbatch_out_3/matrix_3.e%j           # Name of batch job error file
#
###############################################################################

# Initialize Python environment 
module load anaconda/3
# module load openmpi
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=1

# Run job
# mpiexec -np 16 /home/baoyul2/.conda/envs/autompc/bin/python /home/baoyul2/autompc/autompc/model_metalearning/get_portfolio_configurations.py
/home/baoyul2/.conda/envs/meta_autompc/bin/python /home/baoyul2/autompc/autompc/model_metalearning/matrix/creat_matrix_3.py
