#!/bin/bash
#SBATCH -J res50_generic_loop
#SBATCH --mail-user=gregor.volberg@ur.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --time=7:00:00
 
#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR
 
 
# gpu_burn
source ~/.bashrc
source /usr/local/anaconda3/etc/profile.d/conda.sh
# /usr/local/anaconda3/envs/pytorch/bin/python ResNet50_trainingSplit_happyPvsC.py > happyPvsC.txt
conda activate pytorch
python ResNet50_training_generic.py


