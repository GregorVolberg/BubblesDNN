#!/bin/bash
#SBATCH -J sadPvsC
#SBATCH --mail-user=gregor.volberg@ur.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
 
#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR
 
# https://github.com/conda/conda/issues/8536
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate pytorch
python ResNet50_trainingSplit_sadPvsC.py > sadPvsC.txt

