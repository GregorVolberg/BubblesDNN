#!/bin/bash
#SBATCH -J get_dset_FullFaceCrunchy2.m
#SBATCH --mail-user=gregor.volberg@ur.de
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --time=6:00:00
 
#SBATCH -o slurm.%N.%J.%u.out # STDOUT
#SBATCH -e slurm.%N.%J.%u.err # STDERR
 
source ~/.bashrc
matlab -batch "cd('./img'); get_dset_FullFaceCrunchy2.m" > mtext.txt



