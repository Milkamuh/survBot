#!/bin/bash
ulimit -s 8192

#$ -l low
#$ -l h_vmem=5G
#$ -cwd
#$ -pe smp 1
#$ -N survBot_bg
#$ -l os=*stretch

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate py37

# environment variables for numpy to prevent multi threading
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python survBot.py -html '/data/www/~marcel/'
