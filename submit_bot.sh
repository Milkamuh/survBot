#!/bin/bash

#$ -l low
#$ -l h_vmem=2.5G
#$ -l mem=2.5G
#$ -l h_stack=INFINITY
#$ -cwd
#$ -pe smp 1
#$ -binding linear:1
#$ -N survBot
#$ -o /data/www/~kasper/survBot/survBot_bg.log
#$ -e /data/www/~kasper/survBot/survBot_bg.err
#$ -m e
#$ -M kasper.fischer@rub.de

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate survBot

# environment variables for numpy to prevent multi threading
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

python survBot.py -html '/data/www/~kasper/survBot'
