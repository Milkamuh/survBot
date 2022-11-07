#!/bin/bash
ulimit -s 8192

#$ -l low
#$ -l os=*stretch
#$ -cwd
#$ -pe smp 1
##$ -q "*@minos15"

export PYTHONPATH="$PYTHONPATH:/home/marcel/git/"
export PYTHONPATH="$PYTHONPATH:/home/marcel/git/code_base/"

source /opt/anaconda3/etc/profile.d/conda.sh
conda activate py37

python survBot.py
