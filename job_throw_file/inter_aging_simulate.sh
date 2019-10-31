#!/bin/sh
#$ -S /bin/sh
#$ -cwd
export PATH=/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims:$PATH

python3 inter_aging_simulate.py
