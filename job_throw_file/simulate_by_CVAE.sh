#!/bin/sh
#$ -S /bin/sh
#$ -cwd
export PATH=/lustre7/home/lustre4/ryoyokosaka/python/.pyenv/shims:$PATH

python3 /lustre7/home/lustre4/ryoyokosaka/python/aging_simulation/script/simulate_by_CVAE.py
