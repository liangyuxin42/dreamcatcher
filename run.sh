#!/bin/bash
#SBATCH --job-name=DreamCatcher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --gres=gpu:1 

ARGS="\
    --config_path /config/config.yml \
"

export options=" \
    $ARGS \
    "

python3 dreamcatcher.py $options

