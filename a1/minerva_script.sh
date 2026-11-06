#!/bin/bash

#SBARCH -J nlpA1-training
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --output=~/resutls/nlpA1/.
#SBATCH --gres=gpu:L4:1

source ~/.venv/bin/activate
python3 ~/projects/nlp/a1/A1_skeleton.py

