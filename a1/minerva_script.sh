#!/bin/bash

#SBATCH -J nlpA1-training
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 0:30:00
#SBATCH --output=/data/users/arvidko/resutls/nlpA1/.
#SBATCH --gres=gpu:L4:1

cd ~/project/nlp/a1
source ~/.venv/bin/activate
python3 A1_skeleton.py

