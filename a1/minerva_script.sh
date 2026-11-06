#!/bin/bash
#SBATCH -p long
#SBATCH -J nlpA1-training
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 3:00:00
#SBATCH --output=/data/users/arvidko/results/nlpA1/%x-%j.out
#SBATCH --gres=gpu:L4:1

cd ~/project/nlp/a1
source ~/.venv/bin/activate
python3 A1_skeleton.py

