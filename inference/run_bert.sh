#!/bin/bash
#SBATCH --job-name=bert_inference
#SBATCH --output=logs/bert_%j.out
#SBATCH --error=logs/bert_%j.err
#SBATCH --partition=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --mail-user=batuhan.sencer@ttu.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --nodelist=rpg-93-2

module load cuda
source ~/.bashrc
conda activate distilbert

python run_inference_bert.py
