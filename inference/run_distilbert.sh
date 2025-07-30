#!/bin/bash
#SBATCH --job-name=distilbert_inference
#SBATCH --output=logs/distilbert_%j.out
#SBATCH --error=logs/distilbert_%j.err
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

python run_inference_distilbert.py
