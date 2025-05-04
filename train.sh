#! /bin/sh 
#SBATCH --job-name=benchmark 
#SBATCH --partition gpu-a100-q 
#SBATCH --gres=gpu:a100:1 
#SBATCH --cpus-per-task=16 
#SBATCH --mem=80G 
#SBATCH --output=%j.out
#SBATCH --error=%j.err 

python3 benchmark.py
