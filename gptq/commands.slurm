#!/bin/bash
#SBATCH -J allnasspace
#SBATCH -o /home/ya255/projects/shadow_llm/gptq/large_scale_run_logs/rt%j.out
#SBATCH -e /home/ya255/projects/shadow_llm/gptq/large_scale_run_logs/rt%j.err
#SBATCH -N 1
#SBATCH --mem=100000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-02
#SBATCH --gres=gpu:1
#SBATCH -n 4


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate shadow_llm

cd /home/ya255/projects/shadow_llm/gptq


$1