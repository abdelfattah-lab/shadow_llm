#!/bin/bash
#SBATCH -J DELAYEDPROG
#SBATCH -o /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/slurm_delayed/rt%j.out
#SBATCH -e /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/slurm_delayed/rt%j.err
#SBATCH -N 1
#SBATCH --mem=10000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-01
#SBATCH -n 1


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate opt

cd /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/emnlp_commands

# wait 4 hours
sleep 14400

bash pred_ablation_1.3b.sh