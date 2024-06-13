#!/bin/bash
#SBATCH -J allnasspace
#SBATCH -o /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/slurm_13b/rt%j.out
#SBATCH -e /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness/slurm_13b/rt%j.err
#SBATCH -N 1
#SBATCH --mem=150000
#SBATCH -t 48:00:00
#SBATCH --account=abdelfattah
#SBATCH --partition=abdelfattah
#SBATCH --nodelist=abdelfattah-compute-01
#SBATCH --gres=gpu:2
#SBATCH -n 4


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate opt

cd /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness

# sed -i 's|https://s3.amazonaws.com/research.metamind.io/wikitext|https://wikitext.smerity.com|g' /scratch/ya255/huggingface/modules/datasets_modules/datasets/wikitext/790c0ace3eff86700b3c4719bba9d60facc86849425baae42e49db8c94ec641c/wikitext.py

$1