#!/bin/bash
#SBATCH -J emnlp_shadowllm
#SBATCH -o /home/slurm_logs/rt%j.out
#SBATCH -e /home/slurm_logs/rt%j.err
#SBATCH -N 1
#SBATCH --mem=150000
#SBATCH -t 48:00:00
#SBATCH --account=YOURACC
#SBATCH --partition=YOURPARTN
#SBATCH --nodelist=YOURNODE
#SBATCH --gres=gpu:2
#SBATCH -n 4


source /share/apps/anaconda3/2021.05/etc/profile.d/conda.sh 

conda activate opt

cd /path/to/shadow_llm/llm-interpret/lm-evaluation-harness

# May have to run this once on the server for WikiText2 eval.
# sed -i 's|https://s3.amazonaws.com/research.metamind.io/wikitext|https://wikitext.smerity.com|g' /scratch/UNAME/huggingface/modules/datasets_modules/datasets/wikitext/790c0ace3eff86700b3c4719bba9d60facc86849425baae42e49db8c94ec641c/wikitext.py

$1