
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16 --gen_train_headmaps
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16 --gen_train_headmaps
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16
# CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16 --gen_train_headmaps


# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16"
# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16 --headgen_train"


# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16"
# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16 --headgen_train"

# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16"
# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16 --headgen_train"

sbatch --requeue commands.slurm "python activation_predictor.py --dataset wikitext2 --model opt-1.3b"
sbatch --requeue commands.slurm "python activation_predictor.py --dataset c4 --model opt-1.3b"
sbatch --requeue commands.slurm "python activation_predictor.py --dataset ptb --model opt-1.3b"

# cd /home/ya255/projects/shadow_llm/llm-interpret/lm-evaluation-harness
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/1shot_piqa.pkl --num_fewshot 1

# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/1shot_piqa.pkl,head_percent_mask=30 --tasks piqa --output_path results/1.3b/piqa/1shot_piqa_percent.txt --batch_size 2 --num_fewshot 1
