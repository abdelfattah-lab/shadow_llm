
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16 --gen_train_headmaps
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16 --gen_train_headmaps
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16
CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16 --gen_train_headmaps


# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16"
# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b wikitext2 --check --faster-kernel --wbits 16 --headgen_train"


# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16"
# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b ptb --check --faster-kernel --wbits 16 --headgen_train"

# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16"
# sbatch --requeue commands.slurm "CUDA_VISIBLE_DEVICES=0 python opt.py facebook/opt-1.3b c4 --check --faster-kernel --wbits 16 --headgen_train"

# sbatch --requeue commands.slurm "python activation_predictor.py --dataset wikitext2 --model opt-1.3b"
# sbatch --requeue commands.slurm "python activation_predictor.py --dataset c4 --model opt-1.3b"
# sbatch --requeue commands.slurm "python activation_predictor.py --dataset ptb --model opt-1.3b"
