######################### Starting Task piqa #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_piqa_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_piqa_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_piqa_original.pkl --num_fewshot 5 --method predictor"
######################### Starting Task copa #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_copa_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_copa_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_copa_original.pkl --num_fewshot 5 --method predictor"
######################### Starting Task openbookqa #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_openbookqa_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_openbookqa_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_openbookqa_original.pkl --num_fewshot 5 --method predictor"
######################### Starting Task winogrande #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_winogrande_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_winogrande_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_winogrande_original.pkl --num_fewshot 5 --method predictor"
######################### Starting Task rte #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_rte_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_rte_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_rte_original.pkl --num_fewshot 5 --method predictor"
######################### Starting Task hellaswag #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_hellaswag_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_hellaswag_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_hellaswag_original.pkl --num_fewshot 5 --method predictor"
######################### Starting Task arc_easy #########################
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_arc_easy_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=fisher,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_arc_easy_original.pkl --num_fewshot 5 --method predictor"
sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_arc_easy_original.pkl --num_fewshot 5 --method predictor"