######################### Starting Task piqa #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_piqa_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_piqa_original.pkl --num_fewshot 5 --method predictor
######################### Starting Task copa #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_copa_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_copa_original.pkl --num_fewshot 5 --method predictor
######################### Starting Task openbookqa #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_openbookqa_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_openbookqa_original.pkl --num_fewshot 5 --method predictor
######################### Starting Task winogrande #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_winogrande_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_winogrande_original.pkl --num_fewshot 5 --method predictor
######################### Starting Task rte #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_rte_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_rte_original.pkl --num_fewshot 5 --method predictor
######################### Starting Task hellaswag #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_hellaswag_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_hellaswag_original.pkl --num_fewshot 5 --method predictor
######################### Starting Task arc_easy #########################
python main.py --model opt --model_args zcp_calc=plainact,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_arc_easy_original.pkl --num_fewshot 5 --method predictor
python main.py --model opt --model_args zcp_calc=l2_norm,pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_arc_easy_original.pkl --num_fewshot 5 --method predictor