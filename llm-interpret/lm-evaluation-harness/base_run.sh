
# Create using original method
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/1shot_piqa_original.pkl --num_fewshot 1 --method original
# Create using headimportance magnitude method (ours)
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/1shot_piqa_headmagn.pkl --num_fewshot 1 --method headmagn

# Test using original method
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/1shot_piqa_original.pkl,head_percent_mask=30 --tasks piqa --output_path results/1.3b/piqa/1shot_piqa_percent_original.txt --batch_size 2 --num_fewshot 1 --method original
# Test using headimportance magnitude method (ours)
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/1shot_piqa_headmagn.pkl,head_percent_mask=30 --tasks piqa --output_path results/1.3b/piqa/1shot_piqa_percent_headmagn.txt --batch_size 2 --num_fewshot 1 --method headmagn
