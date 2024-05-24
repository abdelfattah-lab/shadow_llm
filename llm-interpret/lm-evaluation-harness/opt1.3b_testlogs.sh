# Base setup
# 1. Generate all the 0shot original pickles for headimportance
# [EXECUTED]
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_piqa_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_copa_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_winogrande_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_rte_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks record --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_record_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_hellaswag_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_arc_easy_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks wsc --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_wsc_original.pkl --num_fewshot 0 --method original
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks wic --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_wic_original.pkl --num_fewshot 0 --method original
# 2. Generate all the trace_inp_hact pickles for per-sample headimportance
# [EXECUTED]
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_piqa_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks copa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_copa_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks openbookqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks winogrande --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_winogrande_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks rte --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_rte_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks record --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_record_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks hellaswag --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_hellaswag_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks arc_easy --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_arc_easy_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks wsc --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_wsc_original.pkl --num_fewshot 0 --method predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks wic --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_wic_original.pkl --num_fewshot 0 --method predictor
# 2. Generate all the 0shot predictor models for headimportance
# [EXECUTED]
# sbatch --requeue slurmrunner.slurm "python activation_predictor.py --dataset piqa --execmode train"
# sbatch --requeue slurmrunner.slurm "python activation_predictor.py --dataset copa --execmode train"
# sbatch --requeue slurmrunner.slurm "python activation_predictor.py --dataset openbookqa --execmode train"
# sbatch --requeue slurmrunner.slurm "python activation_predictor.py --dataset winogrande --execmode train"
# sbatch --requeue slurmrunner.slurm "python activation_predictor.py --dataset combined --execmode train"

### Hypothesis: Input-based pruning should be better than pruning the same heads for the entire test dataset
############################################################ PIQA ############################################################
### Fixed Head Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=10,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.7144|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=20,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.7051|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=30,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.6980|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=40,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.6888|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=50,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.6817|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.6692|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=70,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.6409|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.6023|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=90,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|piqa|      0|acc     |0.5707|   |      |
### Token Based Pruning NOTE THAT HERE WE USE PIQA PREDICTOR MODEL, NOT COMBINED MODEL
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=10,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.7122|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=20,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.7111|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=30,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.7040|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=40,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.7040|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=50,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.6970|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.6899|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=70,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.6616|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.6126|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=90,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|piqa|      0|acc     |0.5767|   |      |
############################################################ WINOGRANDE ############################################################
### Fixed Head Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=10,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5888|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=20,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5888|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=30,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5856|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=40,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5896|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=50,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5817|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.577|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=70,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5241|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5343|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=90,maskmethod=original,predictor_=winogrande --tasks winogrande --output_path results/1.3b/winogrande/0shotwinograndea_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5304|   |      |
### Token Based Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=10,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5919|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=20,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5691|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=30,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5651|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=40,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5691|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=50,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5754|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5525|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=70,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5367|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5257|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=90,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/winogrande/0shot_winogrande_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5138|   |      |
############################################################ COPA ############################################################
### Fixed Head Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=10,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.79|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=20,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   |  0.8|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=30,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.79|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=40,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.77|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=50,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.77|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.77|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=70,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.68|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.67|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=90,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.55|   |      |
### Token Based Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=10,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.79|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=20,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   |  0.8|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=30,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.79|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=40,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.78|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=50,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.76|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.73|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=70,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.68|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.65|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=90,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.63|   |      |
############################################################ OpenbookQA ############################################################
### Fixed Head Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=10,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.226|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=20,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.224|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=30,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.220|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=40,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.226|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=50,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.204|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.196|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=70,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.190|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.156|   |      |
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=90,maskmethod=original,predictor_=openbookqa --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|openbookqa|      0|acc     |0.130|   |      |
### Token Based Pruning
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=10,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=20,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=30,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=40,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=50,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=70,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=90,maskmethod=predictor,predictor_=combined --tasks openbookqa --output_path results/1.3b/openbookqa/0shot_openbookqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor

### Hypothesis: Using another models head-importance should give poor results, when compared to using our combined predictor.
############################################################ PIQA ############################################################

### winogrande --> winogrande
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.577|   |      |
### COPA --> winogrande
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks winogrande --output_path results/1.3b/piqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5233|   |      |
### PIQA --> winogrande
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks winogrande --output_path results/1.3b/piqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
|winogrande|      0|acc   |0.5367|   |      |
### Openbookqa --> winogrande
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks winogrande --output_path results/1.3b/piqa/0shot_openbookqa_original.txt --batch_size 1 --num_fewshot 0 --method original
### Combined Predictor --> winogrande
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=combined --tasks winogrande --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
|winogrande|      0|acc   |0.5525|   |      |
### winogrande predictor --> winogrande
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=winogrande --tasks winogrande --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor


### winogrande --> copa
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.68|   |      |
### COPA --> copa
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=copa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.67|   |      |
### PIQA --> copa
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=piqa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.63|   |      |
### Openbookqa --> copa
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=piqa --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|copa|      0|acc   | 0.66|   |      |
### Combined Predictor --> copa
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=combined --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.65|   |      |
### winogrande Predictor --> copa
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=winogrande --tasks copa --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method predictor
|copa|      0|acc   | 0.65|   |      |



# Using RTE importance values
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_rte_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=copa --tasks rte --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|rte |      0|acc   |0.5018|   |      |
# Using winogrande importance values
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=copa --tasks rte --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|rte |      0|acc   |0.5487|   |      |
# Using COPA importance values
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=80,maskmethod=original,predictor_=copa --tasks rte --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|rte |      0|acc   |0.5126|   |      |
# Using combined predictor importance values
python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=80,maskmethod=predictor,predictor_=combined --tasks rte --output_path results/1.3b/copa/0shot_copa_original.txt --batch_size 1 --num_fewshot 0 --method original
|rte |      0|acc   |0.5632|   |      |


# ### Generate 0-shot head importance for 15000 points from the piqa training data-set as well as the aggregate activation
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks piqa --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_piqa_original.pkl --num_fewshot 0 --method original

# ### Use the 0-shot head importance for inference per-token, and compare that with the aggregated head importance
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=10,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6936|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=20,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6924|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=30,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6892|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=40,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6816|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=50,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6780|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6692|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=70,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6532|   |      |
# # Explicitly modify code to use our predictor
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=10,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6940|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=20,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6948|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=30,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6908|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=40,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6896|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=50,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6836|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6720|   |      |
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=70,maskmethod=predictor,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6680|   |      |

# ### COPA --> PIQA
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_copa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |Task|Version| Metric |Value |   |Stderr|
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6795|   |      |
# |    |       |acc_norm|0.6817|   |      |
# ### OpenBookQA --> PIQA
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_openbookqa_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |Task|Version| Metric |Value |   |Stderr|
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6866|   |      |
# |    |       |acc_norm|0.6910|   |      |
# ### Winogrande --> PIQA
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_winogrande_original.pkl,head_percent_mask=60,maskmethod=original,predictor_=piqa --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original
# |Task|Version| Metric |Value |   |Stderr|
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6801|   |      |
# |    |       |acc_norm|0.6801|   |      |
# ### Combined Predictor --> PIQA
# python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=logs/head_importance/opt1.3b/0shot_piqa_original.pkl,head_percent_mask=60,maskmethod=predictor,predictor_=combined --tasks piqa --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor
# |Task|Version| Metric |Value |   |Stderr|
# |----|------:|--------|-----:|---|------|
# |piqa|      0|acc     |0.6823|   |      |
# |    |       |acc_norm|0.6790|   |      |
