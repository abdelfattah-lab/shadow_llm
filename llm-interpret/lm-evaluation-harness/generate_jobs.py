# ##### Generate commands for model evaluation

# tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
# tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# # zip with task as key
# tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
# zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip', 'oracle_ga'] 
# for task in tasks:
#     print(f'######################### Starting Task {task} #########################')
#     for zcp in zcps:
#         if 'oracle' in zcp:
#             continue
#             # print(f'######################### Starting ZCP {zcp} #########################')
#             # # for prune_perc in [50, 60, 70, 80, 90, 92, 94, 96]:
#             # for prune_perc in [0.5, 0.6, 0.7, 0.8, 0.9, 0.92, 0.94, 0.96]:
#             #     print(f'python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-1.3b/oracle_ga_top_1_{prune_perc}_{task}_0.pkl,head_percent_mask={int(prune_perc*100)},maskmethod=original,predictor_={task} --tasks {task} --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original')
#         else:
#             print(f'######################### Starting ZCP {zcp} #########################')
#             for prune_perc in [50, 60, 70, 80, 90, 92, 94, 96]:
#                 print(f'python main.py --model opt --model_args pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-1.3b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks {task} --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original')


#### Generate commands for ZCP calculation


# zcp_list = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
# datasets = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
# for dataset in datasets:
#     print(f'######################### Starting Task {dataset} #########################')
#     for zcp_tom in zcp_list:
#         base_cmd = f'sbatch --requeue slurmrunner.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_{dataset}_original.pkl --num_fewshot 0 --method predictor"'
#         print(base_cmd)



# #### Train predictors for heads

# zcp_list = ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
# tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
# tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
# model_types = ["b1e", "ble", "b1e_seq"]
# for task in tasks:
#     print(f'######################### Starting Task {task} #########################')
#     for zcp_tom in zcp_list:
#         for model_type in model_types:
#             base_cmd = f'sbatch --requeue slurmrunner.slurm "python emnlp_activation_predictor.py --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-1.3b --execmode train --emb_style {model_type} --rerun"'
#             print(base_cmd)