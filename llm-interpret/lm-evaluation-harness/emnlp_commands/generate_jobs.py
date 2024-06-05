###############################################################################################
###############################################################################################
####### First, we want to generate all ZCPs for both 'aggregate' and predictor training #######
###############################################################################################
###############################################################################################

"""
Insights to extract:
    1. Cross correlation between ZCPs for head-ranking
    2. Range/Distribution of every ZCP (min, max, etc.)
    3. Average correlation of aggregate ZCP based head ranking vs per-prompt ZCP based head ranking
    4. Compare ZCPs designed for aggregate (epenas/jacov)
"""

zcp_list = ['epenas', 'jacov', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
datasets = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
command_list = []
for dataset in datasets:
    print(f'######################### Starting Task {dataset} #########################')
    command_list.append(f'######################### Starting Task {dataset} #########################')
    for zcp_tom in zcp_list:
        base_cmd = f'sbatch --requeue slurmrunner.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt1.3b/0shot_{dataset}_original.pkl --num_fewshot 0 --method predictor"'
        command_list.append(base_cmd)
        print(base_cmd)

# Save every command to a file with name "zcp_calc_1.3b.sh"
with open("zcp_calc_1.3b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

zcp_list = ['epenas', 'jacov', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
datasets = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
command_list = []
for dataset in datasets:
    print(f'######################### Starting Task {dataset} #########################')
    command_list.append(f'######################### Starting Task {dataset} #########################')
    for zcp_tom in zcp_list:
        base_cmd = f'sbatch --requeue slurmrunner_2gpu.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-6.7b,model_cache_dir=opt6.7b_checkpoints,tokenizer_cache_dir=opt6.7b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt6.7b/0shot_{dataset}_original.pkl --num_fewshot 0 --method predictor"'
        command_list.append(base_cmd)
        print(base_cmd)

# Save every command to a file with name "zcp_calc_6.7b.sh"
with open("zcp_calc_6.7b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")


zcp_list = ['epenas', 'jacov', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
datasets = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
command_list = []
for dataset in datasets:
    print(f'######################### Starting Task {dataset} #########################')
    command_list.append(f'######################### Starting Task {dataset} #########################')
    for zcp_tom in zcp_list:
        base_cmd = f'sbatch --requeue slurmrunner_2gpu.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_{dataset}_original.pkl --num_fewshot 0 --method predictor"'
        command_list.append(base_cmd)
        print(base_cmd)

# Save every command to a file with name "zcp_calc_13b.sh"
with open("zcp_calc_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")


#################################################################################
#################################################################################
####### Now, we want to train all ZCP predictor for OPT-1.3b and OPT-6.7b #######
#################################################################################
#################################################################################

"""
Insights to extract:
    1. Best style of predictor (Shadow, Shadow_Full, DejaVu)
    2. Best predictive ability, diagnosis (Grasp range?)
    3. Per-dataset variance of predictor performance
    4. Sample efficiency and number of samples required. [TESTS NOT DONE HERE.]                                 [NOT IMPLEMENTED]
"""

zcp_list = ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "hellaswag", "ARC-Easy"]
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
model_types = ["b1e", "ble", "b1e_seq"]
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp_tom in zcp_list:
        for model_type in model_types:
            base_cmd = f'sbatch --requeue slurmrunner_small.slurm "python emnlp_activation_predictor.py --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-1.3b --execmode train --emb_style {model_type} --rerun"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command with name "train_predictor_1.3b.sh"
with open("train_predictor_1.3b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

zcp_list = ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
model_types = ["b1e", "ble", "b1e_seq"]
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp_tom in zcp_list:
        for model_type in model_types:
            base_cmd = f'sbatch --requeue slurmrunner_small.slurm "python emnlp_activation_predictor.py --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-6.7b --execmode train --emb_style {model_type} --rerun"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command with name "train_predictor_6.7b.sh"
with open("train_predictor_6.7b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

    
zcp_list = ['fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
model_types = ["b1e", "ble", "b1e_seq"]
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp_tom in zcp_list:
        for model_type in model_types:
            base_cmd = f'sbatch --requeue slurmrunner_small.slurm "python emnlp_activation_predictor.py --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-13b --execmode train --emb_style {model_type} --rerun"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command with name "train_predictor_13b.sh"
with open("train_predictor_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

###############################################################################################
###############################################################################################
############################## Generate commands for model evaluation #########################
###############################################################################################
###############################################################################################


"""
Insights to extract:
    1. Accuracy vs Sparsity for each data-set, for each proxy
    2. Accuracy vs Sparsity for GEOMEAN across data-sets, for each proxy
    3. Choose 1-3 best average-case proxies.
    4. Combined aggregate head importance based performance (AVERAGE CASE)                                  [NOT IMPLEMENTED]
"""

tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# zip with task as key
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcps:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
            base_cmd = f'sbatch --requeue slurmrunner.slurm "python main.py --model opt --model_args zcp_calc={zcp},pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-1.3b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks {task} --output_path results/1.3b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_1.3b.sh"
with open("eval_aggr_zcp_1.3b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")


tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# zip with task as key
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcps:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
            base_cmd = f'sbatch --requeue slurmrunner_2gpu.slurm "python main.py --model opt --model_args zcp_calc={zcp},pretrained=facebook/opt-6.7b,model_cache_dir=opt6.7b_checkpoints,tokenizer_cache_dir=opt6.7b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-6.7b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks {task} --output_path results/6.7b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_6.7b.sh"
with open("eval_aggr_zcp_6.7b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")


tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# zip with task as key
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcps:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
            base_cmd = f'sbatch --requeue slurmrunner_2gpu.slurm "python main.py --model opt --model_args zcp_calc={zcp},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-13b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks {task} --output_path results/13b/piqa/0shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_13b.sh"
with open("eval_aggr_zcp_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

###############################################################################################
###############################################################################################
######################## Generate commands for predictor evaluation ###########################
###############################################################################################
###############################################################################################

"""
Insights to extract:
    1. Difference between predictor based vs aggregate based accuracy (GEOMEAN/AVG)
    2. Difference between predictor based vs aggregate based accuracy on 1-3 best average-case proxies.
    3. Train COMBINED head importance predictor (AVERAGE DYNAMIC CASE)                                          [NOT IMPLEMENTED]
    4. COMPARE COMBINED PREDICTOR VS COMBINED AGGREGATE PERFORMANCE                                             [NOT IMPLEMENTED]
"""


tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# zip with task as key
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcps:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
            base_cmd = f'sbatch --requeue slurmrunner.slurm "python main.py --model opt --model_args zcp_calc={zcp},pretrained=facebook/opt-1.3b,model_cache_dir=opt1.3b_checkpoints,tokenizer_cache_dir=opt1.3b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-1.3b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=predictor,predictor_={task} --tasks {task} --output_path results/1.3b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_1.3b.sh"
with open("eval_pred_zcp_1.3b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

# ##### Generate commands for model evaluation

tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# zip with task as key
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcps:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
            base_cmd = f'sbatch --requeue slurmrunner_2gpu.slurm "python main.py --model opt --model_args zcp_calc={zcp},pretrained=facebook/opt-6.7b,model_cache_dir=opt6.7b_checkpoints,tokenizer_cache_dir=opt6.7b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-6.7b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=predictor,predictor_={task} --tasks {task} --output_path results/6.7b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_6.7b.sh"
with open("eval_pred_zcp_6.7b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")


# ##### Generate commands for model evaluation

tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "record", "hellaswag", "arc_easy", "wsc", "wic"]
tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
# zip with task as key
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcps = ['epenas', 'fisher', 'grad_norm', 'grasp', 'l2_norm', 'nwot', 'plainact', 'snip']
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcps:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 65, 70, 75, 80, 85, 90, 91, 92, 93, 94, 95, 96]:
            base_cmd = f'sbatch --requeue slurmrunner_2gpu.slurm "python main.py --model opt --model_args zcp_calc={zcp},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-13b/{zcp}_{tasks_imp_path_dict[task]}_0.pkl,head_percent_mask={prune_perc},maskmethod=predictor,predictor_={task} --tasks {task} --output_path results/13b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_13b.sh"
with open("eval_pred_zcp_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

