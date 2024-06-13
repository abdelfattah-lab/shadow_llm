### NOTE! This generates many .sh files, which should have commands you would need ###
### However, please read the comments and modify the commands as needed.           ###
### The slurm should also be modified to your needs.                               ###

### Evaluate on WikiText2.
### Note that this enforces 'all' eval (predictor and aggregate scores are calculated ACROSS downstream tasks.)
### To change this, add task to {zcp}_all_5.pkl as {zcp}_{task}_5.pkl or {zcp}_{tasks_imp_path_dict[task]}_5.pkl
tasks = ["wikitext"]
tasks_imp_path = ["wikitext"]
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
prune_modes = ["perlayer", "global"]
predmethods = ["dejavu", "predictor", "predictorL"]
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcp_list:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for basemethod in predmethods:
            print(f'######################### Starting Method {basemethod} #########################')
            command_list.append(f'######################### Starting Method {basemethod} #########################')
            for prune_mode in prune_modes:
                for prune_perc in [25, 30, 35, 40, 45, 50, 55, 60]:
                    base_cmd = f'sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args prune_style={prune_mode},ffn_percent_mask={prune_perc},fcmaskmethod=fc,aggr_all=False,zcp_calc={zcp},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-13b/{zcp}_all_5.pkl,head_percent_mask={prune_perc},maskmethod={basemethod},predictor_=all --tasks {task} --output_path results/13b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor"'
                    command_list.append(base_cmd)
                    print(base_cmd)

with open("wikitext_pred_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

# Downstream task evaluation
# Note that this also enforces 'all' eval (described above)
zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
tasks_imp_path = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
prune_modes = ["global", "perlayer"]
predmethods = ["dejavu", "predictorL", "predictor"] # predictorL is shadowLLM trained with per-layer criteria normalization.
fews = 5
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcp_list:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for basemethod in predmethods:
            # if zcp == 'plainact' and basemethod == 'dejavu':
            #     continue
            if f'{zcp}_{basemethod}' not in ["plainact_predictorL", "l2_norm_dejavu"]:
                continue
            print(f'######################### Starting Method {basemethod} #########################')
            command_list.append(f'######################### Starting Method {basemethod} #########################')
            for prune_mode in prune_modes:
                for prune_perc in [30, 40, 50, 60, 70]:
                    base_cmd = f'sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args prune_style={prune_mode},ffn_percent_mask={prune_perc},fcmaskmethod=fc,aggr_all=False,zcp_calc={zcp},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-13b/{zcp}_all_5.pkl,head_percent_mask={prune_perc},maskmethod={basemethod},predictor_=all --tasks {task} --output_path results/13b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot {fews}--method predictor"'
                    command_list.append(base_cmd)
                    print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_13b.sh"
with open("downstream_pred_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")



### Train Predictor On Every Task and ZCP (We train on 'combined' dataset, which is a combination of all tasks, instruction below)
### For training a predictor on EVERY task, run combine_generator.py with the right paths, and then add to tasks and tasks_imp_path: ["all"]
zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
tasks_imp_path = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
model_types = ["b1e", "b1e_seq", "b1eL"] ### b1e_seq may be disabled! Enable it to train Full Seq. ShadowLLM
fews = 5
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp_tom in zcp_list:
        for model_type in model_types:
            base_cmd = f'sbatch --requeue slurmrunner_medium.slurm "python emnlp_activation_predictor.py --fewshot {fews} --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-13b --execmode train --emb_style {model_type} --rerun"'
            command_list.append(base_cmd)
            base_cmd = f'sbatch --requeue slurmrunner_medium.slurm "python emnlp_activation_ffn_predictor.py --fewshot {fews} --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-13b --execmode train --emb_style {model_type} --rerun"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command with name "train_predictor_13b.sh"
with open("medium_train_predictor_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

### Evaluate zero-cost proxy in AGGREGATED setting (averaged over respective dataset)
zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
tasks_imp_path = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
prune_mode = "global"
fews = 5
command_list = []
for task in tasks:
    print(f'######################### Starting Task {task} #########################')
    command_list.append(f'######################### Starting Task {task} #########################')
    for zcp in zcp_list:
        print(f'######################### Starting ZCP {zcp} #########################')
        command_list.append(f'######################### Starting ZCP {zcp} #########################')
        for prune_perc in [50, 60, 70, 80, 85, 90, 91, 92, 93, 94, 95]:
            base_cmd = f'sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args prune_style={prune_mode},ffn_percent_mask={prune_perc},fcmaskmethod=fc,aggr_all=False,zcp_calc={zcp},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-13b/{zcp}_all_{fews}.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks wikitext --output_path results/13b/piqa/{fews}shot_piqa_original.txt --batch_size 1 --num_fewshot {fews} --method original"'
            command_list.append(base_cmd)
            print(base_cmd)

# Save every command to a file with name "eval_aggr_zcp_13b.sh"
with open("medium_wikitext_eval_aggr_zcp_13b.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

### Generate ZCPs for major datasets
zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
tasks_imp_path = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
fewshot_list = [0, 3, 5]
command_list = []
for dataset in tasks:
    print(f'######################### Starting Task {dataset} #########################')
    command_list.append(f'######################### Starting Task {dataset} #########################')
    for zcp_tom in zcp_list:
        for fews in fewshot_list:    
            base_cmd = f'sbatch --requeue slurmrunner_medium.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-13b,model_cache_dir=opt13b_checkpoints,tokenizer_cache_dir=opt13b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt13b/0shot_{dataset}_original.pkl --num_fewshot {fews} --method predictor"'
            command_list.append(base_cmd)
            print(base_cmd)

with open("zcp_generator.sh", "w") as f:
    for command in command_list:
        f.write(command + "\n")

