if True:
    # tasks = ["piqa", "copa", "winogrande", "record", "hellaswag", "arc_easy"]
    # tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "record", "hellaswag", "ARC-Easy", "wsc", "wic"]
    tasks = ["wikitext"]
    tasks_imp_path = ["wikitext"]
    tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
    zcp_list = ["plainact", "l2_norm"]
    prune_modes = ["perlayer", "global"]
    # predmethods = ["dejavu", "predictor", "predictorL"]
    predmethods = ["dejavu", "predictorL"]
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
                print(f'######################### Starting Method {basemethod} #########################')
                command_list.append(f'######################### Starting Method {basemethod} #########################')
                for prune_mode in prune_modes:
                    for prune_perc in [30, 40, 50, 60, 70]:
                        base_cmd = f'sbatch --requeue slurmrunner_large.slurm "python main.py --model opt --model_args prune_style={prune_mode},ffn_percent_mask={prune_perc},fcmaskmethod=fc,aggr_all=False,zcp_calc={zcp},pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-30b/{zcp}_all_5.pkl,head_percent_mask={prune_perc},maskmethod={basemethod},predictor_=all --tasks {task} --output_path results/30b/piqa/0shot_piqa_predictor.txt --batch_size 1 --num_fewshot 0 --method predictor"'
                        command_list.append(base_cmd)
                        print(base_cmd)

    # Save every command to a file with name "eval_aggr_zcp_30b.sh"
    with open("large_wikitext_pred_30b.sh", "w") as f:
        for command in command_list:
            f.write(command + "\n")


exit(0)
if True:
    zcp_list = ["plainact", "l2_norm"]
    tasks = ["all"]
    tasks_imp_path = ["all"]
    tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
    model_types = ["b1e", "b1e_seq", "b1eL"]
    fews = 5
    command_list = []
    for task in tasks:
        print(f'######################### Starting Task {task} #########################')
        command_list.append(f'######################### Starting Task {task} #########################')
        for zcp_tom in zcp_list:
            for model_type in model_types:
                base_cmd = f'sbatch --requeue slurmrunner.slurm "python emnlp_activation_predictor.py --fewshot {fews} --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-30b --execmode train --emb_style {model_type} --rerun"'
                command_list.append(base_cmd)
                base_cmd = f'sbatch --requeue slurmrunner.slurm "python emnlp_activation_ffn_predictor.py --fewshot {fews} --dataset {task} --dataset_cname {tasks_imp_path_dict[task]} --zcp_metric {zcp_tom} --basemodel opt-30b --execmode train --emb_style {model_type} --rerun"'
                command_list.append(base_cmd)
                print(base_cmd)

    # Save every command with name "train_predictor_30b.sh"
    with open("large_train_predictor_30b.sh", "w") as f:
        for command in command_list:
            f.write(command + "\n")



if True:
    zcp_list = ["plainact", "l2_norm"]
    tasks = ["piqa"]
    tasks_imp_path = ["piqa"]
    tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
    prune_mode = "perlayer"
    fews = 5
    command_list = []
    for task in tasks:
        print(f'######################### Starting Task {task} #########################')
        command_list.append(f'######################### Starting Task {task} #########################')
        for zcp in zcp_list:
            print(f'######################### Starting ZCP {zcp} #########################')
            command_list.append(f'######################### Starting ZCP {zcp} #########################')
            for prune_perc in [50, 60, 70, 80, 85, 90, 91, 92, 93, 94, 95]:
                base_cmd = f'sbatch --requeue slurmrunner_large.slurm "python main.py --model opt --model_args prune_style={prune_mode},ffn_percent_mask={prune_perc},fcmaskmethod=fc,aggr_all=False,zcp_calc={zcp},pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-30b/{zcp}_all_{fews}.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks wikitext --output_path results/30b/piqa/{fews}shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original"'
                command_list.append(base_cmd)
                print(base_cmd)

    # Save every command to a file with name "eval_aggr_zcp_30b.sh"
    with open("large_wikitext_eval_aggr_zcp_30b.sh", "w") as f:
        for command in command_list:
            f.write(command + "\n")
            
if True:
    ### Generate ZCPs for major datasets

    # zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
    # zcp_list = ["fisher", "plainact", "l2_norm"]
    zcp_list = ["plainact", "l2_norm"]
    tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
    tasks_imp_path = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
    tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
    fewshot_list = [5]
    command_list = []
    for dataset in tasks:
        print(f'######################### Starting Task {dataset} #########################')
        command_list.append(f'######################### Starting Task {dataset} #########################')
        for zcp_tom in zcp_list:
            for fews in fewshot_list:    
                base_cmd = f'sbatch --requeue slurmrunner_large.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_{dataset}_original.pkl --num_fewshot {fews} --method predictor"'
                command_list.append(base_cmd)
                print(base_cmd)

    # Save every command to a file with name "zcp_calc_30b.sh"
    with open("large_fewshot_zcp_calc_30b.sh", "w") as f:
        for command in command_list:
            f.write(command + "\n")


if False:
    # # ### Generate ZCPs for major datasets

    # zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
    zcp_list = ["plainact", "l2_norm"]
    tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
    tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "hellaswag", "ARC-Easy"]
    tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
    command_list = []
    fews = 5
    for dataset in tasks:
        print(f'######################### Starting Task {dataset} #########################')
        command_list.append(f'######################### Starting Task {dataset} #########################')
        for zcp_tom in zcp_list:
            base_cmd = f'sbatch --requeue slurmrunner_large.slurm "python main.py --model opt --model_args zcp_calc={zcp_tom},pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer --tasks {dataset} --head_importance_calc --save_importance_path logs/head_importance/opt30b/0shot_{dataset}_original.pkl --num_fewshot {fews} --method predictor"'
            command_list.append(base_cmd)
            print(base_cmd)

    # Save every command to a file with name "zcp_calc_30b.sh"
    with open("large_zcp_calc_30b.sh", "w") as f:
        for command in command_list:
            f.write(command + "\n")

        

if False:
    # zcp_list = ["fisher", "grasp", "grad_norm", "plainact", "snip", "nwot", "l2_norm", "epenas", "jacov"]
    zcp_list = ["plainact", "l2_norm"]
    tasks = ["piqa", "copa", "openbookqa", "winogrande", "rte", "hellaswag", "arc_easy"]
    tasks_imp_path = ["piqa", "copa", "main", "winogrande_xl", "rte", "hellaswag", "ARC-Easy"]
    tasks_imp_path_dict = dict(zip(tasks, tasks_imp_path))
    prune_mode = "perlayer"
    fews = 5
    command_list = []
    for task in tasks:
        print(f'######################### Starting Task {task} #########################')
        command_list.append(f'######################### Starting Task {task} #########################')
        for zcp in zcp_list:
            print(f'######################### Starting ZCP {zcp} #########################')
            command_list.append(f'######################### Starting ZCP {zcp} #########################')
            for prune_perc in [50, 60, 70, 80, 85, 90, 91, 92, 93, 94, 95]:
                base_cmd = f'sbatch --requeue slurmrunner_large.slurm "python main.py --model opt --model_args prune_style={prune_mode},ffn_percent_mask={prune_perc},fcmaskmethod=fc,aggr_all=False,zcp_calc={zcp},pretrained=facebook/opt-30b,model_cache_dir=opt30b_checkpoints,tokenizer_cache_dir=opt30b_tokenizer,mask_heads=1,head_importance_path=zcps/opt-30b/{zcp}_{tasks_imp_path_dict[task]}_{fews}.pkl,head_percent_mask={prune_perc},maskmethod=original,predictor_={task} --tasks {task} --output_path results/30b/piqa/{fews}shot_piqa_original.txt --batch_size 1 --num_fewshot 0 --method original"'
                command_list.append(base_cmd)
                print(base_cmd)

    # Save every command to a file with name "eval_aggr_zcp_30b.sh"
    with open("large_fewshot_eval_aggr_zcp_30b.sh", "w") as f:
        for command in command_list:
            f.write(command + "\n")