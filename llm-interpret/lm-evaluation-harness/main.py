import argparse
import json
import logging
import fnmatch
import os
from lm_eval import tasks, evaluator

logging.getLogger("openai").setLevel(logging.WARNING)


class MultiChoice:
    def __init__(self, choices):
        self.choices = choices

    # Simple wildcard support (linux filename patterns)
    def __contains__(self, values):
        for value in values.split(","):
            if len(fnmatch.filter(self.choices, value)) == 0:
                return False

        return True

    def __iter__(self):
        for choice in self.choices:
            yield choice


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model_args", default="")
    parser.add_argument("--tasks", default=None, choices=MultiChoice(tasks.ALL_TASKS))
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--method", type=str, default="original")
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no_cache", action="store_true")
    parser.add_argument("--head_importance_calc", action="store_true", default = False)
    parser.add_argument("--local_rank", type=int, default=None) 
    parser.add_argument("--save_importance_path", type=str, default=None)
    parser.add_argument("--decontamination_ngrams_path", default=None)
    parser.add_argument("--description_dict_path", default=None)
    parser.add_argument("--check_integrity", action="store_true")

    return parser.parse_args()


# Returns a list containing all values of the source_list that
# match at least one of the patterns
def pattern_match(patterns, source_list):
    task_names = set()
    for pattern in patterns:
        for matching in fnmatch.filter(source_list, pattern):
            task_names.add(matching)
    return list(task_names)


def main():
    args = parse_args()

    if args.limit:
        print(
            "WARNING: --limit SHOULD ONLY BE USED FOR TESTING. REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT."
        )

    if args.tasks is None:
        task_names = tasks.ALL_TASKS
    else:
        task_names = pattern_match(args.tasks.split(","), tasks.ALL_TASKS)

    print(f"Selected Tasks: {task_names}")


    assert not args.provide_description  # not implemented
    if args.head_importance_calc:
        import lm_eval.tasks
        task_dict = lm_eval.tasks.get_task_dict(task_names)
        model_args = args.model_args
        import lm_eval.models
        
        for name, task in task_dict.items():
            from types import SimpleNamespace
            nspmodel_args = SimpleNamespace(**{item.split('=')[0]: item.split('=')[1] for item in model_args.split(',')})
            task_name = task.DATASET_NAME if task.DATASET_NAME is not None else task.DATASET_PATH
            org_task_name = name
            mname = nspmodel_args.pretrained.split("/")[-1]
            # check if zcps/opt-{model_size}/{zcp_calc}_{task_name}_{num_fewshot}.pkl exists
            if os.path.exists(f"zcps/{mname}/{nspmodel_args.zcp_calc}_{task_name}_{args.num_fewshot}.pkl"):
                print("Already calculated for ", task_name, " with ", nspmodel_args.zcp_calc)
                exit(0)
            elif os.path.exists(f"zcps/{mname}/{nspmodel_args.zcp_calc}_{org_task_name}_{args.num_fewshot}.pkl"):
                print("Already calculated for ", org_task_name, " with ", nspmodel_args.zcp_calc)
                exit(0)

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)
    
    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        no_cache=args.no_cache,
        limit=args.limit,
        description_dict=description_dict,
        decontamination_ngrams_path=args.decontamination_ngrams_path,
        check_integrity=args.check_integrity,
        head_importance_calc=args.head_importance_calc,
        save_importance_path=args.save_importance_path,
        method=args.method
        )
    
    # if args.local_rank == 0:
    if not args.head_importance_calc:
        dumped = json.dumps(results, indent=2)
        print(dumped)
        os.makedirs(os.path.dirname(args.output_path), exist_ok = True)
        if args.output_path:
            with open(args.output_path, "w") as f:
                f.write(dumped)

        print(
            f"{args.model} ({args.model_args}), limit: {args.limit}, provide_description: {args.provide_description}, "
            f"num_fewshot: {args.num_fewshot}, batch_size: {args.batch_size}"
        )
        print(evaluator.make_table(results))


if __name__ == "__main__":
    main()
