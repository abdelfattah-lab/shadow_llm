import os
import collections
import itertools
import numpy as np
import random
import torch
import pickle
import lm_eval.metrics
import lm_eval.models
import lm_eval.tasks
import lm_eval.base
from lm_eval.utils import positional_deprecated, run_task_tests
from transformers import AutoTokenizer

@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=[],
    num_fewshot=0,
    batch_size=None,
    device=None,
    no_cache=False,
    limit=None,
    bootstrap_iters=100000,
    description_dict=None,
    check_integrity=False,
    decontamination_ngrams_path=None,
    head_importance_calc=False,
    save_importance_path=None,
    method="original"
):

    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int, optional
        Batch size for model
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param no_cache: bool
        Whether or not to cache
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :return
        Dictionary of results
    """
    random.seed(1234)
    np.random.seed(1234)

    assert tasks != [], "No tasks specified"

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.models.get_model(model).create_from_arg_string(
            model_args, {"batch_size": batch_size, "device": device}
        )
    else:
        assert isinstance(model, lm_eval.base.LM)
        lm = model

    task_dict = lm_eval.tasks.get_task_dict(tasks)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if not head_importance_calc:
        results = evaluate(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            description_dict=description_dict,
            decontamination_ngrams_path=decontamination_ngrams_path,
            model_args=model_args,
        )
        results["config"] = {
            "model": model,
            "model_args": model_args,
            "num_fewshot": num_fewshot,
            "batch_size": batch_size,
            "device": device,
            "no_cache": no_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "description_dict": description_dict,
        }   
    else:
        # if os.path.exists(save_importance_path):
        #     with open(save_importance_path, 'rb') as handle:
        #         results = pickle.load(handle)
        # else:
        results = head_importance(
            lm=lm,
            task_dict=task_dict,
            num_fewshot=num_fewshot,
            description_dict=description_dict,
            save_importance_path=save_importance_path,
            method=method,
            model_args=model_args,
        )    

    return results


decontaminate_suffix = "_decontaminate"


def head_importance(
    lm,
    task_dict,
    num_fewshot=0,
    description_dict=None,
    save_importance_path=None,
    method="original",
    model_args=None,
):
    '''
        Docstring
    '''
    for name, task in task_dict.items():
        print(name)
        print(task)
        print(model_args)
        split = 'train' if task.has_training_docs() else 'valid' if task.has_validation_docs() else None
        if not split:
            raise RuntimeError("Task has neither train nor validation")

        tokenizer = lm.get_tokenizer()
        try:
            dataloader = task.get_dataloader(tokenizer, split, subset_size = 2500, batch_size = lm.batch_size, num_fewshot = num_fewshot)
        except:
            try:
                dataloader = task.get_dataloader(tokenizer, split, batch_size = lm.batch_size, num_fewshot = num_fewshot)
            except:
                dataloader = task.get_dataloader(tokenizer, split, batch_size = lm.batch_size)


        """
        ####### epe_nas #######
        Evaluating their behavior using local linear operators at different data points
        1. Calculate Jacobian for different data-points.
            -> Evaluate correlation of different points in same class.
        2. Covariance of Jacobian with the average of the class it belongs to
            -> Each corr.mat of diff. size. Take an average measure. 
        Discussion:
            There is no class measure of transformers. Thus, how can we calculate this variance?
            NOTE: We do an interesting per-token clustering lol
                  for FFN, each neuron has a 'context' independent embedding
                  will have to see how it works out
            [ ** FFN IMP CALC DEPRECATED ** ]
        ####### fisher #######
        2nd Order dLoss on actvn removal -> empirical estimate for binary mask param used to toggle channel
        1. (a*dL/da)**2
        ####### flops  #######
        Every head is of the same parameter count, thus it is irrelevant
        ####### GradNorm #######
        1. |dL/da|
        ####### Grasp ######
        1. -(H*dL/dW).W
                NOTE: We use dA, A and H_{a}, making it a second order GBIS
                NOTE: Second order MAY not be happening? head_grad_w ==  head_grad_attn_x
        ####### jacov ######
        Correlation of jacobians with different inputs. Class invariant alternative to epe_nas?
        1. -Sum_{i} (log(eigv_{i}(corr(J)) + e) + 1/(eigv_{i}(corr(J)) + e))
                NOTE: Class is found by using each token predicted as a class.
        ####### naswot ######
        'Shatter' capacity of neuron. **May not make sense for attention map**
        1. K += (x @ x.T) + ((1.-x) @ (1.-x.T))
        2. slogdet(K)
                NOTE: Does not make sense without ReLU? 
                      Does scaled attention 'shatter'
        ###### plain ###### NOTE: RENAMED TO plainact
        1. |dL/dW * W|
                NOTE: We use dA and A, making it same as GBIS from 66B paper.
                "plain_act"
        ###### snip ###### NOTE: BECOMES SAME AS PLAINACT when converted to activation (?)
        1. |dL/dWmask| -> Investigate dWmask more, dynamic reversal of network by 
            having placeholder weights seem important, but may be 'equivalent' to plain.
            NOTE: We add a mask and measure its gradients for attention. 
                  We take mean of this mask, but max/norm/abs may be more helpful?
        ###### synflow ######
        NOTE: It seems like the synethtic_loss (sensitivity sum maximization) backward
        is the same as grad_attn_x? which makes this simply grad_norm?
        Also, it is for weights / params. 

        ##### zen ######
        
        """

        from types import SimpleNamespace
        nspmodel_args = SimpleNamespace(**{item.split('=')[0]: item.split('=')[1] for item in model_args.split(',')})
        task_name = task.DATASET_NAME if task.DATASET_NAME is not None else task.DATASET_PATH
        org_task_name = name
        # check if zcps/opt-{model_size}/{zcp_calc}_{task_name}_{num_fewshot}.pkl exists
        if os.path.exists(f"zcps/opt-1.3b/{nspmodel_args.zcp_calc}_{task_name}_{num_fewshot}.pkl"):
            print("Already calculated for ", task_name, " with ", nspmodel_args.zcp_calc)
            continue
        elif os.path.exists(f"zcps/opt-1.3b/{nspmodel_args.zcp_calc}_{org_task_name}_{num_fewshot}.pkl"):
            print("Already calculated for ", org_task_name, " with ", nspmodel_args.zcp_calc)
            continue
        else:
            measures = [nspmodel_args.zcp_calc]
            for measure in measures:
                _ = getattr(lm, f'calculate_{measure}')(dataloader, method, task, num_fewshot)
        return _

@positional_deprecated
def evaluate(
    lm,
    task_dict,
    provide_description=None,
    num_fewshot=0,
    limit=10000,
    bootstrap_iters=100000,
    description_dict=None,
    decontamination_ngrams_path=None,
    model_args=None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param provide_description: bool
        Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
    :param num_fewshot: int
        Number of examples in few-shot context
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param description_dict: dict[str, str]
        Dictionary of custom task descriptions of the form: `task_name: description`
    :return
        Dictionary of results
    """
    # TODO: completely refactor this entire function to not be a huge mess, ideally breaking it down into smaller pieces

    # TODO: todo: implement proper description-providing system
    assert not provide_description  # not implemented.
    if provide_description is not None:
        # nudge people to not specify it at all
        print(
            "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict"
        )

    decontaminate = decontamination_ngrams_path is not None

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]

    task_dict_items = [
        (name, task)
        for name, task in task_dict.items()
        if (task.has_validation_docs() or task.has_test_docs())
    ]
    results = collections.defaultdict(dict)
    versions = collections.defaultdict(dict)

    requests = collections.defaultdict(list)
    requests_origin = collections.defaultdict(list)

    overlaps = collections.defaultdict(list)  # {task_name: contaminated_docs}

    # If we ever run into issues where the eval tasks don't fit in memory and we can't afford a machine with bigger
    # memory, we can always modify this plumbing to support that, but I didn't want to include it just yet because
    # over-engineering is bad (or we could make it write the requests to disk and then read them back out again
    #  - probably using an sqlite db because of all the moving parts we have

    # TODO: we need unit tests & sanity checks or something to ensure that the return of `validation_docs` is stable
    docs = {}

    docs_for_decontamination = collections.defaultdict(list)

    # get lists of each type of request
    for task_name, task in task_dict_items:
        versions[task_name] = task.VERSION
        # default to test doc, fall back to val doc if validation unavailable
        # TODO: the test-fallback-to-val system isn't final, we should revisit it at some point
        if task.has_test_docs():
            task_doc_func = task.test_docs
            task_set = "test"  # Required for caching in the decontamination
        elif task.has_validation_docs():
            task_set = "val"  # Required for caching in the decontamination
            task_doc_func = task.validation_docs
        else:
            raise RuntimeError("Task has neither test_docs nor validation_docs")

        # deterministically shuffle docs and chop off the first `limit` because sometimes docs are in some kind of order
        # can remove 1000 during actual evaluation
        task_docs = list(task_doc_func())
        # Only do first 2500
        # task_docs = list(task.training_docs())[:2500]
        rnd = random.Random()
        rnd.seed(42)
        rnd.shuffle(task_docs)
        
        # Limit the number of task documents to a maximum of 5000

        description = (
            description_dict[task_name]
            if description_dict and task_name in description_dict
            else ""
        )

        total_requests = 0  # Initialize a counter for total requests
        limit_requests = 5000  # Set the maximum number of requests
        # Here, we should limit requests to the latter 70% of the data-set.
        for doc_id, doc in enumerate(itertools.islice(task_docs, 0, limit)):

            # if total_requests >= limit_requests:
            #     break  # Stop processing if the limit is reached

            if decontaminate and task.should_decontaminate():
                docs_for_decontamination[(task_name, task_set)].append(
                    task.doc_to_decontamination_query(doc)
                )

            docs[(task_name, doc_id)] = doc
            ctx = task.fewshot_context(
                doc=doc, num_fewshot=num_fewshot, rnd=rnd, description=description
            )
            reqs = task.construct_requests(doc, ctx)
            if not isinstance(reqs, (list, tuple)):
                reqs = [reqs]

            total_requests += len(reqs)  # Update the counter with the number of new requests

            for i, req in enumerate(reqs):
                requests[req.request_type].append(req)
                # i: index in requests for a single task instance
                # doc_id: unique id that we can get back to a doc using `docs`
                requests_origin[req.request_type].append((i, task_name, doc, doc_id))

    # Compare all tasks/sets at once to ensure a single training set scan
    if decontaminate:
        from lm_eval.decontamination.decontaminate import get_train_overlap

        print("Finding train/test overlap, please wait...")
        overlaps = get_train_overlap(
            docs_for_decontamination, decontamination_ngrams_path, limit
        )

    # all responses for each (task, doc)
    process_res_queue = collections.defaultdict(list)

    # execute each type of request
    for reqtype, reqs in requests.items():
        # TODO: right now, this code runs multiple separate LM requests for multiple Requests differing
        #       only in index. We could implement some kind of caching, but that would be more of a band-aid
        #       solution. we could also implement some kind of auto-grouping here;
        #       they should end up next to each other.

        print("Running", reqtype, "requests")
        resps = getattr(lm, reqtype)([req.args for req in reqs])
        resps = [
            x if req.index is None else x[req.index] for x, req in zip(resps, reqs)
        ]

        for resp, (i, task_name, doc, doc_id) in zip(resps, requests_origin[reqtype]):
            process_res_queue[(task_name, doc_id)].append((i, resp))

    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for (task_name, doc_id), requests in process_res_queue.items():
        requests.sort(key=lambda x: x[0])
        requests = [x[1] for x in requests]

        task = task_dict[task_name]
        doc = docs[(task_name, doc_id)]

        metrics = task.process_results(doc, requests)
        for metric, value in metrics.items():
            vals[(task_name, metric)].append(value)

            # Re-use the evaluation for the decontaminated set by just ignoring the overlaps
            if decontaminate and task_name in overlaps:
                if doc_id not in overlaps[task_name]:
                    vals[(task_name, metric + decontaminate_suffix)].append(value)

    from types import SimpleNamespace
    nspmodel_args = SimpleNamespace(**{item.split('=')[0]: item.split('=')[1] for item in model_args.split(',')})
    # comma separate EVERY argument of nspmodel_args without modifying
    run_descr = ",".join([str(item).split("=")[-1] for item in model_args.split(',')]) + ","
    tasklist = ",".join([str(task_name) for (task_name, _), _ in vals.items()])
    run_descr += tasklist
    # if nspmodel_args.predictor_ == "all":
    #     run_descr = nspmodel_args.head_importance_path.replace("zcps/opt-1.3b/", "").replace("_0.pkl", "") + "," + str(nspmodel_args.head_percent_mask) + "," + task_name + "-all" + "," + nspmodel_args.maskmethod + "," + task_name
    # else:
    #     run_descr = nspmodel_args.head_importance_path.replace("zcps/opt-1.3b/", "").replace("_0.pkl", "") + "," + str(nspmodel_args.head_percent_mask) + "," + nspmodel_args.predictor_ + "," + nspmodel_args.maskmethod + "," + task_name
    # aggregate results
    for (task_name, metric), items in vals.items():
        task = task_dict[task_name]
        real_metric = metric  # key when looking up the metric with task.aggregation
        if metric.endswith(decontaminate_suffix):
            real_metric = metric.replace(
                decontaminate_suffix, ""
            )  # decontaminated still uses the same metric
        # save the model_args.head_importance_path.replace("zcps/opt-1.3b/", "").replace("_0.pkl", "") as well as model_args.predictor_ and then the real_metric
        
        run_descr += "," + str(task.aggregation()[real_metric](items))
        # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
        # so we run them less iterations. still looking for a cleaner way to do this
        # if metric == 'bleu':
        #     stderr = lm_eval.metrics.stderr_for_metric(
        #         metric=task.aggregation()[real_metric],
        #         bootstrap_iters=min(bootstrap_iters, 1000)
        #         if metric in ["bleu", "chrf", "ter"]
        #         else bootstrap_iters,
        #     )

        #     if stderr is not None:
        #         results[task_name][metric + "_stderr"] = stderr(items)
    if nspmodel_args.aggr_all:
        if not os.path.exists("all_ind_aggrzcp_res"):
            os.makedirs("all_ind_aggrzcp_res")
        f_2write = "all_ind_aggrzcp_res"
    else:
        # Make a new directory called 'individual_results'
        if not os.path.exists("ind_aggrzcp_res"):
            os.makedirs("ind_aggrzcp_res")
        f_2write = "ind_aggrzcp_res"
    # Save as a new row to 'overall_results.csv' file
    # with open("zcp_execution_results.csv", "a") as f:
    # generate a random integer between 0 and 10000000000 , make sure it is unseeded
    state = random.getstate()
    random.seed()
    run_id = random.randint(0, 10000000000)
    random.setstate(state)
    with open(f"{f_2write}/{run_id}_zcpaggr_result.csv", "w") as f:
        f.write(f"{run_descr}\n")
    results[task_name][metric] = task.aggregation()[real_metric](items)

    return {"results": dict(results), "versions": dict(versions)}


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m, "%.4f" % v, "±", "%.4f" % se])
            else:
                values.append([k, version, m, "%.4f" % v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()
