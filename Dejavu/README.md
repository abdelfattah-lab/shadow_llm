This code is based on the code of the paper [Deja Vu: Contextual Sparsity for Efficient LLMs at Inference Time](https://proceedings.mlr.press/v202/liu23am.html). The original code is available at [FMInference/DejaVu](https://github.com/FMInference/DejaVu).

## Setup

A Dockerfile is provided. To build the image, run the following command:

```
docker build -t shadowllm_perf .
```

# Generate Results

Execute the following commands to execute inside the docker container:
```
docker run --gpus all -ti -v $(pwd):/home/user shadowllm_perf /bin/bash
```

Then, run the following commands to generate the results:
```
set -a
source .customenv
./run_experiments.sh
```

This will generate results for the configurations present in `experiments.csv`. Feel free to add/modify rows with the configurations you want results for. The results will be generated inside the subfolder `experiment_results`