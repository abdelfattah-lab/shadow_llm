#!/bin/bash

# Input and output CSV files
input_csv="experiments.csv"
time=$(date +"%Y-%m-%d_%H-%M-%S")
dir="experiment_results/$time"
mkdir -p $dir
output_csv="$dir/results.csv"
result_file="time.txt"

# Read the header and write it to the output CSV
header=$(head -n 1 "$input_csv")
echo "$header" > "$output_csv"

# Process the CSV file line by line, skipping the header
tail -n +2 "$input_csv" | while IFS=, read -r model gpus sparsity n_heads n_neurons prompt_len gen_len mode time; do
  rm $result_file

  # If mode is dejavu, ffn_flag and att_flag are not needed
  if [[ $mode == "dejavu" ]]; then
    ffn_flag=""
    att_flag=""
  # If it is shadowllm, ffn_flag and att_flag are needed
  else
    if [[ $mode == "shadowllm" ]]; then
      ffn_flag="--replace-ffn-predictor"
      att_flag="--replace-att-predictor"
    # If it is static, ffn_flag and att_flag are needed
    else
      ffn_flag="--skip-ffn-predictor"
      att_flag="--skip-att-predictor"
    fi
  fi

  torchrun --nproc_per_node=$gpus benchmarks/benchmark_generation_opt_dejavu.py --model-name $model --mlp-K $n_neurons --att-K1 $n_heads --att-K2 $n_heads $ffn_flag $att_flag --genlen $gen_len --promptlen $prompt_len


  # Read the result from the result file
  result=$(cat "$result_file")

  # Construct the new CSV line with the result
  new_line="$model,$gpus,$sparsity,$n_heads,$n_neurons,$prompt_len,$gen_len,$mode,$result"

  # Append the new line to the output CSV
  echo "$new_line" >> "$output_csv"
done
