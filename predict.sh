#!/bin/bash

set -e

split_method="random"
dataset_name="consolidation" # 可选数据集：consolidation
task_name="log_molar_absorptivity" # 可选任务：absorption、emission、quantum_yield、log_molar_absorptivity
cuda_id="${1:-${CUDA_ID-0}}"

for fold in 0 1 2 3 4
do
    model_path="models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}.pth"
    echo "Predicting fold ${fold} on CUDA device ${cuda_id}"
    CUDA_VISIBLE_DEVICES=${cuda_id} python predict.py --config base --model_path "${model_path}" --dataset "${task_name}" --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" --dataset_type regression --metric r2 --split splits --results_dir "results/${split_method}/${dataset_name}_fold${fold}/${task_name}"
done
