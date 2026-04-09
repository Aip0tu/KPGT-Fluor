#!/bin/bash

set -e

split_method="random"
dataset_name="consolidation" # 可选数据集：consolidation
task_name="absorption" # 可选任务：absorption、emission、quantum_yield、log_molar_absorptivity
cuda_id="${1:-${CUDA_ID-0}}"
model_path="models/pretrained/base.pth"

for fold in 0 1 2 3 4
do
    split_file="datasets/${split_method}/${dataset_name}_fold${fold}/${task_name}/splits.npy"
    if [ ! -f "$split_file" ]; then
        echo "Splits file ($split_file) not found, creating splits..."
        python preprocess_downstream_dataset.py --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" --dataset "${task_name}"
    fi

    echo "Running fold ${fold} on CUDA device ${cuda_id}"
    CUDA_VISIBLE_DEVICES=${cuda_id} python finetune.py --config base --model_path "${model_path}" --data_path "datasets/${split_method}/${dataset_name}_fold${fold}" --dataset "${task_name}" --dataset_type regression --metric r2 --split splits --weight_decay 0 --dropout 0.1 --lr 3e-5 --save_dir "models/downstream/${split_method}/${dataset_name}_fold${fold}/${task_name}"
done
