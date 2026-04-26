#!/bin/bash

set -e

source_split_method="random"
target_split_method="random"
source_dataset_name="consolidation" # 预先在 FluorDB / consolidation 上训练好的模型
target_dataset_name="cyanine" # 可选外部数据集：cyanine、xanthene
cuda_id="${1:-${CUDA_ID-0}}"

tasks=(
    "absorption"
    "emission"
    "quantum_yield"
    "log_molar_absorptivity"
)

for task_name in "${tasks[@]}"
do
    for fold in 0 1 2 3 4
    do
        base_data_path="datasets/${target_split_method}/${target_dataset_name}_fold${fold}"
        split_file="${base_data_path}/${task_name}/splits.npy"
        model_path="models/downstream/${source_split_method}/${source_dataset_name}_fold${fold}/${task_name}.pth"
        save_dir="models/downstream/${target_split_method}/${target_dataset_name}_fold${fold}/${task_name}"

        if [ ! -f "$split_file" ]; then
            echo "Splits file ($split_file) not found, creating splits..."
            python preprocess_downstream_dataset.py --data_path "$base_data_path" --dataset "$task_name"
        fi

        if [ ! -f "$model_path" ]; then
            echo "Missing source model: $model_path"
            exit 1
        fi

        echo "Finetuning external dataset=${target_dataset_name}, task=${task_name}, fold=${fold}, cuda=${cuda_id}"
        CUDA_VISIBLE_DEVICES=${cuda_id} python finetune_external.py \
            --config base \
            --model_path "${model_path}" \
            --data_path "${base_data_path}" \
            --dataset "${task_name}" \
            --dataset_type regression \
            --metric r2 \
            --split splits \
            --weight_decay 0 \
            --dropout 0.1 \
            --lr 3e-5 \
            --save_dir "${save_dir}"
    done
done
