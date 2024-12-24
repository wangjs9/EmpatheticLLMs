#!/bin/bash

SCRIPT="eval_models.infer_qwen"

PARAMS_LIST=(
    "--yaml_path eval_models/qwen-2.5.yaml"
    "--yaml_path eval_models/soulchat.yaml"
    "--yaml_path eval_models/vanilla-cot.yaml"
)

echo "run $SCRIPT..."

# 遍历参数列表并运行脚本
for PARAMS in "${PARAMS_LIST[@]}"; do
    echo "run $SCRIPT with args: $PARAMS"
    CUDA_VISIBLE_DEVICES=0 python -m $SCRIPT $PARAMS
    if [ $? -ne 0 ]; then
        echo "$SCRIPT fails with args: $PARAMS"
        exit 1
    fi
done

echo "Done"