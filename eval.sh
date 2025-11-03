#!/bin/bash

model="Deepfakecla_bu_pred_cl"

CUDA_VISIBLE_DEVICES=0 python -u evaluate.py \
    --config ./config/default.toml \
    --model $model \
    --data_root <data_path> \
    --checkpoint <model_path>

