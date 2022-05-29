#!/usr/bin/env bash
# -*- coding: utf-8 -*-

# file: eval.sh

REPO_PATH=/trinity/home/e.sharaborin/Diplom/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

OUTPUT_DIR=/beegfs/home/e.sharaborin/Diplom/outputs_rus/mrc_ner/nerel_ver4/warmup200lr2e-5_drop0.3_norm1.0_weight0.1_warmup200_maxlen120
# find best checkpoint on dev in ${OUTPUT_DIR}/train_log.txt
BEST_CKPT_DEV=${OUTPUT_DIR}/epoch=18.ckpt
PYTORCHLIGHT_HPARAMS=${OUTPUT_DIR}/lightning_logs/version_1/hparams.yaml
GPU_ID=0

python3 ${REPO_PATH}/evaluate/mrc_ner_evaluate.py ${BEST_CKPT_DEV} ${PYTORCHLIGHT_HPARAMS} ${GPU_ID} >eval_nerel_out 2> eval_nerel_err
