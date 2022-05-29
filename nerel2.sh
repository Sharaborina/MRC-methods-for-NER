#!/usr/bin/env bash
# -*- coding: utf-8 -*-
#SBATCH --job-name=bert              # Job name
#SBATCH --mail-type=END,FAIL         # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=sharaborinaly@mail.ru  # Where to send mail.  Set this to your email address
#SBATCH --cpus-per-task=4            # Number of cores per MPI task
#SBATCH --nodes=1                    # Maximum number of nodes to be allocated
#SBATCH --mem-per-cpu=30gb           # Memory (i.e. RAM) per processor
#SBATCH --time=2-00:00:00            # Wall time limit (days-hrs:min:sec)
#SBATCH --output=mpi_test_%j.log     # Path to the standard output and error files relative to the working directory
#SBATCH -p gpu:4 						 # gpu_a100

REPO_PATH=/beegfs/home/e.sharaborin/Diplom/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/trinity/home/e.sharaborin/Diplom/mrc-for-flat-nested-ner-wanglaiqi/datasets/nerel
#DATA_DIR=/beegfs/home/e.sharaborin/Diplom/mrc-for-flat-nested-ner/datasets/nerel
BERT_DIR=/beegfs/home/e.sharaborin/Diplom/rusbert/sbert_large_nlu_ru
TAG2QUERY_PATH=/beegfs/home/e.sharaborin/Diplom/mrc-for-flat-nested-ner/ner2mrc/queries/nerel.json

GPUS="4" #Number of GPUs to train on (int) or which GPUs to train on (list or str) applied per NODE Default:
NUM_NODES=1
CUDA_VISIBLE_DEVICES_LIST=0,1,2,3 #LIST of ALL GPUS in ALL NODES: 0,1,2,3,4,5,6,7

BERT_DROPOUT=0.1 # The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
MRC_DROPOUT=0.1 #Во время обучения случайным образом обнуляет некоторые элементы входного тензора с вероятностью p  Это оказалось эффективным методом регуляризации и предотвращения коадаптации нейронов,
LR=2e-5
LR_MINI=3e-7 #used only in polydecay
LR_SCHEDULER=onecycle #polydecay
SPAN_WEIGHT=0.1 #weight_span is used in total loss computation. It is coeff before match_loss
WARMUP=0 # you use a very low learning rate for a set number of training steps (warmup steps). After your warmup steps you use your "regular" learning rate or learning rate scheduler. You can also gradually increase your learning rate over the number of warmup steps.
MAXLEN=120 #max_length: int, max length of query+context
MAXNORM=1.0 #Gradient clipping can be enabled to avoid exploding gradients. B
INTER_HIDDEN=2048

BATCH_SIZE=6
PREC=16 #Double precision (64), full precision (32), half precision (16) or bfloat16 precision (bf16). Can be used on CPU, GPU, TPUs, HPUs or IPUs. Default: 32
VAL_CKPT=0.25 #How often to check the validation set. 0.25 means 4 times per epoch will be computed Pass a float in the range [0.0, 1.0] to check after a fraction of the training epoch. 
ACC_GRAD=1 #Accumulates grads every k batches or as set up in the dict. Trainer also calls optimizer.step() for the last indivisible step number. Then effective batch_size=ACC_GRAD*BATCH_SIZE
MAX_EPOCH=100
SPAN_CANDI=pred_and_gold #used in compute_loss match_candidates
PROGRESS_BAR=1
OPTIM=torch.adam

EVERY_N_EPOCHS=1
MAX_KEEP_CKPT=50


OUTPUT_DIR=/beegfs/home/e.sharaborin/Diplom/outputs_rus/mrc_ner/nerel_ver6/warmup${WARMUP}lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
mkdir -p ${OUTPUT_DIR}
#CHECKPTR_PATH=${OUTPUT_DIR}/epoch=18.ckpt

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST} python3 ${REPO_PATH}/train/mrc_ner_trainer.py \
--gpus=${GPUS} \
--num_nodes=${NUM_NODES} \
--distributed_backend=ddp \
--workers 8 \
--data_dir ${DATA_DIR} \
--tag2query_file ${TAG2QUERY_PATH} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAXLEN} \
--batch_size ${BATCH_SIZE} \
--precision=${PREC} \
--progress_bar_refresh_rate ${PROGRESS_BAR} \
--lr ${LR} \
--lr_mini ${LR_MINI} \
--lr_scheduler ${LR_SCHEDULER} \
--val_check_interval ${VAL_CKPT} \
--accumulate_grad_batches ${ACC_GRAD} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CANDI} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--gradient_clip_val ${MAXNORM} \
--optimizer ${OPTIM} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--every_n_epochs ${EVERY_N_EPOCHS} \
--max_keep_ckpt ${MAX_KEEP_CKPT} >out_nerel5 2> err_nerel5
#--pretrained_checkpoint ${CHECKPTR_PATH} \
