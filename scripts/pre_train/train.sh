#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# Parameters.
LEARNING_METHOD=super
MODEL=UnifiedTransformer
LOAD_MODEL_NAME=convbert
PROJECT_ROOT=/home/myself/TRACE
SAVE_ROOT=/data_hdd/myself/TRACE
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
INIT_CHECKPOINT=${PROJECT_ROOT}/model/${LOAD_MODEL_NAME}
DATA_DIR=${PROJECT_ROOT}/data/pre_train
WITH_CONTRASTIVE=true
WITH_POOL=true
WITH_MLM=true
DYNAMIC_SCORE=true
TOKENIZER_TYPE=Bert
TRIGGER_DATA=
TRIGGER_ROLE=user
DROPOUT_RATIO=0.2
TEMPERATURE=0.07
MLM_RATIO=0.1
LR=1e-5
PROMPT_NUM_FOR_UNDERSTAND=5
BATCH_SIZE_LABEL=160
BATCH_SIZE_NOLABEL=0
NUM_PROCESS=1
NUM_EPOCH=20
NUM_GPU=8
SEED=11
SAVE_DIR=${SAVE_ROOT}/outputs/pre_train/${LOAD_MODEL_NAME}-${LEARNING_METHOD}-${BATCH_SIZE_LABEL}-${BATCH_SIZE_NOLABEL}-drop${DROPOUT_RATIO}-mlm${MLM_RATIO}-tem${TEMPERATURE}-ppt${PROMPT_NUM_FOR_UNDERSTAND}-epoch${NUM_EPOCH}-lr${LR}-${TRIGGER_ROLE}-${TRIGGER_DATA}-seed${SEED}

# Data preprocess.
python -u preprocess.py \
  --data_dir=${DATA_DIR} \
  --with_mlm=${WITH_MLM} \
  --vocab_path=${VOCAB_PATH} \
  --num_process=${NUM_PROCESS} \
  --trigger_data=${TRIGGER_DATA} \
  --trigger_role=${TRIGGER_ROLE} \
  --dynamic_score=${DYNAMIC_SCORE} \
  --tokenizer_type=${TOKENIZER_TYPE} \
  --prompt_num_for_understand=${PROMPT_NUM_FOR_UNDERSTAND}

# Main run.
python -u run.py \
  --do_train=true \
  --model=${MODEL} \
  --data_dir=${DATA_DIR} \
  --vocab_path=${VOCAB_PATH} \
  --num_process=${NUM_PROCESS} \
  --trigger_data=${TRIGGER_DATA} \
  --trigger_role=${TRIGGER_ROLE} \
  --dynamic_score=${DYNAMIC_SCORE} \
  --tokenizer_type=${TOKENIZER_TYPE} \
  --prompt_num_for_understand=${PROMPT_NUM_FOR_UNDERSTAND} \
  --batch_size_label=${BATCH_SIZE_LABEL} \
  --batch_size_nolabel=${BATCH_SIZE_NOLABEL} \
  --save_dir=${SAVE_DIR} \
  --init_checkpoint=${INIT_CHECKPOINT} \
  --learning_method=${LEARNING_METHOD} \
  --temperature=${TEMPERATURE} \
  --with_contrastive=${WITH_CONTRASTIVE} \
  --with_pool=${WITH_POOL} \
  --with_mlm=${WITH_MLM} \
  --mlm_ratio=${MLM_RATIO} \
  --dropout=${DROPOUT_RATIO} \
  --embed_dropout=${DROPOUT_RATIO} \
  --attn_dropout=${DROPOUT_RATIO} \
  --ff_dropout=${DROPOUT_RATIO} \
  --num_epoch=${NUM_EPOCH} \
  --gpu=${NUM_GPU} \
  --seed=${SEED} \
  --lr=${LR} \
  --log_steps=20 \
  --valid_steps=0 \
  --num_type_embeddings=2 \
  --save_checkpoint=true \
  --token_loss=true \
  --max_len=256