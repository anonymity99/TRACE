#!/bin/bash
set -ux

# CUDA environment settings.
export CUDA_VISIBLE_DEVICES=2

# Parameters.
LEARNING_METHOD=super
MODEL=IntentUnifiedTransformer
PROJECT_ROOT=/home/myself/TRACE
SAVE_ROOT=/data_hdd/myself/TRACE
DATA_DIR=${PROJECT_ROOT}/data/pre_train
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
LOAD_MODEL_NAME=state_epoch_60
LOAD_MODEL_DIR=convbert-super-128-0-drop0.2-mlm0.1-tem0.07-ppt5-epoch60-lr1e-5-user-banking,clinc,hwu_few-seed11
INIT_CHECKPOINT=${SAVE_ROOT}/outputs/pre_train/few/${LOAD_MODEL_DIR}/${LOAD_MODEL_NAME}
EXAMPLE=true
WITH_CONTRASTIVE=false
WITH_RDROP=false
WITH_POOL=false
WITH_MLM=true
DYNAMIC_SCORE=true
TOKENIZER_TYPE=Bert
TRIGGER_DATA=hwu_few
TRIGGER_ROLE=user
DROPOUT_RATIO=0.2
TEMPERATURE=0.07
MLM_RATIO=0.1
KL_RATIO=5.0
LR=1e-4
PROMPT_NUM_FOR_UNDERSTAND=5
BATCH_SIZE_LABEL=128
BATCH_SIZE_NOLABEL=0
NUM_PROCESS=1
NUM_INTENT=64
NUM_EPOCH=60
NUM_GPU=1
SEED=11
SAVE_DIR=${SAVE_ROOT}/outputs/${TRIGGER_DATA}/${LOAD_MODEL_DIR}-${LOAD_MODEL_NAME}_${BATCH_SIZE_LABEL}-drop${DROPOUT_RATIO}-rdropkl${KL_RATIO}${WITH_RDROP}-mlm${MLM_RATIO}${WITH_MLM}-ppt${PROMPT_NUM_FOR_UNDERSTAND}-epoch${NUM_EPOCH}-lr${LR}-pool${WITH_POOL}-cl${WITH_CONTRASTIVE}-ex${EXAMPLE}-seed${SEED}

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
python -u run_intent.py \
  --do_train=true \
  --do_infer=true \
  --model=${MODEL} \
  --example=${EXAMPLE} \
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
  --with_rdrop=${WITH_RDROP} \
  --with_pool=${WITH_POOL} \
  --with_mlm=${WITH_MLM} \
  --mlm_ratio=${MLM_RATIO} \
  --kl_ratio=${KL_RATIO} \
  --dropout=${DROPOUT_RATIO} \
  --embed_dropout=${DROPOUT_RATIO} \
  --attn_dropout=${DROPOUT_RATIO} \
  --ff_dropout=${DROPOUT_RATIO} \
  --num_intent=${NUM_INTENT} \
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