#!/bin/bash
set -ux

# Parameters.
PROJECT_ROOT=/home/myself/TRACE
VOCAB_PATH=${PROJECT_ROOT}/model/Bert/vocab.txt
DATA_DIR=${PROJECT_ROOT}/data/pre_train
WITH_MLM=true
DYNAMIC_SCORE=true
TOKENIZER_TYPE=Bert
TRIGGER_DATA=
TRIGGER_ROLE=user
NUM_PROCESS=64
PROMPT_NUM_FOR_UNDERSTAND=5

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