#!/bin/bash

python -u -c 'import torch; print(torch.__version__)'

CODE_PATH=codes
DATA_PATH=data
SAVE_PATH=models

MODE=train
MODEL=RotatE
DATASET=DBPEDIA-clean
GPU_DEVICE=0
SAVE_ID=0

FULL_DATA_PATH=$DATA_PATH/$DATASET
SAVE=$SAVE_PATH/"$MODEL"_"$DATASET"_"$SAVE_ID"

BATCH_SIZE=1024
NEGATIVE_SAMPLE_SIZE=256
KGE_DIM=1000
GAMMA=24.0
ALPHA=1.0
LEARNING_RATE=0.0005
MAX_STEPS=75000
TEST_BATCH_SIZE=4

CUDA_VISIBLE_DEVICES=$GPU_DEVICE python -u $CODE_PATH/run_trt.py --do_train \
    --cuda \
    --do_valid \
    --do_test \
    --data_path $FULL_DATA_PATH \
    --model $MODEL \
    -n $NEGATIVE_SAMPLE_SIZE -b $BATCH_SIZE -d $KGE_DIM \
    -g $GAMMA -a $ALPHA -adv \
    -lr $LEARNING_RATE --max_steps $MAX_STEPS \
    -save $SAVE --test_batch_size $TEST_BATCH_SIZE \
    -de \
    -dt \
    --type_dim 250 \
    --valid_steps 5000 \
    --e2t_batch_size 4096 \
    --e2t_training_steps 75000 \
