#! /bin/bash
time=$(date "+%Y%m%d%H%M%S")
# Runs the "345M" parameter model
export MASTER_ADDR=localhost
export MASTER_PORT=6000
RANK=0
WORLD_SIZE=1

DATA_PATH=./datas/openwebtext_text_document
CHECKPOINT_PATH=./checkpoints/gpt2_medium_sparse_test


python pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 512 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 2000 \
       --lr-decay-iters 1600 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ./datas/gpt2-vocab-50256.json \
       --merge-file ./datas/gpt2-merges-50256.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0003 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --use-wandb \
       --log-interval 1 \
       --save-interval 100 \
       --eval-interval 100 \
       --eval-iters 10 \
       --fp16 \
       --finetune \
       --no-load-optim | tee logs/gpt2_medium_sparse_test_$time.log


