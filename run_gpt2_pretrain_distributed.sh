#! /bin/bash
time=$(date "+%Y%m%d%H%M%S")
# Runs the "345M" parameter model
GPUS_PER_NODE=8
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=./datas/my-gpt2_text_document
CHECKPOINT_PATH=./checkpoints/medium_wikicorpus_50256_128_lamb_bs16384

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 256 \
       --global-batch-size 16384 \
       --seq-length 128 \
       --max-position-embeddings 128 \
       --train-iters 50000 \
       --lr-decay-iters 40000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file ./datas/gpt2-vocab-50256.json \
       --merge-file ./datas/gpt2-merges-50256.txt \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --optimizer lamb \
       --lr 0.006 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --activations-checkpoint-method uniform \
       --use-wandb \
       --log-interval 1 \
       --save-interval 1000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 \
       --no-save-optim \
       --no-save-rng 2>&1 | tee logs/medium_wikicorpus_50256_128_lamb_bs16384_$time.log
