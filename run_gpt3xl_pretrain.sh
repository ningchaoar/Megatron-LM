#! /bin/bash
time=$(date "+%Y%m%d%H%M%S")
# Runs the "345M" parameter model
GPUS_PER_NODE=2
# Change for multinode config
export MASTER_ADDR=localhost
export MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=./datas/openwebtext_text_document
CHECKPOINT_PATH=./checkpoints/gpt3xl_openwebtext_bs16_gbs512_lr2e-4_sparse0.97_block16_init0_start1000_end6000

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --micro-batch-size 16 \
       --global-batch-size 512 \
       --seq-length 2048 \
       --max-position-embeddings 2048 \
       --enable-sparse-mode \
       --train-iters 10000 \
       --lr-decay-iters 8000 \
       --save $CHECKPOINT_PATH \
       --load ./checkpoints/gpt3-xl_openwebtext_gbs512_lr2e-4 \
       --data-path $DATA_PATH \
       --vocab-file ./datas/gpt2-vocab-50256.json \
       --merge-file ./datas/gpt2-merges-50256.txt \
       --data-impl mmap \
       --split 980,19,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style cosine \
       --min-lr 5.0e-6 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction 0.1 \
       --activations-checkpoint-method uniform \
       --use-wandb \
       --log-interval 1 \
       --save-interval 100 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --fp16 2>&1 \
       --finetune | tee logs/gpt3xl_openwebtext_bs16_gbs512_lr2e-4_$time.log
