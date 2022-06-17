#! /bin/bash
TASK="WIKITEXT103"

export MASTER_ADDR=localhost
export MASTER_PORT=6000
VALID_DATA=./datas/wikitext-103-v1/wikitext-103/wiki.test.tokens
VOCAB_FILE=./datas/gpt2-vocab-50256.json
MERGE_FILE=./datas/gpt2-merges-50256.txt
CHECKPOINT_PATH=checkpoints/gpt3xl_openwebtext_bs16_gbs512_lr2e-4

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 2048 \
                  --num-attention-heads 16 \
                  --seq-length 2048 \
                  --max-position-embeddings 2048 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --activations-checkpoint-method uniform \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng