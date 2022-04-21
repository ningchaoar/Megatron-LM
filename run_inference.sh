#! /bin/bash
TASK="WIKITEXT103"

export MASTER_ADDR=localhost
export MASTER_PORT=6000
VALID_DATA=./datas/wikitext-103-v1/wikitext-103/wiki.test.tokens
VOCAB_FILE=./datas/gpt2-vocab-50256.json
MERGE_FILE=./datas/gpt2-merges-50256.txt

COMMON_TASK_ARGS="--num-layers 12 \
                  --hidden-size 768 \
                  --num-attention-heads 12 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

# COMMON_TASK_ARGS="--num-layers 48 \
#                   --hidden-size 1600 \
#                   --num-attention-heads 25 \
#                   --seq-length 1024 \
#                   --max-position-embeddings 1024 \
#                   --fp16 \
#                   --vocab-file $VOCAB_FILE"

python inference_gpt.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --merge-file $MERGE_FILE \
       --micro-batch-size 1 \
       --activations-checkpoint-method uniform \
       --log-interval 1 \
       --no-load-optim \
       --no-load-rng