#! /bin/bash
TASK="LAMBADA"

export MASTER_ADDR=localhost
export MASTER_PORT=6000
VALID_DATA=./datas/lambada_test.jsonl
VOCAB_FILE=./datas/gpt2-vocab-30522.json
MERGE_FILE=./datas/gpt2-merges-30522.txt
CHECKPOINT_PATH=checkpoints/gpt2_medium_1024
# VOCAB_FILE=./datas/gpt2-vocab-50256.json
# MERGE_FILE=./datas/gpt2-merges-50256.txt
# CHECKPOINT_PATH=checkpoints/megatron_lm_345m_v0.0

COMMON_TASK_ARGS="--num-layers 24 \
                  --hidden-size 1024 \
                  --num-attention-heads 16 \
                  --seq-length 1024 \
                  --max-position-embeddings 1024 \
                  --fp16 \
                  --vocab-file $VOCAB_FILE"

python tasks/main.py \
       --task $TASK \
       $COMMON_TASK_ARGS \
       --valid-data $VALID_DATA \
       --tokenizer-type GPT2BPETokenizer \
       --strict-lambada \
       --merge-file $MERGE_FILE \
       --load $CHECKPOINT_PATH \
       --micro-batch-size 8 \
       --activations-checkpoint-method uniform \
       --log-interval 10 \
       --no-load-optim \
       --no-load-rng