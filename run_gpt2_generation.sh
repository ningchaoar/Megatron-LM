#!/bin/bash
# This example will start serving the 345M model.
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --node_rank 0 \
                  --master_addr localhost \
                  --master_port 6000"

CHECKPOINT_PATH=checkpoints/gpt2_medium
VOCAB_FILE=datas/gpt2-30522-vocab.json
MERGE_FILE=datas/gpt2-30522-merges.txt

pip install flask-restful

python -m torch.distributed.launch $DISTRIBUTED_ARGS tools/run_text_generation_server.py  \
       --tensor-model-parallel-size 1  \
       --pipeline-model-parallel-size 1  \
       --num-layers 24  \
       --hidden-size 1024  \
       --load $CHECKPOINT_PATH  \
       --num-attention-heads 16  \
       --max-position-embeddings 1024  \
       --tokenizer-type GPT2BPETokenizer  \
       --fp16  \
       --micro-batch-size 1  \
       --seq-length 1024  \
       --out-seq-length 1024  \
       --temperature 1.0  \
       --vocab-file $VOCAB_FILE  \
       --merge-file $MERGE_FILE  \
       --top_p 0.9  \
       --seed 42
