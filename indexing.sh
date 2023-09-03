export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1

CUDA_VISIBLE_DEVICES="3" \
    python  -m \
    colbert.index --amp --doc_maxlen 180 --mask-punctuation --bsize 256 --similarity l2 \
    --checkpoint /home/jhkim980112/workspace/code/ColBERT/experiments/ZS_MSMARCO/train.py/msmarco.clir.l2/checkpoints/colbert-200000.dnn \
    --collection /home/jhkim980112/workspace/dataset/miracl/ko_docs.tsv \
    --index_root ./indexes/ --index_name MSMARCO.ZS.L2.32x200k \
    --root ./experiments/ --experiment ZS_MSMARCO