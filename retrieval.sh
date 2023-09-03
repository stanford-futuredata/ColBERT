
CHECKPOINT=/home/jhkim980112/workspace/code/ColBERT/experiments/LPLOSS_MSMARCO/train.py/msmarco.clir.l2/checkpoints/colbert-200000.dnn
QUERY=/home/jhkim980112/workspace/dataset/miracl/topics.miracl-v1.0-ko-dev.tsv

CUDA_VISIBLE_DEVICES="2" \
    python -m colbert.retrieve \
    --amp --doc_maxlen 180 --mask-punctuation --bsize 256 \
    --queries $QUERY \
    --nprobe 32 --faiss_depth 1024 --depth 1000 --similarity l2 \
    --index_root ./indexes/ --index_name MSMARCO.LP.L2.32x200k \
    --checkpoint $CHECKPOINT --root ./experiments/ --experiment LPLOSS_MSMARCO