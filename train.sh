export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1
export TRIPLET=/home/jhkim980112/workspace/dataset/msmarco/triples.train.small.tsv
export PARALLEL=/home/jhkim980112/workspace/dataset/ko_en_parallel_corpus/ko_en_parallel_corpus/대화체.xlsx

CUDA_VISIBLE_DEVICES="3" \
    python -m colbert.train \
    --amp --doc_maxlen 180 --bsize 32 --accum 1 \
    --triples $TRIPLET --maxsteps 400_000 --parallel $PARALLEL --lp_loss True \
    --root ./experiments/ --experiment LPLOSS_MSMARCO --similarity l2 --run msmarco.clir.l2