export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1

export EXP=xlmr_base-msmarco-aihub-en_only

export TRIPLET=/home/jhkim980112/workspace/dataset/msmarco/triples.train.small.tsv
export PARALLEL=/home/jhkim980112/workspace/dataset/ko_en_parallel_corpus/ko_en_parallel_corpus/aihub_ko_en_parallel.tsv
export BASE_MODEL=xlm-roberta-base
export using_lp_loss=True

CUDA_VISIBLE_DEVICES="0" \
    python -m colbert.train \
    --triples $TRIPLET \
    --parallel $PARALLEL \
    --base_model $BASE_MODEL \
    --amp --doc_maxlen 180 --bsize 32 --accum 1 --maxsteps 200_000 --similarity l2\
    --root ./experiments/ --experiment $EXP  --run msmarco.clir.l2