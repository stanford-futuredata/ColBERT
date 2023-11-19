export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1

export EXP=Col-mBERT-ZS

export TRIPLET=/home/jhkim980112/workspace/dataset/msmarco/qidpidtriples.rnd-shuf.train.tsv
export BASE_MODEL=bert-base-multilingual-cased

CUDA_VISIBLE_DEVICES="0" \
    python -m colbert.train \
    --triples $TRIPLET \
    --base_model $BASE_MODEL \
    --amp --doc_maxlen 180 --bsize 32 --accum 1 --maxsteps 200_000 --similarity l2\
    --root ./experiments/ --experiment $EXP  --run mMARCO_en-en