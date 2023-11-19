#!/bin/bash

#SBATCH -J  Col-mBERT-ZS
#SBATCH -o  sbatch_output/col-mBERT-ZS.%j.out
#SBATCH -p  A100-80GB
#SBATCH -t  3-00:00:00
#SBATCH   --gres=gpu:1
#SBATCH   --nodes=1
#SBTACH   --ntasks=1
#SBATCH   --tasks-per-node=1
#SBATCH   --cpus-per-task=1

export NCCL_DEBUG=INFO
export NCCL_SHM_DISABLE=1
export GIT_PYTHON_REFRESH=quiet

export EXP=Col-mBERT-ZS
export BASE_MODEL=bert-base-multilingual-cased

export TRIPLET=/home/jhkim980112/workspace/dataset/mMARCO/zeroshot/qidpidtriples.rnd-shuf.train.tsv
export COLLECTION=/home/jhkim980112/workspace/dataset/mMARCO/google_translations/collections/english_collection.tsv
export QUERY=/home/jhkim980112/workspace/dataset/mMARCO/google_translations/queries/train/english_queries.train.tsv

python -m colbert.train \
    --base_model $BASE_MODEL \
    --triples $TRIPLET --collection $COLLECTION --queries $QUERY \
    --amp --doc_maxlen 512 --bsize 64 --accum 1 --maxsteps 200_000 --similarity l2 \
    --root ./experiments/ --experiment $EXP  --run mMARCO_en-en

exit 0