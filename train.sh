export triples=/docker_workspace/dataset/ms_marco/qidpidtriples.train.full.json
export queries=/docker_workspace/dataset/ms_marco/queries.train.tsv
export collection=/docker_workspace/dataset/ms_marco/collection_origin.tsv
export checkpoint=/docker_workspace/workspace/ColBERT/experiments/msmarco/none/2023-06/24/10.07.35/checkpoints/colbert-400000

export batch_size=24
export nranks=1
export max_steps=16_000_000

CUDA_VISIBLE_DEVICES="1" python train.py --triples $triples --queries $queries --collection $collection \
    --batch_size $batch_size --max_steps $max_steps  --nranks $nranks --model_name bert-base-multilingual-uncased \
    --ignore_scores True --checkpoint $checkpoint