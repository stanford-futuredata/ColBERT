export queries=/docker_workspace/dataset/ms_marco/queries.dev.tsv
export collection=/docker_workspace/dataset/ms_marco/collection_origin.tsv
export checkpoint=/docker_workspace/workspace/ColBERT/experiments/msmarco/none/2023-06/24/colbert_BERT_base_90k_bs64
export model=bert-base-uncased

export top_k=1000
export batch_size=64
export nranks=1

CUDA_VISIBLE_DEVICES="0" python retrieval.py --queries $queries --collection $collection \
    --batch_size $batch_size --nranks $nranks --model_name $model \
    --ignore_scores True --checkpoint $checkpoint --top_k $top_k