# ColBERT: Contextualized Late Interaction over BERT

This is version 0.2.0 of the reference implementation of the SIGIR'20 paper **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**. This version is a near-complete rewrite with lots of additional functionality and considerably higher efficiency.

For conda installation, use the provided `conda_env.yml`. ColBERT (new version 0.2.x) requires Python 3.7 and Pytorch 1.6 and uses [HuggingFace Transformers](https://github.com/huggingface/transformers) 3.0. 

Here are some example commands for use:

### Training:

```
python -m torch.distributed.launch --nproc_per_node=4 -m colbert.train --amp --doc_maxlen 180 --mask-punctuation --bsize 32 --accum 1 --triples /path/to/MSMARCO/triples.train.small.tsv --root /root/to/experiments/ --experiment MSMARCO-psg --similarity l2 --run msmarco.psg.l2
```

You can compare a few checkpoints on your validation set by re-ranking (prior to indexing) with `colbert.test`.

### Indexing:

Once you have determined a good checkpoint, you can index your collection for fast retrieval and ranking:

```
CUDA_VISBILE_DEVICES="0,1,2,3" OMP_NUM_THREADS=6 python -m torch.distributed.launch --nproc_per_node=4 -m colbert.index --root /root/to/experiments/ --doc_maxlen 180 --mask-punctuation --bsize 256 --amp --checkpoint /root/to/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn --index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k --collection /path/to/MSMARCO/collection.tsv
```

### Faiss Indexing, for end-to-end retrieval:

```
python -m colbert.index_faiss --index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k --partitions 32768 --root /root/to/experiments/ --sample 0.3
```

### End-to-end retrieval:

```
python -m colbert.retrieve --nprobe 32 --partitions 32768 --faiss_depth 1024 --index_root /root/to/indexes/ --index_name MSMARCO.L2.32x200k --doc_maxlen 180 --mask-punctuation --bsize 256 --amp --checkpoint /root/to/experiments/MSMARCO-psg/train.py/msmarco.psg.l2/checkpoints/colbert-200000.dnn  --root /root/to/experiments/ --queries /path/to/MSMARCO/queries.dev.small.tsv
```
