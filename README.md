### **UPDATE:** See the new version 0.2 on [this branch](https://github.com/stanford-futuredata/ColBERT/tree/v0.2).

----

# ColBERT: Contextualized Late Interaction over BERT

This is the reference implementation of the paper **ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT**, to appear at SIGIR'20 ([preprint](https://arxiv.org/abs/2004.12832)).

### Dependencies

ColBERT requires Python 3 and Pytorch 1 and uses the [HuggingFace Transformers](https://github.com/huggingface/transformers) library. You can create a conda environment with the required dependencies using the `conda_requirements.txt` file.

```
conda create --name <env> --file conda_requirements.txt
```

----


### Data

This repository works directly with the data format of the [MS MARCO Passage Ranking](https://github.com/microsoft/MSMARCO-Passage-Ranking) dataset. You will need the training triples (`triples.train.small.tar.gz`), the official top-1000 ranked lists for the dev set queries (`top1000.dev`), and the dev set relevant passages (`qrels.dev.small.tsv`). For indexing the full collection, you will also need the list of passages (`collection.tar.gz`).

To avoid specifying the directory where you download this data on every command, it might be useful to modify `DEFAULT_DATA_DIR` in `src/parameters.py` to your data directory.

----

### Training

Training requires a list of _<query, positive passage, negative passage>_ tab-separated triples. Out of the box, this works with MS MARCO Passage Ranking's `triples.train.small.tsv` (see above for Data).

Example command:
```
python -m src.train --triples triples.train.small.tsv [--data_dir <path>] [--dim 128] [--maxsteps 400000] [--bsize 32] [-accum 2] [...]
```

Refer to `src/train.py` for the complete list of arguments and their defaults.


#### Pretrained model

To be released soon.

----

### Evaluation

Before indexing into ColBERT, you can evaluate the model at re-ranking a pre-defined top-k set per query. This evaluation will use ColBERT _on-the-fly_. That is, it will compute document representations _during_ query evaluation. For offline indexing and efficient ranking, see Indexing below.

This script requires the top-k list per query, provieded as a tab-separated file whose every line contains a quadruple _<query ID, passage ID, query text, passage text>_. This is the format of MS MARCO's `top1000.dev` and `top1000.eval`. Additionally, you can optionally supply the relevance judgements (qrels) for evaluation. This is a tab-separated file whose every line has a quadruple _<query ID, 0, passage ID, 1>_, like `qrels.dev.small.tsv`.

Example command:

```
python -m src.test --checkpoint colbert.dnn --topk top1000.dev [--qrels qrels.dev.small.tsv] [--output_dir <path>] [...]
```

Refer to `src/test.py` for the complete list of arguments and their defaults.


----

### Indexing

For efficient retrieval and much faster re-ranking, you can precompute the document representations with ColBERT. This step requires a tab-separated file, whose every line contains a passage ID alongside the passage's content. Out of the box, this works with MS MARCO Passage Ranking's `collection.tsv`.

Example command:

```
python -m src.index --index <index_name> --collection collection.tsv --checkpoint colbert.dnn [--bsize <n>] [--output_dir <path>] [...]
```

Indexing uses all GPUs visible to the process. To limit those to, say, GPUs #0 and #2, you can prepend `CUDA_VISIBLE_DEVICES="0,2"` to the command.



#### Using the index for efficient re-ranking

Example command:
```
python -m src.rerank --index <index_name> --checkpoint colbert.dnn --topk top1000.dev [--qrels qrels.dev.small.tsv]
```


#### Indexing for end-to-end retrieval from the full collection

To be released soon. This step uses [faiss](https://github.com/facebookresearch/faiss) for fast vector-similarity search.


#### Using the index for end-to-end retrieval

To be released soon.

```
python -m src.retrieve --index <index_name> --checkpoint colbert.dnn [--qrels qrels.dev.small.tsv]
```
