from colbert.data import Queries
from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Searcher
from argparse import ArgumentParser

import os, torch


def main(args):
    
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            model_name = args.model_name, 
            checkpoint = args.checkpoint,
            root=args.root,
            doc_maxlen = args.doc_maxlen
        )
        name = args.checkpoint.split("/")[-1] + ".msmarco.nbits=" + str(args.nbits)
        searcher = Searcher(index=name, collection = args.collection, config=config)
        queries = Queries(args.queries)
        ranking = searcher.search_all(queries, k=args.top_k)
        ranking.save(name+".top_"  + str(args.top_k) +"ranking.tsv")


if __name__=='__main__':

    print(torch.cuda.device_count())    
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--root', dest='root', default='./experiments', type=str)
    parser.add_argument('--collection', dest='collection',required=True, type=str)
    parser.add_argument('--queries', dest='queries', required=True, type=str)

    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--checkpoint', type=str)

    parser.add_argument('--top_k', default=1000, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--doc_maxlen', default=180, type=int)
    parser.add_argument('--nranks', default=1, type=int)
    parser.add_argument('--nbits', default=1, type=int)
    parser.add_argument('--ignore_scores', dest='ignore_scores', default=True)
        
    args = parser.parse_args()
    
    main(args)

