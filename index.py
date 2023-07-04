from argparse import ArgumentParser
import os, torch

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

def main(args):
    
    print(torch.cuda.device_count())
    
    with Run().context(RunConfig(nranks=args.nranks, experiment="msmarco")):

        config = ColBERTConfig(
            model_name = args.model_name, 
            checkpoint = args.checkpoint,
            bsize=args.batch_size,
            root=args.root,
            ignore_scores = args.ignore_scores,
            doc_maxlen = args.doc_maxlen,
            nbits = args.nbits
        )
        
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        name = args.checkpoint.split("/")[-1] + ".msmarco.nbits=" + str(args.nbits)
        indexer.index(name=name, collection=args.collection, overwrite=args.overwrite)


if __name__=='__main__':
    
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--root', dest='root', default='./experiments', type=str)
    parser.add_argument('--collection', dest='collection',required=True, type=str)
    
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--overwrite', default=True, type=str)

    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--doc_maxlen', default=180, type=int)
    parser.add_argument('--nranks', default=1, type=int)
    parser.add_argument('--nbits', default=1, type=int)
    parser.add_argument('--ignore_scores', dest='ignore_scores', default=True)
        
    args = parser.parse_args()
    
    main(args)