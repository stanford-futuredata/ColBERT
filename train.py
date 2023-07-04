from argparse import ArgumentParser
import os, torch

from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Trainer

def main(args):
    
    print(torch.cuda.device_count())
    
    with Run().context(RunConfig(nranks=args.nranks, experiment="msmarco")):

        config = ColBERTConfig(
            model_name = args.model_name, 
            checkpoint = args.checkpoint,
            bsize=args.batch_size,
            root=args.root,
            maxsteps = args.max_steps, 
            ignore_scores = args.ignore_scores,
            doc_maxlen = args.doc_maxlen
        )
        trainer = Trainer(
            triples = args.triples,
            queries = args.queries,
            collection = args.collection,
            config=config,
        )

        checkpoint_path = trainer.train()

        print(f"Saved checkpoint to {checkpoint_path}...")


if __name__=='__main__':
    
    parser = ArgumentParser(description="msmarco_passages.")

    # Input Arguments.
    parser.add_argument('--root', dest='root', default='./experiments', type=str)
    parser.add_argument('--triples', dest='triples', required=True, type=str)
    parser.add_argument('--queries', dest='queries', required=True, type=str)
    parser.add_argument('--collection', dest='collection',required=True, type=str)
    
    parser.add_argument('--model_name', default='bert-base-uncased', type=str)
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--doc_maxlen', default=180, type=int)
    parser.add_argument('--nranks', default=1, type=int)
    parser.add_argument('--ignore_scores', dest='ignore_scores', default=True)
    parser.add_argument('--max_steps', default=200_000, type=int)
        
    args = parser.parse_args()
    
    main(args)