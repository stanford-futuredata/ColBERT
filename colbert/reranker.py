import os
import json
import pandas as pd
from typing import List
from ranx import Run, fuse

from colbert.utils.utils import print_message
from colbert.infra.run import Run as ColbertRun
from colbert.distillation.scorer import Scorer
from colbert.distillation.ranking_scorer import RankingScorer

class Reranker:
    """
        The Reranker class is responsible for reranking a given ranking.

        Example:-
            with Run().context(RunConfig(
                nranks=number_of_gpu_devices,
                root="path/to/experiments", 
                experiment="awesome_experiment", 
                name='awesome_run_name'
            )):
                ranking = Ranking(path = 'path/to/ranking/file')
                queries = Queries(path = 'path/to/query/file')
                collection = Collection(path = 'path/to/collection/file')

                reranker = Reranker(ranking=ranking, queries=queries, collection=collection)
                score_file = reranker.rerank(checkpoint)
    """

    def __init__(self, ranking=None, queries=None, collection=None):
        """
            The constructor for the Reranker class. The Reranker class is responsible
            for reranking a given set of queries and collections.
        """
        self.ranking = ranking
        self.queries = queries
        self.collection = collection


    def rerank(self, checkpoint, model_type="encoder-only", bsize=32, maxlen=180):
        """
            The rerank method is responsible for reranking a given set of queries and collections.
        """
        print_message("#> Starting..")
        scorer = Scorer(self.queries, self.collection, checkpoint, model_type=model_type, bsize=bsize, maxlen=maxlen)
        rank_scorer = RankingScorer(scorer, self.ranking)

        print(f">>> Generating on ckpt {checkpoint}")
        
        return rank_scorer.run()


    def fuse_rankings(self, rankings: List[str], strategy: str, params:dict = None):
        """
            The fuse_rankings method is responsible for fusing a given set of rankings. The
            strategy parameter is used to determine the fusion strategy to be used. The
            strategy parameter can be any one supported by the ranx library.
        """
        print_message("#> Starting..")
        
        runs = []
        for i, rank_file in enumerate(rankings):
            assert os.path.exists(rank_file), f"Ranking file {rank_file} does not exist."
            assert (rank_file.endswith('.tsv') or rank_file.endswith('.json')), f"Ranking file {rank_file} is not a tsv or json file."

            if rank_file.endswith('.json'):
                output_filename = f"rankings-{i+1}.tsv"
                rank_file = self.convert_json_scores_to_tsv(rank_file, output_filename)

            data = pd.read_csv(rank_file, sep = '\t', names=['qid', 'pid', 'rank', 'score'])
            data = data.drop(['rank'], axis=1)
            data['qid'] = data['qid'].map(str)
            data['pid'] = data['pid'].map(str)
            
            run = Run.from_df(
                df=data,
                q_id_col="qid",
                doc_id_col="pid",
                score_col="score",
            )
            runs.append(run)

        print_message("#> Fusing rankings..")

        combined_run = fuse(
            runs=runs,
            method=strategy,
            params=params
        )

        print_message("#> Saving fused ranking..")

        with ColbertRun().open('ensemble_ranking.json', 'w') as f:
            combined_run.save(f.name)
            ensemble_ranking_path = f.name

        print_message(f"#> Fusion complete. Saved fused ranking to {ensemble_ranking_path}")

        return ensemble_ranking_path


    def convert_json_scores_to_tsv(self, score_json:str, output_filename:str='rankings.tsv'):
        assert os.path.exists(score_json), f"Score file {score_json} does not exist."
        assert output_filename.endswith('.tsv'), f"Output filename {output_filename} is not a tsv file."

        with ColbertRun().open(score_json, 'r') as f:
            # Load the file line-by-line and attempt to parse each line as a JSON object
            data_list = []
            for line in f:
                data_list.append(json.loads(line))

        with ColbertRun().open(output_filename, 'w') as f:
            for item in data_list:
                qid = item[0]
                scores = item[1]
                
                scores.sort(reverse=True)  # Sort scores in descending order
                
                for rank, (score, pid) in enumerate(scores, start=1):
                    f.write(f"{qid}\t{pid}\t{rank}\t{score}\n")
            
            output_path = f.name

        return output_path