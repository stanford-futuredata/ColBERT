import os
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from collections import defaultdict, namedtuple
import time
import heapq
from typing import Dict
import tqdm
import orjson


InvertedIndex = namedtuple(
    "InvertedIndex",
    ["inverted_index", "document_lengths", "num_documents", "avg_document_length"],
)


class Bm25Searcher:
    def __init__(self, index_path):
        super().__init__()

        nltk.download("stopwords")
        self.stopwords = set(stopwords.words("english"))

        self.load_inverted_index(index_path)
        self.index["document_lengths"] = {
            int(k): int(v) for k, v in self.index["document_lengths"].items()
        }

        self.index = InvertedIndex(**self.index)

        self.tokens = defaultdict(list)
        self.running_scores = defaultdict(lambda: defaultdict(float))

    def load_inverted_index(self, json_file_path: str):
        print(f"Loading inverted index...")

        file_size = os.path.getsize(json_file_path)
        with open(json_file_path, "rb") as f:
            progress_bar = tqdm.tqdm(
                total=file_size, unit="B", unit_scale=True, desc="Reading JSON"
            )

            json_bytes = bytearray()

            chunk_size = 1024 * 1024
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                json_bytes.extend(chunk)
                progress_bar.update(len(chunk))

            progress_bar.close()

        print(f"Parsing inverted index JSON...")
        start = time.time()
        self.index = orjson.loads(json_bytes)
        end = time.time()
        print(f"Finished parsing in {(end - start):.2f} seconds")

    def bm25_update_scores_for_token(
        self, token: str, scores: Dict[int, float], k1: float = 1.5, b: float = 0.75
    ):
        if token not in self.index.inverted_index:
            return

        postings = self.index.inverted_index[token]

        df = len(postings)
        idf = np.log((self.index.num_documents - df + 0.5) / (df + 0.5) + 1.0)

        for doc_id, tf in postings:
            doc_length = self.index.document_lengths[doc_id]
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (
                1 - b + b * (doc_length / self.index.avg_document_length)
            )
            score = idf * (numerator / denominator)

            scores[doc_id] += score

    def bm25_update_scores(self, query: str, scores: Dict[int, float]):
        query = query.translate(str.maketrans("", "", string.punctuation))
        tokens = [
            token for token in query.lower().split() if token not in self.stopwords
        ]

        for token in tokens:
            self.bm25_update_scores_for_token(token, scores)

    def get_top_k(self, query, k):
        running_scores = defaultdict(int)
        self.bm25_update_scores(query.output, running_scores)
        pids_and_scores = heapq.nlargest(
            k, list(running_scores.items()), key=lambda x: x[1]
        )
        return tuple(pids_and_scores)
