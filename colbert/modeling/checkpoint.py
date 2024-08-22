import torch
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster


from colbert.modeling.tokenization import QueryTokenizer, DocTokenizer
from colbert.utils.amp import MixedPrecisionManager
from colbert.modeling.colbert import ColBERT


def pool_embeddings_hierarchical(
    p_embeddings,
    token_lengths,
    pool_factor,
    protected_tokens: int = 0,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    p_embeddings = p_embeddings.to(device)
    pooled_embeddings = []
    pooled_token_lengths = []
    start_idx = 0

    for token_length in tqdm(token_lengths, desc="Pooling tokens"):
        # Get the embeddings for the current passage
        passage_embeddings = p_embeddings[start_idx : start_idx + token_length]

        # Remove the tokens at protected_tokens indices
        protected_embeddings = passage_embeddings[:protected_tokens]
        passage_embeddings = passage_embeddings[protected_tokens:]

        # Cosine similarity computation (vector are already normalized)
        similarities = torch.mm(passage_embeddings, passage_embeddings.t())

        # Convert similarities to a distance for better ward compatibility
        similarities = 1 - similarities.cpu().numpy()

        # Create hierarchical clusters using ward's method
        Z = linkage(similarities, metric="euclidean", method="ward")
        # Determine the number of clusters we want in the end based on the pool factor
        max_clusters = (
            token_length // pool_factor if token_length // pool_factor > 0 else 1
        )
        cluster_labels = fcluster(Z, t=max_clusters, criterion="maxclust")

        # Pool embeddings within each cluster
        for cluster_id in range(1, max_clusters + 1):
            cluster_indices = torch.where(
                torch.tensor(cluster_labels == cluster_id, device=device)
            )[0]
            if cluster_indices.numel() > 0:
                pooled_embedding = passage_embeddings[cluster_indices].mean(dim=0)
                pooled_embeddings.append(pooled_embedding)

        # Re-add the protected tokens to pooled_embeddings
        pooled_embeddings.extend(protected_embeddings)

        # Store the length of the pooled tokens (number of total tokens - number of tokens from previous passages)
        pooled_token_lengths.append(len(pooled_embeddings) - sum(pooled_token_lengths))
        start_idx += token_length

    pooled_embeddings = torch.stack(pooled_embeddings)
    return pooled_embeddings, pooled_token_lengths


class Checkpoint(ColBERT):
    """
    Easy inference with ColBERT.

    TODO: Add .cast() accepting [also] an object instance-of(Checkpoint) as first argument.
    """

    def __init__(self, name, colbert_config=None, verbose: int = 3):
        super().__init__(name, colbert_config)
        assert self.training is False

        self.verbose = verbose

        self.query_tokenizer = QueryTokenizer(self.colbert_config, verbose=self.verbose)
        self.doc_tokenizer = DocTokenizer(self.colbert_config)

        self.amp_manager = MixedPrecisionManager(True)

    def query(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                Q = super().query(*args, **kw_args)
                return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        with torch.no_grad():
            with self.amp_manager.context():
                D = super().doc(*args, **kw_args)

                if to_cpu:
                    return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

                return D

    def queryFromText(
        self, queries, bsize=None, to_cpu=False, context=None, full_length_search=False
    ):
        if bsize:
            batches = self.query_tokenizer.tensorize(
                queries,
                context=context,
                bsize=bsize,
                full_length_search=full_length_search,
            )
            batches = [
                self.query(input_ids, attention_mask, to_cpu=to_cpu)
                for input_ids, attention_mask in batches
            ]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer.tensorize(
            queries, context=context, full_length_search=full_length_search
        )
        return self.query(input_ids, attention_mask)

    def docFromText(
        self,
        docs,
        bsize=None,
        keep_dims=True,
        to_cpu=False,
        showprogress=False,
        return_tokens=False,
        pool_factor=1,
        protected_tokens=0,
        clustering_mode: str = "hierarchical",
    ):
        assert keep_dims in [True, False, "flatten"]
        assert clustering_mode in ["hierarchical"]

        if bsize:
            text_batches, reverse_indices = self.doc_tokenizer.tensorize(
                docs, bsize=bsize
            )

            returned_text = []
            if return_tokens:
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]

            keep_dims_ = "return_mask" if keep_dims == "flatten" else keep_dims
            batches = [
                self.doc(input_ids, attention_mask, keep_dims=keep_dims_, to_cpu=to_cpu)
                for input_ids, attention_mask in tqdm(
                    text_batches, disable=not showprogress
                )
            ]

            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == "flatten":
                D, mask = [], []

                for D_, mask_ in batches:
                    D.append(D_)
                    mask.append(mask_)

                D, mask = (
                    torch.cat(D)[reverse_indices],
                    torch.cat(mask)[reverse_indices],
                )

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.colbert_config.dim)
                D = D[mask.bool().flatten()].cpu()

                if pool_factor > 1:
                    print(f"Clustering tokens with a pool factor of {pool_factor}")
                    D, doclens = pool_embeddings_hierarchical(
                        D,
                        doclens,
                        pool_factor=pool_factor,
                        protected_tokens=protected_tokens,
                    )

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.doc_tokenizer.tensorize(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)

    def lazy_rank(self, queries, docs):
        Q = self.queryFromText(queries, bsize=128, to_cpu=True)
        D = self.docFromText(docs, bsize=128, to_cpu=True)

        assert False, "Implement scoring"

    def score(self, Q, D, mask=None, lengths=None):
        assert False, "Call colbert_score"
        # EVENTUALLY: Just call the colbert_score function!

        if lengths is not None:
            assert mask is None, "don't supply both mask and lengths"

            mask = torch.arange(D.size(1), device=self.device) + 1
            mask = mask.unsqueeze(0) <= lengths.to(self.device).unsqueeze(-1)

        scores = D @ Q
        scores = scores if mask is None else scores * mask.unsqueeze(-1)
        scores = scores.max(1)

        return scores.values.sum(-1).cpu()


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(
        bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype
    )

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, : x.size(1)] = x
        offset = endpos

    return output


"""
TODO:

def tokenize_and_encode(checkpoint, passages):
    embeddings, token_ids = checkpoint.docFromText(passages, bsize=128, keep_dims=False, showprogress=True, return_tokens=True)
    tokens = [checkpoint.doc_tokenizer.tok.convert_ids_to_tokens(ids.tolist()) for ids in token_ids]
    tokens = [tokens[:tokens.index('[PAD]') if '[PAD]' in tokens else -1] for tokens in tokens]
    tokens = [[tok for tok in tokens if tok not in checkpoint.skiplist] for tokens in tokens]

    return embeddings, tokens

"""
