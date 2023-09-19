import os
import ujson
import torch
import numpy as np
import tqdm

from colbert.search.index_loader import IndexLoader
from colbert.indexing.index_saver import IndexSaver
from colbert.indexing.collection_encoder import CollectionEncoder

from colbert.utils.utils import lengths2offsets, print_message, dotdict, flatten
from colbert.indexing.codecs.residual import ResidualCodec
from colbert.indexing.utils import optimize_ivf
from colbert.search.strided_tensor import StridedTensor
from colbert.modeling.checkpoint import Checkpoint
from colbert.utils.utils import print_message, batch
from colbert.data import Collection
from colbert.indexing.codecs.residual_embeddings import ResidualEmbeddings
from colbert.indexing.codecs.residual_embeddings_strided import (
    ResidualEmbeddingsStrided,
)
from colbert.indexing.utils import optimize_ivf

# For testing writing into new chunks, can set DEFAULT_CHUNKSIZE smaller (e.g. 1 or 2)
DEFAULT_CHUNKSIZE = 25000


class IndexUpdater:

    """
    IndexUpdater takes in a searcher and adds/remove passages from the searcher.
    A checkpoint for passage-encoding must be provided for adding passages.
    IndexUpdater can also persist the change of passages to the index on disk.

    Sample usage:

        index_updater = IndexUpdater(config, searcher, checkpoint)

        added_pids = index_updater.add(passages) # all passages added to searcher with their pids returned
        index_updater.remove(pids) # all pid within pids removed from searcher

        searcher.search() # the search now reflects the added & removed passages

        index_updater.persist_to_disk() # added & removed passages persisted to index on disk
        searcher.Searcher(index, config) # if we reload the searcher now from disk index, the changes we made persists

    """

    def __init__(self, config, searcher, checkpoint=None):
        self.config = config
        self.searcher = searcher
        self.index_path = searcher.index

        self.has_checkpoint = False
        if checkpoint:
            self.has_checkpoint = True
            self.checkpoint = Checkpoint(checkpoint, config)
            self.encoder = CollectionEncoder(config, self.checkpoint)

        self._load_disk_ivf()

        # variables to track removal / append of passages
        self.removed_pids = []
        self.first_new_emb = torch.sum(self.searcher.ranker.doclens).item()
        self.first_new_pid = len(self.searcher.ranker.doclens)

    def remove(self, pids):
        """
        Input:
            pids: list(int)
        Return: None

        Removes a list of pids from the searcher,
        these pids will no longer apppear in future searches with this searcher
        to erase passage data from index, call persist_to_disk() after calling remove()
        """
        invalid_pids = self._check_pids(pids)
        if invalid_pids:
            raise ValueError("Invalid PIDs", invalid_pids)

        print_message(f"#> Removing pids: {pids}...")
        self._remove_pid_from_ivf(pids)
        self.removed_pids.extend(pids)

    def create_embs_and_doclens(
        self, passages, embs_path="embs.pt", doclens_path="doclens.pt", persist=False
    ):
        # Extend doclens and embs of self.searcher.ranker
        embs, doclens = self.encoder.encode_passages(passages)
        compressed_embs = self.searcher.ranker.codec.compress(embs)

        if persist:
            torch.save(compressed_embs, embs_path)
            torch.save(doclens, doclens_path)
        return compressed_embs, doclens

    def update_searcher(self, compressed_embs, doclens, curr_pid):
        # Update searcher
        # NOTE: For codes and residuals, the tensors end with padding of length 512,
        # hence we concatenate the new appendage in front of the padding
        self.searcher.ranker.embeddings.codes = torch.cat(
            (
                self.searcher.ranker.embeddings.codes[:-512],
                compressed_embs.codes,
                self.searcher.ranker.embeddings.codes[-512:],
            )
        )
        self.searcher.ranker.embeddings.residuals = torch.cat(
            (
                self.searcher.ranker.embeddings.residuals[:-512],
                compressed_embs.residuals,
                self.searcher.ranker.embeddings.residuals[-512:],
            ),
            dim=0,
        )

        self.searcher.ranker.doclens = torch.cat(
            (self.searcher.ranker.doclens, torch.tensor(doclens))
        )

        # Build partitions for each pid and update IndexUpdater's current ivf
        start = 0
        ivf = self.curr_ivf.tolist()
        ivf_lengths = self.curr_ivf_lengths.tolist()
        for doclen in doclens:
            end = start + doclen
            codes = compressed_embs.codes[start:end]
            partitions, _ = self._build_passage_partitions(codes)
            ivf, ivf_lengths = self._add_pid_to_ivf(partitions, curr_pid, ivf, ivf_lengths)

            start = end
            curr_pid += 1
        
        assert start == sum(doclens)

        # Replace the current ivf with new_ivf
        self.curr_ivf = torch.tensor(ivf, dtype=self.curr_ivf.dtype)
        self.curr_ivf_lengths = torch.tensor(ivf_lengths, dtype=self.curr_ivf_lengths.dtype)

        # Update new ivf in searcher
        new_ivf_tensor = StridedTensor(
            self.curr_ivf, self.curr_ivf_lengths, use_gpu=False
        )
        assert new_ivf_tensor != self.searcher.ranker.ivf
        self.searcher.ranker.ivf = new_ivf_tensor

        # Rebuild StridedTensor within searcher
        self.searcher.ranker.set_embeddings_strided()

    def add(self, passages):
        """
        Input:
            passages: list(string)
        Output:
            passage_ids: list(int)

        Adds new passages to the searcher,
        to add passages to the index, call persist_to_disk() after calling add()
        """
        if not self.has_checkpoint:
            raise ValueError(
                "No checkpoint was provided at IndexUpdater initialization."
            )

        # Find pid for the first added passage
        start_pid = len(self.searcher.ranker.doclens)
        curr_pid = start_pid

        compressed_embs, doclens = self.create_embs_and_doclens(passages)
        self.update_searcher(compressed_embs, doclens, curr_pid)

        print_message(f"#> Added {len(passages)} passages from pid {start_pid}.")
        new_pids = list(range(start_pid, start_pid + len(passages)))
        return new_pids

    def persist_to_disk(self):
        """
        Persist all previous stored changes in IndexUpdater to index on disk,
        changes include all calls to IndexUpdater.remove() and IndexUpdater.add()
        before persist_to_disk() is called.
        """

        print_message("#> Persisting index changes to disk")

        # Propagate all removed passages to disk
        self._load_metadata()
        for pid in self.removed_pids:
            self._remove_passage_from_disk(pid)

        # Propagate all added passages to disk
        # Rationale: keep record of all added passages in IndexUpdater.searcher,
        # divide passages into chunks and create / write chunks here

        self._load_metadata()  # Reload after removal

        # Calculate avg number of passages per chunk
        curr_num_chunks = self.metadata["num_chunks"]
        last_chunk_metadata = self._load_chunk_metadata(curr_num_chunks - 1)
        if curr_num_chunks == 1:
            avg_chunksize = DEFAULT_CHUNKSIZE
        else:
            avg_chunksize = last_chunk_metadata["passage_offset"] / (
                curr_num_chunks - 1
            )
        print_message(f"#> Current average chunksize is: {avg_chunksize}.")

        # Calculate number of additional passages we can write to the last chunk
        last_chunk_capacity = max(
            0, avg_chunksize - last_chunk_metadata["num_passages"]
        )
        print_message(
            f"#> The last chunk can hold {last_chunk_capacity} additional passages."
        )

        # Find the first and last passages to be persisted
        pid_start = self.first_new_pid
        emb_start = self.first_new_emb
        pid_last = len(self.searcher.ranker.doclens)
        emb_last = (
            emb_start + torch.sum(self.searcher.ranker.doclens[pid_start:]).item()
        )

        # First populate the last chunk
        if last_chunk_capacity > 0:
            pid_end = min(pid_last, pid_start + last_chunk_capacity)
            emb_end = (
                emb_start
                + torch.sum(self.searcher.ranker.doclens[pid_start:pid_end]).item()
            )

            # Write to last chunk
            self._write_to_last_chunk(pid_start, pid_end, emb_start, emb_end)
            pid_start = pid_end
            emb_start = emb_end

        # Then create new chunks to hold the remaining added passages
        while pid_start < pid_last:
            pid_end = min(pid_last, pid_start + avg_chunksize)
            emb_end = (
                emb_start
                + torch.sum(self.searcher.ranker.doclens[pid_start:pid_end]).item()
            )

            # Write new chunk with id = curr_num_chunks
            self._write_to_new_chunk(
                curr_num_chunks, pid_start, pid_end, emb_start, emb_end
            )

            curr_num_chunks += 1
            pid_start = pid_end
            emb_start = emb_end

        assert pid_start == pid_last
        assert emb_start == emb_last

        # Update metadata
        print_message("#> Updating metadata for added passages...")
        self.metadata["num_chunks"] = curr_num_chunks
        self.metadata["num_embeddings"] += torch.sum(
            self.searcher.ranker.doclens
        ).item()
        metadata_path = os.path.join(self.index_path, "metadata.json")
        with open(metadata_path, "w") as output_metadata:
            ujson.dump(self.metadata, output_metadata)

        # Save current IVF to disk
        optimized_ivf_path = os.path.join(self.index_path, "ivf.pid.pt")
        torch.save((self.curr_ivf, self.curr_ivf_lengths), optimized_ivf_path)
        print_message(f"#> Persisted updated IVF to {optimized_ivf_path}")

        self.removed_pids = []
        self.first_new_emb = torch.sum(self.searcher.ranker.doclens).item()
        self.first_new_pid = len(self.searcher.ranker.doclens)

    # HELPER FUNCTIONS BELOW

    def _load_disk_ivf(self):
        print_message(f"#> Loading IVF...")

        if os.path.exists(os.path.join(self.index_path, "ivf.pid.pt")):
            ivf, ivf_lengths = torch.load(
                os.path.join(self.index_path, "ivf.pid.pt"), map_location="cpu"
            )
        else:
            assert os.path.exists(os.path.join(self.index_path, "ivf.pt"))
            ivf, ivf_lengths = torch.load(
                os.path.join(self.index_path, "ivf.pt"), map_location="cpu"
            )
            ivf, ivf_lengths = optimize_ivf(ivf, ivf_lengths, self.index_path)

        self.curr_ivf = ivf
        self.curr_ivf_lengths = ivf_lengths

    def _load_metadata(self):
        with open(os.path.join(self.index_path, "metadata.json")) as f:
            self.metadata = ujson.load(f)

    def _load_chunk_doclens(self, chunk_idx):
        doclens = []

        print_message("#> Loading doclens...")

        with open(os.path.join(self.index_path, f"doclens.{chunk_idx}.json")) as f:
            chunk_doclens = ujson.load(f)
            doclens.extend(chunk_doclens)

        doclens = torch.tensor(doclens)
        return doclens

    def _load_chunk_codes(self, chunk_idx):
        codes_path = os.path.join(self.index_path, f"{chunk_idx}.codes.pt")
        return torch.load(codes_path, map_location="cpu")

    def _load_chunk_residuals(self, chunk_idx):
        residuals_path = os.path.join(self.index_path, f"{chunk_idx}.residuals.pt")
        return torch.load(residuals_path, map_location="cpu")

    def _load_chunk_metadata(self, chunk_idx):
        with open(os.path.join(self.index_path, f"{chunk_idx}.metadata.json")) as f:
            chunk_metadata = ujson.load(f)
        return chunk_metadata

    def _get_chunk_idx(self, pid):
        for i in range(self.metadata["num_chunks"]):
            chunk_metadata = self._load_chunk_metadata(i)
            if (
                chunk_metadata["passage_offset"] <= pid
                and chunk_metadata["passage_offset"] + chunk_metadata["num_passages"]
                > pid
            ):
                return i
        raise ValueError("Passage ID out of range")

    def _check_pids(self, pids):
        invalid_pids = []
        for pid in pids:
            if pid < 0 or pid >= len(self.searcher.ranker.doclens):
                invalid_pids.append(pid)
        return invalid_pids

    def _remove_pid_from_ivf(self, pids):
        # Helper function for IndexUpdater.remove()

        new_ivf = []
        new_ivf_lengths = []
        runner = 0
        pids = set(pids)

        # Construct mask of where pids to be removed appear in ivf
        mask = torch.isin(self.curr_ivf, torch.tensor(list(pids)))
        indices = mask.nonzero()

        # Calculate end-indices of each centroid section in ivf
        section_end_indices = []
        c = 0
        for length in self.curr_ivf_lengths.tolist():
            c += length
            section_end_indices.append(c)

        # Record the number of pids removed from each centroid section
        removed_len = [0 for _ in range(len(section_end_indices))]
        j = 0
        for ind in indices:
            while ind >= section_end_indices[j]:
                j += 1
            removed_len[j] += 1

        # Update changes
        new_ivf = torch.masked_select(self.curr_ivf, ~mask)
        new_ivf_lengths = self.curr_ivf_lengths - torch.tensor(removed_len)

        new_ivf_tensor = StridedTensor(new_ivf, new_ivf_lengths, use_gpu=False)
        assert new_ivf_tensor != self.searcher.ranker.ivf
        self.searcher.ranker.ivf = new_ivf_tensor

        self.curr_ivf = new_ivf
        self.curr_ivf_lengths = new_ivf_lengths

    def _build_passage_partitions(self, codes):
        # Helper function for IndexUpdater.add()
        # Return a list of ordered, unique centroid ids from codes of a passage
        codes = codes.sort()
        ivf, values = codes.indices, codes.values
        partitions, ivf_lengths = values.unique_consecutive(return_counts=True)
        return partitions, ivf_lengths

    def _add_pid_to_ivf(self, partitions, pid, old_ivf, old_ivf_lengths):
        """
        Helper function for IndexUpdater.add()

        Input:
            partitions: list(int), centroid ids of the passage
            pid: int, passage id
        Output: None

        Adds the pid of new passage into the ivf.
        """
        new_ivf = []
        new_ivf_lengths = []

        partitions_runner = 0
        ivf_runner = 0
        for i in range(len(old_ivf_lengths)):
            # First copy existing partition pids to new ivf
            new_ivf.extend(old_ivf[ivf_runner : ivf_runner + old_ivf_lengths[i]])
            new_ivf_lengths.append(old_ivf_lengths[i])
            ivf_runner += old_ivf_lengths[i]

            # Add pid if partition_index i is in the passage's partitions
            if (
                partitions_runner < len(partitions)
                and i == partitions[partitions_runner]
            ):
                new_ivf.append(pid)
                new_ivf_lengths[-1] += 1
                partitions_runner += 1

        assert ivf_runner == len(old_ivf)
        assert sum(new_ivf_lengths) == len(new_ivf)

        return new_ivf, new_ivf_lengths

    def _write_to_last_chunk(self, pid_start, pid_end, emb_start, emb_end):
        # Helper function for IndexUpdater.persist_to_disk()

        print_message(f"#> Writing {pid_end - pid_start} passages to the last chunk...")
        num_chunks = self.metadata["num_chunks"]

        # Append to current last chunk
        curr_embs = ResidualEmbeddings.load(self.index_path, num_chunks - 1)
        curr_embs.codes = torch.cat(
            (curr_embs.codes, self.searcher.ranker.embeddings.codes[emb_start:emb_end])
        )
        curr_embs.residuals = torch.cat(
            (
                curr_embs.residuals,
                self.searcher.ranker.embeddings.residuals[emb_start:emb_end],
            )
        )
        path_prefix = os.path.join(self.index_path, f"{num_chunks - 1}")
        curr_embs.save(path_prefix)

        # Update doclen of last chunk
        curr_doclens = self._load_chunk_doclens(num_chunks - 1).tolist()
        curr_doclens.extend(self.searcher.ranker.doclens.tolist()[pid_start:pid_end])
        doclens_path = os.path.join(self.index_path, f"doclens.{num_chunks - 1}.json")
        with open(doclens_path, "w") as output_doclens:
            ujson.dump(curr_doclens, output_doclens)

        # Update metadata of last chunk
        chunk_metadata = self._load_chunk_metadata(num_chunks - 1)
        chunk_metadata["num_passages"] += pid_end - pid_start
        chunk_metadata["num_embeddings"] += emb_end - emb_start
        chunk_metadata_path = os.path.join(
            self.index_path, f"{num_chunks - 1}.metadata.json"
        )
        with open(chunk_metadata_path, "w") as output_chunk_metadata:
            ujson.dump(chunk_metadata, output_chunk_metadata)

    def _write_to_new_chunk(self, chunk_idx, pid_start, pid_end, emb_start, emb_end):
        # Helper function for IndexUpdater.persist_to_disk()

        # Save embeddings to new chunk
        curr_embs = ResidualEmbeddings(
            self.searcher.ranker.embeddings.codes[emb_start:emb_end],
            self.searcher.ranker.embeddings.residuals[emb_start:emb_end],
        )
        path_prefix = os.path.join(self.index_path, f"{chunk_idx}")
        curr_embs.save(path_prefix)

        # Create doclen json file for new chunk
        curr_doclens = self.searcher.ranker.doclens.tolist()[pid_start:pid_end]
        doclens_path = os.path.join(self.index_path, f"doclens.{chunk_idx}.json")
        with open(doclens_path, "w+") as output_doclens:
            ujson.dump(curr_doclens, output_doclens)

        # Create metadata json file for new chunk
        chunk_metadata = {
            "passage_offset": pid_start,
            "num_passages": pid_end - pid_start,
            "embedding_offset": emb_start,
            "num_embeddings": emb_end - emb_start,
        }
        chunk_metadata_path = os.path.join(
            self.index_path, f"{chunk_idx}.metadata.json"
        )
        with open(chunk_metadata_path, "w+") as output_chunk_metadata:
            ujson.dump(chunk_metadata, output_chunk_metadata)

    def _remove_passage_from_disk(self, pid):
        # Helper function for IndexUpdater.persist_to_disk()

        chunk_idx = self._get_chunk_idx(pid)

        chunk_metadata = self._load_chunk_metadata(chunk_idx)
        i = pid - chunk_metadata["passage_offset"]
        doclens = self._load_chunk_doclens(chunk_idx)
        codes, residuals = (
            self._load_chunk_codes(chunk_idx),
            self._load_chunk_residuals(chunk_idx),
        )

        # Remove embeddings from codes and residuals
        start = sum(doclens[:i])
        end = start + doclens[i]
        codes = torch.cat((codes[:start], codes[end:]))
        residuals = torch.cat((residuals[:start], residuals[end:]))

        codes_path = os.path.join(self.index_path, f"{chunk_idx}.codes.pt")
        residuals_path = os.path.join(self.index_path, f"{chunk_idx}.residuals.pt")

        torch.save(codes, codes_path)
        torch.save(residuals, residuals_path)

        # Change doclen for passage to 0
        doclens = doclens.tolist()
        doclen_to_remove = doclens[i]
        doclens[i] = 0
        doclens_path = os.path.join(self.index_path, f"doclens.{chunk_idx}.json")
        with open(doclens_path, "w") as output_doclens:
            ujson.dump(doclens, output_doclens)

        # Modify chunk_metadata['num_embeddings'] for chunk_idx
        chunk_metadata["num_embeddings"] -= doclen_to_remove
        chunk_metadata_path = os.path.join(
            self.index_path, f"{chunk_idx}.metadata.json"
        )
        with open(chunk_metadata_path, "w") as output_chunk_metadata:
            ujson.dump(chunk_metadata, output_chunk_metadata)

        # Modify chunk_metadata['embedding_offset'] for all later chunks (minus num_embs_removed)
        for idx in range(chunk_idx + 1, self.metadata["num_chunks"]):
            metadata = self._load_chunk_metadata(idx)
            metadata["embedding_offset"] -= doclen_to_remove
            metadata_path = os.path.join(self.index_path, f"{idx}.metadata.json")
            with open(metadata_path, "w") as output_chunk_metadata:
                ujson.dump(metadata, output_chunk_metadata)

        # Modify num_embeddings in overall metadata (minus num_embs_removed)
        self.metadata["num_embeddings"] -= doclen_to_remove
        metadata_path = os.path.join(self.index_path, "metadata.json")
        with open(metadata_path, "w") as output_metadata:
            ujson.dump(self.metadata, output_metadata)
