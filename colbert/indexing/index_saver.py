import os
import queue
import ujson
import threading

from contextlib import contextmanager

from colbert.indexing.codecs.residual import ResidualCodec

from colbert.utils.utils import print_message


class IndexSaver():
    def __init__(self, config):
        self.config = config

    def save_codec(self, codec):
        codec.save(index_path=self.config.index_path_)

    def load_codec(self):
        return ResidualCodec.load(index_path=self.config.index_path_)

    @contextmanager
    def thread(self):
        self.codec = self.load_codec()

        self.saver_queue = queue.Queue(maxsize=3)
        thread = threading.Thread(target=self._saver_thread)
        thread.start()

        try:
            yield

        finally:
            self.saver_queue.put(None)
            thread.join()

            del self.saver_queue
            del self.codec

    def save_chunk(self, chunk_idx, offset, embs, doclens):
        compressed_embs = self.codec.compress(embs)

        self.saver_queue.put((chunk_idx, offset, compressed_embs, doclens))

    def _saver_thread(self):
        for args in iter(self.saver_queue.get, None):
            self._write_chunk_to_disk(*args)

    def _write_chunk_to_disk(self, chunk_idx, offset, compressed_embs, doclens):
        path_prefix = os.path.join(self.config.index_path_, str(chunk_idx))
        compressed_embs.save(path_prefix)

        doclens_path = os.path.join(self.config.index_path_, f'doclens.{chunk_idx}.json')
        with open(doclens_path, 'w') as output_doclens:
            ujson.dump(doclens, output_doclens)

        metadata_path = os.path.join(self.config.index_path_, f'{chunk_idx}.metadata.json')
        with open(metadata_path, 'w') as output_metadata:
            metadata = {'passage_offset': offset, 'num_passages': len(doclens), 'num_embeddings': len(compressed_embs)}
            ujson.dump(metadata, output_metadata)
