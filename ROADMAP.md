# stanford-futuredata/ColBERT Roadmap

## Guiding Philosophy
stanford-futuredata/ColBERT should remain a stable, canonical reference implementation of late interaction, especially for newcomers to late interaction and the GPU-poor*. Folks looking for the bleeding-edge of late interaction should check out the fantastic [lightonai/PyLate library](https://github.com/lightonai/pylate).

\*stanford-futuredata/ColBERT's coupled model-index design enables [batched encoding](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/indexing/collection_indexer.py#L376) with immediate compression, maintaining [sub-5GB memory usage](https://vishalbakshi.github.io/blog/posts/2025-02-14-RAGatouille-ColBERT-Memory-Profiling/#profiling-results) even for multi-million document collections.

## Immediate Goal: Dependency Stabilization (~3 months)

- Upgrade PyTorch to 2.x
- Upgrade transformers (remove deprecated AdamW)
- Replace faiss with [fastkmeans](https://github.com/AnswerDotAI/fastkmeans) 
- Test Python 3.9-3.12 compatibility
- Resolve crypt.h/ninja errors.
- Merge distributed training fix in [#258](https://github.com/stanford-futuredata/ColBERT/pull/258/files#diff-12632f8041dc63139b026f92118749d36110bc0fbbbd6180206b3109fc694c7f) (potentially related: [#132](https://github.com/stanford-futuredata/ColBERT/issues/132) and [#233](https://github.com/stanford-futuredata/ColBERT/issues/233))
- Replace git-python with GitPython in PyPI (already changed in repo in commit [736f88b](https://github.com/stanford-futuredata/ColBERT/commit/736f88b981078a2c8687c8ee33c0f390e01284cd))

## Medium-Term Goals: Documentation and Bug Fixes (~6 months)

- Update documentation
  - Address documentation updates in issues/PRs ([#316](https://github.com/stanford-futuredata/ColBERT/pull/316), [#153](https://github.com/stanford-futuredata/ColBERT/issues/153), [#167](https://github.com/stanford-futuredata/ColBERT/issues/167), etc.).
  - Create an llms.txt and llms_ctx.txt for the repo.
- Investigate issues:
  - Bug ([#159](https://github.com/stanford-futuredata/ColBERT/issues/159), [#317](https://github.com/stanford-futuredata/ColBERT/issues/317), [#360](https://github.com/stanford-futuredata/ColBERT/issues/360), etc.) 
  - Training ([#262](https://github.com/stanford-futuredata/ColBERT/issues/262), [#265](https://github.com/stanford-futuredata/ColBERT/issues/265), [#291](https://github.com/stanford-futuredata/ColBERT/issues/291), etc.).
  - IndexUpdater ([#180](https://github.com/stanford-futuredata/ColBERT/issues/180), [#261](https://github.com/stanford-futuredata/ColBERT/issues/261), [#276](https://github.com/stanford-futuredata/ColBERT/issues/276), etc.).
  - Multi-GPU ([#158](https://github.com/stanford-futuredata/ColBERT/issues/158), [#265](https://github.com/stanford-futuredata/ColBERT/issues/265), [#318](https://github.com/stanford-futuredata/ColBERT/issues/318), etc.).
  - Ready-to-close issues after review/repro ([#139](https://github.com/stanford-futuredata/ColBERT/issues/139), [#179](https://github.com/stanford-futuredata/ColBERT/issues/179), [#335](https://github.com/stanford-futuredata/ColBERT/issues/335), etc.).
  - etc.
  
## Long-Term Goals: Feature Requests (~3 months)

- Resuming training from checkpoint ([#307](https://github.com/stanford-futuredata/ColBERT/issues/307)).
- Allow string pids ([#326](https://github.com/stanford-futuredata/ColBERT/pull/326)).
- Explore [batch size handling options](https://github.com/stanford-futuredata/ColBERT/blob/8627585ad290c21720eaa54e325e7c8c301d15f6/colbert/search/index_storage.py#L121) to resolve OOM during search.
- etc.
