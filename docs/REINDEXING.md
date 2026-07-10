# Reindexing Guide

When the pipeline, parser, chunker, or embedding model version changes, existing
documents must be reindexed. The procedure ensures no downtime for search and
no data loss.

## Version Components

| Component | Constant | Location |
|---|---|---|
| Pipeline version | `PIPELINE_VERSION` | `src/rag_platform/ingestion/service.py` |
| Parser version | `PARSER_VERSION` | `src/rag_platform/ingestion/service.py` |
| Embedding model | `openai_embedding_model` | Config / `.env` |

Bump the relevant version before starting a reindex. Derivative artifacts in
object storage use `pipeline_version` and `embedding_model` in their keys, so
old artifacts remain available while new ones are generated.

## Zero-Downtime Reindex Procedure

### 1. Bump Versions

```python
# src/rag_platform/ingestion/service.py
PIPELINE_VERSION = "2026-07-10.1"   # old
PIPELINE_VERSION = "2026-07-15.1"   # new
```

If the embedding model changes, update the config:

```dotenv
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 2. Create New Qdrant Collection

A new collection is automatically created at startup because
`QDRANT_COLLECTION_PREFIX` + model name produces a new identifier.
Old collections are not deleted.

### 3. Re-ingest Documents

Run the reindex script against all active documents:

```bash
uv run python -m rag_platform.scripts.reindex \
  --tenant-id 00000000-0000-0000-0000-000000000001 \
  --pipeline-version 2026-07-15.1
```

This creates new `DocumentVersion` rows and re-processes all clean PDFs
through the updated pipeline. Old versions remain ACTIVE during processing
so queries continue to return results.

### 4. Validate New Index

```bash
uv run python -m rag_platform.scripts.validate-reindex \
  --old-pipeline 2026-07-10.1 \
  --new-pipeline 2026-07-15.1 \
  --sample-count 1000
```

Compares chunk counts, SHA-256 distribution, and retrieves a sample
from both old and new indexes. Reports discrepancies.

### 5. Switch Read Alias

Update `documents.current_version_id` for all reindexed documents
to point to the new version. The retrieval service post-validates
every result against the current version, so this switch is atomic.

```bash
uv run python -m rag_platform.scripts.promote-versions \
  --pipeline-version 2026-07-15.1
```

### 6. Shadow Query and Validate Quality

Run the evaluation suite against both old and new indexes:

```bash
curl -X POST http://localhost:8000/v1/evaluation/run \
  -H "X-Tenant-ID: ..." \
  -H "X-Subject-ID: ..." \
  --data '{"corpus_id": "...", "dataset_name": "golden-v1"}'
```

Compare Recall@20, faithfulness, and citation precision.
Promote only when the new pipeline meets or exceeds the old baseline.

### 7. Clean Up Old Projections

After a rollback window (default 7 days), delete old vector points,
BM25 entries, graph nodes, and derivative artifacts:

```bash
uv run python -m rag_platform.scripts.purge-pipeline \
  --pipeline-version 2026-07-10.1
```

Set the pipeline's `state` to `PURGED` and send deletion events to Kafka
for the asynchronous delete worker.

## Rollback

If the new pipeline produces worse results:

1. Revert `PIPELINE_VERSION` and config changes.
2. Switch `documents.current_version_id` back to the previous version.
3. Delete the new collection and derivative artifacts.
4. The old index is untouched and immediately authoritative.

## Embedding Model Migration

When changing the embedding model:

1. Set `MODEL_PROVIDER=openai` and update `OPENAI_EMBEDDING_MODEL`.
2. The new collection name changes automatically.
3. Reindex all documents (the new embedding model is used for all vectors).
4. Both old and new collections coexist. Queries use the model-specific
   collection based on the current config.
5. Validate recall quality on a golden dataset before promoting.
6. Delete the old collection after the rollback window.

No downtime. No deleted vectors. No unsearchable gap.
