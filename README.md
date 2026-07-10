# Atlas RAG

> An open, production-oriented starting guide for building controlled retrieval and agentic AI systems over millions of documents.

[![Python 3.12+](https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-16-000000?style=flat-square)](https://nextjs.org/)
[![Kubernetes](https://img.shields.io/badge/Kubernetes-ready-326CE5?style=flat-square)](https://kubernetes.io/)
[![Terraform](https://img.shields.io/badge/Terraform-AWS-844FBA?style=flat-square)](infra/stacks/aws)
[![License](https://img.shields.io/badge/License-Apache--2.0-4c1?style=flat-square)](LICENSE)

Atlas RAG is a working reference platform for teams that want to move beyond a single-vector-database demo. It combines secure PDF ingestion, structure-aware chunking, exact deduplication, hybrid retrieval, reranking, grounded citations, bounded tool use, streaming responses, observability, and cloud deployment assets in one repository.

Use the complete stack, adopt individual services, or use the architecture and contracts as a guide for your own deployment. The repository is intentionally modular: the application contracts stay stable while storage, model, search, identity, and infrastructure providers can be replaced.

The design target is demanding by intention: **10,000,000+ PDFs and hundreds of millions of chunks without an unfiltered vector flood**.

<p align="center">
  <img src="public-img/Ask%20the%20corpus.png" width="48%" alt="Grounded chat with citations, source metadata, and token usage" />
  <img src="public-img/Document%20inventory.png" width="48%" alt="Live document inventory showing real PDFs, processing states, and upload form" />
</p>
<p align="center">
  <img src="public-img/Infrastructure%20%26%20SLOs.png" width="48%" alt="Live SLO dashboard with real P95 latency, cache hit rate, dependency status, and budget burn" />
  <img src="public-img/Retrieval%20operations.png" width="48%" alt="Operations overview with live corpus metrics, throughput charts, and latency time series" />
</p>

## Why Atlas RAG

Most RAG examples stop after `embed -> vector search -> prompt`. That path is useful for a prototype, but it breaks down under large corpora, multi-tenancy, access control, document updates, deletion requirements, and production latency.

Atlas starts with different rules:

- Filter by tenant, ACL, corpus, type, date, classification, and active version before expensive search.
- Run PostgreSQL lexical search and filtered Qdrant ANN in parallel.
- Fuse ranks with Reciprocal Rank Fusion instead of mixing incompatible raw scores.
- Rerank a small candidate set, remove redundancy with MMR, and compress extractively.
- Register every source and validate citations before presenting a grounded answer.
- Treat PDFs, websites, tool output, and model output as untrusted data.
- Keep object storage authoritative and make search indexes rebuildable projections.
- Separate online query workloads from offline parsing, embedding, and indexing workloads.
- Prove quality, latency, isolation, deletion, backup, and rollback rather than assuming them.

The full rationale, capacity model, diagrams, SLOs, and phased roadmap are in [`HYPERSCALE_RAG_ARCHITECTURE.md`](HYPERSCALE_RAG_ARCHITECTURE.md).

## What Is Implemented

### Data and retrieval plane

- Quarantined PDF upload to MinIO or S3.
- SHA-256 source and chunk hashing.
- Active-content and optional ClamAV scanning.
- PyMuPDF structure extraction with page coordinates and metadata.
- Heading-aware semantic chunking with bounded token sizes and overlap.
- Idempotent Kafka ingestion with PostgreSQL stage state and bounded retries.
- Offline deterministic or OpenAI embeddings.
- Partition-ready Qdrant collections and indexed authorization payloads.
- PostgreSQL full-text retrieval with a production BM25/OpenSearch extension path.
- Metadata and ACL pre-filtering, dense ANN, sparse retrieval, RRF, reranking, MMR, and extractive compression.
- Redis exact, embedding, and vector-similarity semantic caches with corpus epochs.
- HMAC-backed source identifiers, page/section metadata, and citation validation.
- Tombstone-first deletion with asynchronous vector, object, graph, and cache cleanup.

### Agent and security plane

- Deterministic routes for internal RAG, GraphRAG, current web search, composite retrieval, and calculator requests.
- OpenAI-compatible generation with a key-free deterministic local provider.
- Tavily web search adapter.
- Neo4j document, chunk, entity, and relationship projection with bounded multi-hop retrieval.
- Auth0 JWT validation, tenant context, permissions, groups, and clearance.
- Browser-side Auth0 SPA support with in-memory tokens.
- Prompt-injection and credential-pattern controls.
- Bounded tool execution, safe arithmetic parsing, output inspection, abstention, and cancellation.
- Server-Sent Events for status, tools, sources, tokens, citations, usage, completion, and errors.

### Platform and interface

- Responsive Next.js operations console and grounded chat workspace.
- Live document inventory, upload, chat, health, and retrieval metrics when the API is available.
- Clearly labeled fixtures for screens whose operational/evaluation backends are not connected.
- Hardened backend and frontend containers.
- One-command Docker Compose environment.
- Helm chart with API, frontend, ingestion/deletion workers, migrations, HPA, KEDA, PDBs, topology spread, network policy, External Secrets, and ServiceMonitor support.
- Terraform/OpenTofu AWS foundation for VPC, EKS, RDS PostgreSQL, ElastiCache IAM, MSK Serverless IAM, S3, KMS, ECR, Secrets Manager, and workload identity.
- AWS IAM token generation for MSK and ElastiCache, with no static broker/cache passwords.
- GitHub Actions for backend/frontend quality, container scanning, SBOMs, Terraform validation, Helm validation, and digest-based GitOps promotion.

## Live Data Versus Fixtures

The UI does not silently present sample values as production telemetry.

| Screen | Data source | Status |
|---|---|---|
| Grounded chat | Live API, retrieval, generation, and SSE | Live when the backend is connected |
| Documents | Live PostgreSQL and ingestion API | Live when the backend is connected |
| Overview counts/latency/cache | Live PostgreSQL, Qdrant, and request logs | Live after migrations; fallback is labeled |
| Ingestion operations | Representative queue fixture | Explicitly labeled until the queue telemetry API is enabled |
| Evaluation | Representative RAGAS fixture | Explicitly labeled until evaluation datasets and judge models are configured |
| Infrastructure operations | Representative telemetry fixture | Explicitly labeled until the observability API is connected |

Local deterministic embeddings and answers are real executable code paths intended for integration testing, but they are not semantically equivalent to a production model. Set OpenAI and Tavily credentials to test the production provider paths.

## Architecture

```text
PDF sources
    |
    v
Quarantine -> scan -> parse/OCR -> semantic chunks -> SHA-256 dedup
    |                                          |
    |                                          v
    |                                 offline embeddings
    v                                          |
Object storage <-> PostgreSQL metadata/BM25 <-> Qdrant ANN
                         \                     /
                          \--- RRF fusion ----/
                                  |
                         CPU cross-encoder
                                  |
                    MMR + extractive compression
                                  |
                     evidence registry + citations
                                  |
User -> Auth0 -> router -> RAG / GraphRAG / Tavily / tools
                                  |
                        grounded LLM response
                                  |
                    SSE tokens, sources, and usage
```

## Quick Start

### Requirements

- Docker Desktop or Docker Engine with Compose.
- At least 8 GB of available memory for the complete local stack.
- `make` and `curl` for the convenience commands.

### Start the complete local platform

```bash
make quickstart
```

This builds and starts PostgreSQL, Redis Stack, MinIO, Qdrant, Kafka, migrations, API, ingestion worker, deletion worker, and the Next.js frontend. No paid API key is required because local mode uses deterministic embeddings and extractive answers.

Open:

- Frontend: [http://localhost:3000](http://localhost:3000)
- API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- MinIO console: [http://localhost:9001](http://localhost:9001)

Check the stack:

```bash
make status
make smoke
```

Stop it without deleting data:

```bash
make down
```

Delete local containers and persistent volumes:

```bash
make clean-local
```

### Developer mode without application containers

```bash
cp .env.example .env
uv sync
docker compose up -d postgres redis minio qdrant kafka
uv run alembic upgrade head
uv run uvicorn rag_platform.api.app:app --reload
```

In another terminal:

```bash
uv run python -m rag_platform.workers.ingestion
```

## Enable Production Providers

Keep secrets in `.env` locally and in a secret manager in production. Never commit keys or paste them into frontend variables.

```dotenv
MODEL_PROVIDER=openai
OPENAI_API_KEY=...
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_CHAT_MODEL=gpt-4.1-mini

WEB_SEARCH_ENABLED=true
TAVILY_API_KEY=...

AUTH_DISABLED=false
AUTH0_DOMAIN=your-tenant.us.auth0.com
AUTH0_AUDIENCE=https://api.atlas-rag.example.com
```

The browser-safe Auth0 domain, SPA client ID, and audience belong in `frontend/.env.local`; the OpenAI and Tavily keys never do.

Detailed account setup and secret locations are documented in [`docs/SETUP.md`](docs/SETUP.md).

## Deployment Options

### Docker Compose

Best for local development, demonstrations, and architecture evaluation. It is one command, but it is not an HA production topology.

```bash
make quickstart
```

### Existing Kubernetes cluster

The Helm chart is provider-neutral. Supply PostgreSQL, Kafka, Redis, S3-compatible storage, and Qdrant endpoints, then deploy:

```bash
helm upgrade --install atlas-rag deploy/charts/atlas-rag \
  --namespace atlas-rag \
  --create-namespace \
  -f deploy/charts/atlas-rag/values-production.example.yaml
```

Required controllers and production values are listed in [`deploy/charts/atlas-rag/README.md`](deploy/charts/atlas-rag/README.md).

### AWS

The AWS stack provisions one regional data plane. It is designed for AWS SSO and reviewed plan/apply workflows, not long-lived access keys.

```bash
aws sso login --profile atlas-platform
export AWS_PROFILE=atlas-platform

terraform -chdir=infra/stacks/aws init -backend-config=backend.hcl
terraform -chdir=infra/stacks/aws plan -var-file=dev.tfvars -out=tfplan
terraform -chdir=infra/stacks/aws apply tfplan
```

Follow [`infra/stacks/aws/README.md`](infra/stacks/aws/README.md) for remote-state bootstrap, costs, secret population, EKS access, Qdrant, and GitOps release steps.

### Is it one-click deployable everywhere?

**Local: yes, with `make quickstart`.**

**Kubernetes: one Helm command after its managed dependencies and controllers exist.**

**AWS: one reviewed infrastructure workflow after the AWS account, remote state, DNS/certificates, Qdrant, Auth0, provider secrets, and GitHub OIDC are configured.** The repository deliberately does not hide an expensive production plan behind an unaudited “Deploy” button.

GCP and Azure can run the portable containers and Helm chart, but this repository currently ships first-class Terraform for AWS. Provider-specific GCP and Azure modules are roadmap work, not something claimed as complete today.

## API Example

Upload a PDF in local mode:

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000001" \
  -H "X-Subject-ID: local-developer" \
  -F "file=@document.pdf;type=application/pdf" \
  -F "corpus_id=00000000-0000-0000-0000-000000000100" \
  -F "title=Example policy" \
  -F "document_type=policy"
```

Search the corpus:

```bash
curl -X POST http://localhost:8000/v1/search \
  -H "Content-Type: application/json" \
  -H "X-Tenant-ID: 00000000-0000-0000-0000-000000000001" \
  -H "X-Subject-ID: local-developer" \
  --data '{
    "query": "What is the recovery objective?",
    "filters": {
      "corpus_ids": ["00000000-0000-0000-0000-000000000100"]
    },
    "top_k": 5
  }'
```

## Repository Map

```text
src/rag_platform/       Python APIs, domain logic, adapters, retrieval, agents, workers
frontend/               Next.js chat and operations console
database/migrations/    PostgreSQL schema and RLS migrations
deploy/charts/          Portable Kubernetes Helm chart
deploy/environments/    Staging and production GitOps values
infra/bootstrap/aws/    Encrypted Terraform remote-state bootstrap
infra/stacks/aws/       Regional AWS production foundation
observability/          OpenTelemetry and Prometheus local configuration
tests/                  Unit, contract, and integration tests
.github/workflows/      CI, security, and digest-based release automation
```

## Quality and Verification

```bash
# Backend
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest

# Frontend
npm --prefix frontend run lint
npm --prefix frontend run test
npm --prefix frontend run build

# Deployment contracts
docker compose config --quiet
terraform -chdir=infra/stacks/aws validate
helm lint deploy/charts/atlas-rag
```

CI additionally builds and scans containers, publishes SBOMs, validates Kubernetes resources, and proposes releases by immutable image digest.

## Scale Targets

The architecture is designed around the following acceptance targets:

| Target | Initial objective |
|---|---:|
| Logical PDFs | 10,000,000+ |
| Active chunks | Approximately 300,000,000 |
| Sustained backfill | 2,500 documents/minute |
| Dense vector search p95 | Under 200 ms |
| Complete retrieval p95 | Under 350 ms |
| Retrieval availability | 99.95% |
| Cross-tenant leakage | Zero tolerated |

These are engineering targets that require a representative corpus, production-sized indexes, load tests, relevance evaluation, and provider-specific tuning. They are not benchmark claims for a laptop or an empty cloud account.

## Security Model

- Tenant context is derived from authenticated identity.
- PostgreSQL RLS and vector payload filters enforce isolation in depth.
- Auth0 validates issuer, audience, signature, expiry, subject, permissions, and tenant claims.
- PDFs are quarantined before parsing.
- Parser and tool containers are designed for non-root, read-only, restricted execution.
- AWS uses IRSA, KMS, private networking, IAM-authenticated MSK/Redis, and Secrets Manager.
- The model never grants authorization or invents source IDs.
- Unknown citations, insufficient evidence, and contradictory evidence result in abstention.

Review the threat controls before exposing any deployment to untrusted users. A reference implementation is not a substitute for your organization's security review, data classification, legal requirements, and incident-response process.

## Build on It

Atlas is meant to be extended. Common next steps include:

- Add a source connector for SharePoint, Google Drive, Confluence, or an internal object catalog.
- Replace PostgreSQL FTS with `pg_search`/ParadeDB or OpenSearch BM25 at very large lexical scale.
- Add provider modules for GCP and Azure.
- Connect the fixture-labeled evaluation and operations screens to your RAGAS and observability APIs.
- Add domain-specific entity extraction and graph relationship schemas.
- Add delegated Jira, Salesforce, or internal action tools behind approval policies.

See [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a change and [`SECURITY.md`](SECURITY.md) for vulnerability reporting.

## License

Licensed under the [Apache License 2.0](LICENSE). You may use, modify, and distribute the project subject to that license.
