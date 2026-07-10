# Atlas RAG Helm Chart

The chart deploys the API, Next.js frontend, ingestion worker, deletion worker, migration job, services, autoscalers, disruption budgets, network policies, quota, and optional ingress/monitoring resources. Images accept either a tag or an immutable `sha256:` digest; a digest takes precedence.

## Required controllers

- External Secrets Operator when `externalSecrets.enabled=true`.
- KEDA when `keda.enabled=true`.
- Metrics Server for HPA resource metrics.
- Prometheus Operator when `serviceMonitor.enabled=true`.
- An ingress controller when `ingress.enabled=true`.

Install CRDs/controllers before this release. The chart uses the IRSA-annotated `atlas-rag` service account for AWS access and disables the default Kubernetes API token mount.

```bash
helm lint deploy/charts/atlas-rag -f deploy/charts/atlas-rag/values-local.yaml
helm template atlas-rag deploy/charts/atlas-rag \
  --namespace atlas-rag \
  -f deploy/charts/atlas-rag/values-production.example.yaml
```

`values-production.example.yaml` contains identifiers and replacement markers only, never credentials. Replace those markers from the Terraform `platform_contract`. External Secrets renders `DATABASE_URL` from the RDS-managed secret and copies provider values into `atlas-rag-runtime`.

The frontend's `NEXT_PUBLIC_*` values are compiled into the image by Next.js. Build production with `deploy/docker/frontend.Dockerfile` and the same API, corpus, and public Auth0 values used here; runtime environment variables do not rewrite an already-built client bundle.

## Qdrant

Qdrant is always external to this release. Set `config.QDRANT_URL` to Qdrant Cloud or an independently operated cluster with tested shard placement, replication, snapshots, restore, encryption, and capacity. The chart deliberately embeds neither an operator nor a Qdrant workload.

## Current application constraint

The current Python Kafka and Redis adapters need IAM token-provider support before they can connect to the Terraform stack's MSK Serverless and ElastiCache IAM endpoints. Keep affected workloads disabled until that application support is available; do not weaken managed-service authentication as a workaround.
