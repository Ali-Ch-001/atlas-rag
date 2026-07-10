locals {
  name = "${var.name_prefix}-${var.environment}"
  azs  = slice(data.aws_availability_zones.available.names, 0, 3)

  public_subnets  = [for index, _ in local.azs : cidrsubnet(var.vpc_cidr, 4, index)]
  private_subnets = [for index, _ in local.azs : cidrsubnet(var.vpc_cidr, 4, index + 4)]
  data_subnets    = [for index, _ in local.azs : cidrsubnet(var.vpc_cidr, 4, index + 8)]

  tags = merge(
    {
      Environment = var.environment
      ManagedBy   = "terraform"
      Project     = "atlas-rag"
      Region      = var.aws_region
    },
    var.tags,
  )

  bucket_name_base = substr(
    "${local.name}-${data.aws_caller_identity.current.account_id}-${var.aws_region}",
    0,
    52,
  )
  bucket_names = {
    quarantine = "${local.bucket_name_base}-quarantine"
    clean      = "${local.bucket_name_base}-clean"
    derived    = "${local.bucket_name_base}-derived"
  }

  secret_names = toset(["openai", "tavily", "auth0", "citation", "qdrant"])

  github_oidc_provider_arn = coalesce(
    var.github_oidc_provider_arn,
    "arn:${data.aws_partition.current.partition}:iam::${data.aws_caller_identity.current.account_id}:oidc-provider/invalid",
  )
  github_repository = coalesce(var.github_repository, "invalid/invalid")

  system_node_groups = {
    for index, az in local.azs : "system-${substr(az, length(az) - 1, 1)}" => {
      name           = "${local.name}-system-${substr(az, length(az) - 1, 1)}"
      subnet_ids     = [module.vpc.private_subnets[index]]
      min_size       = 1
      max_size       = 3
      desired_size   = 1
      instance_types = var.system_instance_types
      capacity_type  = "ON_DEMAND"
      labels = {
        "atlas-rag.io/node-pool" = "system"
      }
      taints = {
        critical = {
          key    = "CriticalAddonsOnly"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      }
    }
  }

  application_secret_arns = concat(
    [for secret in aws_secretsmanager_secret.application : secret.arn],
    module.postgresql.db_instance_master_user_secret_arn == null ? [] : [module.postgresql.db_instance_master_user_secret_arn],
  )

  kafka_topic_arn_prefix = replace(aws_msk_serverless_cluster.this.arn, ":cluster/", ":topic/")
  kafka_group_arn_prefix = replace(aws_msk_serverless_cluster.this.arn, ":cluster/", ":group/")
}
