module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "20.37.1"

  cluster_name    = local.name
  cluster_version = var.eks_cluster_version

  cluster_endpoint_private_access      = true
  cluster_endpoint_public_access       = var.eks_endpoint_public_access
  cluster_endpoint_public_access_cidrs = var.eks_public_access_cidrs

  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.platform.arn
    resources        = ["secrets"]
  }
  create_kms_key = false

  cloudwatch_log_group_kms_key_id        = aws_kms_key.platform.arn
  cloudwatch_log_group_retention_in_days = 90

  enable_irsa                              = true
  enable_cluster_creator_admin_permissions = var.enable_cluster_creator_admin

  cluster_addons = {
    coredns = {
      addon_version = var.eks_addon_versions["coredns"]
    }
    kube-proxy = {
      addon_version = var.eks_addon_versions["kube-proxy"]
    }
    vpc-cni = {
      addon_version            = var.eks_addon_versions["vpc-cni"]
      before_compute           = true
      service_account_role_arn = aws_iam_role.vpc_cni.arn
      configuration_values = jsonencode({
        enableNetworkPolicy = "true"
        env = {
          ENABLE_PREFIX_DELEGATION = "true"
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
  }

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.private_subnets

  node_security_group_additional_rules = {
    ingress_self_all = {
      description = "Node-to-node traffic"
      protocol    = "-1"
      self        = true
      from_port   = 0
      to_port     = 0
      type        = "ingress"
    }
  }

  eks_managed_node_group_defaults = {
    ami_type                   = "AL2023_x86_64_STANDARD"
    use_name_prefix            = true
    enable_monitoring          = true
    iam_role_attach_cni_policy = false

    update_config = {
      max_unavailable_percentage = 33
    }
  }

  eks_managed_node_groups = merge(
    local.system_node_groups,
    {
      online-cpu = {
        name           = "${local.name}-online-cpu"
        min_size       = 3
        max_size       = 30
        desired_size   = 3
        instance_types = var.online_instance_types
        capacity_type  = "ON_DEMAND"
        labels = {
          "atlas-rag.io/node-pool" = "online-cpu"
        }
        taints = {
          dedicated = {
            key    = "atlas-rag.io/dedicated"
            value  = "online-cpu"
            effect = "NO_SCHEDULE"
          }
        }
      }
      batch-spot = {
        name           = "${local.name}-batch-spot"
        min_size       = 0
        max_size       = 100
        desired_size   = 0
        instance_types = var.batch_instance_types
        capacity_type  = "SPOT"
        labels = {
          "atlas-rag.io/node-pool" = "batch-cpu"
        }
        taints = {
          dedicated = {
            key    = "atlas-rag.io/dedicated"
            value  = "batch-cpu"
            effect = "NO_SCHEDULE"
          }
        }
      }
      sandbox = {
        name           = "${local.name}-sandbox"
        min_size       = 0
        max_size       = 20
        desired_size   = 0
        instance_types = var.sandbox_instance_types
        capacity_type  = "ON_DEMAND"
        labels = {
          "atlas-rag.io/node-pool" = "sandbox"
        }
        taints = {
          dedicated = {
            key    = "atlas-rag.io/dedicated"
            value  = "sandbox"
            effect = "NO_SCHEDULE"
          }
        }
      }
    },
    var.enable_gpu_batch ? {
      gpu-batch = {
        name           = "${local.name}-gpu-batch"
        min_size       = 0
        max_size       = 20
        desired_size   = 0
        instance_types = var.gpu_batch_instance_types
        capacity_type  = "SPOT"
        ami_type       = "AL2023_x86_64_NVIDIA"
        labels = {
          "atlas-rag.io/node-pool" = "gpu-batch"
        }
        taints = {
          nvidia = {
            key    = "nvidia.com/gpu"
            value  = "true"
            effect = "NO_SCHEDULE"
          }
        }
      }
    } : {},
  )
}

data "aws_iam_policy_document" "vpc_cni_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type        = "Federated"
      identifiers = [module.eks.oidc_provider_arn]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:aud"
      values   = ["sts.amazonaws.com"]
    }

    condition {
      test     = "StringEquals"
      variable = "${module.eks.oidc_provider}:sub"
      values   = ["system:serviceaccount:kube-system:aws-node"]
    }
  }
}

resource "aws_iam_role" "vpc_cni" {
  name               = "${local.name}-vpc-cni"
  assume_role_policy = data.aws_iam_policy_document.vpc_cni_assume.json
}

resource "aws_iam_role_policy_attachment" "vpc_cni" {
  role       = aws_iam_role.vpc_cni.name
  policy_arn = "arn:${data.aws_partition.current.partition}:iam::aws:policy/AmazonEKS_CNI_Policy"
}
