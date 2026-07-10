# Deployment workflow

Configure the `staging` and `production` GitHub environments with required reviewers. Define these environment variables (GitHub Actions configuration variables, not secrets):

- `AWS_DEPLOY_ROLE_ARN`
- `AWS_REGION`
- `API_ECR_REPOSITORY`
- `FRONTEND_ECR_REPOSITORY`
- `NEXT_PUBLIC_API_URL`
- `NEXT_PUBLIC_DEFAULT_CORPUS_ID`
- `NEXT_PUBLIC_AUTH0_DOMAIN`
- `NEXT_PUBLIC_AUTH0_CLIENT_ID`
- `NEXT_PUBLIC_AUTH0_AUDIENCE`

`release-gitops.yml` assumes the AWS role through GitHub OIDC only after environment approval, publishes immutable images, and opens a promotion pull request containing ECR digests. It never stores AWS access keys, changes the cluster directly, or applies Terraform.
