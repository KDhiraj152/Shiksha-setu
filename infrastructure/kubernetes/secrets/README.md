# Kubernetes Secrets Configuration for ShikshaSetu

This directory contains Kubernetes secrets configuration for production deployment.

## ⚠️ Security Notice

**NEVER commit actual secrets to version control!** This directory contains templates and instructions only.

## Setup Methods

### Method 1: Using Sealed Secrets (Recommended for GitOps)

Sealed Secrets allows you to encrypt secrets and safely store them in Git.

#### 1. Install Sealed Secrets Controller

```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/controller.yaml
```

#### 2. Install kubeseal CLI

```bash
# macOS
brew install kubeseal

# Linux
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.24.0/kubeseal-0.24.0-linux-amd64.tar.gz
tar xfz kubeseal-0.24.0-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal
```

#### 3. Create and Seal Secrets

```bash
# Create a secret file (DO NOT COMMIT THIS)
cat > /tmp/shiksha-setu-secrets.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: shiksha-setu-secrets
  namespace: shiksha-setu
type: Opaque
stringData:
  DATABASE_URL: "postgresql://user:password@host:5432/dbname"
  JWT_SECRET_KEY: "your-jwt-secret-key"
  REDIS_URL: "redis://redis:6379/0"
  OPENAI_API_KEY: "sk-..."
  SUPABASE_URL: "https://your-project.supabase.co"
  SUPABASE_KEY: "your-supabase-key"
EOF

# Seal the secret
kubeseal --format=yaml < /tmp/shiksha-setu-secrets.yaml > k8s/base/sealed-secrets.yaml

# Clean up temp file
rm /tmp/shiksha-setu-secrets.yaml

# Now you can safely commit k8s/base/sealed-secrets.yaml
```

### Method 2: Using External Secrets Operator

External Secrets Operator integrates with AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, HashiCorp Vault, etc.

#### 1. Install External Secrets Operator

```bash
helm repo add external-secrets https://charts.external-secrets.io
helm install external-secrets external-secrets/external-secrets -n external-secrets-system --create-namespace
```

#### 2. Configure SecretStore

See `k8s/base/external-secrets.yaml` for configuration examples.

### Method 3: Manual Secrets (Development Only)

For local development only:

```bash
# Create namespace
kubectl create namespace shiksha-setu

# Create secrets from .env file
kubectl create secret generic shiksha-setu-secrets \
  --from-env-file=.env \
  --namespace=shiksha-setu

# Or create individual secrets
kubectl create secret generic shiksha-setu-secrets \
  --from-literal=DATABASE_URL="postgresql://..." \
  --from-literal=JWT_SECRET_KEY="..." \
  --from-literal=REDIS_URL="redis://..." \
  --namespace=shiksha-setu
```

## Environment-Specific Secrets

Use Kustomize overlays for environment-specific configurations:

```
k8s/
├── base/
│   ├── sealed-secrets.yaml       # Common sealed secrets
│   └── external-secrets.yaml     # External secrets config
└── overlays/
    ├── dev/
    │   └── kustomization.yaml    # Dev-specific patches
    ├── staging/
    │   └── kustomization.yaml    # Staging-specific patches
    └── prod/
        └── kustomization.yaml    # Production-specific patches
```

## Required Secrets

### Application Secrets

- `DATABASE_URL`: PostgreSQL connection string
- `JWT_SECRET_KEY`: JWT signing key (64+ random bytes)
- `REDIS_URL`: Redis connection string
- `SECRET_KEY`: FastAPI secret key

### External Service Keys

- `OPENAI_API_KEY`: OpenAI API key (optional)
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase anonymous key
- `SMTP_USERNAME`: Email service username (for alerts)
- `SMTP_PASSWORD`: Email service password (for alerts)

### Monitoring Secrets

- `GRAFANA_ADMIN_PASSWORD`: Grafana admin password
- `PROMETHEUS_BASIC_AUTH`: Prometheus basic auth (optional)

## Rotation Policy

- **JWT Secrets**: Rotate every 90 days
- **API Keys**: Rotate when compromised or every 180 days
- **Database Passwords**: Rotate every 90 days
- **Service Accounts**: Rotate every 30 days

## Best Practices

1. **Never commit plaintext secrets** to version control
2. **Use different secrets** for each environment (dev/staging/prod)
3. **Rotate secrets regularly** according to the policy above
4. **Audit secret access** using Kubernetes audit logs
5. **Use RBAC** to restrict secret access to specific service accounts
6. **Enable encryption at rest** for etcd in production clusters
7. **Monitor secret usage** with tools like Falco or OPA

## Verification

After creating secrets, verify they're available:

```bash
# List secrets
kubectl get secrets -n shiksha-setu

# Describe secret (won't show values)
kubectl describe secret shiksha-setu-secrets -n shiksha-setu

# View secret values (use carefully!)
kubectl get secret shiksha-setu-secrets -n shiksha-setu -o jsonpath='{.data.DATABASE_URL}' | base64 -d
```

## Troubleshooting

### Secret not found

```bash
# Check if secret exists in correct namespace
kubectl get secrets -n shiksha-setu

# Check deployment logs
kubectl logs -n shiksha-setu deployment/shiksha-setu-backend
```

### Permission denied

```bash
# Check service account has access
kubectl auth can-i get secrets --as=system:serviceaccount:shiksha-setu:default -n shiksha-setu
```

### Sealed secret not decrypting

```bash
# Check sealed secrets controller logs
kubectl logs -n kube-system deployment/sealed-secrets-controller

# Verify certificate
kubeseal --fetch-cert
```

---

## 👨‍💻 Author

**K Dhiraj** • [k.dhiraj.srihari@gmail.com](mailto:k.dhiraj.srihari@gmail.com) • [@KDhiraj152](https://github.com/KDhiraj152) • [LinkedIn](https://www.linkedin.com/in/k-dhiraj-83b025279/)

*Last updated: November 2025*
