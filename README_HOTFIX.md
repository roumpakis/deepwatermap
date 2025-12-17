# DeepWaterMap UI — hotfix (checkpoint path)

## Why you were getting:
`Failed to find any matching files for /models/checkpoints/cp.135.ckpt`

Because inside the Docker image your checkpoint lives in:
`/app/checkpoints/cp.135.ckpt.{index,data-*}`
…but the service was configured to look in `/models/checkpoints/...`.

This hotfix makes the app auto-resolve the checkpoint prefix and sets Docker default:
`CHECKPOINT_PATH=/app/checkpoints/cp.135.ckpt`

## Docker (recommended quick run)
From the project folder (where Dockerfile.dwm is):
```bash
docker build -t deepwatermap-ui:latest -f Dockerfile.dwm .
docker run --rm -p 8080:8080 deepwatermap-ui:latest
```
Open: http://localhost:8080

Health:
```bash
curl http://localhost:8080/health
```

## Docker (if you want checkpoint via volume)
```bash
docker run --rm -p 8080:8080 ^
  -e CHECKPOINT_PATH=/models/checkpoints/cp.135.ckpt ^
  -v "C:\Users\csdro\Downloads\with_cluster\models\deepwatermap\checkpoints:/models/checkpoints:ro" ^
  deepwatermap-ui:latest
```

## Kubernetes (NodePort)
```bash
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml
```
Then: http://localhost:30080

## Kubernetes with hostPath PV (Docker Desktop on Windows)
1) Edit `k8s/pv-pvc-hostpath.yaml` hostPath if needed
2) Apply:
```bash
kubectl apply -f k8s/pv-pvc-hostpath.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment-with-pv.yaml
```
