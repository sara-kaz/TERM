# NRP Baseline Experiments — Language-Table

Fine-tunes **Octo** and **OpenVLA** on Language-Table under the exact same
protocol as TERM (3,000 episodes, 90/10 split, 3 seeds, 8-bin action accuracy).

## Prerequisites

```bash
# Confirm namespace access
kubectl get pods -n csun-ehuang-era

# Confirm GPU availability
kubectl describe nodes | grep -A5 "nvidia.com/gpu"
```

## Step 1 — Upload pre-processed data (one-time)

If you already have `language_table_episodes.pkl` locally:

```bash
# Create a PVC for shared data (optional but faster than GCS download per pod)
kubectl apply -f data_pvc.yaml -n csun-ehuang-era

# Upload the pkl file into the PVC via a temporary pod
kubectl run data-uploader --image=busybox --restart=Never \
    --overrides='{"spec":{"volumes":[{"name":"pvc","persistentVolumeClaim":{"claimName":"lt-data"}}],"containers":[{"name":"data-uploader","image":"busybox","command":["sleep","3600"],"volumeMounts":[{"name":"pvc","mountPath":"/data"}]}]}}' \
    -n csun-ehuang-era

kubectl cp language_table_episodes.pkl \
    data-uploader:/data/language_table_episodes.pkl \
    -n csun-ehuang-era

kubectl delete pod data-uploader -n csun-ehuang-era
```

If you do NOT have the pkl locally, the jobs auto-download from GCS
(`gs://gresearch/robotics/language_table/0.0.1`) via `convert_language_table.py`.

## Step 2 — Deploy Octo jobs (3 seeds in parallel)

```bash
kubectl apply -f nrp/octo_job.yaml -n csun-ehuang-era

# Watch progress
kubectl get pods -n csun-ehuang-era -l app=octo-lt -w

# Follow logs for seed 42
kubectl logs -f job/octo-lt-s42 -n csun-ehuang-era
```

Expected wall-clock: **~5 hours** (all 3 seeds run simultaneously on 1×A100 each).

## Step 3 — Deploy OpenVLA jobs (3 seeds in parallel)

```bash
kubectl apply -f nrp/openvla_job.yaml -n csun-ehuang-era

# Watch progress
kubectl get pods -n csun-ehuang-era -l app=openvla-lt -w

# Follow logs for seed 42
kubectl logs -f job/openvla-lt-s42 -n csun-ehuang-era
```

Expected wall-clock: **~12–14 hours** (2×A100 per seed, all 3 run simultaneously).

## Step 4 — Retrieve results

```bash
# Get Octo results
for SEED in 42 123 456; do
  POD=$(kubectl get pods -n csun-ehuang-era -l "app=octo-lt,seed=${SEED}" \
        -o jsonpath='{.items[0].metadata.name}')
  kubectl cp ${POD}:/checkpoints/octo_lt_s${SEED}/results.json \
      ./results/octo_s${SEED}.json -n csun-ehuang-era
done

# Get OpenVLA results
for SEED in 42 123 456; do
  POD=$(kubectl get pods -n csun-ehuang-era -l "app=openvla-lt,seed=${SEED}" \
        -o jsonpath='{.items[0].metadata.name}')
  kubectl cp ${POD}:/checkpoints/openvla_lt_s${SEED}/results.json \
      ./results/openvla_s${SEED}.json -n csun-ehuang-era
done
```

## Step 5 — Aggregate into paper numbers

```bash
cd TERM/

python -m baselines.aggregate_results \
    --dirs results/octo_s42 results/octo_s123 results/octo_s456 \
    --model Octo

python -m baselines.aggregate_results \
    --dirs results/openvla_s42 results/openvla_s123 results/openvla_s456 \
    --model OpenVLA
```

Output format:
```
Octo
  Seeds: [68.1, 67.8, 68.4]
  Mean ± Std: 68.1 ± 0.2%

LaTeX table row:
  Octo & 68.1 & 67.8 & 68.4 & $68.1 \pm 0.2$ \\
```

## Clean up

```bash
# Delete all Octo jobs
kubectl delete jobs -l app=octo-lt -n csun-ehuang-era

# Delete all OpenVLA jobs
kubectl delete jobs -l app=openvla-lt -n csun-ehuang-era

# Delete ConfigMaps
kubectl delete configmap octo-lt-scripts openvla-lt-scripts -n csun-ehuang-era
```

## Comparison protocol

| | TERM | Octo | OpenVLA |
|---|---|---|---|
| Base model | CLIP ViT-B/32 (frozen) | octo-small (backbone frozen) | openvla-7b (LoRA r=16) |
| Trainable params | 7.5M | ~1M (head only) | ~50M (LoRA) |
| Training data | 3,000 LT episodes | 3,000 LT episodes | 3,000 LT episodes |
| Train/val split | 90/10 (seed=42) | 90/10 (seed=42) | 90/10 (seed=42) |
| Seeds reported | 42, 123, 456 | 42, 123, 456 | 42, 123, 456 |
| Metric | 8-bin action accuracy | 8-bin action accuracy | 8-bin action accuracy |
| Random baseline | 12.5% | 12.5% | 12.5% |
