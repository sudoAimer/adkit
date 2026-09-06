# Bottle / AnomalyDINO baseline

Completed: 2026-09-07. This is a single-category official-code reproduction, not a claim to reproduce the paper's all-category mean.

## Protocol

- Dataset: MVTec AD bottle; 209 normal training images, 83 test images (20 normal + 63 anomalous), 63 masks.
- 1-shot, three official repetitions: sorted reference images `000.png`, `001.png`, `002.png`; this is not random sampling.
- DINOv2 ViT-S/14, frozen FP32 weights; shorter edge 448; ImageNet normalization; eight rotations; no masking or coreset; exact 1-NN cosine distance.
- Official DINOv2 positional interpolation (+0.1 offset, no antialias), patch top-1% image score, original-resolution linear resize followed by Gaussian sigma=4.
- Each memory bank has 8,192 patches x 384 dimensions; reference bank stored on CPU.
- Test images and masks are used only for evaluation. No thresholds selected from the test set.

## Measured metrics

Values are percentages; standard deviations are percentage points (population standard deviation, ddof=0).

| Metric | Mean | Std | Official-code mean | Maximum absolute difference |
|---|---:|---:|---:|---:|
| Image AUROC | 99.4709 | 0.3685 | 99.4709 | 0.00000000 pp |
| Image AP | 99.8267 | 0.1247 | 99.8267 | 0.00000000 pp |
| Pixel AUROC | 98.8511 | 0.0728 | 98.8511 | 0.00000019 pp |
| Pixel AP | 82.1590 | 0.8225 | 82.1590 | 0.00000274 pp |
| AUPRO (FPR <= 0.3) | 96.1143 | 0.2777 | 96.1143 | 0.00000090 pp |

| Reference image | Image AUROC | Image AP | Pixel AUROC | Pixel AP | AUPRO |
|---|---:|---:|---:|---:|---:|
| 000.png | 99.8413 | 99.9500 | 98.9526 | 83.3112 | 96.4145 |
| 001.png | 99.6032 | 99.8743 | 98.8152 | 81.4448 | 95.7450 |
| 002.png | 98.9683 | 99.6559 | 98.7854 | 81.7210 | 96.1834 |

## Implementation checks

- Independent original DINOv2 backbone versus runtime: maximum feature error `0` on three test images; preprocessing tensors identical.
- Original AnomalyDINO detection script ran all 83 test images for each of three repetitions (249 comparisons). Maximum patch-distance difference: `1.25169754e-06`; maximum image-score difference: `2.08616257e-07`.
- Maximum metric difference across all metrics/repetitions: `2.74160593e-08` on the 0-1 scale. Image AUROC and AP match exactly.
- AUPRO independently checked against the original evaluator on repetition 0: difference `1.54676272e-12`.
- Local save/reload: identical full-resolution anomaly map (maximum error 0). Weights are excluded from detector checkpoints and SHA256 checked on reload.
- Nine pytest checks passed, including exact chunked kNN vs dense search, reference sampling, non-square geometry, perfect/tied metrics, offline checkpoint loading and reference-bank replacement.
- Plain-folder fit/predict and separate YAML checkpoint inference both completed.

## Interpretation and limits

- This establishes agreement with the official implementation for bottle. It does not establish accuracy on other MVTec categories, SubspaceAD, or custom industrial data.
- Pixel AP (82.16%) remains lower than pixel AUROC (98.85%); the latter alone does not establish precise defect boundaries. No mask, box, production threshold or pass/fail decision is provided.
- Heatmap colours are normalized per image for visualization only. Raw scores/maps are used for metrics.
- Initial uncorrected timm-position-interpolation runs remain in `outputs/bottle/` as exploratory results. The final baseline is `outputs/bottle_official/`.
- Timing logs are diagnostic: some checks overlapped, and timing was not an isolated warmed-up throughput benchmark. No speed claim is made.
- Exact native-resolution pixel metrics process 67,230,000 pixels per repetition and can use several GB of system RAM.

## Artifacts and reproduction

```powershell
conda run --no-capture-output -n detect python run.py --config configs/bottle.yaml
conda run --no-capture-output -n detect python tools/compare_official.py
conda run -n detect python -m pytest -q
```

Choose a new `output.directory` to rerun completed experiments. For the comparison script, keep `configs/bottle.yaml` pointing to the corresponding runtime result directory.

- Final metric JSON: `outputs/bottle_official/summary.json`
- Independent comparison: `outputs/official_reference/summary.json`
- Configuration: `configs/bottle.yaml`
- Dependency snapshot: `requirements-tested.txt`
- Sources and revisions: `THIRD_PARTY.md`
- Weight provenance: `weights/dinov2_vits14/source.json`
- Dataset manifest: `datasets/mvtec-ad/bottle-download-manifest.json`

## Example localization

Bottle broken_large/000.png, repetition 0. Red indicates higher relative anomaly score, not a calibrated binary defect label.

![Bottle anomaly overlay](../outputs/bottle_official/seed_0/images/broken_large/000.overlay.png)
