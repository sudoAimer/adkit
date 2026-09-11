# Sources and changes

The runtime has no dependency on anomalib or the research repositories.
Original source snapshots and Apache-2.0 licenses are retained in `references/`.

| Source | Revision | Use |
|---|---|---|
| open-edge-platform/anomalib | db4e8cb2bc8a03c501333ae0d5802fe67dcf5ba8 | AnomalyDINO normalized patch bank, cosine nearest neighbours, top-1% score and foreground-mask structure |
| dammsi/AnomalyDINO | b9d1c2648e3a5247437d4d953d907a8f3d994457 | Reference selection, rotation, preprocessing and original-resolution map rendering; independent benchmark |
| facebookresearch/dinov2 | 7764ea0f912e53c92e82eb78a2a1631e92725fc8 | Original positional interpolation and independent backbone verification |
| CLendering/SubspaceAD | ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7 | Multi-layer feature mean, two-pass FP64 PCA, reconstruction scoring, map processing and independent reference run |

Adaptations: removed Lightning, anomalib data types and dynamic-buffer dependencies;
explicit CPU memory bank serialization; exact chunked search; local safetensors;
official DINOv2 position interpolation; standalone configuration/data/evaluation.
AnomalyDINO omits coreset sampling and retains every reference patch; SuperADD uses per-layer distance-based coreset sampling.

The `official` interpolation option implements the original non-register DINOv2
bicubic scale factor `(grid + 0.1) / pretrained_grid`, without antialiasing.
The optional `timm` mode preserves timm's default position interpolation for comparison.
The runtime foreground mask follows the author's flattened morphology rather than
anomalib's 2D morphology; bottle does not enable masking.

Downloaded dataset and weight repositories/revisions/checksums are recorded alongside
the assets. Their original license metadata is retained separately from the source code.

The historical SubspaceAD Giant backend uses the official Hugging Face DINOv2-with-registers-Giant weights and
Transformers 4.57.6 eager attention. Runtime inference does not request attention
maps, because the official bottle benchmark does not use saliency masking.
The independent reference harness disables only unused attention retention and
saliency calculation, leaving feature extraction, two-pass augmentation/PCA,
scoring and main evaluation code unchanged.
The reference harness aliases removed NumPy `trapz` to `trapezoid` for NumPy 2.4
compatibility; both implement the same trapezoidal integration.

The official feature generator draws new rotations in the second PCA pass.
`ReferenceBatches` preserves that behavior. A single-use Tensor iterator instead
caches its extracted features and performs both PCA passes on those features.
SubspaceAD pixel metrics use its 672x672 output space, and masks are resized with
nearest-neighbor interpolation, matching the repository. Native-resolution maps
are saved separately for display. The shared exact AUPRO and the repository's
300-threshold AUPRO are reported separately.


## SuperADD

Source: open-edge-platform/anomalib, revision `3759687e76395c4d6d239552d3bf6d72e003da78`, Apache-2.0. The unmodified torch model, revision and license are retained in `references/super_add/`.

Adapted overlapping tile ownership, DINOv3 unnormalized intermediate features, random-subset density coreset sampling, Euclidean 1-NN/channel scaling, Gaussian maps and top-quantile scoring. Local timm weights replace downloads; CPU banks and both-axis chunking bound device memory. Tiles execute sequentially. Added minimum-tile padding, bounded degenerate sampling and a strict bank cap. Defaults use S/16 instead of upstream H+/16. The anomalib normal-validation threshold postprocessor and paper-specific morphology/downsampling are not included. See `docs/superadd.md`.

## SubspaceAD Small default

The default now uses the same local timm `vit_small_patch14_dinov2.lvd142m` state dict and official positional interpolation as AnomalyDINO. Intermediate hidden states are averaged without final LayerNorm, preserving SubspaceAD feature semantics. Small/Base layer selection `[-4,-5]` follows the pinned CLendering/SubspaceAD `scripts/backbone_ablation.sh` (revision `ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7`). Indexing includes embedding state 0. The Transformers Giant backend and old checkpoints remain supported; its reproduction configs have explicit `subspace_giant_*` names. Existing Giant reports are not Small accuracy claims.
