# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapted from anomalib SuperADD; source revision in THIRD_PARTY.md.
"""Offline SuperADD: overlapping DINOv3 features and per-layer Euclidean banks.

Inputs use ImageNet normalization, as supplied by adkit.data.prepare.
The public patch_size is the backbone token stride; tile_size is the upstream
SuperADD patch_size. No anomalib, Lightning, or automatic weight downloads.
"""
import hashlib
import itertools
import math
from pathlib import Path

import timm
import torch
from safetensors.torch import load_file
from torch.nn import functional as F

from .base import BaseDetector
from .data import pad_to_patch


TARGET_LAYERS = {'small': (3, 5, 8, 10), 'base': (3, 5, 8, 10),
                 'large': (5, 11, 17, 23), 'huge': (7, 15, 23, 31)}


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def axis_patch_split(size, tile_size, overlap, stride):
    """Upstream centered-overlap ownership, in pixel and token coordinates."""
    if (any(type(v) is not int for v in (size, tile_size, overlap, stride))
            or stride < 1 or overlap <= 0 or tile_size <= 2 * overlap
            or size < tile_size or any(v % stride for v in (size, tile_size, overlap))):
        raise ValueError('Invalid aligned tile geometry')
    dim, tile, border = size // stride, tile_size // stride, overlap // stride
    count = math.ceil((dim - tile) / (tile - 2 * border)) + 1
    starts = [i * (dim - tile) // max(1, count - 1) for i in range(count)]
    result = []
    for i, start in enumerate(starts):
        lo = 0 if i == 0 else math.ceil((starts[i - 1] + tile - start) / 2)
        hi = tile if i == count - 1 else tile - math.floor((start + tile - starts[i + 1]) / 2)
        result.append(((start * stride, (start + tile) * stride),
                       (lo, hi), (start + lo, start + hi)))
    return result


def euclidean_knn(query, bank, k=1, query_chunk=1024, bank_chunk=16384):
    """Exact unnormalized Euclidean distances, bounded on both matrix axes."""
    if (query.ndim != 2 or bank.ndim != 2 or query.shape[1] != bank.shape[1]
            or not len(query) or not 1 <= k <= len(bank)
            or query_chunk < 1 or bank_chunk < 1):
        raise ValueError('Invalid nearest-neighbor inputs')
    result = []
    for q in query.split(query_chunk):
        best = q.new_empty((len(q), 0))
        for keys in bank.split(bank_chunk):
            keys = keys.to(q.device)
            distances = (-2 * (q @ keys.T) + q.square().sum(-1, keepdim=True)
                         + keys.square().sum(-1).unsqueeze(0)).clamp_min_(0).sqrt_()
            candidates = torch.cat((best, distances), dim=1)
            best = candidates.topk(min(k, candidates.shape[1]), largest=False).values
        result.append(best)
    return torch.cat(result)


def distance_coreset(features, target, iterations=100, *, device='cpu',
                     query_chunk=1024, bank_chunk=16384):
    """Upstream random-subset density sampling with bounded degenerate cases.

    A zero distance threshold or a saturated density estimate cannot shrink
    indefinitely. In those cases randomly trim the surviving candidates. Also
    enforce the requested cap when iterations exceeds the target bank size.
    """
    if (type(target) is not int or target < 1 or type(iterations) is not int or iterations < 1
            or features.ndim != 2 or not len(features) or not torch.isfinite(features).all()):
        raise ValueError('Expected finite nonempty features and positive target/iterations')
    features = features.detach().float().cpu()
    if len(features) <= target:
        return features
    keep = torch.zeros(len(features), dtype=torch.bool)
    subset_size = max(1, len(features) // iterations)
    subset_target = max(1, target // iterations)
    for _ in range(iterations):
        candidates = torch.where(~keep)[0]
        if not len(candidates):
            break
        indices = candidates[torch.randperm(len(candidates))[:subset_size]]
        if len(indices) <= subset_target:
            keep[indices] = True
            continue
        x = features[indices].to(device)
        distances = euclidean_knn(x, x, min(100, len(x)), query_chunk, bank_chunk)
        threshold = float(distances.mean()) / 10
        maximum = float(distances.max())
        random_numbers = torch.rand(len(x), device=device)
        selected = torch.ones(len(x), dtype=torch.bool, device=device)
        for _ in range(256):
            if threshold <= 0:
                break
            density = (distances < threshold).sum(-1) + 1
            selected = random_numbers < 1 / density.float()
            if int(selected.sum()) <= subset_target or threshold > maximum:
                break
            threshold *= 1.1
        survivors = torch.where(selected.cpu())[0]
        if len(survivors) > subset_target:
            survivors = survivors[torch.randperm(len(survivors))[:subset_target]]
        keep[indices[survivors]] = True
    chosen = torch.where(keep)[0]
    if len(chosen) > target:
        keep[chosen[torch.randperm(len(chosen))[target:]]] = False
    elif len(chosen) < target:
        remaining = torch.where(~keep)[0]
        keep[remaining[torch.randperm(len(remaining))[:target - len(chosen)]]] = True
    return features[keep]


def gaussian_blur(amap, sigma):
    """Finite 4-sigma Gaussian with reflection padding, matching anomalib."""
    radius = int(4 * sigma + .5)
    if radius == 0:
        return amap
    coords = torch.arange(-radius, radius + 1, device=amap.device, dtype=amap.dtype)
    kernel = torch.exp(-coords.square() / (2 * sigma * sigma))
    kernel /= kernel.sum()
    # Replication extends reflection's domain to inputs smaller than the kernel.
    horizontal = 'reflect' if amap.shape[-1] > radius else 'replicate'
    vertical = 'reflect' if amap.shape[-2] > radius else 'replicate'
    amap = F.conv2d(F.pad(amap, (radius, radius, 0, 0), mode=horizontal), kernel[None, None, None, :])
    return F.conv2d(F.pad(amap, (0, 0, radius, radius), mode=vertical), kernel[None, None, :, None])


class SuperADDDetector(BaseDetector):
    def __init__(self, weights: str | Path, encoder_name: str = 'vit_small_patch16_dinov3',
                 device: str = 'cuda', layers: list[int] | tuple[int, ...] | None = None,
                 tile_size: int = 448, patch_overlap: int = 16,
                 max_database_size: int = 100000, subsampling_iterations: int = 100,
                 gaussian_blur_sigma: float = 4.0, score_quantile: float = 1e-3,
                 query_chunk: int = 1024, bank_chunk: int = 16384):
        """Load a matching local timm state dict; all backbone parameters freeze.

        layers are zero-based transformer block indices, in increasing order.
        Tiles execute sequentially to bound backbone activation memory.
        """
        for name, value in [('tile_size', tile_size), ('patch_overlap', patch_overlap),
                            ('max_database_size', max_database_size),
                            ('subsampling_iterations', subsampling_iterations),
                            ('query_chunk', query_chunk), ('bank_chunk', bank_chunk)]:
            if type(value) is not int or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(gaussian_blur_sigma) or gaussian_blur_sigma < 0:
            raise ValueError('gaussian_blur_sigma must be finite and non-negative')
        if not 0 < score_quantile <= 1:
            raise ValueError('score_quantile must be in (0, 1]')
        if 'dinov3' not in encoder_name:
            raise ValueError('SuperADD requires a timm DINOv3 encoder')
        if layers is None:
            layers = next((v for key, v in TARGET_LAYERS.items() if key in encoder_name), None)
        if (not layers or any(type(v) is not int or v < 0 for v in layers)
                or list(layers) != sorted(set(layers))):
            raise ValueError('layers must be nonempty, unique, increasing block indices')
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f'Local backbone weights not found: {weights}')
        self.weights, self.encoder_name = str(weights), encoder_name
        self.device, self.layers = torch.device(device), tuple(layers)
        self.tile_size, self.patch_overlap = tile_size, patch_overlap
        self.max_database_size, self.subsampling_iterations = max_database_size, subsampling_iterations
        self.gaussian_blur_sigma, self.score_quantile = gaussian_blur_sigma, score_quantile
        self.query_chunk, self.bank_chunk = query_chunk, bank_chunk
        self.encoder = timm.create_model(encoder_name, pretrained=False, num_classes=0)
        if max(self.layers) >= len(self.encoder.blocks):
            raise ValueError('layers exceed the encoder depth')
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        axis_patch_split(tile_size, tile_size, patch_overlap, self.patch_size)
        self.weight_sha256 = file_sha256(weights)
        state = load_file(str(weights)) if weights.suffix == '.safetensors' else torch.load(
            weights, map_location='cpu', weights_only=True)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.eval().requires_grad_(False).to(self.device)
        self.memory_bank = torch.empty(len(self.layers), 0, self.encoder.num_features)
        self.metadata = {}

    @property
    def reference_patches(self):
        """Retained reference tokens per layer (not multiplied by layer count)."""
        return self.memory_bank.shape[1]

    @torch.inference_mode()
    def extract_features(self, batch):
        if (batch.ndim != 4 or batch.shape[1] != 3 or not len(batch)
                or min(batch.shape[-2:]) < 1 or not torch.isfinite(batch).all()):
            raise ValueError('Expected finite nonempty [B,3,H,W] tensor')
        batch = pad_to_patch(batch, self.patch_size).to(self.device, dtype=torch.float32)
        h, w = batch.shape[-2:]
        batch = F.pad(batch, (0, max(0, self.tile_size - w), 0, max(0, self.tile_size - h)), mode='replicate')
        height, width = batch.shape[-2:]
        ys = axis_patch_split(height, self.tile_size, self.patch_overlap, self.patch_size)
        xs = axis_patch_split(width, self.tile_size, self.patch_overlap, self.patch_size)
        features = batch.new_empty((len(self.layers), len(batch), height // self.patch_size,
                                    width // self.patch_size, self.encoder.num_features))
        side = self.tile_size // self.patch_size
        for (iy, py, ry), (ix, px, rx) in itertools.product(ys, xs):
            tokens = self.encoder.forward_intermediates(
                batch[:, :, iy[0]:iy[1], ix[0]:ix[1]], indices=list(self.layers),
                norm=False, output_fmt='NLC', intermediates_only=True)
            if len(tokens) != len(self.layers):
                raise RuntimeError('Encoder returned an unexpected number of feature layers')
            for index, layer in enumerate(tokens):
                grid = layer.reshape(len(batch), side, side, self.encoder.num_features)
                features[index, :, ry[0]:ry[1], rx[0]:rx[1]] = grid[:, py[0]:py[1], px[0]:px[1]]
        # Minimum-tile padding is context only; never enter the bank or score.
        return features[:, :, :h // self.patch_size, :w // self.patch_size]

    @torch.inference_mode()
    def fit(self, batches):
        stored = [[] for _ in self.layers]
        for batch in batches:
            for index, features in enumerate(self.extract_features(batch)):
                stored[index].append(features.reshape(-1, self.encoder.num_features).cpu())
        if not stored[0]:
            raise ValueError('No normal reference features were collected')
        banks = [distance_coreset(torch.cat(layer), self.max_database_size,
                                 self.subsampling_iterations, device=self.device,
                                 query_chunk=self.query_chunk, bank_chunk=self.bank_chunk) for layer in stored]
        self.memory_bank = torch.stack(banks)

    @torch.inference_mode()
    def predict(self, batch):
        if not self.reference_patches:
            raise RuntimeError('Reference bank is empty; call fit or load first')
        shape = batch.shape[-2:]
        features = self.extract_features(batch)
        _, b, h, w, channels = features.shape
        maps = []
        for layer, bank in zip(features, self.memory_bank):
            distances = euclidean_knn(layer.reshape(-1, channels), bank, 1,
                                      self.query_chunk, self.bank_chunk)
            maps.append(distances.reshape(b, h, w) / channels)
        amap = F.interpolate(torch.stack(maps, dim=1),
                             size=(h * self.patch_size, w * self.patch_size),
                             mode='bilinear', align_corners=False).mean(1, keepdim=True)
        amap = gaussian_blur(amap, self.gaussian_blur_sigma)[:, :, :shape[0], :shape[1]]
        pixels = amap.flatten(1)
        scores = pixels.topk(max(1, int(pixels.shape[1] * self.score_quantile)), dim=1).values.mean(1)
        return {'pred_score': scores.cpu(), 'anomaly_map': amap.cpu()}

    def _init_params(self):
        names = ('weights', 'encoder_name', 'layers', 'tile_size', 'patch_overlap',
                 'max_database_size', 'subsampling_iterations', 'gaussian_blur_sigma',
                 'score_quantile', 'query_chunk', 'bank_chunk')
        return {**{name: getattr(self, name) for name in names}, 'device': str(self.device)}

    def save(self, path):
        if not self.reference_patches:
            raise RuntimeError('Cannot save an empty reference bank')
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save({'format_version': 2, 'algorithm': 'superadd', 'init_params': self._init_params(),
                    'weight_sha256': self.weight_sha256, 'memory_bank': self.memory_bank.cpu(),
                    'metadata': self.metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device='cpu', weights=None):
        state = torch.load(path, map_location='cpu', weights_only=True)
        if state.get('format_version') != 2 or state.get('algorithm') != 'superadd':
            raise ValueError('Unsupported detector checkpoint')
        params = dict(state['init_params'], device=device)
        if weights is not None:
            params['weights'] = weights
        model = cls(**params)
        if model.weight_sha256 != state['weight_sha256']:
            raise ValueError('Backbone weight checksum does not match the reference bank')
        bank = state['memory_bank']
        if (not isinstance(bank, torch.Tensor) or bank.ndim != 3 or bank.shape[0] != len(model.layers)
                or not 0 < bank.shape[1] <= model.max_database_size
                or bank.shape[2] != model.encoder.num_features or not torch.isfinite(bank).all()):
            raise ValueError('Invalid reference bank')
        model.memory_bank = bank.float()
        model.metadata = state.get('metadata', {})
        return model
