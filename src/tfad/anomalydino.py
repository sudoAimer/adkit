# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
# Adapted from anomalib AnomalyDINOModel, revision recorded in THIRD_PARTY.md.
"""Anomalib's normalized patch memory bank with local timm weights.

Differences: explicit lifecycle, exact chunked search, CPU reference storage,
and official AnomalyDINO map rendering. No anomalib/Lightning dependency.
"""
from pathlib import Path
import hashlib

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import PCA
import timm
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

from .base import BaseDetector


def render_map(patch_map, shape, sigma=4.0):
    """Official dists2map: linear resize first, Gaussian at target resolution."""
    return gaussian_filter(cv2.resize(np.asarray(patch_map), (shape[1], shape[0]),
                                     interpolation=cv2.INTER_LINEAR), sigma=sigma)


def exact_knn(query, bank, k=1, query_chunk=1024, bank_chunk=16384):
    if k < 1 or k > len(bank):
        raise ValueError(f"num_neighbours must be between 1 and bank size ({len(bank)})")
    result = []
    for q in query.split(query_chunk):
        best = q.new_empty((len(q), 0))
        for b in bank.split(bank_chunk):
            distances = (1 - q @ b.to(q.device).T).clamp(0, 2)
            candidates = torch.cat((best, distances), dim=1)
            best = candidates.topk(min(k, candidates.shape[1]), largest=False).values
        result.append(best.mean(dim=1))
    return torch.cat(result)


class AnomalyDinoDetector(BaseDetector):
    @property
    def reference_patches(self):
        return len(self.memory_bank)

    def __init__(self, weights, encoder_name="vit_small_patch14_dinov2.lvd142m",
                 device="cuda", num_neighbours=1, masking=False,
                 mask_ref_images=False, query_chunk=1024, bank_chunk=16384,
                 sigma=4.0, positional_encoding="official"):
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Local backbone weights not found: {weights}")
        if query_chunk < 1 or bank_chunk < 1 or num_neighbours < 1 or sigma < 0:
            raise ValueError("Chunk sizes/neighbours must be positive and sigma non-negative")
        if positional_encoding not in {'official', 'timm'}:
            raise ValueError("positional_encoding must be official or timm")
        self.config = dict(weights=str(weights), encoder_name=encoder_name, device=device,
                           num_neighbours=num_neighbours, masking=masking,
                           mask_ref_images=mask_ref_images, query_chunk=query_chunk,
                           bank_chunk=bank_chunk, sigma=sigma, positional_encoding=positional_encoding)
        self.weight_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        self.device = torch.device(device)
        self.encoder = timm.create_model(encoder_name, pretrained=False,
                                        num_classes=0, dynamic_img_size=True)
        state = load_file(str(weights)) if weights.suffix == ".safetensors" else torch.load(
            weights, map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.eval().requires_grad_(False).to(self.device)
        if positional_encoding == 'official' and self.encoder.num_prefix_tokens != 1:
            raise ValueError("Official positional encoding currently supports DINOv2 without registers")
        self.patch_size = self.encoder.patch_embed.patch_size[0]
        self.memory_bank = torch.empty(0, self.encoder.num_features)
        self.metadata = {}

    @torch.inference_mode()
    def extract_features(self, batch):
        if batch.ndim != 4 or batch.shape[1] != 3 or len(batch) == 0:
            raise ValueError("Expected non-empty [B,3,H,W] tensor")
        if batch.shape[-1] % self.patch_size or batch.shape[-2] % self.patch_size:
            raise ValueError("Image dimensions must be divisible by the patch size")
        batch = batch.to(self.device, dtype=torch.float32)
        if self.config['positional_encoding'] == 'timm':
            tokens = self.encoder.forward_features(batch)
        else:
            # DINOv2's historical +0.1 scale factor and antialias=False differ
            # from timm's dynamic interpolation. Preserve the original math.
            x = self.encoder.patch_embed(batch)
            b, h, w, dim = x.shape
            x = torch.cat((self.encoder.cls_token.expand(b, -1, -1), x.reshape(b, -1, dim)), dim=1)
            positions = self.encoder.pos_embed
            side = int((positions.shape[1]-1)**.5)
            if (h, w) != (side, side):
                patches = F.interpolate(positions[:, 1:].reshape(1, side, side, dim).permute(0, 3, 1, 2),
                                        scale_factor=((h+.1)/side, (w+.1)/side), mode='bicubic', antialias=False)
                positions = torch.cat((positions[:, :1], patches.permute(0, 2, 3, 1).reshape(1, h*w, dim)), dim=1)
            x = self.encoder.norm_pre(self.encoder.pos_drop(x+positions))
            tokens = self.encoder.norm(self.encoder.blocks(x))
        return tokens[:, self.encoder.num_prefix_tokens:]

    def _masks(self, features, grid, enabled):
        masks = np.ones(features.shape[:2], dtype=bool)
        if enabled:
            # Match the official implementation's flattened morphological mask.
            for index, feature in enumerate(features.cpu().numpy()):
                pc = PCA(n_components=1, svd_solver="randomized", random_state=0).fit_transform(feature)
                mask = pc > 10
                h, w = grid
                center = mask.reshape(grid)[int(h*.2):int(h*.8), int(w*.2):int(w*.8)]
                if center.sum() <= center.size * .35:
                    mask = -pc > 10
                kernel = np.ones((3, 3), np.uint8)
                mask = cv2.dilate(mask.astype(np.uint8), kernel)
                masks[index] = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).ravel().astype(bool)
        return torch.from_numpy(masks).to(features.device)

    @torch.inference_mode()
    def fit(self, batches):
        embeddings = []
        for batch in batches:
            features = self.extract_features(batch)
            grid = (batch.shape[-2]//self.patch_size, batch.shape[-1]//self.patch_size)
            masks = self._masks(features, grid, self.config['masking'] and self.config['mask_ref_images'])
            embeddings.append(F.normalize(features[masks], dim=-1).cpu())
        if not embeddings or sum(len(x) for x in embeddings) == 0:
            raise ValueError("No normal reference features were collected")
        bank = torch.cat(embeddings)
        if len(bank) < self.config['num_neighbours']:
            raise ValueError("Fewer reference patches than requested neighbours")
        self.memory_bank = bank

    @torch.inference_mode()
    def predict(self, batch):
        if not len(self.memory_bank):
            raise RuntimeError("Reference bank is empty; call fit or load first")
        features = self.extract_features(batch)
        grid = (batch.shape[-2]//self.patch_size, batch.shape[-1]//self.patch_size)
        masks = self._masks(features, grid, self.config['masking'])
        distances = features.new_zeros(features.shape[:2])
        if masks.any():
            distances[masks] = exact_knn(F.normalize(features[masks], dim=-1), self.memory_bank,
                                        self.config['num_neighbours'], self.config['query_chunk'],
                                        self.config['bank_chunk'])
        score = distances.topk(max(1, int(distances.shape[1]*.01)), dim=1).values.mean(dim=1)
        patches = distances.reshape(-1, 1, *grid).cpu()
        maps = np.stack([render_map(p[0].numpy(), batch.shape[-2:], self.config['sigma']) for p in patches])
        return {'pred_score': score.cpu(), 'anomaly_map': torch.from_numpy(maps[:, None]),
                'patch_map': patches}

    def save(self, path):
        if not len(self.memory_bank):
            raise RuntimeError("Cannot save an empty reference bank")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save({'format_version': 1, 'algorithm': 'anomalydino', 'config': self.config,
                    'weight_sha256': self.weight_sha256, 'memory_bank': self.memory_bank.cpu(),
                    'metadata': self.metadata}, temporary)
        temporary.replace(path)

    @classmethod
    def load(cls, path, device="cpu", weights=None):
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.get('format_version') != 1 or state.get('algorithm') != 'anomalydino':
            raise ValueError("Unsupported detector checkpoint")
        config = dict(state['config'], device=device)
        if weights is not None:
            config['weights'] = weights
        model = cls(**config)
        if model.weight_sha256 != state['weight_sha256']:
            raise ValueError("Backbone weight checksum does not match the reference bank")
        bank = state['memory_bank']
        if bank.ndim != 2 or bank.shape[1] != model.encoder.num_features or not len(bank) or not torch.isfinite(bank).all():
            raise ValueError("Invalid reference bank")
        model.memory_bank = bank.float()
        model.metadata = state.get('metadata', {})
        return model
