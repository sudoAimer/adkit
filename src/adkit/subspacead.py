# SubspaceAD 检测器：多层冻结特征与两遍 PCA 重建评分。
# SPDX-License-Identifier: Apache-2.0
# Adapted from CLendering/SubspaceAD (revision in THIRD_PARTY.md).
"""Frozen multi-layer DINOv2 features and two-pass PCA reconstruction."""
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import torch

from .base import BaseDetector
from .data import pad_to_patch
from .dinov2 import input_tokens


def file_sha256(path):
    """分块计算文件哈希，避免一次性读入大型权重。"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def fit_pca(feature_batches, device='cpu', explained_variance=.99, components=None):
    """两遍统计均值和协方差，以解释方差或指定维数确定 PCA 子空间。
    Two passes over a callable feature source, matching the author's PCA.

    A re-iterable image source regenerates augmentations on pass two, as the
    official implementation does. This detail is recorded in the benchmark.
    """
    total = 0
    mean = None
    logging.info('SubspaceAD PCA pass 1: mean')
    for batch_index,array in enumerate(feature_batches(),1):
        x = torch.as_tensor(array, device=device, dtype=torch.float64)
        if x.ndim != 2 or not len(x) or not torch.isfinite(x).all():
            raise ValueError('Expected finite, nonempty feature matrices')
        if mean is None:
            mean = torch.zeros(x.shape[1], device=device, dtype=torch.float64)
        mean += x.sum(0)
        total += len(x)
        if batch_index % 10 == 0:
            logging.info('PCA mean: %d feature batches',batch_index)
    if total < 2:
        raise ValueError('PCA needs at least two reference tokens')
    mean /= total
    covariance = torch.zeros((len(mean),len(mean)),device=device,dtype=torch.float64)
    second_count = 0
    logging.info('SubspaceAD PCA pass 2: covariance')
    for batch_index,array in enumerate(feature_batches(),1):
        x = torch.as_tensor(array,device=device,dtype=torch.float64)
        if x.ndim != 2 or x.shape[1] != len(mean) or not torch.isfinite(x).all():
            raise ValueError('Invalid second-pass features')
        centered = x-mean
        covariance += centered.T@centered
        second_count += len(x)
        if batch_index % 10 == 0:
            logging.info('PCA covariance: %d feature batches',batch_index)
    if second_count != total:
        raise ValueError('PCA passes must contain the same number of tokens')
    covariance /= total-1
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues,descending=True)
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:,order]
    if eigenvalues.sum() <= 0:
        raise ValueError('Reference features have zero variance')
    if components is None:
        ratio = eigenvalues.cumsum(0)/eigenvalues.sum()
        k = min(len(eigenvalues),torch.searchsorted(ratio,torch.tensor([explained_variance],device=device,dtype=torch.float64)).item()+1)
    else:
        k = min(components,len(eigenvalues))
    logging.info('PCA complete: %d components, %d reference patches',k,total)
    return {'mu':mean.cpu(), 'components':eigenvectors[:,:k].cpu(),
            'eigvals':eigenvalues[:k].cpu(), 'k':k, 'reference_patches':total}


def reconstruction_scores(features, state):
    # Preserve official float32 NumPy scoring after the FP64 PCA fit.
    """以官方 float32 精度计算特征在 PCA 子空间中的重建误差。"""
    x = np.asarray(features,dtype=np.float32)
    mu = state['mu'].numpy().astype(x.dtype)
    basis = state['components'].numpy().astype(x.dtype)
    reconstructed = ((x-mu)@basis)@basis.T+mu
    return np.sum((x-reconstructed)**2,axis=1)


def subspace_map(patches, shape):
    """将 patch 误差缩放到输入尺寸，并采用官方固定参数平滑。"""
    resized = cv2.resize(np.asarray(patches,dtype=np.float32),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(resized,(3,3),4.)


class SubspaceADDetector(BaseDetector):
    def __init__(
        self,
        weights: str | Path,
        device: str = "cuda",
        layers: list[int] | tuple[int, ...] | None = None,
        explained_variance: float = .99,
        components: int | None = None,
        pca_device: str | None = None,
        backend: str = "auto",
        encoder_name: str = "vit_small_patch14_dinov2.lvd142m",
        positional_encoding: str = "official",
    ) -> None:
        """从本地模型目录初始化冻结骨干与空 PCA 状态。

        默认共用 AnomalyDINO 的 timm Small 权重文件；目录自动选择 Transformers。
        layers 指定参与平均的隐藏层，负数按末尾索引，内部保存为元组。
        components 指定维数时优先使用该值，否则按 explained_variance 选取。
        device 用于特征提取；pca_device 默认跟随 device，用于 PCA 统计。
        """
        weights = Path(weights).resolve()
        if backend not in {'auto', 'timm', 'transformers'}:
            raise ValueError('backend must be auto, timm or transformers')
        if positional_encoding not in {'official', 'timm'}:
            raise ValueError('positional_encoding must be official or timm')
        self.backend = ('transformers' if weights.is_dir() else 'timm') if backend == 'auto' else backend
        if not 0 < explained_variance <= 1 or (components is not None and components < 1):
            raise ValueError('explained_variance must be in (0,1]; components must be positive')
        self.weights = str(weights)
        self.encoder_name = encoder_name
        self.positional_encoding = positional_encoding
        self.explained_variance = explained_variance
        self.components = components
        self.pca_device = torch.device(pca_device or device)
        self.device = torch.device(device)
        if self.backend == 'timm':
            # Exactly the same local state dict and model construction as AnomalyDINO.
            import timm
            from safetensors.torch import load_file
            if not weights.is_file():
                raise FileNotFoundError(f'Local backbone weights not found: {weights}')
            if 'dinov2' not in encoder_name:
                raise ValueError('SubspaceAD timm backend requires a DINOv2 backbone')
            self.encoder = timm.create_model(encoder_name, pretrained=False,
                                            num_classes=0, dynamic_img_size=True)
            state = load_file(str(weights)) if weights.suffix == '.safetensors' else torch.load(
                weights, map_location='cpu', weights_only=True)
            self.encoder.load_state_dict(state, strict=True)
            if positional_encoding == 'official' and self.encoder.num_prefix_tokens != 1:
                raise ValueError('Official positional encoding requires DINOv2 without registers')
            self.processor = None  # Shared ImageNet preprocessing in adkit.data.prepare.
            self.patch_size = self.encoder.patch_embed.patch_size[0]
            self.feature_dim = self.encoder.num_features
            depth = len(self.encoder.blocks)
            self.weight_sha256 = file_sha256(weights)
            self.asset_hashes = {}
        else:
            from transformers import AutoImageProcessor, AutoModel
            weight_file = weights/'model.safetensors'
            for file in [weight_file, weights/'config.json', weights/'preprocessor_config.json']:
                if not file.is_file():
                    raise FileNotFoundError(f'Missing local SubspaceAD backbone asset: {file}')
            self.processor = AutoImageProcessor.from_pretrained(str(weights), local_files_only=True, use_fast=False)
            self.encoder = AutoModel.from_pretrained(str(weights), local_files_only=True,
                                                    attn_implementation='eager')
            self.patch_size = self.encoder.config.patch_size
            self.feature_dim = self.encoder.config.hidden_size
            depth = self.encoder.config.num_hidden_layers
            self.weight_sha256 = file_sha256(weight_file)
            self.asset_hashes = {name: file_sha256(weights/name) for name in ['config.json', 'preprocessor_config.json']}
        self.encoder.eval().requires_grad_(False).to(self.device)
        # Match the authors' backbone_ablation.sh, using HF hidden-state indexing:
        # 0 = embedded tokens; i > 0 = output of block i (before final norm).
        defaults = {12: [-4, -5], 24: [-7, -8, -9, -10, -11],
                    40: [-12, -13, -14, -15, -16, -17, -18]}
        layers = list(layers if layers is not None else defaults.get(depth, [-1]))
        count = depth + 1
        if not layers or any(type(i) is not int or not -count <= i < count for i in layers):
            raise ValueError(f'Feature indices must address {count} hidden states')
        self.layers = tuple(layers)
        self.pca_state = None
        self.reference_patches = 0
        self.metadata = {}

    @torch.inference_mode()
    def extract_features(self,batch):
        """提取冻结骨干的 patch 特征，保持算法对应的特征协议。"""
        if batch.ndim != 4 or batch.shape[1] != 3 or len(batch)==0:
            raise ValueError('Expected nonempty [B,3,H,W] image batch')
        batch = pad_to_patch(batch, self.patch_size)
        if getattr(self, 'backend', 'transformers') == 'timm':
            count = len(self.encoder.blocks) + 1
            selected = {i % count for i in self.layers}
            x = input_tokens(self.encoder, batch.to(self.device, dtype=torch.float32), self.positional_encoding)
            hidden = {0: x} if 0 in selected else {}
            for index, block in enumerate(self.encoder.blocks, 1):
                x = block(x)
                if index in selected:
                    hidden[index] = x
                if index == max(selected):
                    break
            drop = self.encoder.num_prefix_tokens
            return torch.stack([hidden[i % count][:, drop:] for i in self.layers]).mean(0)
        outputs = self.encoder(pixel_values=batch.to(self.device,dtype=torch.float32),
                               output_hidden_states=True,output_attentions=False)
        drop = 1+getattr(self.encoder.config,'num_register_tokens',0)
        features = torch.stack([outputs.hidden_states[i][:,drop:,:] for i in self.layers],dim=0).mean(0)
        return features

    def fit(self,batches):
        """使用正常图像批次重建参考状态，不执行反向传播。"""
        if iter(batches) is batches:
            # A single-use Tensor iterator cannot regenerate augmentation. Cache
            # its features and use the same samples on both statistical passes.
            cached = [self.extract_features(batch).reshape(-1,self.feature_dim).cpu().numpy() for batch in batches]
            source = lambda: iter(cached)
        else:
            def source():
                """每次调用重新遍历特征源，为两遍 PCA 统计提供数据。"""
                for batch in batches:
                    yield self.extract_features(batch).reshape(-1,self.feature_dim).cpu().numpy()
        state = fit_pca(source,self.pca_device,self.explained_variance,self.components)
        self.pca_state = state
        self.reference_patches = state['reference_patches']

    @torch.inference_mode()
    def predict(self,batch):
        """基于已建立的参考状态返回 CPU 异常分数和异常图。"""
        if self.pca_state is None:
            raise RuntimeError('PCA state is empty; call fit or load first')
        original_shape = batch.shape[-2:]
        batch = pad_to_patch(batch, self.patch_size)
        features = self.extract_features(batch)
        grid = (batch.shape[-2]//self.patch_size,batch.shape[-1]//self.patch_size)
        patches = reconstruction_scores(features.reshape(-1,features.shape[-1]).cpu().numpy(),self.pca_state).reshape(len(batch),*grid)
        maps = np.stack([subspace_map(p,batch.shape[-2:]) for p in patches])
        maps = maps[:, :original_shape[0], :original_shape[1]]
        flat = maps.reshape(len(batch),-1)
        k = max(1,int(flat.shape[1]*.01))
        scores = np.partition(flat,flat.shape[1]-k,axis=1)[:,-k:].mean(1)
        return {'pred_score':torch.from_numpy(scores),'anomaly_map':torch.from_numpy(maps[:,None])}

    def _init_params(self) -> dict:
        """将构造参数导出为可序列化字典，仅用于保存检查点。"""
        return {
            "weights": self.weights,
            "backend": self.backend,
            "encoder_name": self.encoder_name,
            "positional_encoding": self.positional_encoding,
            "layers": list(self.layers),
            "explained_variance": self.explained_variance,
            "components": self.components,
            "pca_device": str(self.pca_device),
            "device": str(self.device),
        }

    def save(self,path):
        """原子保存构造参数、参考状态与权重校验信息，不保存骨干权重。"""
        if self.pca_state is None:
            raise RuntimeError('Cannot save an unfitted PCA model')
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save({'format_version':2,'algorithm':'subspacead','init_params':self._init_params(),
                    'weight_sha256':self.weight_sha256,'pca_state':self.pca_state,
                    'asset_hashes':self.asset_hashes,
                    'metadata':self.metadata},temporary)
        temporary.replace(path)

    @classmethod
    def load(cls,path,device='cpu',weights=None):
        """加载新格式检查点并校验参考状态，允许覆盖设备与权重路径。"""
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('algorithm')!='subspacead' or saved.get('format_version')!=2:
            raise ValueError('Unsupported SubspaceAD checkpoint')
        params = dict(saved['init_params'],device=device,pca_device=device)
        # Pre-backend checkpoints always used local Transformers directories.
        params.setdefault('backend', 'transformers')
        if weights is not None:
            params['weights'] = weights
        model = cls(**params)
        if model.weight_sha256 != saved['weight_sha256']:
            raise ValueError('Backbone weight checksum does not match PCA state')
        if saved['asset_hashes'] != model.asset_hashes:
            raise ValueError('Backbone/processor config checksum does not match PCA state')
        state = saved['pca_state']
        d = model.feature_dim
        if state['mu'].shape != (d,) or state['components'].shape != (d,state['k']) or state['k']<1:
            raise ValueError('Invalid PCA state dimensions')
        if not all(torch.isfinite(state[k]).all() for k in ['mu','components','eigvals']):
            raise ValueError('Non-finite PCA state')
        model.pca_state = state
        model.reference_patches = state['reference_patches']
        model.metadata = saved.get('metadata',{})
        return model
