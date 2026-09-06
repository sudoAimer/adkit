# SPDX-License-Identifier: Apache-2.0
# Adapted from CLendering/SubspaceAD (revision in THIRD_PARTY.md).
"""Frozen multi-layer DINOv2 features and two-pass PCA reconstruction."""
import hashlib
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import AutoImageProcessor, AutoModel

from .base import BaseDetector


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for block in iter(lambda:file.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def fit_pca(feature_batches, device='cpu', explained_variance=.99, components=None):
    """Two passes over a callable feature source, matching the author's PCA.

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
    x = np.asarray(features,dtype=np.float32)
    mu = state['mu'].numpy().astype(x.dtype)
    basis = state['components'].numpy().astype(x.dtype)
    reconstructed = ((x-mu)@basis)@basis.T+mu
    return np.sum((x-reconstructed)**2,axis=1)


def subspace_map(patches, shape):
    resized = cv2.resize(np.asarray(patches,dtype=np.float32),(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(resized,(3,3),4.)


class SubspaceADDetector(BaseDetector):
    def __init__(self, weights, device='cuda', layers=None, explained_variance=.99,
                 components=None, pca_device=None):
        weights = Path(weights).resolve()
        weight_file = weights/'model.safetensors'
        for file in [weight_file,weights/'config.json',weights/'preprocessor_config.json']:
            if not file.is_file():
                raise FileNotFoundError(f'Missing local SubspaceAD backbone asset: {file}')
        if not 0 < explained_variance <= 1 or (components is not None and components < 1):
            raise ValueError('explained_variance must be in (0,1]; components must be positive')
        layers = list(layers if layers is not None else [-12,-13,-14,-15,-16,-17,-18])
        if not layers:
            raise ValueError('At least one feature layer is required')
        self.config = dict(weights=str(weights),device=device,layers=layers,
                           explained_variance=explained_variance,components=components,
                           pca_device=pca_device or device)
        self.weight_sha256 = file_sha256(weight_file)
        self.asset_hashes = {name:file_sha256(weights/name) for name in ['config.json','preprocessor_config.json']}
        self.device = torch.device(device)
        self.processor = AutoImageProcessor.from_pretrained(str(weights),local_files_only=True,use_fast=False)
        self.encoder = AutoModel.from_pretrained(str(weights),local_files_only=True,
                                                attn_implementation='eager').eval().requires_grad_(False).to(self.device)
        self.patch_size = self.encoder.config.patch_size
        count = self.encoder.config.num_hidden_layers+1
        if any(not -count <= i < count for i in layers):
            raise ValueError(f'Feature indices must address {count} hidden states')
        self.pca_state = None
        self.reference_patches = 0
        self.metadata = {}

    @torch.inference_mode()
    def extract_features(self,batch):
        if batch.ndim != 4 or batch.shape[1] != 3 or len(batch)==0:
            raise ValueError('Expected nonempty [B,3,H,W] image batch')
        if batch.shape[-2]%self.patch_size or batch.shape[-1]%self.patch_size:
            raise ValueError('Input dimensions must be divisible by patch size')
        outputs = self.encoder(pixel_values=batch.to(self.device,dtype=torch.float32),
                               output_hidden_states=True,output_attentions=False)
        drop = 1+getattr(self.encoder.config,'num_register_tokens',0)
        features = torch.stack([outputs.hidden_states[i][:,drop:,:] for i in self.config['layers']],dim=0).mean(0)
        return features

    def fit(self,batches):
        if iter(batches) is batches:
            # A single-use Tensor iterator cannot regenerate augmentation. Cache
            # its features and use the same samples on both statistical passes.
            cached = [self.extract_features(batch).reshape(-1,self.encoder.config.hidden_size).cpu().numpy() for batch in batches]
            source = lambda: iter(cached)
        else:
            def source():
                for batch in batches:
                    yield self.extract_features(batch).reshape(-1,self.encoder.config.hidden_size).cpu().numpy()
        state = fit_pca(source,self.config['pca_device'],self.config['explained_variance'],self.config['components'])
        self.pca_state = state
        self.reference_patches = state['reference_patches']

    @torch.inference_mode()
    def predict(self,batch):
        if self.pca_state is None:
            raise RuntimeError('PCA state is empty; call fit or load first')
        features = self.extract_features(batch)
        grid = (batch.shape[-2]//self.patch_size,batch.shape[-1]//self.patch_size)
        patches = reconstruction_scores(features.reshape(-1,features.shape[-1]).cpu().numpy(),self.pca_state).reshape(len(batch),*grid)
        maps = np.stack([subspace_map(p,batch.shape[-2:]) for p in patches])
        flat = maps.reshape(len(batch),-1)
        k = max(1,int(flat.shape[1]*.01))
        scores = np.partition(flat,flat.shape[1]-k,axis=1)[:,-k:].mean(1)
        return {'pred_score':torch.from_numpy(scores),'anomaly_map':torch.from_numpy(maps[:,None])}

    def save(self,path):
        if self.pca_state is None:
            raise RuntimeError('Cannot save an unfitted PCA model')
        path = Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary = path.with_suffix(path.suffix+'.tmp')
        torch.save({'format_version':1,'algorithm':'subspacead','config':self.config,
                    'weight_sha256':self.weight_sha256,'pca_state':self.pca_state,
                    'asset_hashes':self.asset_hashes,
                    'metadata':self.metadata},temporary)
        temporary.replace(path)

    @classmethod
    def load(cls,path,device='cpu',weights=None):
        saved = torch.load(path,map_location='cpu',weights_only=True)
        if saved.get('algorithm')!='subspacead' or saved.get('format_version')!=1:
            raise ValueError('Unsupported SubspaceAD checkpoint')
        config = dict(saved['config'],device=device,pca_device=device)
        if weights is not None:
            config['weights'] = weights
        model = cls(**config)
        if model.weight_sha256 != saved['weight_sha256']:
            raise ValueError('Backbone weight checksum does not match PCA state')
        if saved.get('asset_hashes',model.asset_hashes) != model.asset_hashes:
            raise ValueError('Backbone/processor config checksum does not match PCA state')
        state = saved['pca_state']
        d = model.encoder.config.hidden_size
        if state['mu'].shape != (d,) or state['components'].shape != (d,state['k']) or state['k']<1:
            raise ValueError('Invalid PCA state dimensions')
        if not all(torch.isfinite(state[k]).all() for k in ['mu','components','eigvals']):
            raise ValueError('Non-finite PCA state')
        model.pca_state = state
        model.reference_patches = state['reference_patches']
        model.metadata = saved.get('metadata',{})
        return model
