"""Shared DINOv2 token preparation for AnomalyDINO and SubspaceAD."""
# DINOv2 positional interpolation extracted unchanged from anomalydino.py.
import torch
from torch.nn import functional as F


def input_tokens(encoder, batch, positional_encoding='official'):
    """Patch/CLS tokens before transformer blocks; preserve official +0.1 resize."""
    if positional_encoding == 'timm':
        return encoder.norm_pre(encoder.patch_drop(encoder._pos_embed(encoder.patch_embed(batch))))
    if positional_encoding != 'official' or encoder.num_prefix_tokens != 1:
        raise ValueError('Official positional encoding requires DINOv2 without registers')
    x = encoder.patch_embed(batch)
    b, h, w, dim = x.shape
    x = torch.cat((encoder.cls_token.expand(b, -1, -1), x.reshape(b, -1, dim)), dim=1)
    positions = encoder.pos_embed
    side = int((positions.shape[1] - 1) ** .5)
    if (h, w) != (side, side):
        patches = F.interpolate(positions[:, 1:].reshape(1, side, side, dim).permute(0, 3, 1, 2),
                                scale_factor=((h + .1) / side, (w + .1) / side),
                                mode='bicubic', antialias=False)
        positions = torch.cat((positions[:, :1], patches.permute(0, 2, 3, 1).reshape(1, h * w, dim)), dim=1)
    return encoder.norm_pre(encoder.pos_drop(x + positions))
