# 图像读取、数据集解析、参考样本选择与批次生成。
"""Image and dataset helpers; no dataset framework required."""
from pathlib import Path
import json
import random

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms


def image_paths(path):
    """收集支持的图像文件并返回绝对路径。"""
    path = Path(path)
    candidates = [path] if path.is_file() else sorted(path.rglob('*'))
    return [p.resolve() for p in candidates if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}]


def read_rgb(path):
    """读取图像并统一转换为 RGB 数组。"""
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def prepare(image, image_size=448, patch_size=14, processor=None):
    """根据算法处理器缩放归一化图像，并满足 patch 尺寸要求。"""
    if image_size < patch_size:
        raise ValueError("image_size must be at least one patch")
    if processor is not None:
        if image_size % patch_size:
            raise ValueError('SubspaceAD image_size must be divisible by patch_size')
        return processor(images=[Image.fromarray(image)],return_tensors='pt',do_resize=True,
                         size={'height':image_size,'width':image_size},do_center_crop=False,
                         crop_size={'height':image_size,'width':image_size}).pixel_values[0]
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.ToTensor(),
        transforms.Normalize((.485, .456, .406), (.229, .224, .225)),
    ])
    tensor = transform(Image.fromarray(image))
    h, w = tensor.shape[-2:]
    # Official DINO wrapper removes only the bottom/right remainder.
    return tensor[:, :h-h % patch_size, :w-w % patch_size]


def augment(image, rotation):
    """按配置生成固定角度旋转图像。"""
    for angle in range(0, 360, 45) if rotation else [None]:
        if angle is None:
            yield image
        else:
            center = tuple(np.array(image.shape[1::-1]) / 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.)
            yield cv2.warpAffine(image, matrix, image.shape[1::-1],
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_DEFAULT)


def reference_batches(paths, config, batch_size=1, processor=None):
    """根据数据配置生成参考批次，尺寸不同时提前结束当前批次。"""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    pending = []
    for path in paths:
        if config.get('augmentation') == 'subspacead':
            original = Image.fromarray(read_rgb(path))
            rotation = transforms.RandomRotation(degrees=(0,345))
            count = config.get('aug_count',30) if config.get('shots',1) != -1 else 0
            if count < 0:
                raise ValueError('aug_count must be non-negative')
            # Generate all rotations before extraction, just like the official source.
            images = [np.array(original)]+[np.array(rotation(original)) for _ in range(count)]
        else:
            images = augment(read_rgb(path), config.get('rotation', False))
        for image in images:
            tensor = prepare(image, config.get('image_size', 448), processor=processor)
            if pending and pending[0].shape != tensor.shape:
                yield torch.stack(pending)
                pending = []
            pending.append(tensor)
            if len(pending) == batch_size:
                yield torch.stack(pending)
                pending = []
    if pending:
        yield torch.stack(pending)


class ReferenceBatches:
    """Re-iterable batches for two-pass PCA; augmentations regenerate per pass."""
    def __init__(self, paths, config, batch_size=1, processor=None):
        """保存路径、数据配置和批大小，延迟到遍历时读取图像。"""
        self.args = paths, config, batch_size, processor

    def __iter__(self):
        """每次遍历重新生成批次，使两遍 PCA 能重新采样增强。"""
        return reference_batches(*self.args)


def samples(config):
    """解析普通目录或 MVTec 数据，检查图像和标注路径。
    Return normal paths and test records with optional masks/labels."""
    kind = config.get('format', 'mvtec')
    if kind == 'folder':
        normal = image_paths(config['normal_dir']) if config.get('normal_dir') else []
        tests = [{'path': str(p), 'key': str(p.relative_to(Path(config['test_path']).resolve()))
                  if Path(config['test_path']).is_dir() else p.name,
                  'label': None, 'mask': None} for p in image_paths(config['test_path'])] if config.get('test_path') else []
    elif kind == 'mvtec':
        root = Path(config['root']).resolve()
        category = config['category']
        if (root / 'samples.json').is_file():
            records = json.loads((root / 'samples.json').read_text(encoding='utf-8'))['samples']
            records = [r for r in records if r['category']['label'] == category]
            normal = sorted(root/r['filepath'] for r in records if r['split']=='train' and r['defect']['label']=='good')
            tests = [{'path': str(root/r['filepath']), 'key': r['defect']['label']+'/'+Path(r['filepath']).name,
                      'label': int(r['defect']['label']!='good'),
                      'mask': str(root/r['defect_mask']['mask_path']) if r.get('defect_mask') else None}
                     for r in records if r['split']=='test']
        else:
            root = root/category
            normal = image_paths(root/'train'/'good')
            tests = []
            for path in image_paths(root/'test'):
                defect = path.parent.name
                tests.append({'path': str(path), 'key': defect+'/'+path.name, 'label': int(defect!='good'),
                              'mask': str(root/'ground_truth'/defect/(path.stem+'_mask.png')) if defect!='good' else None})
        tests.sort(key=lambda r: r['key'])
    else:
        raise ValueError(f"Unknown data format: {kind}")
    for path in [*normal, *[r['path'] for r in tests], *[r['mask'] for r in tests if r['mask']]]:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Missing dataset file: {path}")
    if kind == 'mvtec' and any(r['label'] == 1 and r['mask'] is None for r in tests):
        raise ValueError("Anomalous MVTec sample is missing its ground-truth mask")
    return normal, tests


def select_reference(paths, shots, seed, sampling='official'):
    """按指定采样协议和种子选择正常参考图像。"""
    if not paths:
        raise ValueError("No normal reference images found")
    if shots == -1:
        return list(paths)
    if shots < 1 or seed < 0:
        raise ValueError("shots must be -1 or positive; seed must be non-negative")
    if sampling == 'official':
        selected = sorted(paths)[seed*shots:(seed+1)*shots]
    elif sampling == 'random':
        selected = random.Random(seed).sample(sorted(paths), shots)
    elif sampling == 'subspacead':
        selected = sorted(paths)
        random.Random(seed).shuffle(selected)
        selected = selected[:shots]
    else:
        raise ValueError(f"Unknown sampling method: {sampling}")
    if len(selected) != shots:
        raise ValueError("Not enough reference images for this shots/seed combination")
    return selected
