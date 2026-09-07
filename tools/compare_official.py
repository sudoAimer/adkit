# 使用本地权重运行 AnomalyDINO 官方对照。
"""Run untouched official detection code with offline backbone loading.

Requires references/AnomalyDINO, references/dinov2 and faiss-cpu.
The mock only replaces the original download call; feature extraction, data
augmentation, reference selection and FAISS search use official source code.
"""
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import sys
from unittest.mock import patch
import json

import faiss
import numpy as np
from PIL import Image
from safetensors.torch import load_file
import torch
import yaml

from adkit import create_detector
from adkit.data import samples, prepare, read_rgb
from adkit.anomalydino import render_map
from adkit.metrics import evaluate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'references'/'AnomalyDINO'))
sys.path.insert(0, str(ROOT/'references'/'dinov2'))
from src.backbones import DINOv2Wrapper
from src.detection import run_anomaly_detection
from src.post_eval import compute_pro, trapezoid
from dinov2.hub.backbones import dinov2_vits14


def main():
    """解析命令行参数并启动本文件对应的运行流程。"""
    config = yaml.safe_load((ROOT/'configs/bottle.yaml').read_text())
    destination = ROOT/'outputs/official_reference'
    destination.mkdir(parents=True, exist_ok=True)
    faiss.omp_set_num_threads(4)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    backbone = dinov2_vits14(pretrained=False).eval()
    state = load_file(str(ROOT/config['model']['weights']))
    # timm omits the unused masked-pretraining token. It never enters inference.
    state['mask_token'] = torch.zeros_like(backbone.mask_token)
    backbone.load_state_dict(state, strict=True)
    with patch('torch.hub.load', return_value=backbone):
        wrapper = DINOv2Wrapper('dinov2_vits14', 'cuda', smaller_edge_size=448)
    _, tests = samples(config['data'])
    our_model = create_detector(**config['model'])
    feature_errors = []
    for record in [tests[0], next(t for t in tests if t['label']==0), tests[-1]]:
        image = read_rgb(record['path'])
        official_tensor, _ = wrapper.prepare_image(image)
        our_tensor = prepare(image, 448, alignment='legacy')
        torch.testing.assert_close(official_tensor, our_tensor, rtol=0, atol=0)
        ref = wrapper.extract_features(official_tensor)
        actual = our_model.extract_features(our_tensor.unsqueeze(0))[0].cpu().numpy()
        feature_errors.append(float(np.max(np.abs(ref-actual))))
    del our_model
    print('Backbone max errors:', feature_errors, flush=True)
    results = []
    for seed in [0,1,2]:
        out = destination/f'seed_{seed}'
        out.mkdir(parents=True, exist_ok=True)
        with (out/'official.log').open('w', encoding='utf-8') as log, redirect_stdout(log), redirect_stderr(log):
            scores, fit_seconds, times = run_anomaly_detection(
                wrapper, 'bottle', str(ROOT/'datasets/mvtec-ad'), 1,
                {'bottle':['broken_large','broken_small','contamination']}, str(out),
                save_examples=False, masking=False, mask_ref_images=False, rotation=True,
                knn_metric='L2_normalized', knn_neighbors=1, faiss_on_cpu=True,
                seed=seed, save_patch_dists=True, save_tiffs=False)
        print(f'Official seed {seed}: predictions complete', flush=True)
        maps, masks, deltas = [], [], []
        for record in tests:
            key = record['key']
            patch_path = out/'anomaly_maps'/f'seed={seed}'/'bottle'/'test'/Path(key).with_suffix('.npy')
            patch_map = np.load(patch_path)
            rgb = read_rgb(record['path'])
            maps.append(render_map(patch_map, rgb.shape[:2]))
            if record['mask']:
                with Image.open(record['mask']) as image:
                    masks.append(np.array(image)>0)
            else:
                masks.append(np.zeros(rgb.shape[:2], dtype=bool))
            ours_path = ROOT/config['output']['directory']/f'seed_{seed}'/'images'/Path(key).with_suffix('.patch.npy')
            if ours_path.exists():
                deltas.append(float(np.max(np.abs(np.load(ours_path)-patch_map))))
        metrics = evaluate([t['label'] for t in tests], [float(scores[t['key']]) for t in tests], maps, masks)
        result = {'seed':seed, 'metrics':metrics, 'max_patch_difference':max(deltas) if deltas else None,
                  'compared_images':len(deltas), 'fit_seconds':fit_seconds,
                  'scores':{k:float(v) for k,v in scores.items()}}
        # Validate our PRO calculation against the original evaluator once.
        if seed == 0:
            fpr, pro = compute_pro(maps, masks)
            reference_pro = float(trapezoid(fpr, pro, x_max=.3)/.3)
            result['original_evaluator_aupro'] = reference_pro
            result['aupro_evaluator_difference'] = abs(reference_pro-metrics['aupro'])
        (out/'comparison.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        results.append(result)
        print(json.dumps({k:v for k,v in result.items() if k!='scores'}), flush=True)
        del maps, masks
    (destination/'summary.json').write_text(json.dumps({'backbone_max_errors':feature_errors,'runs':results},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
