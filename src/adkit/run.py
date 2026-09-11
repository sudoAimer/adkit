# 命令行运行入口；相对路径基于当前工作目录解析。
"""YAML entry point. Relative paths resolve against the project working directory."""
import argparse
import csv
import json
import logging
from pathlib import Path
import random
import time

import cv2
import numpy as np
from PIL import Image
import torch
import yaml

from .factory import create_detector, load_detector
from .anomalydino import render_map
from .data import samples, select_reference, ReferenceBatches, read_rgb, prepare, restore_map
from .metrics import evaluate


def write_json(path, data):
    """写入 UTF-8 JSON，禁止非有限浮点值进入运行记录。"""
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def synchronize(device):
    """在 CUDA 设备上同步任务，确保计时覆盖实际计算。"""
    if str(device).startswith('cuda'):
        torch.cuda.synchronize(device)


def run(config):
    """根据 YAML 组织采样、建库、推理与结果输出。"""
    torch.set_num_threads(config['run'].get('cpu_threads', 4))
    mode = config['run'].get('mode', 'evaluate')
    if mode not in {'fit', 'predict', 'fit_predict', 'evaluate'}:
        raise ValueError(f"Unknown run mode: {mode}")
    data = config['data']
    normal, tests = samples(data)
    if mode != 'fit' and not tests:
        raise ValueError("No test images found")
    if mode == 'evaluate' and any(t['label'] is None for t in tests):
        raise ValueError("Unlabelled folders support inference, not evaluation")
    output = Path(config['output']['directory'])
    output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in config['run'].get('seeds', [0]) if mode != 'predict' else [None]:
        destination = output / (f'seed_{seed}' if seed is not None else 'prediction')
        destination.mkdir(parents=True, exist_ok=True)
        if (destination / 'run.json').exists():
            raise FileExistsError(f"Run already exists: {destination}. Choose a new output.directory")
        random.seed(seed or 0)
        np.random.seed(seed or 0)
        torch.manual_seed(seed or 0)
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if mode == 'predict':
            model = load_detector(
                config['model']['name'], config['run']['checkpoint'],
                device=config['model'].get('device', 'cpu'),
                weights=config['model'].get('weights'),
            )
            saved_data = model.metadata.get('data', {})
            if saved_data.get('image_size') != data.get('image_size', 448):
                raise ValueError("Inference image_size differs from saved reference preprocessing")
            if saved_data.get('alignment', 'legacy') != data.get('alignment', 'pad'):
                raise ValueError('Inference alignment differs from saved reference preprocessing')
            selected = model.metadata.get('reference_samples', [])
            fit_seconds = 0.
        else:
            model = create_detector(**config['model'])
            selected = select_reference(normal, data.get('shots', -1), seed, data.get('sampling', 'official'))
            synchronize(model.device)
            start = time.perf_counter()
            model.fit(ReferenceBatches(selected, data, config['run'].get('batch_size', 1),
                                       processor=getattr(model,'processor',None),
                                       patch_size=model.patch_size))
            synchronize(model.device)
            fit_seconds = time.perf_counter()-start
            model.metadata = {'reference_samples': [str(p) for p in selected], 'data': data, 'seed': seed}
            model.save(destination/'model.pt')
        print(f'Seed {seed}: reference_patches={model.reference_patches}, fit={fit_seconds:.2f}s', flush=True)
        run_info = {'seed': seed, 'reference_samples': [str(p) for p in selected],
                    'fit_seconds': fit_seconds, 'weight_sha256': model.weight_sha256,
                    'reference_patches': model.reference_patches}
        if getattr(model,'pca_state',None) is not None:
            run_info['pca_components'] = model.pca_state['k']
        (destination/'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        if mode == 'fit':
            write_json(destination/'run.json', run_info)
            continue
        if str(model.device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(model.device)
        rows, maps, masks = [], [], []
        for i, sample in enumerate(tests):
            rgb = read_rgb(sample['path'])
            tensor, geometry = prepare(rgb, data.get('image_size', 448), model.patch_size,
                             processor=getattr(model,'processor',None),
                             alignment=data.get('alignment', 'pad'), return_geometry=True)
            tensor = tensor.unsqueeze(0)
            synchronize(model.device)
            start = time.perf_counter()
            prediction = model.predict(tensor)
            synchronize(model.device)
            seconds = time.perf_counter()-start
            patch = prediction['patch_map'][0, 0].numpy() if 'patch_map' in prediction else None
            # Official path: directly resize patch distances to the original image,
            # then smooth once; don't resize an already smoothed model-input map.
            input_map = prediction['anomaly_map'][0,0].numpy()
            amap = render_map(patch, rgb.shape[:2], model.sigma) if patch is not None else cv2.resize(
                input_map,(rgb.shape[1],rgb.shape[0]),interpolation=cv2.INTER_LINEAR)
            if geometry['alignment'] != 'legacy':
                amap = restore_map(input_map, geometry)
            score = float(prediction['pred_score'][0])
            rows.append({'key': sample['key'], 'path': sample['path'], 'label': sample['label'],
                         'score': score, 'seconds': seconds})
            artifact = destination/'images'/Path(sample['key'])
            artifact.parent.mkdir(parents=True, exist_ok=True)
            if config['output'].get('save_maps', True):
                np.save(artifact.with_suffix('.npy'), amap)
                if patch is not None:
                    np.save(artifact.with_suffix('.patch.npy'), patch)
                else:
                    np.save(artifact.with_suffix('.input.npy'), input_map)
            if config['output'].get('visualize', True):
                # Per-image contrast for display only; never used for metrics/decisions.
                normalized = (amap-amap.min())/max(float(amap.max()-amap.min()), 1e-12)
                heat = cv2.applyColorMap((normalized*255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                Image.fromarray(heat).save(artifact.with_suffix('.heat.png'))
                overlay = np.clip(.6*rgb+.4*heat, 0, 255).astype(np.uint8)
                Image.fromarray(overlay).save(artifact.with_suffix('.overlay.png'))
            if mode == 'evaluate':
                mask = np.zeros(rgb.shape[:2], dtype=bool)
                if sample['mask']:
                    with Image.open(sample['mask']) as image:
                        mask = np.array(image.convert('L')) > 0
                evaluation_map = input_map if config.get('evaluation',{}).get('resolution')=='input' else amap
                if config.get('evaluation',{}).get('resolution') == 'input' and geometry['alignment'] != 'legacy':
                    h, w = geometry['resized']
                    evaluation_map = evaluation_map[:h, :w]
                if mask.shape != evaluation_map.shape:
                    mask = np.array(Image.fromarray(mask.astype(np.uint8)).resize(
                        (evaluation_map.shape[1],evaluation_map.shape[0]),Image.Resampling.NEAREST))>0
                maps.append(evaluation_map)
                masks.append(mask)
            if (i+1) % 20 == 0 or i+1 == len(tests):
                print(f'Seed {seed}: predicted {i+1}/{len(tests)}', flush=True)
        with (destination/'predictions.csv').open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        run_info['mean_prediction_seconds'] = float(np.mean([r['seconds'] for r in rows]))
        run_info['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(model.device) if str(model.device).startswith('cuda') else 0
        if mode == 'evaluate':
            print(f'Seed {seed}: computing exact pixel metrics ({maps[0].shape})', flush=True)
            metrics = evaluate([r['label'] for r in rows], [r['score'] for r in rows], maps, masks,
                               config.get('evaluation', {}).get('pro_fpr_limit', .3))
            run_info['metrics'] = metrics
            summaries.append(metrics)
            print(json.dumps(metrics), flush=True)
        write_json(destination/'run.json', run_info)
        del model, maps, masks
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if summaries:
        summary = {key: {'mean': float(np.mean([m[key] for m in summaries])),
                         'std': float(np.std([m[key] for m in summaries], ddof=0))}
                   for key in summaries[0] if all(m[key] is not None for m in summaries)}
        write_json(output/'summary.json', {'runs': len(summaries), 'std_ddof': 0, 'metrics': summary})
        return summary


def main():
    """解析命令行参数并启动本文件对应的运行流程。"""
    logging.basicConfig(level=logging.INFO,format='%(message)s')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    with open(args.config, encoding='utf-8') as file:
        config = yaml.safe_load(file)
    run(config)


if __name__ == '__main__':
    main()
