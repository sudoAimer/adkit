# 使用本地权重运行 SubspaceAD 官方对照。
"""Run original SubspaceAD main.py offline and compare every bottle prediction.

Only download/loading and unused saliency attention retention are adapted.
Official augmentation, PCA, scoring, postprocessing and evaluator remain intact.
"""
import argparse
import gc
import importlib.util
import json
import logging
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
import yaml

from adkit.data import samples, prepare, read_rgb
from adkit.metrics import evaluate
from adkit.subspacead import SubspaceADDetector

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'references/SubspaceAD'
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(SOURCE/'src'))
spec = importlib.util.spec_from_file_location('subspace_official_main',SOURCE/'main.py')
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)


def run_seed(seed):
    """运行单个种子的官方 SubspaceAD 对照并记录差异。"""
    config = yaml.safe_load((ROOT/'configs/subspace_giant_bottle.yaml').read_text())
    output = ROOT/'outputs/subspace_official_reference'/f'seed_{seed}'
    output.mkdir(parents=True,exist_ok=True)
    _, records = samples(config['data'])
    feature_errors, captured_maps, captured_pca, sampled_pro = [], [], [], []

    def initialize(self,model_ckpt):
        """将官方骨干加载替换为本地权重，并对照特征提取结果。"""
        local_weights = str(ROOT/config['model']['weights'])
        self.processor = AutoImageProcessor.from_pretrained(local_weights,local_files_only=True,use_fast=False)
        self.model = AutoModel.from_pretrained(local_weights,local_files_only=True,
                                               attn_implementation='eager').eval().requires_grad_(False).cuda()
        forward = self.model.forward
        def without_unused_attention(*args,**kwargs):
            """关闭不使用的注意力输出，同时保留官方接口需要的占位值。"""
            kwargs['output_attentions'] = False
            result = forward(*args,**kwargs)
            # Original extractor checks for non-None even when no mask is used.
            result.attentions = (None,)
            return result
        self.model.forward = without_unused_attention
        probe = SubspaceADDetector.__new__(SubspaceADDetector)
        probe.encoder, probe.device = self.model, torch.device('cuda')
        probe.layers = tuple(config['model']['layers'])
        probe.patch_size = self.model.config.patch_size
        for record in [records[0],next(r for r in records if r['label']==0),records[-1]]:
            rgb = read_rgb(record['path'])
            reference, _, _ = self.extract_tokens([Image.fromarray(rgb)],672,config['model']['layers'],'mean')
            tensor = prepare(rgb,672,processor=self.processor,alignment='legacy').unsqueeze(0)
            ours = probe.extract_features(tensor).cpu().numpy().reshape(reference.shape)
            feature_errors.append(float(np.max(np.abs(reference-ours))))
        print(f'Official seed {seed}: feature extraction comparison {feature_errors}',flush=True)

    def unused_saliency(self,attentions,dino_saliency_layer,num_reg,drop_front,n_expected,batch_size,h_p,w_p):
        """返回占位显著性图，避免缓存未使用的注意力。"""
        return np.zeros((batch_size,h_p,w_p),dtype=np.float32)

    original_post = official.post_process_map
    def record_map(*args,**kwargs):
        """记录官方异常图及其输出文件。"""
        result = original_post(*args,**kwargs)
        captured_maps.append(result.copy())
        key = records[len(captured_maps)-1]['key']
        target = output/'maps'/Path(key).with_suffix('.npy')
        target.parent.mkdir(parents=True,exist_ok=True)
        np.save(target,result)
        if len(captured_maps)%20==0:
            print(f'Official seed {seed}: predicted {len(captured_maps)}/{len(records)}',flush=True)
        return result

    original_fit = official.PCAModel.fit
    def record_fit(self,*args,**kwargs):
        """记录官方 PCA 拟合状态以供离线对照。"""
        result = original_fit(self,*args,**kwargs)
        captured_pca.append(result)
        torch.save({k:torch.from_numpy(v) if isinstance(v,np.ndarray) else v for k,v in result.items()},output/'pca.pt')
        print(f'Official seed {seed}: PCA rank={result["k"]}',flush=True)
        return result

    original_pro = official.compute_aupro
    def record_pro(*args,**kwargs):
        """记录官方采样式 AUPRO 结果。"""
        result = original_pro(*args,**kwargs)
        sampled_pro.append(float(result))
        return result

    argv = ['main.py','--dataset_name','mvtec_ad','--dataset_path',str(ROOT/'datasets/mvtec-ad'),
            '--categories','bottle','--image_res','672','--k_shot','1','--layers=-12,-13,-14,-15,-16,-17,-18',
            '--model_ckpt','facebook/dinov2-with-registers-giant','--aug_count','30','--pca_ev','0.99',
            '--seed',str(seed),'--agg_method','mean','--outdir',str(output.relative_to(ROOT))]
    with patch.object(official.FeatureExtractor,'__init__',initialize), \
         patch.object(official.FeatureExtractor,'_get_saliency_mask',unused_saliency), \
         patch.object(official.PCAModel,'fit',record_fit), \
         patch.object(official,'post_process_map',record_map), \
         patch.object(official,'compute_aupro',record_pro), \
         patch.object(np,'trapz',np.trapezoid,create=True), patch.object(sys,'argv',argv):
        official.main()
    assert len(captured_maps)==len(records), (len(captured_maps),len(records))
    # Release the reference backbone before CPU metrics and the next repetition.
    gc.collect()
    torch.cuda.empty_cache()
    masks, scores, map_errors, score_errors = [], [], [], []
    own_dir = ROOT/config['output']['directory']/f'seed_{seed}'
    import csv
    with (own_dir/'predictions.csv').open() as file:
        own_scores = {r['key']:float(r['score']) for r in csv.DictReader(file)}
    for record,amap in zip(records,captured_maps):
        if record['mask']:
            with Image.open(record['mask']) as image:
                mask = np.array(image.convert('L').resize((672,672),Image.Resampling.NEAREST))>0
        else:
            mask = np.zeros((672,672),dtype=bool)
        masks.append(mask)
        score = float(official.topk_mean(amap,frac=.01))
        scores.append(score)
        ours = np.load(own_dir/'images'/Path(record['key']).with_suffix('.input.npy'))
        map_errors.append(float(np.max(np.abs(ours-amap))))
        score_errors.append(abs(score-own_scores[record['key']]))
    metrics = evaluate([r['label'] for r in records],scores,captured_maps,masks)
    own_state = torch.load(own_dir/'model.pt',map_location='cpu',weights_only=True)['pca_state']
    reference = captured_pca[0]
    result = {'seed':seed,'compared_images':len(records),'feature_max_errors':feature_errors,
              'metrics':metrics,'official_sampled_aupro':sampled_pro[0],
              'max_map_difference':max(map_errors),'max_score_difference':max(score_errors),
              'official_pca_rank':reference['k'],'runtime_pca_rank':own_state['k'],
              'pca_mean_max_difference':float(np.max(np.abs(reference['mu']-own_state['mu'].numpy())))}
    (output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    results = [run_seed(seed) for seed in args.seeds]
    (ROOT/'outputs/subspace_official_reference/summary.json').write_text(json.dumps({'runs':results},indent=2),encoding='utf-8')
