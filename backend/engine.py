"""隔离模型依赖与 Web 接口，使用 adkit 的真实建库和预测流程。"""
import gc
import math
from pathlib import Path


class DetectorEngine:
    """每次作业独立加载骨干，结束时释放显存，避免多任务占满 GPU。"""

    def __init__(self, weights: dict[str, Path], device: str):
        """记录服务器配置的本地权重路径和计算设备。"""
        self.weights = weights
        self.device = device

    def execute(self, task: dict, directory: Path, operation: str, progress) -> dict:
        """在后台线程执行模型作业，完成或失败均释放当前模型资源。"""
        import torch
        from adkit import create_detector, load_detector

        from .assessment import task_algorithms

        algorithms = task_algorithms(task)
        for algorithm_index, algorithm in enumerate(algorithms):
            model = None
            checkpoint = directory / (f'model_{algorithm}.pt' if len(algorithms) > 1 else 'model.pt')
            def report(**changes):
                if 'completed' in changes:
                    changes['completed'] += algorithm_index * len(task['test'])
                if 'message' in changes:
                    changes['message'] = f"{algorithm}: {changes['message']}"
                progress(**changes)
            try:
                report(message='正在加载本地模型…')
                if operation == 'fit':
                    model = create_detector(algorithm, weights=self.weights[algorithm], device=self.device)
                    self.fit(model, task, directory, report, checkpoint)
                else:
                    model = load_detector(algorithm, checkpoint, device=self.device,
                                          weights=self.weights[algorithm])
                    self.predict(model, task, directory, report, algorithm)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        return {'model_ready': True} if operation == 'fit' else {}

    def fit(self, model, task: dict, directory: Path, progress, checkpoint=None) -> None:
        """用全部正常图片建库；SubspaceAD 保留可重复迭代的两遍统计。"""
        from adkit.data import ReferenceBatches

        paths = [directory / "images" / item["file"] for item in task["normal"]]
        data = {"image_size": task["image_size"], "rotation": task["rotation"], "alignment": task.get("alignment", "legacy")}
        batches = ReferenceBatches(paths, data, processor=getattr(model, "processor", None))
        progress(message=f"正在使用 {len(paths)} 张正常图片建立参考库…")
        model.fit(batches)
        model.metadata = {"image_size": task["image_size"], "normal_ids": [item["id"] for item in task["normal"]]}
        model.save(checkpoint or directory / "model.pt")

    def predict(self, model, task: dict, directory: Path, progress, algorithm) -> None:
        """逐图保存原图尺度结果；坏图单独报错，不丢失已完成的检测结果。"""
        import cv2
        import numpy as np
        from PIL import Image
        from adkit.data import prepare, read_rgb, restore_map
        from adkit.anomalydino import render_map

        output = directory / "results"
        output.mkdir(exist_ok=True)
        for index, item in enumerate(task["test"]):
            result = {"id": item["id"], "name": item["name"], "original": item["url"], "algorithm": algorithm}
            try:
                rgb = read_rgb(directory / "images" / item["file"])
                batch, geometry = prepare(rgb, task["image_size"], model.patch_size,
                                processor=getattr(model, "processor", None),
                                alignment=task.get("alignment", "legacy"), return_geometry=True)
                batch = batch.unsqueeze(0)
                prediction = model.predict(batch)
                score = float(prediction["pred_score"][0])
                if not math.isfinite(score):
                    raise ValueError("模型返回了非有限分数")
                if "patch_map" in prediction:
                    amap = render_map(prediction["patch_map"][0, 0].numpy(), rgb.shape[:2], model.sigma)
                else:
                    amap = cv2.resize(prediction["anomaly_map"][0, 0].numpy(),
                                      (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                if geometry["alignment"] != "legacy":
                    amap = restore_map(prediction["anomaly_map"][0, 0].numpy(), geometry)
                if not np.isfinite(amap).all():
                    raise ValueError("模型返回了无效异常图")
                raw_name = f"{item['id']}_{algorithm}.npy"
                np.save(output / raw_name, amap)
                result.update(raw_map=raw_name, pixels=int(amap.size),
                              input_shape=list(batch.shape[-2:]))
                # 颜色仅按单张图拉伸，阈值始终使用未经归一化的图像分数。
                normalized = (amap - amap.min()) / max(float(amap.max() - amap.min()), 1e-12)
                heat = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
                overlay = np.clip(.6 * rgb + .4 * heat, 0, 255).astype(np.uint8)
                for kind, array in [("heatmap", heat), ("overlay", overlay)]:
                    filename = f"{item['id']}_{algorithm}_{kind}.png"
                    Image.fromarray(array).save(output / filename)
                    result[kind] = f"/api/tasks/{task['id']}/files/results/{filename}"
                result["score"] = score
            except Exception as exc:
                result["error"] = f"检测失败：{exc}"
            progress(completed=index + 1, message=f"已检测 {index + 1} / {len(task['test'])} 张", result=result)
