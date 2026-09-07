# 图像级与像素级异常检测评估指标。
"""Exact ranking metrics and region-weighted PRO, without threshold sampling."""
import numpy as np
from scipy.ndimage import label
from sklearn.metrics import roc_auc_score, average_precision_score


def aupro(maps, masks, limit=.3):
    """按精确分数排序积分计算限定假阳性率范围内的区域重叠指标。"""
    if not 0 < limit <= 1:
        raise ValueError("PRO FPR limit must lie in (0,1]")
    changes = []
    regions = 0
    for mask in masks:
        components, count = label(mask, np.ones((3, 3)))
        sizes = np.bincount(components.ravel())
        weights = np.zeros(count+1)
        weights[1:] = 1 / sizes[1:]
        changes.append(weights[components].ravel())
        regions += count
    if regions == 0:
        return None
    scores = np.concatenate([m.ravel() for m in maps])
    negatives = ~np.concatenate([m.ravel().astype(bool) for m in masks])
    if negatives.sum() == 0:
        return None
    order = np.argsort(scores)[::-1]
    sorted_scores = scores[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores)-1]
    fp = np.cumsum(negatives[order], dtype=np.int64)[ends] / negatives.sum()
    pro = np.cumsum(np.concatenate(changes)[order], dtype=np.float64)[ends] / regions
    fp, pro = np.r_[0., fp, 1.], np.r_[0., np.minimum(pro, 1.), 1.]
    inside = fp <= limit
    x, y = fp[inside], pro[inside]
    if x[-1] < limit:
        right = np.searchsorted(fp, limit, side='right')
        value = pro[right-1] + (pro[right]-pro[right-1])*(limit-fp[right-1])/(fp[right]-fp[right-1])
        x, y = np.r_[x, limit], np.r_[y, value]
    return float(np.sum(np.diff(x)*(y[1:]+y[:-1])*.5)/limit)


def evaluate(labels, scores, maps, masks, pro_fpr_limit=.3):
    """汇总图像级和像素级指标，拒绝缺少必要类别的输入。"""
    if len(labels) == 0 or len(set(labels)) != 2:
        raise ValueError("Evaluation requires both normal and anomalous test images")
    if not (len(labels) == len(scores) == len(maps) == len(masks)):
        raise ValueError("Prediction and ground-truth counts differ")
    if not np.isfinite(scores).all():
        raise ValueError("Non-finite image scores")
    for image_map, mask in zip(maps, masks):
        if image_map.shape != mask.shape or not np.isfinite(image_map).all():
            raise ValueError("Invalid anomaly map or ground-truth shape mismatch")
    truth = np.concatenate([m.ravel() for m in masks])
    predictions = np.concatenate([m.ravel() for m in maps])
    result = {'image_auroc': float(roc_auc_score(labels, scores)),
              'image_ap': float(average_precision_score(labels, scores)),
              'pixel_auroc': float(roc_auc_score(truth, predictions)) if len(np.unique(truth)) == 2 else None,
              'pixel_ap': float(average_precision_score(truth, predictions)) if truth.any() else None}
    del truth, predictions
    result['aupro'] = aupro(maps, masks, pro_fpr_limit)
    return result
