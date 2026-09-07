"""Image-level decisions from raw scores and original-resolution region areas."""
import math
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def task_algorithms(task):
    return task['algorithms']


@lru_cache(maxsize=2048)
def largest_region(path, modified_ns, threshold):
    """Area-only slider moves reuse segmentation; reruns invalidate by mtime."""
    amap = np.load(path, allow_pickle=False)
    mask = (amap >= threshold).astype(np.uint8)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def assess(task, directory: Path, algorithm, threshold, area_threshold):
    labels = {item['id']: item.get('label') for item in task['test']}
    counts = dict(false_positive=0, false_negative=0, detected=0, true_negative=0,
                  unlabelled=0, unset=0, failed=0, pending=0, predicted_defect=0)
    rows = []
    for result in task['results']:
        if result.get('algorithm') != algorithm or result.get('data_revision', 0) != task.get('data_revision', 0):
            continue
        row = {'id': result['id'], 'label': labels.get(result['id']), 'area': None}
        score = result.get('score')
        if result.get('error') or score is None or not math.isfinite(score):
            row['classification'] = 'error'
            counts['failed'] += 1
        elif threshold is None:
            row['classification'] = 'unset'
            counts['unset'] += 1
        else:
            positive = score >= threshold
            if area_threshold > 0:
                # Old runs have no raw map; never fabricate a region size.
                raw = result.get('raw_map')
                if not raw or not (directory / 'results' / raw).is_file():
                    row.update(classification='unset', reason='请重新检测以计算缺陷面积')
                    counts['unset'] += 1
                    rows.append(row)
                    continue
                path = directory / 'results' / raw
                row['area'] = largest_region(str(path), path.stat().st_mtime_ns, threshold)
                positive = positive and row['area'] >= area_threshold
            row['classification'] = 'defect' if positive else 'normal'
            counts['predicted_defect'] += int(positive)
            if row['label'] is None:
                counts['unlabelled'] += 1
            elif row['label'] == 'normal':
                counts['false_positive' if positive else 'true_negative'] += 1
            else:
                counts['detected' if positive else 'false_negative'] += 1
        rows.append(row)
    counts['pending'] = max(0, len(task['test']) - len(rows))
    normal = counts['false_positive'] + counts['true_negative']
    defect = counts['detected'] + counts['false_negative']
    return dict(algorithm=algorithm, threshold=threshold, area_threshold=area_threshold,
                counts=counts, rows=rows,
                false_positive_rate=counts['false_positive']/normal if normal else None,
                miss_rate=counts['false_negative']/defect if defect else None,
                recall=counts['detected']/defect if defect else None)
