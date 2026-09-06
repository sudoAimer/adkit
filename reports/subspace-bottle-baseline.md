# SubspaceAD：MVTec AD bottle 实测与原文对比

验证日期：2026-09-07。已接入独立 Python 算法库，通过 `model.name: subspacead` 选择，使用现有 `fit / predict / save / load` 接口。

结论：本库与当前官方实现的 3 次实验共 249 张异常图逐像素一致。论文 bottle 1-shot 的像素 AUROC 为 98.6%，本次为 98.59%；其他指标及协议差异如下。本次只验证 bottle，不代表完整 15 类复现。

## 论文与实测指标

论文来源：[SubspaceAD 附录 Table 5，Bottle / 1-shot](https://arxiv.org/html/2602.23013v1#A1.T5)。数值均为百分数；本库报告 seed 42、43、44 的均值和总体标准差（ddof=0）。论文未提供这组均值对应的完整抽样列表和重复次数，因此不声称使用了与论文完全相同的参考图。

| 指标 | 论文 Table 5 | 本库，均值 ± 标准差 | 当前官方代码，同三组样本 |
|---|---:|---:|---:|
| 图像 AUROC | 100.0 | 99.9206 ± 0.0648 | 99.9206 |
| 图像 AP | 100.0 | 99.9751 ± 0.0204 | 99.9751 |
| 像素 AUROC | 98.6 | 98.5883 ± 0.0614 | 98.5883 |
| AUPRO，FPR≤0.3 | 96.1 | 96.6090 ± 0.1103（精确积分） | 96.6023（官方 300 阈值） |
| 像素 AP | 未列出 | 78.7255 ± 1.0594 | 78.7255 |

本库公共评估器使用精确分数排序积分；官方代码使用 300 个分位数阈值计算 AUPRO。对官方热图采用同一个公共评估器，AUPRO 也为 96.6090%，其余指标同样与本库一致。两种 AUPRO 算法在这里仅相差 0.0067 个百分点，不能据此解释与论文约 0.5 个百分点的差距。论文抽样信息不完整，论文描述与当前代码的处理细节也不完全相同，现有证据不足以确定该差距的具体原因。

## 实验协议

- 数据：`datasets/mvtec-ad/bottle`，209 张正常训练图；83 张测试图（20 张正常、63 张异常），63 张缺陷掩码。
- 每组只选 1 张正常图，测试图和测试掩码不参与 PCA 拟合或参数选择。
- 骨干：DINOv2-with-registers-Giant，FP32 eager attention，输入 672×672，batch size 1；本地加载，无在线权重依赖。
- 特征：官方隐藏层索引 `[-12,-13,-14,-15,-16,-17,-18]` 的均值，去除 CLS 和 register token，特征维度 1536。
- 建模：原图加 30 次随机旋转，每遍 71,424 个 patch；FP64 两遍均值/协方差统计和特征分解，保留 99% 方差。
- 当前官方特征生成器在第二遍 PCA 统计时重新生成旋转。本库 YAML 入口的 `ReferenceBatches` 保留此行为。直接传一次性 Tensor 迭代器则缓存特征，在同一组特征上统计两遍。
- 评分：PCA 重建平方残差；异常图双线性缩放到 672×672 后以 3×3、sigma=4 高斯核平滑；最高 1% 像素均值作为图像分数。
- 像素评估使用 672×672 异常图及最近邻缩放的 GT，遵循当前代码。另存原图尺寸热图供展示。论文文字中的原始分辨率描述不等同于此处代码协议。

当前仓库的 `benchmark_few_shot.sh` 使用 seed 42；额外测量 43、44 用于观察参考图变化。

| seed | 正常参考图 | PCA 维数 | 图像 AUROC | 图像 AP | 像素 AUROC | 像素 AP | 精确 AUPRO | 官方阈值 AUPRO |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 42 | 037.png | 819 | 100.0000 | 100.0000 | 98.6276 | 79.6485 | 96.6706 | 96.6635 |
| 43 | 083.png | 810 | 99.9206 | 99.9752 | 98.6357 | 79.2860 | 96.7023 | 96.6960 |
| 44 | 135.png | 808 | 99.8413 | 99.9500 | 98.5017 | 77.2420 | 96.4541 | 96.4474 |

## 官方实现与保存加载验证

参考代码固定为 [CLendering/SubspaceAD @ ef56d5c](https://github.com/CLendering/SubspaceAD/tree/ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7)，位于 `references/SubspaceAD`。对照脚本直接执行其 `main.py`，保留官方预处理、样本选择、PCA、重建与评估流程。

适配包括本地加载权重、不保存未使用的注意力图及 saliency，以及将 NumPy 2.4 已移除的 `trapz` 别名到数学等价的 `trapezoid`。本次配置未启用 saliency masking。这些适配不是未经修改的上游运行，已在 `THIRD_PARTY.md` 中记录。

- 3 组 × 83 张图的最大像素绝对误差：**0**。
- 9 次特征探针（每组 3 张）的最大误差：**0**。
- PCA 维数及均值与官方一致，均值最大误差：**0**。
- 图像分数最大绝对误差：**0.0000152587890625**。最高 1% 元素的浮点求和顺序有细微差异；未改变此次 AUROC/AP。
- 17 项测试通过，覆盖 PCA 数值、官方抽样、两遍旋转、离线小模型保存加载、资产校验及已有算法回归。
- 实际 Giant seed 42 检查点重新加载后，对 `broken_large/000.png` 推理，输入尺寸及原图尺寸异常图与原运行的最大误差均为 **0**。

机器：RTX 4070 SUPER 12 GB，conda `detect`，PyTorch 2.11.0+cu128，Transformers 4.57.6。每组 PCA 拟合约 40.7–40.9 秒，单图 `predict` 平均约 0.65–0.66 秒；计时不含外部读图、预处理、文件输出及指标计算。推理阶段 PyTorch 峰值已分配显存约 5.78 GiB，不等于整个进程或建模阶段峰值。这些是本机观测，不是正式吞吐基准。

子空间检查点约 10 MB，独立骨干权重约 4.55 GB，不把骨干复制进每个检查点。模型资产：

- HF 仓库：`facebook/dinov2-with-registers-giant`。
- revision：`8d0d49f77fb8b5dd78842496ff14afe7dd4d85cb`。
- safetensors SHA256：`c03832d44691e99b62ae28c4dfa2f134853a3614b3756c94f525109cce5a5051`。

## 重跑与产物

在项目根目录执行；重复运行请先在 YAML 中设置新的输出目录，入口会拒绝覆盖已有运行。

```powershell
conda run --no-capture-output -n detect python run.py --config configs/subspace_bottle.yaml
conda run --no-capture-output -n detect python tools/compare_subspace_official.py
conda run --no-capture-output -n detect python run.py --config configs/subspace_predict.yaml
conda run -n detect python -m pytest -q
```

官方对照默认读取既有 `outputs/subspace_bottle` 结果；改变实验输出目录时同步调整对照脚本读取的配置。依赖准备与权重下载命令见 README。

- `outputs/subspace_bottle/summary.json`：本库均值/标准差；各 `seed_*/run.json` 保留参考图、计时、维数和指标。
- `outputs/subspace_bottle/seed_*/images`：输入分辨率 `.input.npy`、原图分辨率 `.npy`、热力图和叠加图。
- `outputs/subspace_official_reference/summary.json`：逐组官方对照结果；各 seed 下同时保留官方热图、PCA 状态及原始 CSV。
- `outputs/subspace_reloaded/verification.json`：实际 Giant 检查点重新加载的一致性验证。
- `reports/subspace-paper-reference.json`：论文 Table 5 的机器可读参考值。

现有 AnomalyDINO 结果保留。两个算法当前采用各自官方的骨干、分辨率和样本选择协议，不能把两份 bottle 报告直接作为控制变量后的算法优劣排名。
