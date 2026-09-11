# 评估与复现

[返回首页](../README.md) · [使用指南](usage.md)

评估流程用于比较异常分数、异常图及官方实现的一致性。普通 Python 调用和网页工作台无需准备参考仓库或评估数据。

## 准备环境与数据

先创建并激活 [uv 环境](environment.md)，再安装评估依赖：

```bash
uv pip install -e ".[test,reference,subspacead]" --torch-backend=auto
git submodule update --init --recursive
```

按运行机器选择 PyTorch 后端，并准备模型权重、MVTec 数据与缺陷标注。`references/` 中的版本信息见 [THIRD_PARTY.md](../THIRD_PARTY.md)。

检查配置中的数据路径、权重路径、设备与输出目录。每次重跑使用新的 `output.directory`，对照脚本从相应配置读取本库结果位置。

## 运行评估

AnomalyDINO：

```bash
adkit --config configs/bottle.yaml
python tools/compare_official.py
```

SubspaceAD：

```bash
adkit --config configs/subspace_giant_bottle.yaml
python tools/compare_subspace_official.py
```

对照脚本需要先生成对应的本库运行结果。SubspaceAD 官方对照脚本使用 CUDA，应在具备相应 GPU 和驱动的机器上执行。

测试入口：

```bash
python -m pytest -q
```

部分测试依赖本地权重或官方参考代码，缺少对应资源时会跳过。`requirements-tested.txt` 保存历史实验使用的依赖版本，具体实验范围以对应报告为准。

## 评估协议

| 项目 | 约定 |
|---|---|
| 图像指标 | AUROC、AP |
| 像素指标 | AUROC、AP、AUPRO |
| AUPRO | 8 连通区域，FPR 上限默认 0.3，按精确分数排序积分 |
| 多次运行汇总 | 均值与总体标准差，`ddof=0` |
| 数据划分 | 正常训练样本用于建库，测试图像及掩码用于评估 |
| 分数 | 使用未经可视化归一化的异常分数与异常图 |

AnomalyDINO 的 `official` 采样对排序后的正常图片取 `[seed * shots : (seed + 1) * shots]`。预处理保持比例缩放短边，并裁去底部和右侧不足一个 patch 的余量；输出原图尺度异常图时先缩放 patch 距离图，再高斯平滑。

SubspaceAD 历史 Giant 官方配置（`subspace_giant_bottle.yaml`）使用 672×672 输入，GT 掩码按最近邻插值缩放到相同尺寸。少样本增强采用原图加随机旋转，两遍 PCA 统计时重新生成增强；原图尺度异常图另行保存供展示。

历史 Giant 基线与 AnomalyDINO 的骨干、预处理及像素评估分辨率不同，比较时需要同时注明协议。网页工作台的固定角度增强流程也不等同于 SubspaceAD 官方少样本协议。

## 同批数据模型对比

工作台从注册表动态选择一个、多个或全部模型，在同一份正常／待测图片及标注上独立建库和评分。当前内置 AnomalyDINO 与 SubspaceAD，新增注册模型无需修改前端。所有模型共享目标输入几何与固定角度增强，按各自的 patch 要求向上补齐。执行选择和展示选择独立，比较仅纳入当前图片版本的结果。表格按模型独立阈值统计图像级 FP/FN/TP/TN，未标注图片不计入这些指标。该流程不等同于下述历史官方协议；官方 YAML 使用 `data.alignment: legacy`，新工作台使用保留内容的向上补边。

## 官方代码适配

对照流程使用本地权重加载，具体适配记录见 [第三方来源说明](../THIRD_PARTY.md)。SubspaceAD 对照关闭未使用的 saliency attention 存储，并为使用 `np.trapz` 的参考代码提供 `np.trapezoid` 兼容映射。

本库精确 AUPRO 与官方采样阈值 AUPRO 在报告中分别列出。

## 历史报告

| 报告 | 内容 |
|---|---|
| [AnomalyDINO bottle 基线](../reports/bottle-baseline.md) | 单类别少样本评估、特征及分数对照 |
| [SubspaceAD bottle 基线](../reports/subspace-bottle-baseline.md) | PCA、预处理、官方代码与论文参考值对照 |

报告记录对应实验时的环境与结果，适用范围为报告中的数据、配置和代码版本。重新运行请使用本页的环境与命令说明。

默认 `subspace_bottle.yaml` 现在使用 Small，与 AnomalyDINO 共用权重；原报告仍代表 Giant，不代表 Small 精度。Small 配置保留 672×672 和原 few-shot 增强设置，论文骨干消融脚本则使用 448px，不能直接当作本配置的实测降幅。
