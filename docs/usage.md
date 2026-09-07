# 使用指南

[返回首页](../README.md) · [环境配置](environment.md) · [网页工作台](../backend/README.md)

## Python API

两个检测器均通过构造函数接收参数。`fit` 使用正常图片的 Tensor 批次重建参考状态；`predict` 接收 `[B, 3, H, W]` Tensor。再次调用 `fit` 会替换已有参考状态。

### AnomalyDINO

```python
from adkit.anomalydino import AnomalyDinoDetector
from adkit.data import image_paths, prepare, read_rgb, reference_batches

detector = AnomalyDinoDetector(
    weights="weights/dinov2_vits14/model.safetensors",
    device="cpu",
    num_neighbours=1,
    query_chunk=1024,
    bank_chunk=16384,
)
data = {"image_size": 448, "rotation": False}
detector.fit(reference_batches(image_paths("images/normal"), data))
batch = prepare(read_rgb("images/test.png"), 448).unsqueeze(0)
prediction = detector.predict(batch)
```

| 参数 | 用途 |
|---|---|
| `weights` | 本地权重文件，结构需与骨干匹配 |
| `encoder_name` | 骨干名称，默认 `vit_small_patch14_dinov2.lvd142m` |
| `device` | 特征提取设备，默认 `cuda` |
| `num_neighbours` | 近邻数量，默认 1 |
| `query_chunk` / `bank_chunk` | 精确检索分块大小，默认 1024 / 16384 |
| `masking` / `mask_ref_images` | 前景筛选及是否应用于参考图，默认关闭 |
| `sigma` | 异常图平滑参数，默认 4.0 |
| `positional_encoding` | 位置编码插值方式，默认 `official`，可选 `timm` |

### SubspaceAD

先安装 `subspacead` 可选依赖并准备本地模型目录。建库与预测都应使用检测器的 `processor`：

```python
from adkit.subspacead import SubspaceADDetector
from adkit.data import image_paths, prepare, read_rgb, ReferenceBatches

detector = SubspaceADDetector(
    weights="weights/dinov2_with_registers_giant",
    device="cpu",
    explained_variance=0.99,
)
batches = ReferenceBatches(
    image_paths("images/normal"),
    {"image_size": 672},
    processor=detector.processor,
)
detector.fit(batches)
batch = prepare(
    read_rgb("images/test.png"), 672, processor=detector.processor,
).unsqueeze(0)
prediction = detector.predict(batch)
```

| 参数 | 用途 |
|---|---|
| `weights` | 包含权重、模型配置与预处理配置的本地目录 |
| `device` / `pca_device` | 特征提取 / PCA 设备，后者默认跟随前者 |
| `layers` | 参与平均的隐藏层，默认 `[-12,-13,-14,-15,-16,-17,-18]` |
| `explained_variance` | 累计解释方差目标，默认 0.99 |
| `components` | 指定 PCA 维数，设置后优先于解释方差目标 |

`ReferenceBatches` 支持重复迭代，适合两遍 PCA 统计；使用随机增强时，每遍重新生成增强。若传入一次性迭代器，检测器会缓存特征，在同一组特征上完成两遍统计。

### 工厂与检查点

可以按算法名创建检测器：

```python
from adkit import create_detector, load_detector

detector = create_detector(
    "anomalydino", weights="weights/dinov2_vits14/model.safetensors", device="cpu",
)
# 完成 fit 后调用 detector.save("outputs/model.pt")。
restored = load_detector("anomalydino", "outputs/model.pt", device="cpu")
```

也可使用对应类的 `load(path, device=..., weights=...)`。检查点保存参考状态、构造参数、元数据和权重校验信息，不包含骨干权重；换机器后仍需准备相同权重，可用 `weights` 覆盖其路径。

当前检查点格式为 `format_version=2`，构造参数保存在 `init_params`。加载时仅覆盖设备和权重路径；算法参数或正常样本发生变化时应重新建库。旧格式检查点需要重新生成。

Python 示例保存的检查点可通过 Python API 加载。YAML 的预测入口还会校验建库时保存的数据预处理元数据，应使用同一 YAML 运行流程生成的检查点；工作台任务由工作台管理。

### 返回结果

`predict` 返回 CPU Tensor 字典：

| 字段 | 形状 | 含义 |
|---|---|---|
| `pred_score` | `[B]` | 原始图像异常分数，非概率 |
| `anomaly_map` | `[B,1,H,W]` | 输入 Tensor 分辨率的异常图 |
| `patch_map` | `[B,1,H/14,W/14]` | AnomalyDINO 专用的未平滑 patch 距离图 |

`H`、`W` 为预处理后的输入尺寸。SubspaceAD 不返回 `patch_map`。核心 API 不作阈值判定；工作台在图像分数上应用用户设置的阈值。

## YAML 与命令行

所有相对路径基于启动命令的当前工作目录。下面的普通目录配置使用全部正常图片，可保存为 `configs/local.yaml`，并将路径改为自己的文件：

```yaml
model:
  name: anomalydino
  weights: weights/dinov2_vits14/model.safetensors
  device: cpu
data:
  format: folder
  normal_dir: images/normal
  test_path: images/test
  image_size: 448
  rotation: false
  shots: -1
run:
  mode: fit_predict
  seeds: [0]
  batch_size: 1
output:
  directory: outputs/local
  save_maps: true
  visualize: true
```

```bash
adkit --config configs/local.yaml
```

| 配置 | 含义 |
|---|---|
| `run.mode` | `fit` 建库；`predict` 加载后预测；`fit_predict` 建库后预测；`evaluate` 建库后评估 |
| `run.checkpoint` | `predict` 模式的检查点路径 |
| `run.seeds` | 建库重复使用的种子列表，预测模式不使用 |
| `data.shots` | `-1` 使用全部正常图片，正整数使用指定数量 |
| `data.sampling` | `official` 使用排序切片；`random` 随机抽样；`subspacead` 按种子打乱后截取 |
| `data.image_size` | 预处理尺寸，加载后预测须与建库一致 |
| `output.directory` | 运行结果目录，重跑时选择新的目录 |

`official` 切片为 `[seed * shots : (seed + 1) * shots]`。模型参数写在 `model` 中，由入口展开传给检测器构造函数。普通目录无标签时支持建库、预测，不计算评估指标。

现有配置示例：

- [folder.yaml](../configs/folder.yaml)：普通目录建库与预测，需修改其中的示例数据路径。
- [predict.yaml](../configs/predict.yaml)：加载 AnomalyDINO 参考库预测。
- [subspace_predict.yaml](../configs/subspace_predict.yaml)：加载 SubspaceAD 子空间预测。
- [bottle.yaml](../configs/bottle.yaml)、[subspace_bottle.yaml](../configs/subspace_bottle.yaml)：MVTec bottle 评估配置，协议见 [评估文档](evaluation.md)。

## 数据与输出

MVTec 数据支持标准 `category/train/good`、`category/test/*`、`category/ground_truth/*` 目录，以及 FiftyOne 导出的 `samples.json`。普通目录使用 `normal_dir` 和 `test_path`，后者可指向单张图片或目录。

YAML 建库运行保存在 `output.directory/seed_<seed>/`，独立预测保存在 `output.directory/prediction/`。根据运行模式和输出选项生成：

| 文件 | 内容 |
|---|---|
| `config.yaml`、`run.json` | 配置快照与运行记录 |
| `model.pt` | 建库检查点，独立预测不重新保存 |
| `predictions.csv` | 文件名、图像分数与预测耗时 |
| `images/*.npy` | 原图尺寸的浮点异常图 |
| `images/*.patch.npy` | AnomalyDINO patch 距离图 |
| `images/*.input.npy` | SubspaceAD 输入尺寸异常图 |
| `images/*.heat.png`、`*.overlay.png` | 热力图与叠加图 |
| `summary.json` | 评估模式的多次运行指标汇总，位于输出根目录 |

AnomalyDINO 保持比例缩放短边，裁去不能整除 patch 的底部和右侧余量；SubspaceAD 使用模型 processor 缩放为指定正方形尺寸。具体像素评估协议见 [评估文档](evaluation.md)。

网页工作台的任务文件独立保存在 `ADKIT_DATA_DIR` 中，目录和使用规则见 [工作台说明](../backend/README.md)。
