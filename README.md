# adkit

轻量、离线运行的 Python 异常检测库。目前实现 AnomalyDINO 和 SubspaceAD；通过 YAML 选择算法、模型参数和数据，无 Lightning/FastAPI 依赖。

## 安装与运行

在项目根目录执行，所有 YAML 相对路径均相对于当前工作目录。

首次克隆时包含固定版本的官方参考代码：

```powershell
git clone --recurse-submodules https://github.com/sudoAimer/adkit.git
cd adkit
```

如果已普通克隆，执行 `git submodule update --init --recursive`。数据集、权重和运行输出不随 Git 提交，需要在本机准备。

```powershell
conda run -n detect python -m pip install -e . --no-deps
conda run --no-capture-output -n detect python run.py --config configs/bottle.yaml
```

`--no-deps` 适用于已经准备好的本机 `detect` 环境。新环境先安装适合硬件的 PyTorch，再运行 `pip install -e .`。本次测试的具体依赖版本见 `requirements-tested.txt`。

- `configs/bottle.yaml`：bottle / 1-shot / 3 次重复，输出完整指标。
- `configs/folder.yaml`：普通正常图片目录建库，推理单张图片。
- `configs/predict.yaml`：加载已有参考库后推理。
- `configs/subspace_bottle.yaml`：SubspaceAD bottle / 1-shot，seed 42、43、44。
- `configs/subspace_predict.yaml`：加载已有 SubspaceAD 子空间后推理。
- `run.mode` 可选 `fit`、`predict`、`fit_predict`、`evaluate`。
- `data.shots: -1` 使用全部正常图片；正整数指定 k-shot。
- `data.sampling: official` 按排序后的 `[seed*k:(seed+1)*k]` 选择参考图；`random` 才是真正的种子随机抽样。
- 正式配置使用原版位置编码插值。`model.positional_encoding: timm` 仅用于对照 timm 默认行为。
- 修改实验配置后使用新的 `output.directory`，避免覆盖已经完成的运行。

权重固定从 `model.weights` 指定的本地文件加载，不会自动联网下载。

SubspaceAD 首次准备（默认下载代理为 `127.0.0.1:7897`）：

```powershell
conda run -n detect python -m pip install -e .[subspacead]
conda run --no-capture-output -n detect python tools/download_subspace_weights.py
conda run --no-capture-output -n detect python run.py --config configs/subspace_bottle.yaml
```

SubspaceAD 的 `weights` 是包含 `model.safetensors`、`config.json`、`preprocessor_config.json` 的本地目录。官方 Giant 模型约 4.55 GB，不会替换为小骨干。运行采用 FP32 eager attention，不保留未使用的 saliency attention maps。

## Python 调用

```python
import yaml
from adkit import create_detector, load_detector
from adkit.data import samples, select_reference, reference_batches
from adkit.anomalydino import AnomalyDinoDetector

with open('configs/bottle.yaml', encoding='utf-8') as file:
    config = yaml.safe_load(file)
detector = AnomalyDinoDetector(
    weights=config['model']['weights'],
    device='cuda',
    num_neighbours=1,
)
normal, tests = samples(config['data'])
selected = select_reference(normal, shots=1, seed=0)
detector.fit(reference_batches(selected, config['data']))
batch = next(reference_batches(selected, dict(config['data'], rotation=False)))
prediction = detector.predict(batch)
detector.save('outputs/example.pt')
restored = AnomalyDinoDetector.load('outputs/example.pt', device='cuda')
```

基类只有 `fit / predict / save / load`。`predict` 返回 CPU Tensor 字典：

| 字段 | 形状 | 含义 |
|---|---|---|
| `pred_score` | `[B]` | 原始图像异常分数，非概率 |
| `anomaly_map` | `[B,1,H,W]` | 输入 Tensor 分辨率下的异常图 |
| `patch_map` | `[B,1,H/14,W/14]` | AnomalyDINO 的未平滑 patch 距离图，供原图尺度渲染 |

SubspaceAD 返回 `pred_score` 和 `anomaly_map`，不返回 AnomalyDINO 专用的 `patch_map`。使用 `create_detector("anomalydino", weights=path, device="cuda")` 按名称构造算法；使用 `load_detector("subspacead", checkpoint_path, device="cuda")` 加载检查点（从 `adkit` 导入）。

`fit` 重建参考库，不执行反向传播。检查点格式为 2，构造参数存于 `init_params`，不兼容旧格式；项目版本保持 `0.1.0`。检查点只含参考库、参数、元数据和权重 SHA256，不重复包含骨干权重；迁移机器时可在 `load(..., weights=新路径)` 指定同一权重文件。

SubspaceAD 检查点保存均值、PCA 基、特征值及配置，不保存正常样本特征库。其 `fit` 接收 Tensor 批次：单次迭代器缓存特征做两遍统计；`ReferenceBatches` 可重复迭代并在第二遍重新生成旋转，与当前官方代码一致。

## 数据与输出

MVTec 支持标准 `category/train/good`、`category/test/*`、`category/ground_truth/*`，也支持 FiftyOne 导出的 `samples.json`。普通目录通过 `normal_dir` 和 `test_path` 指定，无标签时不计算评估指标。

每次运行保存 `config.yaml`、`model.pt`、`predictions.csv`、`run.json` 及 `images/`。图像产物包括原图分辨率浮点 `.npy`、patch 距离 `.patch.npy`、热力图和叠加图。`summary.json` 汇总各次指标，标准差使用 `ddof=0`。

SubspaceAD 另外保存模型输入分辨率的 `.input.npy`。其官方配置在 672×672 上评估，GT 掩码按最近邻缩放；原图尺寸 `.npy` 用于展示。因此两个算法沿用各自官方协议的像素指标，不能直接视为控制了骨干、分辨率和样本选择的公平横向比较。

热力图为显示而单图拉伸色彩，不能通过最红区域判断是否有缺陷。评估始终使用未归一化异常分数；本版本不输出阈值、缺陷框或正常/异常结论。

AnomalyDINO 预处理遵循官方流程：保持比例缩放短边、从下方/右方裁去不足一个 patch 的余量。运行入口将 patch 图直接缩放回原图尺寸，再做 sigma=4 高斯平滑；与官方一致，非整除尺寸的边缘存在裁剪后的轻微坐标伸缩。SubspaceAD 则使用官方 HF processor 缩放到 672×672，不裁剪；重建误差图双线性插值后使用 3×3、sigma=4 高斯平滑。

## 精度验证

```powershell
conda run -n detect python -m pytest -q
conda run --no-capture-output -n detect python tools/compare_official.py
```

官方对照另需 `pip install -e .[reference]`，以及保存在 `references/` 的原始代码。它只替换官方骨干的联网加载调用，正常执行官方预处理、样本选择、特征提取、FAISS 精确检索和评分。模型权重相同；timm 未保留的预训练 mask token 在无掩码推理中不使用。

指标为图像 AUROC/AP、像素 AUROC/AP、AUPRO（8 连通，FPR≤0.3，精确分数排序积分）。测试图片及掩码不参与建库或参数选择。先测量官方代码一致性，不以全类别论文平均值冒充 bottle 的参考值。

来源、版本与修改说明见 `THIRD_PARTY.md`。本次实测见 `reports/bottle-baseline.md`。

SubspaceAD 官方代码对照：

```powershell
conda run --no-capture-output -n detect python tools/compare_subspace_official.py
```

需要先完成 `configs/subspace_bottle.yaml`。此脚本调用保存于 `references/SubspaceAD` 的官方 `main.py`，调整本地权重加载、未使用的 saliency 存储，并将 NumPy 2.4 已移除的 `trapz` 映射到等价的 `trapezoid`。报告见 `reports/subspace-bottle-baseline.md`，并分别列出论文附录、当前官方代码和本库结果。

## 代码结构与参数约定

- `src/adkit/base.py`：定义检测器生命周期接口。
- `src/adkit/factory.py`：按名称构造或加载检测器，延迟导入算法依赖。
- `src/adkit/anomalydino.py`、`subspacead.py`：显式构造参数、特征提取及算法状态。
- `src/adkit/data.py`、`metrics.py`：数据处理与评估工具。
- `src/adkit/run.py`：解析 YAML 并组织运行，命令为 `adkit --config configs/bottle.yaml`。

模型参数通过 `__init__` 传入，实例使用独立属性，不再提供 `self.config`。
YAML 是命令行实验入口；`samples`、`ReferenceBatches` 继续接收数据配置。
检查点加载以保存的算法参数为准，仅覆盖 `device` 和 `weights`；修改算法参数应重新构造并建库。
旧包导入和旧检查点需要迁移或重新生成。
