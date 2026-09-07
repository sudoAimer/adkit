# adkit

基于正常样本的图像异常检测工具，支持 **AnomalyDINO** 和 **SubspaceAD**。使用预训练骨干提取正常特征并建立参考库，再对待测图片输出异常分数与异常图。

## 功能

- **Python API**：通过构造函数传入参数，使用 `fit / predict / save / load` 管理检测器。
- **命令行**：通过 YAML 配置批量建库、推理和评估。
- **网页工作台**：Vue 3 + FastAPI，支持任意尺寸输入、同批数据多模型对比、实际标签与误报／漏检／检出统计，以及分数／面积长滑条。
- **本地运行**：准备好模型权重后，建库与推理无需联网；`fit` 不执行反向传播。

## 安装

需要 Python 3.10+。推荐使用 [uv](https://docs.astral.sh/uv/getting-started/installation/) 创建独立环境，下面以 Python 3.11 为例。

```bash
git clone https://github.com/sudoAimer/adkit.git
cd adkit
uv venv --python 3.11
```

激活环境：

```bash
# Linux / macOS
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

安装核心库，并让 uv 根据当前机器选择 PyTorch 后端：

```bash
uv pip install -e . --torch-backend=auto
```

按需选择以下安装组合；仅使用 CPU 时将 `auto` 改为 `cpu`：

| 使用方式 | 安装命令 |
|---|---|
| Python / 命令行，AnomalyDINO | `uv pip install -e . --torch-backend=auto` |
| Python / 命令行，包含 SubspaceAD | `uv pip install -e ".[subspacead]" --torch-backend=auto` |
| 网页工作台，AnomalyDINO | `uv pip install -e ".[web]" --torch-backend=auto` |
| 网页工作台，包含两个算法 | `uv pip install -e ".[web,subspacead]" --torch-backend=auto` |

环境激活后直接使用 `python` 和 `adkit` 命令。CPU/CUDA 选择、设备配置和环境复现见 [uv 环境配置](docs/environment.md)。

## 准备权重

从对应模型页面下载文件，放到下列默认位置，或在调用时指定自己的路径。只需准备所选算法的权重。

| 算法 | 权重来源 | 默认位置与文件 |
|---|---|---|
| AnomalyDINO | [timm DINOv2 ViT-S/14](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) | `weights/dinov2_vits14/model.safetensors` |
| SubspaceAD | [DINOv2 with registers Giant](https://huggingface.co/facebook/dinov2-with-registers-giant) | `weights/dinov2_with_registers_giant/`，包含 `model.safetensors`、`config.json`、`preprocessor_config.json` |

权重和图片不随仓库提供，程序不会在建库或推理时自动下载权重。

## 快速开始

### Python

准备 `images/normal/` 中的正常图片，以及另一张待测图片 `images/test.png`：

```python
from adkit.anomalydino import AnomalyDinoDetector
from adkit.data import image_paths, prepare, read_rgb, reference_batches

# 按运行设备选择 "cpu" 或 "cuda"。
detector = AnomalyDinoDetector(
    weights="weights/dinov2_vits14/model.safetensors",
    device="cpu",
    num_neighbours=1,
)
normal_images = image_paths("images/normal")
detector.fit(reference_batches(normal_images, {"image_size": 448}))

# 待测图片使用与建库一致的预处理尺寸。
batch = prepare(read_rgb("images/test.png"), image_size=448).unsqueeze(0)
prediction = detector.predict(batch)
print(prediction["pred_score"].item())
detector.save("outputs/model.pt")
```

参数、SubspaceAD 用法及检查点加载见 [使用指南](docs/usage.md)。

### 命令行

复制 [普通图片目录配置](configs/folder.yaml)，将权重路径、正常图片目录、待测图片路径和设备改为自己的设置，再运行：

```bash
adkit --config configs/folder.yaml
```

### 网页工作台

安装 `web` 可选依赖，并准备 Node.js 22.12+ 与 npm。在仓库根目录执行：

```bash
npm --prefix frontend ci
npm --prefix frontend run build
python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

打开 [图像检测工作台](http://127.0.0.1:8000)，按“上传正常样本 → 建立参考库 → 上传待测图片”的流程操作。后端默认使用 CUDA，CPU 环境需先设置 `ADKIT_DEVICE=cpu`；设置方式见 [工作台说明](backend/README.md)。

## 结果说明

核心算法输出原始异常分数 `pred_score` 和异常图 `anomaly_map`；AnomalyDINO 另外输出 `patch_map`。分数越高表示异常程度越高，分数不是概率。

工作台提供原图、热力图、叠加图，支持动态多选模型建库、测试与网格对比，并允许调整分数和最小连通缺陷面积阈值，按实际标注统计误报、漏检与检出。尺寸阈值为 0 时，仅按图像分数判定。热力图颜色按单张图片拉伸，用于展示异常分布；图像阈值不改变热力图。当前不输出缺陷框。

## 文档

| 文档 | 内容 |
|---|---|
| [uv 环境配置](docs/environment.md) | 虚拟环境、PyTorch 后端、可选依赖与环境复现 |
| [使用指南](docs/usage.md) | Python API、YAML 配置、数据格式、输出与检查点 |
| [工作台说明](backend/README.md) | 页面使用、服务配置与启动 |
| [评估与复现](docs/evaluation.md) | 评估指标、官方对照步骤与历史报告 |
| [第三方来源](THIRD_PARTY.md) | 算法来源、参考代码版本与修改说明 |

## 许可证

本项目使用 [Apache-2.0](LICENSE) 许可证。第三方代码与模型权重遵循各自的许可证，来源见 [THIRD_PARTY.md](THIRD_PARTY.md) 和对应模型页面。
