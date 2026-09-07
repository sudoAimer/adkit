# uv 环境配置

[返回首页](../README.md)

推荐采用 **uv 创建 `.venv` → 通过 `uv pip` 安装依赖 → 激活后运行命令** 的流程。依赖声明来自 `pyproject.toml`，PyTorch 后端在安装时按运行机器选择。

## 创建环境

先按 [uv 安装指南](https://docs.astral.sh/uv/getting-started/installation/) 安装近期版本的 uv。在仓库根目录执行：

```bash
uv venv --python 3.11
```

项目声明支持 Python 3.10+，文档统一以 3.11 为例。创建环境后，按终端类型激活：

| 终端 | 命令 |
|---|---|
| Bash / Zsh | `source .venv/bin/activate` |
| Windows PowerShell | `.venv\Scripts\Activate.ps1` |
| Windows CMD | `.venv\Scripts\activate.bat` |

后续命令均在仓库根目录、已激活的 `.venv` 中执行。新开终端后需要重新激活。不要同时激活其他 Conda 或虚拟环境，以免依赖安装到不同位置。uv 的环境选择规则见 [官方说明](https://docs.astral.sh/uv/pip/environments/)。

## 选择 PyTorch 后端

### 在实际运行机器上安装

```bash
uv pip install -e . --torch-backend=auto
```

`auto` 根据当前机器的 GPU/驱动选择 PyTorch 索引；未检测到支持的 GPU 时使用 CPU 构建。也可以显式选择 CPU：

```bash
uv pip install -e . --torch-backend=cpu
```

`torch` 与 `torchvision` 应由同一次依赖解析选取匹配版本。后端选择行为及支持的取值见 [uv 的 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。

### 为其他机器或容器准备环境

`auto` 检测的是执行安装命令的机器。无 GPU 的构建机器不应据此决定目标 GPU 服务器的环境。应按目标驱动选择显式后端，例如目标环境支持 CUDA 12.8 时：

```bash
uv pip install -e ".[web,subspacead]" --torch-backend=cu128
```

这里的 `cu128` 是选择方式示例，不是项目对 CUDA 版本的统一要求。GPU 型号、驱动、操作系统及可用 wheel 必须与选择相匹配；安装依赖不会安装或升级显卡驱动。

改变 CPU/CUDA 后端时，建议在新的虚拟环境中重新安装，避免复用旧环境中已满足版本约束的 PyTorch 构建。

## 选择功能依赖

将下面的 `auto` 替换为已选定的 `cpu` 或 CUDA 后端，所有追加安装使用同一选择：

| 功能 | 命令 |
|---|---|
| 核心算法与命令行 | `uv pip install -e . --torch-backend=auto` |
| 包含 SubspaceAD | `uv pip install -e ".[subspacead]" --torch-backend=auto` |
| 网页工作台 | `uv pip install -e ".[web]" --torch-backend=auto` |
| 工作台与两个算法 | `uv pip install -e ".[web,subspacead]" --torch-backend=auto` |
| 测试与官方对照 | `uv pip install -e ".[test,reference,subspacead]" --torch-backend=auto` |

uv 会安装所选功能需要的依赖，无需默认使用 `--no-deps`。普通使用不需要安装测试或官方对照依赖。前端依赖仍由 npm 管理，使用 `npm ci` 按 `frontend/package-lock.json` 安装。

## 选择运行设备

PyTorch 构建与 adkit 的运行设备需要分别配置：

| 入口 | 设置方式 |
|---|---|
| Python | 构造函数的 `device="cpu"` 或 `device="cuda"` |
| YAML | `model.device: cpu` 或 `model.device: cuda` |
| 工作台 | 环境变量 `ADKIT_DEVICE=cpu` 或 `ADKIT_DEVICE=cuda` |

SubspaceAD 可通过 `pca_device` 单独指定 PCA 计算设备，默认跟随 `device`。使用 CPU 环境时，将配置中显式写出的 `pca_device: cuda` 一并改为 `cpu`。其他加速设备不在本文的配置范围内。

可按需检查当前解释器和 CUDA 可用状态；这只检查环境，不代表模型精度验证：

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available())"
```

## 日常运行与环境复现

激活 `.venv` 后直接运行 `python`、`adkit` 等已安装的命令。例如：

```bash
adkit --config configs/folder.yaml
```

本流程使用 `uv pip`，不生成项目级 `uv.lock`。不要直接混用默认 `uv sync` 或 `uv run` 来维护此环境：项目级同步需要另外配置 PyTorch 索引和可选依赖，可能改变已经选择的环境。相关区别见 [uv 的 PyTorch 指南](https://docs.astral.sh/uv/guides/integration/pytorch/)。

需要固定某个部署目标的依赖时，可针对该平台、Python 版本及后端生成独立清单。例如在目标 CPU 平台上：

```bash
uv pip compile pyproject.toml --extra web --extra subspacead --python-version 3.11 --torch-backend=cpu -o requirements-cpu.txt
uv pip sync requirements-cpu.txt --torch-backend=cpu
uv pip install -e . --no-deps
```

这里最后一步仅安装本仓库本身，因为依赖已由同步清单安装。清单应随目标平台和后端分别维护；它不锁定模型权重、数据或系统驱动。需要复现实验时，另行记录仓库提交、权重版本与数据配置。[uv 依赖锁定说明](https://docs.astral.sh/uv/pip/compile/)

仓库中的 `requirements-tested.txt` 是历史实验环境记录，入口见 [评估与复现](evaluation.md)。
