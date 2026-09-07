# 图像检测工作台

[返回首页](../README.md) · [uv 环境配置](../docs/environment.md)

Vue 3 + FastAPI 单人工作台，支持上传正常图片建库、批量检测、同批数据双模型对比、查看分数与异常图，以及调整分数和缺陷面积阈值。任务、图片和参考库保存在后端机器，刷新页面后可以继续使用。

## 安装与启动

先按 [环境配置](../docs/environment.md) 创建并激活 `.venv`。在仓库根目录安装工作台依赖；需要 SubspaceAD 时一并选择该功能：

```bash
uv pip install -e ".[web,subspacead]" --torch-backend=auto
```

仅使用 AnomalyDINO 时安装 `.[web]` 即可。CPU 环境将 `auto` 改为 `cpu`。按 [权重说明](../README.md#准备权重) 准备所选算法的本地文件。

准备 Node.js 22.12+ 和 npm，构建前端：

```bash
npm --prefix frontend ci
npm --prefix frontend run build
```

后端默认使用 CUDA。仅使用 CPU 时，启动前设置环境变量：

```bash
# Linux / macOS
export ADKIT_DEVICE=cpu
```

```powershell
# Windows PowerShell
$env:ADKIT_DEVICE = "cpu"
```

启动服务：

```bash
python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

打开 [工作台](http://127.0.0.1:8000)。FastAPI 同时提供页面与接口；模型建库和推理在后端执行。

## 页面使用

1. **创建任务**：填写名称并选择单模型或双模型对比。输入尺寸可保留原图、指定短边或指定高宽，创建后固定。两个模型共享相同图片、尺寸与固定角度增强。现有任务可点击“用当前图片创建双模型对比”，复制图片及标注到独立任务。
2. **上传正常样本**：拖拽或多选同类物品的无缺陷图片，支持预览与删除。
3. **建立参考库**：点击“开始建库”，等待状态变为完成。建库期间显示当前阶段。
4. **批量检测**：上传待测图片并开始检测，按已处理图片数量显示进度；单图失败会显示错误原因。
5. **查看结果**：切换原图、热力图和叠加图，查看原始异常分数，下载当前视图。
6. **标注与统计**：在单图详情选择实际正常／缺陷。表格每个模型一行，显示当前阈值下误报（FP）、漏检（FN）、检出（TP）和正确正常（TN）。未标注的已判定图片、未判定、失败及待测图片单独显示，不计入误报／漏检。检出率为 TP/(TP+FN)，误报率为 FP/(FP+TN)，无相应样本时显示 —。
7. **调整阈值**：输入数值或拖动滑条预览判定，长滑条下方点击保存按钮保留当前模型的设置；清空后保存恢复为“未判定”。

尺寸阈值为 0 时，`pred_score >= threshold` 判为缺陷。尺寸阈值大于 0 时，还要求原图尺度异常图中 `pixel_score >= threshold` 的最大 8 连通区域面积达到尺寸阈值（px²）。等于阈值时包含在内，分散区域不累加。默认不设置分数阈值，保持未判定。分数不是概率，应结合自己的正常与缺陷样本设定阈值，不同模型和任务不共用分数尺度。

阈值仅影响图像级判定，不改变热力图、不生成缺陷框。热力图按单张图片拉伸颜色，用于展示异常分布。

## 图片与任务规则

- 每次上传 1–20 张；单张至多 20 MB、1200 万像素，不限制长宽比。
- 正常图片和待测图片各最多 200 张；支持 JPG、PNG、WebP、BMP、TIFF。
- 增删正常图片后，旧参考库与结果失效，需要重新建库；增删待测图片后需要重新检测。
- 作业排队或运行时不能修改、删除对应任务的样本。
- 删除任务会同时删除其图片、参考库和检测结果。

工作台使用全部上传的正常样本。开启旋转增强时使用 45° 固定角度增强；官方少样本实验的配置见 [评估文档](../docs/evaluation.md)。

## 服务配置

通过启动终端设置环境变量，示例见 [.env.example](../.env.example)。服务不自动读取 `.env` 文件。默认目录位于仓库下；自定义相对路径基于启动命令所在目录。

| 环境变量 | 默认值 / 用途 |
|---|---|
| `ADKIT_DEVICE` | `cuda`，可设为 `cpu` |
| `ADKIT_ANOMALYDINO_WEIGHTS` | `weights/dinov2_vits14/model.safetensors` |
| `ADKIT_SUBSPACEAD_WEIGHTS` | `weights/dinov2_with_registers_giant` |
| `ADKIT_DATA_DIR` | `web-data`，保存任务、图片、检查点与结果 |
| `ADKIT_FRONTEND_DIST` | `frontend/dist`，前端构建目录 |

使用 **`--workers 1`**。模型作业串行执行，最多排队及运行 16 项任务；不要用多个后端实例共享同一数据目录。服务正常关闭时等待作业结束，异常退出后未完成作业在下次启动时标记失败，可重新提交。运行模型任务时不要开启 `--reload`。

服务默认只监听本机，不包含登录鉴权。需要远程访问时，应通过可信网络或带认证的反向代理提供访问。

## 前端开发与 API

后端启动后，在另一个终端运行：

```bash
npm --prefix frontend ci
npm --prefix frontend run dev
```

打开 Vite 输出的地址。开发时 `/api` 代理到 `127.0.0.1:8000`；部署时使用构建产物，由 FastAPI 同源托管。

接口文档位于 [API 文档](http://127.0.0.1:8000/docs)。后端按路由、参数校验、持久化、作业调度和算法执行分为 `app.py`、`schemas.py`、`store.py`、`service.py`、`engine.py`。

## 尺寸与旧任务兼容

新任务高宽分别向上补齐到 14 的倍数，如 12×12 → 14×14、225×319 → 238×322。
右侧和底部复制边缘像素补齐，不裁去原图；异常图移除补边后映射回原图。
混合尺寸参考图片按兼容尺寸组批。原图模式的内存需求随像素数增长，可选择较小短边控制成本。

旧任务保留历史预处理与检查点，分数阈值仍可使用。旧结果缺少原始异常图时，非零尺寸阈值显示未判定并提示重新检测。
双模型任务顺序加载模型，分别保存检查点和结果；正常样本变化会使两套参考库同时失效。
新旧预处理结果不应直接与历史官方报告混比。

新增接口：`POST /api/tasks/{id}/comparison` 复用任务图片；
`PATCH /api/tasks/{id}/images/test/{image_id}/label` 设置 `label: normal|defect|null`；
`POST /api/tasks/{id}/assessment` 预览统计；`PATCH /api/tasks/{id}/threshold` 保存阈值。
后两者接受 `algorithm`、`threshold`、`area_threshold`，双模型任务必须指定模型。
区域计算使用保存的浮点异常图，不使用热力图颜色，不重新运行模型。
