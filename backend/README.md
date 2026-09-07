# adkit 图像检测工作台

Vue 3 + Vite 提供中文白蓝界面，FastAPI 调用真实的 adkit 建库和预测流程。服务为单人、单进程工作台；模型与图片保存在运行后端的机器上，刷新浏览器不会丢失任务。

## 启动

在仓库根目录，使用已安装适合本机硬件的 PyTorch 的 Python 环境：

```bash
python -m pip install -e '.[web]'
# 使用 SubspaceAD 时额外安装：
python -m pip install -e '.[subspacead]'

cd frontend
npm ci
npm run build
cd ..
python -m uvicorn backend.app:create_app --factory --host 127.0.0.1 --port 8000 --workers 1
```

打开 <http://127.0.0.1:8000>。默认使用 CUDA，且从仓库的 `weights/` 读取已准备好的骨干权重。页面提示资源未准备时，先准备对应权重，再启动作业。不需要同时准备两个算法的权重。

- AnomalyDINO：`weights/dinov2_vits14/model.safetensors`
- SubspaceAD：`weights/dinov2_with_registers_giant/`，包含 `model.safetensors`、`config.json` 和 `preprocessor_config.json`

仅安装前端或上传图片不会生成模型权重。在线 fit 指服务端正常样本建库，不执行反向传播。该流程需要可运行 PyTorch 的后端机器，不能在纯静态托管或浏览器中运行。

## 开发前端

启动上面的后端后，在另一个终端运行：

```bash
cd frontend
npm ci
npm run dev
```

打开终端显示的 Vite 地址，`/api` 自动代理到 `127.0.0.1:8000`。生产模式下先构建前端，再启动 FastAPI，由同一服务提供页面与接口，无需 CORS 配置。

## 参数

环境变量通过启动终端设置；`.env.example` 仅提供示例，不自动加载。未设置时使用上述默认路径。自定义相对路径相对于启动命令的当前目录解析。

| 环境变量 | 用途 |
|---|---|
| `ADKIT_DEVICE` | 特征计算设备，默认 `cuda`，可设为 `cpu` |
| `ADKIT_ANOMALYDINO_WEIGHTS` | AnomalyDINO 本地权重文件 |
| `ADKIT_SUBSPACEAD_WEIGHTS` | SubspaceAD 本地模型目录 |
| `ADKIT_DATA_DIR` | 任务、上传图片、检查点和结果目录，默认仓库下 `web-data` |
| `ADKIT_FRONTEND_DIST` | Vue 构建产物目录，默认仓库下 `frontend/dist` |

必须使用 `--workers 1`。所有 GPU 作业进入一个线程队列，每次只运行一个；最多允许 16 个未完成任务。不要启动多个后端实例共享同一数据目录。服务关闭时等待作业完成；非正常退出后的未完成任务在下次启动时标记失败，用户可重新执行。开发模型任务时也不要使用 `--reload`。

默认只监听本机，没有用户登录和鉴权。若用于另一台机器访问，需由部署者在可信网络或具备认证的反向代理后开放服务，不应把单人接口直接开放到公共互联网。

## 使用流程

1. 新建任务，输入名称并选择算法。高级设置可调整输入尺寸和固定角度旋转增强；创建后算法与预处理固定。
2. 上传同类物品的正常图片；支持拖拽、多选、缩略图和删除。
3. 点击“开始建库”，页面展示排队、加载模型、建库、完成或错误状态。建库没有可用的精确百分比，因此只显示真实阶段。
4. 上传待测图片并开始检测，页面按已处理图片数显示进度。单图失败保留错误信息，其他图片继续处理。
5. 查看原图、热力图和叠加图，下载当前视图。每个结果显示原始异常分数。
6. 手动输入或拖动图像阈值，结果立即预览；点击“保存阈值”持久化。清空后保存可以恢复“未判定”。

判定规则是 `pred_score >= threshold` 为“缺陷”，否则“正常”。默认阈值为空，不使用未经校准的固定值。异常分数不是概率，阈值不跨算法或任务通用。阈值只影响图像级结论，不是像素阈值，不改变热力图；热力图按单张图片拉伸颜色，不输出缺陷框。

每批上传 1–20 张，每张至多 20 MB、1200 万像素，长宽比不超过 4:1；每个任务的正常和待测分组分别至多 200 张。服务端校验图片并转为 RGB PNG，保留原文件名供界面展示。修改正常样本会清除旧参考库与结果，必须重新建库；修改待测样本会清除旧检测结果。页面任务处理中禁止修改或删除其样本。

此工作台使用全部上传正常图片。旋转增强为 45° 固定角度增强，不等同于 SubspaceAD 官方少样本随机旋转评估协议。

## 代码结构

| 文件 | 职责 |
|---|---|
| `app.py` | API、上传校验、结果文件白名单与静态页面托管 |
| `schemas.py` | 请求数据校验 |
| `store.py` | 带锁的原子 JSON 持久化与重启恢复 |
| `service.py` | 作业队列、状态更新与样本失效规则 |
| `engine.py` | 真实 adkit 建库、预测与异常图输出 |
| `../frontend/src/App.vue` | 任务、步骤、阈值与结果工作区 |
| `../frontend/src/components/ImageUpload.vue` | 统一图片上传与缩略图组件 |

主要接口：`GET/POST /api/tasks`、`GET/DELETE /api/tasks/{id}`、`POST /api/tasks/{id}/images/{normal|test}`、`DELETE /api/tasks/{id}/images/{normal|test}/{image_id}`、`POST /api/tasks/{id}/jobs/{fit|predict}`、`PATCH /api/tasks/{id}/threshold`。接口文档位于 `/docs`。

## 本次验证范围

```bash
python -m compileall -q backend src/adkit
npm --prefix frontend run build
```

云端仅检查 Python 语法和前端生产构建，不运行模型、不验证精度、不声称真实 GPU 流程已通过。本机准备权重后，可按上面的页面流程运行。
