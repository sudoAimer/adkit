# SuperADD

SuperADD 通过冻结 DINOv3 的多层 token 特征、重叠分块拼接、逐层距离采样参考库和欧氏最近邻检测异常，无需反向传播。实现适配自 [anomalib SuperADD](https://github.com/open-edge-platform/anomalib/tree/3759687e76395c4d6d239552d3bf6d72e003da78/src/anomalib/models/image/super_add)，不依赖 anomalib / Lightning。

## 安装与本地权重

```bash
uv pip install -e '.[superadd]' --torch-backend=auto
# 网页工作台及全部算法
uv pip install -e '.[web,subspacead,superadd]' --torch-backend=auto
```

需要 timm >= 1.0.20。准备与 `vit_small_patch16_dinov3` 完全匹配的 **timm 格式** DINOv3 S/16 权重，保存到 `weights/dinov3_vits16/model.safetensors`，也可传入本地 `.pt` state dict。构造时使用 `pretrained=False` 并严格加载参数；不会自动下载权重。DINOv2、Transformers 或原始研究代码格式权重不能直接替代。权重需自行从授权来源准备，并遵循其许可。

默认选择较小的 S/16，适合先验证流程。若切换骨干，请同时设置匹配的 `encoder_name`、权重和 `layers`；上游代码默认使用 H+/16。该示例不是论文精度复现配置。

## 使用

```bash
adkit --config configs/superadd_bottle.yaml
adkit --config configs/superadd_predict.yaml
```

```python
from adkit import create_detector, load_detector
from adkit.data import ReferenceBatches, prepare, read_rgb

model = create_detector('superadd', weights='weights/dinov3_vits16/model.safetensors', device='cpu')
model.fit(ReferenceBatches(['normal.png'], {'image_size': 448}, patch_size=model.patch_size))
x = prepare(read_rgb('test.png'), 448, patch_size=model.patch_size).unsqueeze(0)
result = model.predict(x)  # CPU: pred_score [B], anomaly_map [B,1,H,W]
model.save('model.pt')
restored = load_detector('superadd', 'model.pt', device='cpu')
```

直接传张量时应使用 ImageNet mean/std 归一化。重复 `fit` 替换原参考库；检查点保存构造参数、逐层 CPU 参考库、元数据和权重 SHA-256，不保存骨干权重。加载时可覆盖设备或相同内容的权重路径。

网页工作台由注册表自动提供 SuperADD 选项。启动服务前设置 `ADKIT_SUPERADD_WEIGHTS`（见 `.env.example`），即可使用既有建库、推理、多模型比较及阈值界面。其他骨干可由管理员在 `backend/registry.py` 的 `ModelSpec.parameters` 配置。

## 参数与几何

| 参数 | 默认值 | 含义 |
|---|---|---|
| `layers` | `[3,5,8,10]`（S/16） | 从 0 开始的 block 索引，递增且不重复；特征不做 LayerNorm/L2 归一化 |
| `tile_size` | 448 | 对应上游 `patch_size`，即输入分块边长 |
| `patch_overlap` | 16 | 上游重叠布局参数，分块步长上限为 `tile_size - 2*patch_overlap` |
| `max_database_size` | 100000 | 每层最多保留的参考 token 数 |
| `subsampling_iterations` | 100 | 距离密度随机子集采样轮数 |
| `gaussian_blur_sigma` | 4.0 | 输出像素空间高斯平滑；0 关闭 |
| `score_quantile` | 0.001 | 最高分像素的平均比例，范围 `(0,1]` |
| `query_chunk` / `bank_chunk` | 1024 / 16384 | 欧氏距离矩阵两轴分块，限制显存 |

公共属性 `model.patch_size` 为骨干 token 步长 **16**，用于 adkit 的输入对齐，不是 `tile_size`。`tile_size` 和 `patch_overlap` 必须为步长的整数倍，且前者大于后者的两倍。输入不足一个分块时复制边缘补齐；这部分上下文不进入参考库和分数。非整除输入输出仍保持原始尺寸。

按各 token 在分块中的中心位置决定拼接归属。每层计算原始特征的 1-NN 欧氏距离，除以通道数，再双线性插值、多层平均、高斯平滑，最终对最高分像素求均值。参考库保存在 CPU，分块搬到计算设备；骨干按单个空间分块执行，避免一次展开所有分块。

## 与上游的边界

- 修复全零/重复特征或密度饱和时采样循环无法终止的问题；退化时随机裁减。采样轮数大于目标库容量时也严格限制容量。
- 对小图增加边缘填充支持；平滑核超过图像尺寸时使用复制边界。
- 使用 adkit 现有阈值流程，不包含 anomalib `SuperADDPostProcessor` 的正常验证集百分位阈值校准；分数和热力图为原始值。不要把可视化颜色范围当作判定阈值。
- 不包含论文额外的多阈值形态学闭运算、闭合区域填充和评估图 4 倍降采样，也未默认加入亮度增强。高分辨率实验可改用 `tile_size: 640`、`patch_overlap: 128` 并保留输入宽高比，但显存和 CPU 参考特征存储会增加。

## 验证

`tests/test_superadd.py` 覆盖分块欧氏检索与稠密矩阵比较、固定上游版本的几何一致性、退化采样、跨批次拼接、分数公式、平滑、微型真实 timm DINOv3 离线保存加载及 CLI 16 像素对齐。运行：

```bash
python -m pytest -q
```

CI 安装 CPU PyTorch 与可选依赖，测试不下载预训练权重。生产权重精度、真实数据指标和 CUDA 性能需在准备好资源后另行验证。
