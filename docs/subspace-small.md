# SubspaceAD 切换 DINOv2 Small

默认使用 `vit_small_patch14_dinov2.lvd142m`（无 registers）和 `weights/dinov2_vits14/model.safetensors`，与 AnomalyDINO 共用同一份本地权重。两者共用输入 token 和官方位置插值逻辑，但 SubspaceAD 平均中间层、建 PCA 子空间；AnomalyDINO 使用最终归一化特征及近邻参考库。

## 论文是否讨论精度损失？

有。[论文 arXiv v1 第 4.7 节、图 5](https://arxiv.org/html/2602.23013v1#S4.F5) 比较了 DINOv2 S/B/L/G。下面是从图线估读的 **图像 AUROC（%）**，不是作者发布的精确数据表：

| 数据集 | shot | Giant | Small | 约下降（百分点） |
|---|---|---|---|---|
| MVTec-AD | 1 | 98.0 | 95.0 | 3.0 |
| MVTec-AD | 4 | 98.4 | 96.0 | 2.4 |
| VisA | 1 | 93.3 | 85.8 | 7.5 |
| VisA | 4 | 94.5 | 86.7 | 7.8 |

降幅与数据集有关，不能推断自有数据或 bottle 单类别的固定损失，也不能把 AUROC 下降等同于实际漏检率上升。论文附录 D 表 9 在 H100、448px、1-shot 下报告 Giant 127ms/图、Small 36ms/图（约 3.5 倍快）；实际硬件和本实现需另测。

[作者的骨干消融脚本](https://github.com/CLendering/SubspaceAD/blob/ef56d5c8ab2f1feb7dda1c93b25cc3f73f0960d7/scripts/backbone_ablation.sh) 使用 `facebook/dinov2-small`、448px、`layers=-4,-5`。adkit 默认 Small 示例保留原 672×672 与 few-shot 增强参数，所以这里的论文读图差值不是本仓库 Small 配置的实测结果。

## 使用及迁移

```bash
adkit --config configs/subspace_bottle.yaml
adkit --config configs/subspace_predict.yaml
```

Small 使用核心 timm 依赖，无需 Transformers。工作台启动环境的 `ADKIT_SUBSPACEAD_WEIGHTS` 默认与 AnomalyDINO 指向同一个文件；若此前显式配置了 Giant 路径，请更新环境变量。

- `backend=auto`：文件选择 timm；目录选择 Transformers。检查点持久化实际后端，路径类型变化不会悄悄更换旧模型。
- Small 默认 `layers=[-4,-5]`。按 13 个隐藏状态计数（包含 embedding），选取 state 9/8，即第 9/8 个 block 的输出；不会误用 Giant 的 `-18`，也不会额外应用最终 LayerNorm。
- 小图和非整除输入仍通过 patch padding 保留像素；PCA、评分和平滑流程保留。
- 更换骨干后重新 fit，并重新校准阈值；不兼容 Giant 的 PCA 状态。
- 新 Small 示例输出到 `outputs/subspace_small_bottle`，避免覆盖旧 Giant 结果。
- 原 Giant 配置保存为 `configs/subspace_giant_bottle.yaml` / `subspace_giant_predict.yaml`，继续使用原 Transformers 目录、层选择、672px 预处理及结果目录。官方对照脚本已指向 Giant 配置。
- 旧 v2 检查点不含 backend 字段时按 Transformers 还原。权重与预处理配置校验继续生效。

验证涵盖同一权重的实际微型 timm DINOv2 特征逐层对比、两种位置插值、Small 保存加载与 refit、旧 Transformers 检查点及工作台默认路径。未下载生产权重或数据，不提供本地精度/速度复现结论。
