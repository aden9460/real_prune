# 基于真实图片的 Head 尖锐度分析说明

本说明文档介绍如何在 VAR-d16 模型上，使用真实 ImageNet 图片、完整序列的注意力权重（softmax 后）来分析 16 层每个 head 的注意力分布模式（尖锐/平缓），并生成热力图与均值/方差统计。

---

## 1. 分析目标
- 对 d16 的每一层、每个 head，计算注意力集中度三指标：
  - Shannon Entropy（越低越尖锐）
  - Effective Span@0.9（越小越尖锐）
  - Gini 系数（越高越尖锐）
- 使用真实图片（ImageNet 验证集）并采用 teacher forcing，从而在完整序列（约 L=680）上得到注意力权重。
- 每层基于上述三指标做 KMeans 聚类（k=2），将 head 标注为“尖锐(1)”或“平缓(0)”。
- 输出热力图（entropy/span/gini/labels）与每层统计（均值与方差）。

---

## 2. 代码入口与位置
- 主脚本：`analyze_head_sharpness_from_images.py`
  - 依赖：`verify_scale_concentration.py` 中的模型加载、数据准备、Attention 提取与指标计算函数。
- 输出目录：默认 `head_sharpness_images/`

---

## 3. 前置依赖
- 模型权重（默认路径）：
  - VAE: `/home/project/daily/AR/model_zoo/vae_ch160v4096z32.pth`
  - VAR-d16: `/home/project/daily/AR/model_zoo/var_d16.pth`
- 数据集：ImageNet 根目录（包含 train/ 与 val/）：`/home/project/ImageNet-1K`

---

## 4. 安装 Python 依赖
用于聚类与绘图：

```bash
python -m pip install -U scipy scikit-learn matplotlib seaborn
```

说明：脚本已在绘图时强制使用非交互式后端（`Agg`），不会弹出窗口。

---

## 5. 运行示例
生成所有 16 层的指标、聚类标签与热力图（建议先用 32 张图片加速）：

```bash
python analyze_head_sharpness_from_images.py \
  --model_depth 16 \
  --num_samples 32 \
  --layers all \
  --imagenet_dir /home/project/ImageNet-1K \
  --plots
```

常用参数：
- `--num_samples`: 真实图片样本数（建议 20~50，更多更稳但更慢/占内存）。
- `--layers`: `all` 或逗号分隔的层索引（如 `0,2,5,7,11,12`）。
- `--vae_ckpt` / `--var_ckpt`: 覆盖默认模型权重路径。
- `--out_dir`: 输出目录（默认 `head_sharpness_images`）。

---

## 6. 输出说明
脚本会在 `head_sharpness_images/` 下生成：

- `metrics_per_layer.json`：每层每个 head 的原始指标与聚类标签，及矩阵形状。
- `layer_stats.csv`：每层指标均值与方差，以及“尖锐” head 数量。
- 热力图（启用 `--plots` 时生成）：
  - `entropy_heatmap.png`：层×head 的熵（越低越尖锐）。
  - `span_heatmap.png`：层×head 的有效范围（越小越尖锐）。
  - `gini_heatmap.png`：层×head 的 Gini（越高越尖锐）。
  - `labels_heatmap.png`：层×head 的二值标签（红=尖锐1，蓝=平缓0）。

### 6.1 指标的均值/方差与直方图
- `metrics_means_vars.png`：
  - 上图：各层 entropy / span / gini 的均值曲线（用于观察随层深的整体集中趋势变化）。
  - 下图：各层 entropy / span / gini 的方差曲线（用于观察层内 head 的差异程度）。
- `metrics_means_vars.csv`：与上图对应的数据表。
- `entropy_hist_grid.png` / `span_hist_grid.png` / `gini_hist_grid.png`：
  - 分别给出 16 层（4×4 网格）的直方图。
  - 观察每层指标的分布形态（是否单峰/是否偏斜/是否有长尾）。

### 6.2 等权综合分数 S_w
- 定义：`S_w = mean( -z(entropy), -z(span), +z(gini) )`，采用全局 z 标准化（跨层可比）。
- `sw_heatmap.png`：层×head 的 S_w 热力图（红=更尖锐，蓝=更平缓；0 为全局均值）。
- `sw_means_vars.png`：各层 S_w 的均值与方差曲线（观察层级整体尖锐度与层内差异）。
- `sw_hist_grid.png`：各层 S_w 的直方图（4×4 网格）。

### 6.3 基于 KMeans 的均匀（逐层）比例剪枝计划（40% 示例）
- 由 `plan_pruning_kmeans_uniform.py` 生成：
  - `pruning_plan_40pct.json`：每层按 KMeans 两簇（平缓/尖锐）比例分配 40% 的删减配额；在各簇内按 S_w 从低到高优先剪除（更平缓先剪）。
  - `pruning_mask.csv`：层×head 的三态掩码（0=保留，1=剪除-平缓簇，2=剪除-尖锐簇）。
  - `pruning_stacked_bars.png`：每层堆叠柱图，显示从“平缓簇/尖锐簇”各剪了多少（虚线为每层配额）。
  - `pruning_mask_heatmap.png`：三色热力图直观展示被剪 head 的位置与所属簇。

---

## 7. 方法细节
- 注意力来源：使用 `AttentionMapExtractor` 在目标层强制 slow 路径，捕获 softmax 后的 `attention_weights`，并在 teacher forcing 模式下喂入真实图片编码的 tokens，以覆盖完整序列（first_l + 679 ≈ 680）。
- 指标计算：
  - Entropy：对每个 query 位置的分布 `p` 计算 `-Σ p*log(p)` 并对样本与 query 取均值。
  - Effective Span@0.9：降序累计到 0.9 的最小位置数，越小越集中。
  - Gini：用标准公式衡量不均匀程度，越高越集中。
- 聚类：
  - 每层对 `[entropy, span, gini]` 标准化后做 `KMeans(k=2)`（`n_init=20, random_state=42`）。
  - 用 `S = -entropy - span + gini` 对两个聚类中心打分，得分高者标为“尖锐(1)”，另一簇为“平缓(0)”。
  - 若环境未装 scikit-learn，脚本会退化到 numpy 简易 k-means，算法逻辑相同（建议安装 sklearn 以更稳）。

---

## 8. 性能与内存建议
- 完整序列 L≈680，单样本注意力矩阵大小约 `H * L * L ≈ 16 * 680 * 680 ≈ 7.4M` 浮点数（约 30MB）。
- `num_samples=32` 时，累计内存约 ~1GB（CPU 上），再加上中间张量与指标计算开销，整体在数分钟到十余分钟量级。
- 如需加速：可先降低 `--num_samples`，或仅分析关键层（如 `--layers 0,1,2,7,11,12`）。

---

## 9. 常见问题
1) ImportError（如 `sklearn` 或 `matplotlib` 未安装）
   - 按第 4 节安装依赖；绘图需要 `matplotlib` 与 `seaborn`。

2) 显存/内存不足
   - 减少 `--num_samples`，或只分析部分层；也可分批运行，拼接结果。

3) 运行很慢
   - 指标计算是对所有 query 的双循环，计算量确实较大。可以先用 16~32 张图做预览，再提高样本数做最终结果。

---

## 10. 结果解读建议
- 若某层“尖锐” head 占比明显更高，说明该层注意力普遍更集中；反之则更平缓。
- 可结合 `entropy_heatmap.png` 和 `labels_heatmap.png` 快速定位极端 head（如极低熵/极高 Gini）。
- 对后续剪枝：
  - 可在“尖锐”组与“平缓”组内分别执行组内补偿或不同阈值策略；
  - 也可与 `scale_mul`、Hessian 相关性、head 输出相关性等指标联动筛选。

---

## 11. 变更记录
- 2025-11-12：新增 `analyze_head_sharpness_from_images.py`；绘图改用非交互式后端 `Agg`；编写本说明。

---

## 12. 按指定 Head 剪枝 + SlimGPT 全局补偿（实现说明）

本节说明如何将“已选定的要剪 head 列表”（例如来自 KMeans 比例均匀剪枝 40% 的计划）投入实际剪枝，并确保对剩余 head 执行 SlimGPT 的全局补偿。该流程适用于 `attn.proj`；`ffn.fc2` 保持原先按稀疏率的结构化剪枝。

### 12.1 启用与参数

运行示例：

```bash
python model_slimming_basic.py \
  --model_depth 16 \
  --use_images --imagenet_dir /home/project/ImageNet-1K \
  --num_samples 256 \
  --sparsity 0.4 \
  --use_selected_heads \
  --selected_heads_json head_sharpness_images/pruning_plan_40pct.json
```

- `--use_selected_heads`：启用“按外部计划剪 head”。
- `--selected_heads_json`：计划文件，建议使用 `plan_pruning_kmeans_uniform.py` 生成的 `pruning_plan_40pct.json`。
- 行为：
  - 对 `attn.proj` 严格按计划删除指定 heads，对剩余列做 SlimGPT 全局补偿；
  - 对 `ffn.fc2` 仍按 `--sparsity` 执行结构化剪枝（“保留”现有策略）。

### 12.2 计划文件格式（摘录）

`head_sharpness_images/pruning_plan_40pct.json` 示例结构：

```json
{
  "rate": 0.4,
  "layers": [
    {
      "layer": 0,
      "num_heads": 16,
      "n_smooth": 8,
      "n_sharp": 8,
      "prune_count": 6,
      "prune_smooth": [ ... ],
      "prune_sharp":  [ ... ],
      "pruned": [1, 3, 4, 5, 8, 12]
    },
    ...
  ]
}
```

- 读取逻辑：每层使用 `layers[i].pruned` 作为要剪的 head 列表。
- 映射规则：`head_idx -> 列索引范围 [head_idx * head_dim, (head_idx+1) * head_dim)`。

### 12.3 SlimGPT 如何“按指定列”做全局补偿

在 `slim_utils/slimgpt.py` 中新增 `struct_prune_with_indices(prune_columns, percdamp, headsize=1)`：

1) 基于已采集的输入激活构建 Hessian H（同原始 SlimGPT 流程）。
2) 将要剪列重排到前段：`column_sort_idx = [pruned..., kept...]`；对 Hinv 做相同的重排。
3) 计算 `Hinv_chol = cholesky(Hinv)[:cnt]`，其中 `cnt = len(prune_columns)`。
4) 局部消元（局部补偿）：
   - 令 `W1 = W[:, :cnt]`（重排后被剪的列块），逐列做 `Err1[:, i] = W1[:, i] / Hinv_chol[i,i]` 并右乘上三角部分，确保稳定；
5) 全局补偿：
   - 对剩余列 `W[:, cnt:] -= Err1 @ Hinv_chol[:, cnt:]`（一次性对剩余列做全局更新）；
6) 将前段列置零（表示剪除），再按原顺序还原列布局，回写权重张量。

这与原 SlimGPT 的设计原理一致：
- 仍基于 Hessian 的二阶近似和 Cholesky 分解进行稳定求解；
- 差别仅在于“列集合由外部指定”，而非由误差指标排序循环选择；
- 局部与全局补偿的数学形式、阻尼 `percdamp` 的使用、`no_compensate` 的开关逻辑均保持一致。

注意：为确保矩阵维度一致，全局补偿使用 `Hinv_chol`（形状 `cnt x columns`），而不是 `Hinv` 的切片。

### 12.4 Torch-Pruning 的联动调整（attn.proj 专用）

当 `attn.proj` 指定列剪枝完成后：
- 更新 Bias：`q_bias / zero_k_bias / v_bias` 仅保留与剩余列对应的下标；
- 更新 `attn.num_heads`：改为实际保留的 head 数量；
- 更新 `scale_mul_1H11`：仅保留未剪 head 的缩放参数，数值不变；
- 剪 `attn.mat_qkv` 的输出通道：对 Q/K/V 三块做相同下标的删除，偏移量使用剪枝前的 `old_in_features` 作为块宽（保证三块对齐）。

这些与既有的 Torch-Pruning 集成逻辑保持一致，区别只是“被剪的列集合”来源不同（计划文件 vs 自动选择）。

### 12.5 变量命名澄清（熵对“注意力权重”，非对 Q 向量）

在集中度指标计算中：
- 熵是对 `attn_weights[b, head, qi, :]`（沿 key 维度的 softmax 概率）逐 query 位置 `qi` 的分布做 `-Σ p log p`，并在样本与 query 维上求均值；
- 代码里的 `q` 表示“query 索引 qi”，不是 Query 向量 Q。为避免歧义，后续可在变量命名上改为 `qi`（不影响行为）。

### 12.6 与原始 SlimGPT 流程的差异与一致性（重要）

- 原始 `struct_prune`（迭代式）
  - 选择阶段：基于 `H^{-1}` 的误差估计对列排序，按批次取最差者；
  - 补偿阶段：对当前批次做“局部消元 + 一次全局补偿”；
  - H 更新：每批次后“模拟移除”当前批次（对应行列清零、对角置 1），再进入下一批；
  - 特点：多批次、多次补偿，选择与数值路径取决于排序与 batch 大小。

- 新增 `struct_prune_with_indices`（一次性块剪）
  - 选择阶段：由外部计划直接给出待剪集合 E（由列索引组成，可来自多个 head）；
  - 补偿阶段：一次性将 E 重排至左端，按块做“局部消元 + 一次全局补偿”（使用 `Hinv_chol` 保证维度与稳定性）；
  - H 使用：使用初始 H 即可完成块解（OBS 的块剪公式不要求中间更新 H）；
  - 特点：单批次、一次性补偿，路径更短，严格对齐外部计划。

- 一致性与差异
  - 两者都遵循 OBS/SlimGPT 的二阶近似最小扰动原则；
  - 若改为“分多批次迭代剪”，每批次都补偿并更新 H，则与“一次性块剪”存在轻微数值差（因每步 W 已改变）；
  - 采用初始 H 做一次性块解，满足块OBS推导；只有拆成多批时，才需要在批间更新 H 表征“已移除”的影响。

- 本实现的取舍
  - `attn.proj`：采用“一次性块剪”（按计划给定的列集合 E 一次处理），严格执行计划；
  - `ffn.fc2`：保留原有“按稀疏率迭代剪”的流程（多批次、批间更新 H）；
  - 形状一致性：`mat_qkv` 的 Q/K/V 三块出通道剪枝偏移使用剪前的 `old_in_features`，并更新 `num_heads`、`scale_mul_1H11` 与 bias 以匹配保留列；
  - 数值稳定：使用 `Hinv_chol` 参与全局补偿；`percdamp` 阻尼与 `no_compensate` 开关保持与原实现一致。
