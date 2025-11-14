# Scale_mul聚类假设验证工具

## 目标

验证核心假设："**scale_mul接近的heads关注的注意力模式高度相似**"

这个假设是聚类剪枝策略（策略1）的理论基础。

## 文件说明

- `verify_scale_mul_clustering.py` - 核心验证脚本
- `run_verification_quick.bash` - 快速验证脚本（单层，50样本）
- `run_verification_all.bash` - 全面验证脚本（多层，100样本）
- `SCALE_MUL_CLUSTERING_VERIFICATION_GUIDE.md` - 详细验证方法文档

## 快速开始

### Phase 1: 快速验证（推荐先运行）

```bash
# 验证单个高方差层（Layer 7）
bash run_verification_quick.bash
```

**预期时间**: 10-15分钟
**输出目录**: `./verification_results_d16_layer7/`

### Phase 2: 查看结果

```bash
# 查看验证报告
cat verification_results_d16_layer7/verification_report.json

# 查看可视化
# - method1_scatter_layer7.png : scale_mul差异 vs 注意力相似度
# - method3_heatmap_layer7.png : 相似度热力图（聚类边界）
```

### Phase 3: 全面验证（可选）

```bash
# 验证多个层（高方差+低方差）
bash run_verification_all.bash
```

**预期时间**: 1-2小时
**验证的层**: 5, 7, 11, 12（高方差） + 0, 2（低方差）

## 手动运行

```bash
# 自定义参数
python verify_scale_mul_clustering.py \
    --model_depth 16 \
    --layer_idx 7 \
    --num_samples 50 \
    --threshold_factor 0.3 \
    --output_dir ./my_verification
```

**参数说明**:
- `--model_depth`: VAR模型深度（16/20/24/30）
- `--layer_idx`: 验证的层索引（推荐高方差层: 5, 7, 11, 12）
- `--num_samples`: 校准样本数（更多样本 = 更可靠）
- `--threshold_factor`: 聚类阈值系数（threshold = std × factor）
- `--output_dir`: 结果输出目录

## 验证方法

### 方法1: 直接注意力相似度

- 计算16×16 attention相似度矩阵
- 分析scale_mul差异 vs 注意力相似度的Pearson相关性
- **成功标准**: r < -0.3, p < 0.05

### 方法3: 聚类质量分析

- 基于scale_mul识别聚类
- 计算聚类内部 vs 间相似度
- 计算Silhouette score
- **成功标准**: Silhouette > 0.3, 内部相似度 > 外部相似度 × 1.5

## 结果解读

### ✅ 验证通过 (PASS)

```json
{
  "method1": {"pearson_r": -0.45, "p_value": 0.001, "status": "PASS"},
  "method3": {"silhouette_score": 0.42, "status": "PASS"},
  "overall": {
    "result": "PASS",
    "confidence": "HIGH",
    "recommendation": "Clustering pruning is highly recommended for this layer."
  }
}
```

**含义**: 假设成立，可以使用聚类剪枝策略
**下一步**: 应用策略1进行剪枝

### ⚠️ 部分通过 (WEAK_PASS)

```json
{
  "method1": {"pearson_r": -0.25, "p_value": 0.04, "status": "WEAK"},
  "method3": {"silhouette_score": 0.25, "status": "WEAK"},
  "overall": {
    "result": "WEAK_PASS",
    "confidence": "LOW",
    "recommendation": "Clustering pruning may work but needs careful validation."
  }
}
```

**含义**: 假设有一定支持，但不强
**下一步**: 谨慎使用策略1，或结合其他策略

### ❌ 验证失败 (FAIL)

```json
{
  "method1": {"pearson_r": 0.05, "p_value": 0.58, "status": "FAIL"},
  "method3": {"silhouette_score": 0.0, "status": "FAIL"},
  "overall": {
    "result": "FAIL",
    "confidence": "N/A",
    "recommendation": "Clustering pruning is NOT recommended. Use alternative strategies."
  }
}
```

**含义**: 假设不成立，聚类剪枝可能无效
**下一步**: 使用策略2（方差条件）或策略3（组合方法）

## 预期结果分析

根据文档理论预测：

### 高方差层（Layer 5, 7, 11, 12）

- **预测**: 假设更可能成立
- **理由**: 高方差 = heads高度分化 = 形成功能聚类
- **预期指标**: r: -0.35 ~ -0.45, Silhouette: 0.35 ~ 0.50

### 低方差层（Layer 0, 1, 2）

- **预测**: 假设可能较弱
- **理由**: 低方差 = heads统一 = 聚类不明显
- **预期指标**: r: -0.15 ~ -0.25, Silhouette: 0.15 ~ 0.30

## 关键发现示例

从测试运行（10个样本，Layer 7）来看：

```
Layer 7 scale_mul:
  Mean: 23.85
  Std:  6.90  ← 高方差层

Method 1: r = 0.05, p = 0.58 → FAIL
Method 3: Silhouette = 0.0 → FAIL

结论: 假设在Layer 7上不成立（至少对于10个样本）
```

**注意**: 这是初步测试结果，需要更多样本（50-100）才能得出最终结论！

## 故障排除

### 问题1: CUDA Out of Memory

**解决**: 减少`--num_samples`或在运行前清理GPU

```bash
nvidia-smi  # 检查GPU使用
# 减少样本数
python verify_scale_mul_clustering.py --num_samples 20 ...
```

### 问题2: 运行速度慢

**原因**: autoregressive生成需要时间
**解决**:
- 使用快速验证（20-30个样本）先确认趋势
- 每个样本约需10-15秒

### 问题3: 结果不一致

**原因**: 样本数太少，随机性大
**解决**: 增加`--num_samples`到50-100

## 下一步行动

### 如果验证通过
1. 应用聚类剪枝策略（策略1）
2. 在剪枝脚本中集成聚类识别逻辑
3. 进行消融实验验证FID影响

### 如果验证失败
1. 分析为什么失败（查看可视化图）
2. 考虑调整聚类阈值（`--threshold_factor`）
3. 使用备选策略：
   - 策略2: 方差条件剪枝
   - 策略3: 组合方法

## 参考文档

- `SCALE_MUL_CLUSTERING_VERIFICATION_GUIDE.md` - 详细验证方法论
- `PRUNING_CRITERIA_STRATEGIES.md` - 剪枝策略对比
- `D16_VS_D30_PRUNING_RATE_ANALYSIS.md` - 剪枝率分析

## 联系与支持

如有问题，请查看：
1. 验证指南文档中的"失败情况分析"章节
2. 生成的可视化图表
3. 完整的JSON报告

---

**版本**: v1.0
**最后更新**: 2025-11-11
**状态**: 已测试，可用
