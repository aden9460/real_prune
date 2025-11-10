# VAR Training-Free Pruning 调查计划

> **问题**：VAR 模型剪枝后 FID=178，但训练 1 epoch 后恢复到 FID=6
>
> **已修复**：scale_mul_1H11 现在保留原始学习值而非重置
>
> **待调查**：找出剩余性能差距的根本原因

---

## 当前状态

### 已完成的修复
- ✅ **scale_mul_1H11 保留修复**（2024-11-09）
  - 位置：`model_slimming_basic.py:421-442`
  - 改动：从剪枝索引计算保留的 head，提取对应的 scale_mul 值
  - 预期改善：保持注意力粒度的多样性

### 待测试
- [ ] 运行修复后的剪枝代码，测量 FID
- [ ] 对比修复前后的 FID 差异

---

## 调查假设（按优先级排序）

### 假设 1：离散 Token 采样的脆弱性 ⭐⭐⭐

**理论依据**：
- VAR 通过 argmax 从 logits 中选择 codebook 索引（0-4095）
- 即使 logits 数值差异很小，argmax 结果可能完全不同
- LLM 使用连续的 log-likelihood 评估，对 logits 扰动更鲁棒

**数学表示**：
```python
# 剪枝前某个 token 的 logits：
logits_original = [2.1, 1.9, 1.8, 1.7, ...]
idx_original = argmax(logits_original) = 0  # 选择 index 0

# 剪枝后（即使数值接近）：
logits_pruned = [1.8, 2.0, 1.9, 1.6, ...]
idx_pruned = argmax(logits_pruned) = 1  # 选择 index 1 ✗

# 结果：680 个 token 中如果 30% 选错，FID 暴涨
```

**验证实验**：

1. **Logits Top-1 Accuracy 测试**
   ```python
   # 对比剪枝前后模型在相同输入上的 argmax 结果
   with torch.no_grad():
       logits_original = model_original(calibration_data)
       logits_pruned = model_pruned(calibration_data)

       idx_original = logits_original.argmax(dim=-1)
       idx_pruned = logits_pruned.argmax(dim=-1)

       top1_acc = (idx_original == idx_pruned).float().mean()
       print(f"Top-1 token match rate: {top1_acc:.2%}")

   # 预期：如果是离散采样问题，top1_acc 会很低（如 60-70%）
   ```

2. **KL 散度测试**
   ```python
   # 测量 logits 分布的相似度
   kl_div = F.kl_div(
       F.log_softmax(logits_pruned, dim=-1),
       F.softmax(logits_original, dim=-1),
       reduction='batchmean'
   )
   print(f"KL Divergence: {kl_div:.4f}")

   # 预期：KL 散度可能不大，但 argmax 结果差异大
   ```

3. **Temperature 采样测试**
   ```python
   # 在推理时使用 temperature 平滑分布
   # 修改 var.py:175 的采样逻辑

   # 原始：
   idx_Bl = logits_BlV.argmax(dim=-1)

   # 测试不同 temperature：
   for temp in [0.5, 0.7, 0.9, 1.0, 1.2]:
       logits_scaled = logits_BlV / temp
       idx_Bl = logits_scaled.argmax(dim=-1)
       # 生成图像并计算 FID

   # 预期：较高的 temperature 可能改善 FID
   ```

**实施优先级**：⭐⭐⭐ 最高
**预期时间**：2-3 小时
**成功标准**：找到 top-1 accuracy 与 FID 的相关性

---

### 假设 2：AdaLN Gamma 数值分布不匹配 ⭐⭐

**理论依据**：
- AdaLN 的 gamma1/gamma2 是为剪枝前的注意力/FFN 输出学习的
- 剪枝后 proj 的权重矩阵变化（1024×1024 → 1024×960）
- 虽然输出维度仍是 1024，但数值分布（幅度、方差）可能不同
- gamma 不再是最优的缩放因子

**验证实验**：

1. **激活统计量对比**
   ```python
   # 收集剪枝前后的激活统计
   stats_original = {}
   stats_pruned = {}

   def hook_fn(name):
       def hook(module, input, output):
           stats[name] = {
               'mean': output.mean().item(),
               'std': output.std().item(),
               'min': output.min().item(),
               'max': output.max().item(),
           }
       return hook

   # 对比每层的 attn 和 ffn 输出统计
   # 查看哪些层变化最大
   ```

2. **Gamma 值分析**
   ```python
   # 打印每层的 gamma1, gamma2 分布
   for i, block in enumerate(model.blocks):
       cond_BD = model.class_emb(torch.tensor([0]))
       if hasattr(block, 'ada_lin'):
           gammas = block.ada_lin(cond_BD)
           gamma1, gamma2 = gammas[:, :, 0], gammas[:, :, 1]
           print(f"Layer {i}: gamma1={gamma1.mean():.3f}, gamma2={gamma2.mean():.3f}")
   ```

3. **缩放因子调整实验**
   ```python
   # 临时缩放 gamma 生成器的权重
   scale_ratio = 0.9  # 960/1024 ≈ 0.9375

   for block in model.blocks:
       if hasattr(block, 'ada_lin'):
           # 调整 gamma1 部分的权重
           C = block.C
           block.ada_lin[1].weight.data[:C] *= scale_ratio

   # 测试 FID 是否改善
   ```

**实施优先级**：⭐⭐ 中等
**预期时间**：3-4 小时
**成功标准**：发现激活统计显著变化的层

---

### 假设 3：SlimGPT 误差补偿在 VAR 上效果差 ⭐

**理论依据**：
- SlimGPT 的误差补偿假设输入分布不变
- VAR 的输入是高度结构化的（多尺度 token + 条件信息）
- Hessian 近似 H = X^T X 可能不准确

**验证实验**：

1. **无补偿对比实验**
   ```python
   # 运行剪枝时使用 --no_compensate 标志
   # 对比有无补偿的 FID 差异

   python model_slimming_basic.py \
       --sparsity 0.2 \
       --no_compensate  # 关闭误差补偿

   # 如果无补偿反而更好，说明补偿有问题
   ```

2. **Hessian 质量检查**
   ```python
   # 检查 Hessian 的条件数
   H = pruner.H
   eigenvalues = torch.linalg.eigvalsh(H)
   condition_number = eigenvalues.max() / eigenvalues.min()
   print(f"Condition number: {condition_number:.2e}")

   # 条件数太大说明 Hessian 病态
   ```

**实施优先级**：⭐ 较低
**预期时间**：2 小时
**成功标准**：确定补偿是否有害

---

### 假设 4：多尺度结构的影响 （理论依据不足）

**注**：经过讨论，VAR 的参数在 10 个尺度间共享，不存在"头专业化到某尺度"的机制。但可能存在统计上的模式。

**可选验证**：
- 分尺度计算 FID（Scale 1-3, 4-6, 7-10）
- 查看不同尺度的生成质量下降程度

**实施优先级**：⭐ 最低（除非其他假设都被排除）

---

## 实验执行计划

### Phase 1：快速验证（1-2 天）
1. **测试 scale_mul 修复效果**
   - 运行修复后的剪枝代码
   - 记录 FID 变化

2. **离散采样测试（假设 1）**
   - Top-1 accuracy 测试
   - Temperature 采样测试
   - 如果验证成功，直接跳到解决方案

### Phase 2：深度分析（3-5 天）
3. **AdaLN 分析（假设 2）**
   - 激活统计对比
   - Gamma 调整实验

4. **SlimGPT 诊断（假设 3）**
   - 无补偿对比
   - Hessian 质量分析

### Phase 3：综合方案
- 整合所有发现
- 设计最终修复方案
- 撰写技术报告

---

## 成功标准

### 短期目标
- Training-free 剪枝 FID < 50（从 178 改善）
- 理解 1 epoch 训练为何有效

### 长期目标
- Training-free 剪枝 FID < 30
- 可选：50 步 quick calibration 后 FID < 15

---

## 实验记录模板

```markdown
### 实验 X：[实验名称]
**日期**：YYYY-MM-DD
**假设**：[测试哪个假设]
**方法**：[实验步骤]
**结果**：
- FID: 原始 X.X → 修改后 Y.Y
- 其他指标：...
**结论**：[发现和分析]
**下一步**：[后续行动]
```

---

## 参考资料

### 相关文件
- `model_slimming_basic.py` - 主要剪枝脚本
- `slim_utils/slimgpt.py` - SlimGPT 实现
- `VAR/models/var.py` - VAR 模型定义
- `VAR/models/basic_var.py` - 注意力和 AdaLN 实现

### 关键代码位置
- scale_mul_1H11 使用：`basic_var.py:101-104`
- AdaLN 缩放：`basic_var.py:158-159`
- Token 采样：`var.py:175`
- Logits 生成：`var.py:124`

---

## 附录：为什么 LLM 不受影响？

| 特性 | LLM (LLaMA) | VAR |
|------|-------------|-----|
| 输出类型 | 连续分布 (log-prob) | 离散采样 (argmax) |
| 评估指标 | PPL (连续) | FID (依赖生成质量) |
| 注意力缩放 | 固定 (1/√d) | 可学习 (scale_mul) |
| 层归一化 | RMSNorm (固定) | AdaLN (条件自适应) |
| 对 logits 扰动的敏感度 | 低（log-likelihood 平滑） | 高（argmax 不连续） |

**关键差异**：PPL 计算的是整个分布的对数似然，对 logits 的小偏移不敏感；而 FID 依赖 argmax 采样，一个 bit 的翻转都可能导致完全不同的图像。

---

最后更新：2024-11-09
维护者：Claude & 用户
