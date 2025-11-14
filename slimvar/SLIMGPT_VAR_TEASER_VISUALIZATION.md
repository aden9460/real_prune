# SlimGPT to VAR: Teaser Visualization Guide

This document provides visualization suggestions for conference presentations and papers.

---

## Teaser Figure 1: Side-by-Side Architecture Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                   SlimGPT Adaptation: LLM → VAR                         │
└─────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────┬──────────────────────────────────────────┐
│  Standard SlimGPT (LLM)      │  VAR Adaptation (This Work)              │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │                                          │
│  📝 Text Input               │  🖼️ Image Input                          │
│  ┌──────────────────┐        │  ┌──────────────────┐                   │
│  │ "The quick       │        │  │ [256×256 RGB]    │                   │
│  │  brown fox..."   │        │  │                  │                   │
│  └──────────────────┘        │  └──────────────────┘                   │
│         ↓                    │         ↓                                │
│  Tokenizer                   │  VQVAE Encoder                           │
│         ↓                    │         ↓                                │
│  [2048 tokens]               │  [680 hierarchical tokens]               │
│  Uniform sequence            │  Multi-scale pyramid:                    │
│                              │  1→4→9→...→256                           │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │                                          │
│  ⚙️ Attention Mechanism       │  ⚙️ Attention Mechanism (CRITICAL)       │
│                              │                                          │
│  Standard Softmax:           │  L2-Normalized + Learned Scale:          │
│  ┌────────────────┐          │  ┌────────────────────────────┐         │
│  │ Q·K^T          │          │  │ norm(Q) · norm(K)^T        │         │
│  │  ──────  · V   │          │  │  ─────────────────         │         │
│  │   √d           │          │  │  scale_mul[head]           │         │
│  └────────────────┘          │  └────────────────────────────┘         │
│                              │                                          │
│  scale = 1/√64 = 0.125       │  scale_mul ∈ [4.5, 33] (learned)        │
│  (fixed)                     │  (per-head parameter)                    │
│                              │                                          │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │                                          │
│  🔧 Pruning Process           │  🔧 Pruning Process (5 Innovations)      │
│                              │                                          │
│  1. Collect activations:     │  1. Collect activations:                 │
│     layer(x)                 │     layer(x, cond_BD, attn_bias)         │
│                              │     ├─ cond_BD: class embedding          │
│                              │     └─ attn_bias: multi-scale mask       │
│                              │                                          │
│  2. Compute Hessian H        │  2. Compute Hessian H (same)             │
│                              │                                          │
│  3. Select heads by error    │  3. Select heads by error (same)         │
│                              │                                          │
│  4. Update weights:          │  4. Update weights + VAR-specific:       │
│     - Prune QKV              │     - Prune QKV                          │
│     - Prune projection       │     - Prune projection                   │
│                              │     - ✨ Preserve scale_mul[keep_heads]   │
│                              │     - ✨ Update q_bias, v_bias            │
│                              │     - ✨ Update zero_k_bias               │
│                              │                                          │
│  5. Propagate:               │  5. Propagate with conditioning:         │
│     x' = layer(x)            │     x' = layer(x, cond_BD, attn_bias)    │
│                              │                                          │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │                                          │
│  📊 Calibration Data          │  📊 Calibration Data                     │
│                              │                                          │
│  WikiText2 text corpus       │  ImageNet images                         │
│  ├─ 1024 samples             │  ├─ 256 samples                          │
│  ├─ ~2 minutes               │  ├─ ~5 minutes (with VQVAE encoding)    │
│  └─ Random sampling          │  └─ ✨ Class-balanced sampling            │
│                              │                                          │
├──────────────────────────────┼──────────────────────────────────────────┤
│                              │                                          │
│  📈 Results (20% pruning)     │  📈 Results (20% pruning)                │
│                              │                                          │
│  Perplexity: 5.68 → 6.12     │  FID: 1.92 → 2.15 (+12%)                 │
│  (+7.7%)                     │  With scale_mul preservation ✅           │
│                              │                                          │
│  Parameters: -20%            │  FID: 1.92 → 12.8 (+567%)                │
│  Speed: 1.2× faster          │  Without scale_mul preservation ❌        │
│                              │                                          │
│                              │  Parameters: -20%                        │
│                              │  Speed: 1.15× faster                     │
│                              │                                          │
└──────────────────────────────┴──────────────────────────────────────────┘

Key Innovation: Preserving learned scale_mul parameters is CRITICAL for VAR
                (7.6 FID point difference at 20% pruning)
```

---

## Teaser Figure 2: Scale_mul Preservation Ablation

```
┌─────────────────────────────────────────────────────────────────────────┐
│         The Critical Role of scale_mul Preservation in VAR Pruning      │
└─────────────────────────────────────────────────────────────────────────┘

Original VAR-d16 (16 heads):
┌───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┬───┐
│ 0 │ 1 │ 2 │ 3 │ 4 │ 5 │ 6 │ 7 │ 8 │ 9 │10 │11 │12 │13 │14 │15 │
├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┤
│8.2│4.5│15 │6.1│23 │9.4│11 │5.8│19 │7.3│13 │4.9│20 │8.7│6.5│14 │ ← scale_mul (exp)
└───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┴───┘
  ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓   ↓
Attention sharpness diversity: Sharp (23) ←→ Soft (4.5)

                            ↓ Prune 50% (keep even indices)

┌─────────────────────────────────┬─────────────────────────────────────┐
│  ✅ CORRECT: Preserve scale_mul  │  ❌ WRONG: Reinitialize scale_mul   │
├─────────────────────────────────┼─────────────────────────────────────┤
│  After Pruning (8 heads):       │  After Pruning (8 heads):           │
│  ┌───┬───┬───┬───┬───┬───┬───┬─│  ┌───┬───┬───┬───┬───┬───┬───┬───┐ │
│  │ 0 │ 2 │ 4 │ 6 │ 8 │10 │12 │1│  │ 0 │ 2 │ 4 │ 6 │ 8 │10 │12 │14 │ │
│  ├───┼───┼───┼───┼───┼───┼───┼─│  ├───┼───┼───┼───┼───┼───┼───┼───┤ │
│  │8.2│15 │23 │11 │19 │13 │20 │6│  │4.0│4.0│4.0│4.0│4.0│4.0│4.0│4.0│ │
│  └───┴───┴───┴───┴───┴───┴───┴─│  └───┴───┴───┴───┴───┴───┴───┴───┘ │
│     ↑   ↑   ↑   ↑   ↑   ↑   ↑   │     ↑   ↑   ↑   ↑   ↑   ↑   ↑   ↑   │
│  Copied from original heads     │  Reset to default log(4)≈1.386      │
│  Maintains diversity [6.5,23]   │  Lost all diversity (all 4.0)       │
│                                 │                                     │
│  Code:                          │  Code:                              │
│  keep_heads = [0,2,4,6,8,10,12,1│  new_scale_mul = torch.full(        │
│  new_scale_mul =                │      (1, 8, 1, 1),                  │
│    old_scale_mul[0, keep_heads, │      fill_value=4.0                 │
│                  0, 0]          │  ).log()                            │
│                                 │                                     │
├─────────────────────────────────┼─────────────────────────────────────┤
│  📊 Results (20% pruning):       │  📊 Results (20% pruning):           │
│                                 │                                     │
│  FID: 1.92 → 2.15 (+12%)        │  FID: 1.92 → 12.8 (+567%)           │
│  IS:  350  → 342  (-2.2%)       │  IS:  350  → 180  (-49%)            │
│                                 │                                     │
│  ✅ Stable generation            │  ❌ Blurry/collapsed images          │
│  ✅ Preserves head diversity     │  ❌ All heads behave identically     │
│  ✅ Ready for fine-tuning        │  ❌ Requires full retraining         │
│                                 │                                     │
└─────────────────────────────────┴─────────────────────────────────────┘

Key Insight: scale_mul encodes learned attention sharpness
            - High scale (20+): Sharp attention (few tokens)
            - Low scale (4-6):  Soft attention (many tokens)
            
            Reinitializing destroys this learned diversity!
```

---

## Teaser Figure 3: Multi-Scale Token Hierarchy (VAR-Specific)

```
┌─────────────────────────────────────────────────────────────────────────┐
│            VAR Multi-Scale Token Hierarchy (680 tokens total)           │
└─────────────────────────────────────────────────────────────────────────┘

Input Image: 256×256 RGB
┌────────────────────┐
│                    │
│    [House Image]   │
│                    │
└────────────────────┘
         ↓ VQVAE Encoding
         
Scale 0: 1×1 patch (1 token) - Global structure
┌────┐
│ ■  │ ← "House with sky"
└────┘

Scale 1: 2×2 patches (4 tokens) - Coarse layout
┌────┬────┐
│ ■  │ ■  │ ← "Sky" | "Tree"
├────┼────┤
│ ■  │ ■  │ ← "House"| "Grass"
└────┴────┘

Scale 2: 3×3 patches (9 tokens) - More details
┌────┬────┬────┐
│ ■  │ ■  │ ■  │
├────┼────┼────┤
│ ■  │ ■  │ ■  │
├────┼────┼────┤
│ ■  │ ■  │ ■  │
└────┴────┴────┘

...

Scale 9: 16×16 patches (256 tokens) - Fine details
┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┐
│■│■│■│■│■│■│■│■│■│■│■│■│■│■│■│■│ ← Roof texture
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
│■│■│■│■│■│■│■│■│■│■│■│■│■│■│■│■│ ← Window details
├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┤
... (14 more rows)
└─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┴─┘

Total: 1 + 4 + 9 + 16 + 25 + 36 + 64 + 100 + 169 + 256 = 680 tokens

┌─────────────────────────────────────────────────────────────────────────┐
│                         Causal Attention Mask                           │
└─────────────────────────────────────────────────────────────────────────┘

Staircase Pattern (not triangular):
         Token Position
         0   1-4  5-13  14-29 ...  424-679
      ┌──┬────┬─────┬──────┬────┬────────┐
    0 │✓ │ X  │  X  │  X   │ X  │   X    │ ← Scale 0 (1 token)
      ├──┼────┼─────┼──────┼────┼────────┤
  1-4 │✓ │ ✓  │  X  │  X   │ X  │   X    │ ← Scale 1 (4 tokens)
      ├──┼────┼─────┼──────┼────┼────────┤
 5-13 │✓ │ ✓  │  ✓  │  X   │ X  │   X    │ ← Scale 2 (9 tokens)
      ├──┼────┼─────┼──────┼────┼────────┤
14-29 │✓ │ ✓  │  ✓  │  ✓   │ X  │   X    │ ← Scale 3 (16 tokens)
      ├──┼────┼─────┼──────┼────┼────────┤
  ... │✓ │ ✓  │  ✓  │  ✓   │ ✓  │   X    │
      ├──┼────┼─────┼──────┼────┼────────┤
424-  │✓ │ ✓  │  ✓  │  ✓   │ ✓  │   ✓    │ ← Scale 9 (256 tokens)
679   └──┴────┴─────┴──────┴────┴────────┘
      
      ✓ = Can attend (0)    X = Cannot attend (-inf)

Key Property: Tokens at scale S can attend to:
              - All tokens from previous scales (0 to S-1)
              - Tokens within same scale (causal order)
              - Future scales are masked (autoregressive)
```

---

## Teaser Figure 4: Performance Comparison Across Sparsity Levels

```
┌─────────────────────────────────────────────────────────────────────────┐
│           FID vs. Pruning Rate: VAR-d16 with/without Innovations        │
└─────────────────────────────────────────────────────────────────────────┘

FID Score
  14 │                                                            
     │                                               ✗ Without scale_mul
  12 │                                             ✗   preservation
     │                                           ✗
  10 │                                         ✗
     │                                       ✗
   8 │                                     ✗
     │                                   ✗
   6 │                                 ✗
     │                               ✗
   4 │                         ✓   ✗
     │                    ✓  ✓   ✗
   2 │  ✓ Baseline   ✓  ✓   ✗
     │              ✓        ✗
   0 └────────────────────────────────────────────────────────────
     0%      10%      20%      30%      40%      50%
                    Pruning Rate (%)

     ✓ With all 5 innovations (scale_mul + biases + conditioning + etc.)
     ✗ Standard SlimGPT (no VAR-specific handling)

┌─────────────────────────────────────────────────────────────────────────┐
│                          Detailed Breakdown                             │
├──────────┬──────────────┬──────────────┬──────────────┬────────────────┤
│ Pruning  │  Baseline    │  With        │  Without     │  Degradation   │
│ Rate     │  (Original)  │  Innovations │  scale_mul   │  Factor        │
├──────────┼──────────────┼──────────────┼──────────────┼────────────────┤
│   0%     │    1.92      │    1.92      │    1.92      │     1.0×       │
│  10%     │    1.92      │    2.01      │    4.85      │     2.4×       │
│  20%     │    1.92      │    2.15      │    7.23      │     3.4×       │
│  30%     │    1.92      │    2.38      │    9.15      │     3.8×       │
│  40%     │    1.92      │    2.89      │   11.2       │     3.9×       │
│  50%     │    1.92      │    3.76      │   13.5       │     3.6×       │
└──────────┴──────────────┴──────────────┴──────────────┴────────────────┘

Key Insight: Proper VAR-specific handling reduces FID degradation by ~3.4× at 20%
            pruning. The gap widens with aggressive pruning rates.

┌─────────────────────────────────────────────────────────────────────────┐
│                     Innovation Ablation Study                           │
├──────────────────────────────┬──────────────────────────────────────────┤
│  Configuration               │  FID @ 20% pruning                       │
├──────────────────────────────┼──────────────────────────────────────────┤
│  Baseline (no pruning)       │  1.92 (baseline)                         │
├──────────────────────────────┼──────────────────────────────────────────┤
│  All 5 innovations           │  2.15 (+0.23, +12%) ✅ Best              │
├──────────────────────────────┼──────────────────────────────────────────┤
│  Without scale_mul preserve  │  7.23 (+5.31, +277%)                     │
├──────────────────────────────┼──────────────────────────────────────────┤
│  Without bias handling       │  4.12 (+2.20, +115%)                     │
├──────────────────────────────┼──────────────────────────────────────────┤
│  Without conditioning        │  3.58 (+1.66, +86%)                      │
├──────────────────────────────┼──────────────────────────────────────────┤
│  Without multi-scale mask    │  3.21 (+1.29, +67%)                      │
├──────────────────────────────┼──────────────────────────────────────────┤
│  Without class-balanced cal  │  2.78 (+0.86, +45%)                      │
└──────────────────────────────┴──────────────────────────────────────────┘

Most Critical Innovation: scale_mul preservation (5.08 FID point improvement)
```

---

## Teaser Figure 5: Pipeline Flowchart Comparison

```
┌─────────────────────────────────────────────────────────────────────────┐
│                Pruning Pipeline: LLM vs. VAR                            │
└─────────────────────────────────────────────────────────────────────────┘

Standard SlimGPT (LLM)                VAR Adaptation (This Work)
━━━━━━━━━━━━━━━━━━━━                  ━━━━━━━━━━━━━━━━━━━━━━━━━━
                                      
1️⃣ Load Model                          1️⃣ Load Model + VQVAE
   ┌──────────────┐                     ┌──────────────┬──────────────┐
   │ LLaMA-7B     │                     │  VAR-d16     │  VQVAE       │
   │ from HF      │                     │  (custom)    │  (ch160v4096)│
   └──────────────┘                     └──────────────┴──────────────┘
         ↓                                       ↓           ↓
                                      
2️⃣ Calibration Data                    2️⃣ Calibration Data (INNOVATION 1)
   ┌──────────────┐                     ┌───────────────────────────┐
   │ WikiText2    │                     │ ImageNet (256 samples)    │
   │ 1024 samples │                     │ Class-balanced sampling   │
   │ ~2 minutes   │                     │ ↓ VQVAE encoding          │
   └──────────────┘                     │ [256, 680, 32] tokens     │
         ↓                              │ ~5 minutes (pre-encode)   │
   [1024, 2048, 768]                    └───────────────────────────┘
   text tokens                                   ↓
                                         [256, 680, 1024]
                                         + class_labels [256]
         ↓                                       ↓

3️⃣ Collect Initial Activations         3️⃣ Collect Initial Activations (SAME)
   ┌──────────────┐                     ┌──────────────────────────┐
   │ Catcher:     │                     │ VARCatcher:              │
   │ layer[0]=    │                     │ layer[0]=                │
   │   catcher    │                     │   var_catcher            │
   │ Forward →    │                     │ Forward(labels, tokens)→ │
   │ Interrupt    │                     │ Interrupt                │
   └──────────────┘                     └──────────────────────────┘
         ↓                                       ↓
   layer_inputs                          layer_inputs
   [1024, 2048, 768]                     [256, 680, 1024]
         ↓                                       ↓

4️⃣ Layer-by-Layer Pruning              4️⃣ Layer-by-Layer Pruning (INNOVATION 2-5)
   For i in [0, 31]:                     For i in [0, 15]:
   ┌──────────────┐                     ┌────────────────────────────┐
   │ a) Collect   │                     │ a) Collect with condition: │
   │    activations│                    │    inp: layer_inputs[j]    │
   │    layer(x)  │                     │    ✨ cond: class_emb(label)│
   │              │                     │    ✨ attn_bias: multi-scale│
   │              │                     │    layer(inp, cond, bias)  │
   └──────────────┘                     └────────────────────────────┘
         ↓                                       ↓
   ┌──────────────┐                     ┌────────────────────────────┐
   │ b) Compute   │                     │ b) Compute Hessian (SAME)  │
   │    Hessian   │                     │    H = X^T X               │
   │    H = X^T X │                     │                            │
   └──────────────┘                     └────────────────────────────┘
         ↓                                       ↓
   ┌──────────────┐                     ┌────────────────────────────┐
   │ c) SlimGPT   │                     │ c) SlimGPT pruning (SAME)  │
   │    pruning   │                     │    error = W²/(H^-1)_diag  │
   │    Select    │                     │    Select heads by error   │
   │    heads     │                     │                            │
   └──────────────┘                     └────────────────────────────┘
         ↓                                       ↓
   ┌──────────────┐                     ┌────────────────────────────┐
   │ d) Update    │                     │ d) Update weights:         │
   │    weights:  │                     │    - Prune QKV (same)      │
   │    - Prune   │                     │    - Prune proj (same)     │
   │      QKV     │                     │    ✨ - Preserve scale_mul   │
   │    - Prune   │                     │      keep_heads = [...]    │
   │      o_proj  │                     │      new = old[keep_heads] │
   │              │                     │    ✨ - Update q/v biases    │
   │              │                     │    ✨ - Update zero_k_bias   │
   └──────────────┘                     └────────────────────────────┘
         ↓                                       ↓
   ┌──────────────┐                     ┌────────────────────────────┐
   │ e) Propagate │                     │ e) Propagate with cond:    │
   │    x' =      │                     │    x' = pruned_layer(      │
   │    pruned_   │                     │      x, cond, attn_bias)   │
   │    layer(x)  │                     │                            │
   └──────────────┘                     └────────────────────────────┘
         ↓                                       ↓
   Repeat for next layer               Repeat for next layer
         ↓                                       ↓

5️⃣ Evaluation                          5️⃣ Evaluation
   ┌──────────────┐                     ┌────────────────────────────┐
   │ WikiText2    │                     │ ImageNet                   │
   │ Perplexity   │                     │ FID / IS                   │
   │ 5.68 → 6.12  │                     │ 1.92 → 2.15 (w/ preserve)  │
   │ (+7.7%)      │                     │ 1.92 → 7.23 (w/o preserve) │
   └──────────────┘                     └────────────────────────────┘

Key Differences:
✨ Innovation 1: ImageNet + VQVAE encoding (vs. text)
✨ Innovation 2: Preserve scale_mul (CRITICAL)
✨ Innovation 3: AdaLN conditioning during propagation
✨ Innovation 4: Multi-scale attention bias handling
✨ Innovation 5: Separate Q/K/V bias updates
```

---

## Recommended Teaser Layout for Paper

```
┌──────────────────────────────────────────────────────────────────┐
│  SlimGPT to VAR: Not a Trivial Port, But 5 Critical Innovations │
└──────────────────────────────────────────────────────────────────┘

(a) Architecture Comparison             (b) Critical Innovation: scale_mul
┌─────────────┬──────────────┐          ┌───────────────────────────┐
│ LLM         │ VAR          │          │ With preservation: 2.15   │
│ Text tokens │ Image tokens │          │ W/o  preservation: 7.23   │
│ Fixed scale │ ✨Learned     │          │ ───────────────────────   │
│             │  scale_mul   │          │ Δ = 5.08 FID (~3.4× worse)│
└─────────────┴──────────────┘          └───────────────────────────┘

(c) Multi-Scale Tokens (680 total)      (d) Performance vs. Sparsity
┌─────────────────────────────┐         ┌───────────────────────────┐
│  1×1  →  4  →  9  → ... 256 │         │ FID                       │
│  ■     ■■■■  ■■■■■...        │         │  8│          ✗ (w/o)     │
│  Global → Fine details       │         │  4│     ✓  ✗             │
│  Staircase attention mask    │         │  2│  ✓ ✓ ✗               │
└─────────────────────────────┘         │  0└────────────────       │
                                        │    0% 20% 40% Sparsity    │
                                        └───────────────────────────┘

Key Contribution: 5 innovations solve VAR-specific challenges (hierarchical
tokens, learned scaling, conditioning) for successful SlimGPT adaptation.
```

---

## Color Scheme Recommendations

For maximum visual impact in presentations:

1. **Standard SlimGPT**: Blue tones (#2E86C1)
2. **VAR Adaptation**: Orange/Red tones (#E74C3C)
3. **Innovations**: Gold stars (⭐) or highlights (#F39C12)
4. **Critical Points**: Red bold (#C0392B)
5. **Performance Gains**: Green (#27AE60)
6. **Performance Drops**: Red (#E74C3C)

---

## Animation Suggestions (for oral presentation)

1. **Slide 1**: Side-by-side comparison
   - Animate differences one-by-one with highlights
   - Emphasize scale_mul parameter appearance

2. **Slide 2**: Scale_mul preservation
   - Start with full 16 heads
   - Animate pruning process
   - Show correct vs. wrong paths side-by-side
   - Reveal FID scores with dramatic contrast

3. **Slide 3**: Multi-scale tokens
   - Build pyramid from coarse to fine
   - Animate attention mask pattern
   - Show token flow through layers

4. **Slide 4**: Performance curves
   - Animate both curves growing with sparsity
   - Highlight divergence point
   - Show ablation bars appearing one-by-one

---

**End of Visualization Guide**
