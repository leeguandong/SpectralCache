# SpectralCache 实现文档

## 概述

SpectralCache 是论文中提出的 FastCache++ 方法，包含三个核心模块：

1. **TADS (Timestep-Aware Dynamic Scheduling)**: 时间步自适应动态调度
2. **CEB (Cumulative Error Budget)**: 累积误差预算
3. **FDC (Frequency-Decomposed Caching)**: 频率分解缓存

## 实现位置

- **核心实现**: `xfuser/model_executor/cache/utils.py` 中的 `FastCachePPTransformerBlocks` 类
- **Flux 适配器**: `xfuser/model_executor/cache/diffusers_adapters/flux.py`
- **测试脚本**: `benchmark/test_spectralcache.py`
- **消融实验**: `benchmark/spectralcache_ablation.py`

## 核心模块实现

### 1. TADS (Timestep-Aware Dynamic Scheduling)

**论文描述**：使用余弦调度函数动态调整缓存阈值

```python
s(t) = s_min + (s_max - s_min) * (1 - cos(2π * t/T)) / 2
τ^eff(t) = τ_base * s(t)
```

**实现位置**: `FastCachePPTransformerBlocks.tads_scale()`

**参数**:
- `tads_s_min`: 最小缩放因子 (默认 0.5)
- `tads_s_max`: 最大缩放因子 (默认 1.5)
- `enable_tads`: 是否启用 TADS (默认 True)

### 2. CEB (Cumulative Error Budget)

**论文描述**：限制连续缓存的时间步数量，防止误差累积

```
if c_t < C_max:
    允许缓存，c_t += 1
else:
    强制全计算，c_t = 0
```

**实现位置**: `FastCachePPTransformerBlocks.are_two_tensor_similar()`

**参数**:
- `ceb_c_max`: 最大连续缓存步数 C_max (默认 2)
- `enable_ceb`: 是否启用 CEB (默认 True)

### 3. FDC (Frequency-Decomposed Caching)

**论文描述**：将特征维度分为低频和高频两个频段，使用不对称阈值

```
δ^low = mean(|M_t^low - M_{t-1}^low|) / mean(|M_{t-1}^low|)
δ^high = mean(|M_t^high - M_{t-1}^high|) / mean(|M_{t-1}^high|)

缓存条件: δ^low ≤ τ^eff * γ_low AND δ^high ≤ τ^eff * γ_high
```

**实现位置**: `FastCachePPTransformerBlocks._fdc_check()`

**参数**:
- `fdc_freq_ratio`: 频率分割比例 (默认 0.5)
- `gamma_low`: 低频阈值缩放因子 (默认 0.8，更严格)
- `gamma_high`: 高频阈值缩放因子 (默认 1.5，更宽松)
- `enable_fdc`: 是否启用 FDC (默认 True)

## 使用方法

### 方法 1: 使用 Flux 适配器（推荐）

```python
from diffusers import FluxPipeline
from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer

# 加载模型
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-schnell",
    torch_dtype=torch.bfloat16,
).to("cuda")

# 应用 SpectralCache
apply_cache_on_transformer(
    pipe.transformer,
    rel_l1_thresh=0.6,  # 论文中的 τ = 0.6
    num_steps=20,
    use_cache="SpectralCache",  # 或 "FastPP"
    # TADS 参数
    tads_s_min=0.5,
    tads_s_max=1.5,
    enable_tads=True,
    # CEB 参数
    ceb_c_max=2,
    enable_ceb=True,
    # FDC 参数
    fdc_freq_ratio=0.5,
    gamma_low=0.8,
    gamma_high=1.5,
    enable_fdc=True,
)

# 运行推理
result = pipe(
    prompt="a photo of an astronaut riding a horse on the moon",
    num_inference_steps=20,
    height=512,
    width=512,
)
```

### 方法 2: 直接使用类

```python
from xfuser.model_executor.cache.utils import FastCachePPTransformerBlocks

# 创建缓存包装器
cached_blocks = FastCachePPTransformerBlocks(
    transformer.transformer_blocks,
    transformer.single_transformer_blocks,
    transformer=transformer,
    rel_l1_thresh=0.6,
    num_steps=20,
    # TADS
    tads_s_min=0.5,
    tads_s_max=1.5,
    enable_tads=True,
    # CEB
    ceb_c_max=2,
    enable_ceb=True,
    # FDC
    fdc_freq_ratio=0.5,
    gamma_low=0.8,
    gamma_high=1.5,
    enable_fdc=True,
)
```

## 消融实验

运行完整的消融实验（复现论文表 4）：

```bash
python benchmark/spectralcache_ablation.py \
    --model_type flux \
    --num_inference_steps 20 \
    --height 512 \
    --width 512 \
    --output_dir spectralcache_ablation_results
```

这将测试以下 9 种配置：
1. Baseline (No Cache)
2. None (基础缓存，无 TADS/CEB/FDC)
3. TADS only
4. CEB only
5. FDC only
6. TADS + CEB
7. TADS + FDC
8. CEB + FDC
9. TADS + CEB + FDC (Full)

## 快速测试

运行快速测试验证实现：

```bash
python benchmark/test_spectralcache.py
```

## 论文参数设置

根据论文实验部分（Section 4.1），推荐参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `rel_l1_thresh` (τ) | 0.6 | 基础缓存阈值 |
| `tads_s_min` | 0.5 | TADS 最小缩放 |
| `tads_s_max` | 1.5 | TADS 最大缩放 |
| `ceb_c_max` (C_max) | 2 | 最大连续缓存步数 |
| `fdc_freq_ratio` (r) | 0.5 | 频率分割比例 |
| `gamma_low` (γ_low) | 0.8 | 低频阈值缩放 |
| `gamma_high` (γ_high) | 1.5 | 高频阈值缩放 |

## 实现细节

### 累积距离机制

论文算法 1 中的累积距离 A_t：

```python
# 计算相对 L1 距离
d_t = ||M_t - M_{t-1}||_1 / ||M_{t-1}||_1

# 多项式重缩放
P(d_t) = rescale_func(d_t)

# 累积
A_t = A_{t-1} + P(d_t)

# 缓存条件
if A_t < τ^eff:
    可以缓存
```

### 模态输入

使用第一个 transformer block 的 norm1 层计算模态输入：

```python
M_t = Norm_1(H_{t,0}; e_t)
```

这提供了比原始隐藏状态更好的相似度信号。

### 缓存决策流程

```
1. 计算 A_t = A_{t-1} + P(d_t)
2. 计算 τ^eff = τ * s(t)  [TADS]
3. 检查 c_t < C_max  [CEB]
4. 检查 δ^low ≤ τ^eff * γ_low AND δ^high ≤ τ^eff * γ_high  [FDC]
5. 检查 A_t < τ^eff
6. 如果所有条件满足：
   - 缓存命中，使用缓存的残差
   - c_t += 1
7. 否则：
   - 全计算
   - A_t = 0, c_t = 0
```

## 性能指标

获取缓存统计信息：

```python
# 从 transformer blocks 获取统计
for block in pipe.transformer.transformer_blocks:
    if hasattr(block, 'get_cache_statistics'):
        stats = block.get_cache_statistics()
        print(f"Cache hits: {stats['cache_hits']}")
        print(f"Total steps: {stats['total_steps']}")
        print(f"Hit rate: {stats['cache_hit_rate']:.2%}")
```

## 论文预期结果

根据论文表 4（FLUX.1-schnell, 512×512, 20 steps）：

| 配置 | 加速比 | LPIPS↓ | SSIM↑ |
|------|--------|--------|-------|
| Baseline | 1.00× | - | - |
| None | 2.29× | 0.207 | 0.723 |
| TADS | 2.04× | 0.213 | 0.717 |
| CEB | 2.08× | 0.207 | 0.723 |
| FDC | 2.12× | 0.207 | 0.723 |
| TADS+CEB | 1.79× | **0.205** | **0.726** |
| TADS+FDC | 1.74× | 0.213 | 0.717 |
| CEB+FDC | 1.95× | 0.207 | 0.723 |
| **Full (TADS+CEB+FDC)** | **1.86×** | **0.205** | **0.726** |

## 故障排除

### 问题 1: 缓存命中率为 0

**原因**: 阈值设置过于严格

**解决**: 增大 `rel_l1_thresh` 或调整 TADS/FDC 参数

### 问题 2: 质量下降明显

**原因**: 缓存过于激进

**解决**:
- 减小 `ceb_c_max`
- 减小 `tads_s_max`
- 减小 `gamma_high`

### 问题 3: 加速不明显

**原因**: 缓存命中率太低

**解决**:
- 增大 `rel_l1_thresh`
- 增大 `ceb_c_max`
- 增大 `gamma_low` 和 `gamma_high`

## 与其他方法的对比

| 方法 | 时间步自适应 | 误差控制 | 频率感知 |
|------|-------------|---------|---------|
| TeaCache | ❌ | ❌ | ❌ |
| FastCache | ❌ | ❌ | ❌ |
| **SpectralCache** | ✅ TADS | ✅ CEB | ✅ FDC |

## 参考文献

论文: "SpectralCache: Frequency-Aware Error-Bounded Caching for Accelerating Diffusion Transformers"

相关代码:
- TeaCache: https://github.com/ali-vilab/TeaCache
- FastCache: 本仓库的 FastCachedTransformerBlocks
