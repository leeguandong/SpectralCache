# SpectralCache Implementation Documentation

## Overview

SpectralCache is a unified caching framework for accelerating Diffusion Transformer (DiT) inference. It exploits three orthogonal axes of non-uniformity in the denoising process:

1. **Temporal non-uniformity**: Sensitivity varies across timesteps
2. **Depth non-uniformity**: Consecutive caching accumulates errors
3. **Feature non-uniformity**: Different feature dimensions have heterogeneous dynamics

## Architecture

### Core Components

#### 1. TADS (Timestep-Aware Dynamic Scheduling)

**Purpose**: Adapt caching aggressiveness based on timestep phase.

**Implementation**:
```python
def compute_tads_scale(t, T, s_min=0.5, s_max=1.5):
    """Compute TADS scaling factor for timestep t."""
    normalized_t = t / T
    scale = s_min + (s_max - s_min) * (1 - np.cos(2 * np.pi * normalized_t)) / 2
    return scale

def get_effective_threshold(tau_base, t, T, s_min, s_max):
    """Get TADS-adjusted threshold."""
    s_t = compute_tads_scale(t, T, s_min, s_max)
    return tau_base * s_t
```

**Key Parameters**:
- `s_min`: Minimum scale (conservative caching at endpoints), default: 0.5
- `s_max`: Maximum scale (aggressive caching at midpoint), default: 1.5
- `tau_base`: Base cache threshold, default: 0.6

**Intuition**: The cosine bell schedule aligns with the U-shaped sensitivity profile observed empirically. Early and late timesteps are sensitive (low scale → strict threshold), while middle timesteps are tolerant (high scale → lenient threshold).

#### 2. CEB (Cumulative Error Budget)

**Purpose**: Prevent error cascading by limiting consecutive cached timesteps.

**Implementation**:
```python
class CEBCounter:
    def __init__(self, C_max=2):
        self.C_max = C_max
        self.consecutive_count = 0

    def can_cache(self):
        """Check if caching is allowed under CEB constraint."""
        return self.consecutive_count < self.C_max

    def on_cache_hit(self):
        """Increment counter on cache hit."""
        self.consecutive_count += 1

    def on_cache_miss(self):
        """Reset counter on full computation."""
        self.consecutive_count = 0
```

**Key Parameters**:
- `C_max`: Maximum consecutive cached timesteps, default: 2

**Intuition**: When multiple consecutive timesteps reuse the same cached residual, the residual becomes increasingly stale. CEB forces periodic full computation to re-anchor the trajectory and prevent exponential error accumulation.

#### 3. FDC (Frequency-Decomposed Caching)

**Purpose**: Apply differentiated thresholds to feature bands with heterogeneous dynamics.

**Implementation**:
```python
def fdc_check(M_t, M_t_prev, tau_eff, r=0.5, gamma_low=0.8, gamma_high=1.5):
    """
    Frequency-decomposed caching check.

    Args:
        M_t: Current modulated input [B, N, D]
        M_t_prev: Previous modulated input [B, N, D]
        tau_eff: TADS-adjusted threshold
        r: Frequency split ratio (0 < r < 1)
        gamma_low: Low-band scale (< 1, stricter)
        gamma_high: High-band scale (> 1, lenient)

    Returns:
        bool: Whether both bands pass threshold checks
    """
    D = M_t.shape[-1]
    split_idx = int(D * r)

    # Split into low and high bands
    M_low_t = M_t[..., :split_idx]
    M_high_t = M_t[..., split_idx:]
    M_low_prev = M_t_prev[..., :split_idx]
    M_high_prev = M_t_prev[..., split_idx:]

    # Compute relative L1 change per band
    delta_low = torch.mean(torch.abs(M_low_t - M_low_prev)) / torch.mean(torch.abs(M_low_prev))
    delta_high = torch.mean(torch.abs(M_high_t - M_high_prev)) / torch.mean(torch.abs(M_high_prev))

    # Apply asymmetric thresholds
    tau_low = tau_eff * gamma_low
    tau_high = tau_eff * gamma_high

    return (delta_low <= tau_low) and (delta_high <= tau_high)
```

**Key Parameters**:
- `r`: Frequency split ratio, default: 0.5
- `gamma_low`: Low-band scale (stricter), default: 0.8
- `gamma_high`: High-band scale (lenient), default: 1.5

**Intuition**: The modulated input's first half captures structural changes (higher volatility), while the second half captures stable fine details (lower volatility). Asymmetric thresholds prevent caching when structural components change significantly, even if fine details are stable.

## Complete Algorithm

```python
def spectralcache_forward(H_t0, e_t, R_prev, A_prev, ceb_counter,
                          tau_base, s_min, s_max, C_max,
                          gamma_low, gamma_high, r, t, T):
    """
    SpectralCache forward pass at timestep t.

    Args:
        H_t0: Hidden states at block 0 [B, N, D]
        e_t: Timestep embedding
        R_prev: Cached residual from previous timestep
        A_prev: Accumulated polynomial distance
        ceb_counter: CEB counter object
        tau_base: Base cache threshold
        s_min, s_max: TADS parameters
        C_max: CEB max consecutive cached steps
        gamma_low, gamma_high, r: FDC parameters
        t: Current timestep
        T: Total timesteps

    Returns:
        H_tL: Output hidden states
        R_t: Updated residual
        A_t: Updated accumulated distance
    """
    # 1. Compute modulated input
    M_t = norm1(H_t0, e_t)  # First block's norm1

    # 2. Polynomial-rescaled accumulated distance
    d_t = torch.norm(M_t - M_prev, p=1) / torch.norm(M_prev, p=1)
    A_t = A_prev + polynomial_rescale(d_t)

    # 3. TADS: timestep-adaptive threshold
    tau_eff = get_effective_threshold(tau_base, t, T, s_min, s_max)

    # 4. Check caching conditions
    if (R_prev is not None and
        t != 0 and t != T-1 and
        ceb_counter.can_cache()):

        # FDC: frequency-decomposed gating
        if (A_t < tau_eff and
            fdc_check(M_t, M_prev, tau_eff, r, gamma_low, gamma_high)):

            # Cache hit: skip all blocks
            H_tL = H_t0 + R_prev
            ceb_counter.on_cache_hit()
            return H_tL, R_prev, A_t

    # Cache miss: full computation
    H_tL = H_t0
    for block in transformer_blocks:
        H_tL = block(H_tL)

    R_t = H_tL - H_t0
    ceb_counter.on_cache_miss()
    A_t = 0  # Reset accumulator

    return H_tL, R_t, A_t
```

## Hyperparameter Tuning

### Default Configuration

```python
default_config = {
    'tau_base': 0.6,      # Base cache threshold
    's_min': 0.5,         # TADS min scale
    's_max': 1.5,         # TADS max scale
    'C_max': 2,           # CEB consecutive limit
    'gamma_low': 0.8,     # FDC low-band scale
    'gamma_high': 1.5,    # FDC high-band scale
    'r': 0.5              # FDC split ratio
}
```

### Tuning Guidelines

**Quality-Speed Tradeoff** (via `tau_base`):
- Lower `tau` (0.3-0.5): Better quality, lower speedup
- Higher `tau` (0.6-0.8): Higher speedup, acceptable quality
- Very high `tau` (>0.8): Diminishing returns, quality degradation

**Temporal Adaptation** (via `s_min`, `s_max`):
- Wider range (s_min=0.3, s_max=2.0): More aggressive temporal adaptation
- Narrower range (s_min=0.6, s_max=1.2): More conservative, stable quality

**Error Control** (via `C_max`):
- Smaller `C_max` (1-2): Better quality, lower speedup
- Larger `C_max` (3-4): Higher speedup, risk of error accumulation

**Frequency Sensitivity** (via `gamma_low`, `gamma_high`):
- More asymmetric (gamma_low=0.6, gamma_high=2.0): Stricter structural protection
- Less asymmetric (gamma_low=0.9, gamma_high=1.2): More uniform treatment

## Integration with xFuser

SpectralCache is implemented as an extension to the xFuser/FastCache framework:

```python
# In xfuser/model_executor/cache/utils.py
class SpectralCacheContext(CacheContext):
    """Extended cache context with TADS, CEB, FDC state."""

    def __init__(self, ...):
        super().__init__(...)
        self.ceb_counter = CEBCounter(C_max)
        self.tads_config = {'s_min': s_min, 's_max': s_max}
        self.fdc_config = {'r': r, 'gamma_low': gamma_low, 'gamma_high': gamma_high}

# In xfuser/model_executor/pipelines/fastcache_pipeline.py
class xFuserFastCachePipelineWrapper:
    def enable_spectralcache(self, tau, s_min, s_max, C_max,
                            gamma_low, gamma_high, r):
        """Enable SpectralCache with specified parameters."""
        self.cache_context = SpectralCacheContext(...)
```

## Performance Characteristics

**Computational Overhead**:
- TADS: O(1) per timestep (cosine computation)
- CEB: O(1) per timestep (counter check)
- FDC: O(D) per timestep (feature splitting + L1 computation)
- Total overhead: < 0.5% of full transformer computation

**Memory Overhead**:
- Cached residual: O(B × N × D) per timestep
- Modulated input history: O(B × N × D) for previous timestep
- CEB counter: O(1)
- Total: ~2× hidden state size

**Speedup Characteristics**:
- Cache hit rate: 40-50% at tau=0.6
- Per-hit speedup: ~2× (skips entire transformer backbone)
- Overall speedup: 1.8-2.2× depending on configuration

## Limitations and Future Work

1. **Fixed frequency partition**: Current implementation uses a simple 50-50 split. Learning the optimal spectral basis from data could improve performance.

2. **Global error budget**: `C_max` is constant across all timesteps. Adaptive per-timestep budgets could better exploit temporal variation.

3. **Model-specific tuning**: Default hyperparameters work well for FLUX.1, but other DiT architectures may benefit from different configurations.

4. **Video extension**: Current implementation targets image generation. Video DiTs have additional temporal redundancy that could be exploited.

## References

- TeaCache: [arXiv:2410.17812](https://arxiv.org/abs/2410.17812)
- FastCache: [arXiv:2505.20353](https://arxiv.org/abs/2505.20353)
- xDiT/xFuser: [GitHub](https://github.com/xdit-project/xDiT)
