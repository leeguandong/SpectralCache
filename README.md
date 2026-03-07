# SpectralCache: Frequency-Aware Error-Bounded Caching for Accelerating Diffusion Transformers

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

Official implementation of **SpectralCache**, a unified caching framework for accelerating Diffusion Transformer (DiT) inference by exploiting non-uniformity across time, depth, and frequency dimensions.

> **Paper**: [SpectralCache](https://arxiv.org/html/2603.05315v1)

## 🔥 Highlights

- **2.46× speedup** on FLUX.1-schnell - **16% faster than TeaCache** with comparable quality
- **Training-free** and **plug-and-play** - works with existing DiT architectures
- **Three orthogonal components**:
  - **TADS** (Timestep-Aware Dynamic Scheduling): Adapts caching aggressiveness across denoising phases
  - **CEB** (Cumulative Error Budget): Prevents error cascading via consecutive caching limits
  - **FDC** (Frequency-Decomposed Caching): Applies asymmetric thresholds to feature bands

## 📊 Performance

| Method | Speedup | LPIPS↓ | SSIM↑ | PSNR↑ |
|--------|---------|--------|-------|-------|
| No Cache | 1.00× | - | - | - |
| FBCache | 1.87× | 0.145 | 0.792 | 22.45 |
| TeaCache | 2.12× | 0.215 | 0.734 | 20.51 |
| FastCache | 4.51× | 0.559 | 0.360 | 14.53 |
| **SpectralCache** | **2.46×** | **0.217** | **0.727** | **20.41** |

*Evaluated on FLUX.1-schnell at 512×512 resolution, 20 steps*

**Key Result**: SpectralCache achieves 16% higher speedup than TeaCache (2.46× vs 2.12×) while maintaining near-identical quality (LPIPS difference < 1%).

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/leeguandong/SpectralCache.git
cd SpectralCache

# Install dependencies (requires xfuser)
pip install xfuser
```

### Basic Usage

```python
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
from diffusers import FluxPipeline

# Load FLUX model
model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell")

# Wrap with SpectralCache
wrapper = xFuserFastCachePipelineWrapper(model)

# Enable SpectralCache with default parameters (τ=0.8 for best speed)
wrapper.enable_spectralcache(
    tau=0.8,              # Base cache threshold (0.8 for 2.46× speedup)
    s_min=0.5,            # TADS min scale
    s_max=1.5,            # TADS max scale
    C_max=2,              # CEB consecutive limit
    gamma_low=0.8,        # FDC low-band scale
    gamma_high=1.5,       # FDC high-band scale
    r=0.5                 # FDC frequency split ratio
)

# Generate images
result = wrapper(
    prompt="a photo of an astronaut riding a horse on the moon",
    num_inference_steps=20,
)
```

## 📁 Repository Structure

```
SpectralCache/
├── benchmark/              # Benchmark scripts
│   ├── spectral_sweep.py          # Threshold sensitivity analysis
│   ├── spectralcache_ablation.py  # Ablation study
│   ├── main_comparison.py         # Main results comparison
│   ├── quality_eval.py            # Quality metrics evaluation
│   └── cache_error_analysis.py    # Error analysis
├── examples/               # Example usage scripts
│   └── basic_usage.py             # Basic SpectralCache usage example
├── docs/                   # Documentation
│   ├── IMPLEMENTATION.md          # Technical implementation details
│   └── spectralcache.md           # Method documentation
└── README.md               # This file
```

## 🧪 Running Benchmarks

### Main Results (Table 1 & 2)

```bash
python benchmark/main_comparison.py \
    --model_type flux \
    --model "black-forest-labs/FLUX.1-schnell" \
    --num_inference_steps 20 \
    --height 512 \
    --width 512 \
    --output_dir results/main
```

### Ablation Study (Table 3)

```bash
python benchmark/spectralcache_ablation.py \
    --model_type flux \
    --num_inference_steps 20 \
    --output_dir results/ablation
```

### Threshold Sensitivity (Table 4)

```bash
python benchmark/spectral_sweep.py \
    --model_type flux \
    --thresholds 0.3 0.4 0.5 0.6 0.8 \
    --output_dir results/threshold
```

### Quality Evaluation

```bash
python benchmark/quality_eval.py \
    --model_type flux \
    --cache_methods spectralcache teacache fbcache \
    --num_samples 10 \
    --output_dir results/quality
```

## 🔬 Method Overview

SpectralCache exploits three axes of non-uniformity in DiT denoising:

### 1. Temporal Non-Uniformity (TADS)

Sensitivity to caching errors follows a U-shaped curve across timesteps. TADS uses a cosine bell schedule to apply conservative caching at sensitive early/late steps and aggressive caching in the tolerant middle regime:

```
s(t) = s_min + (s_max - s_min) · (1 - cos(2πt/T)) / 2
τ_eff(t) = τ_base · s(t)
```

### 2. Depth Non-Uniformity (CEB)

Consecutive cached timesteps accumulate errors without correction. CEB limits consecutive caching to force periodic full computation:

```
cache_allowed = (c_t < C_max) AND (other_conditions)
```

### 3. Feature Non-Uniformity (FDC)

Different feature dimensions exhibit heterogeneous temporal dynamics. FDC partitions features into two bands with asymmetric thresholds:

```
δ_low ≤ τ_eff · γ_low   (stricter, γ_low < 1)
δ_high ≤ τ_eff · γ_high (lenient, γ_high > 1)
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This work builds upon:
- [xDiT/xFuser](https://github.com/xdit-project/xDiT) - Base framework for DiT acceleration
- [TeaCache](https://arxiv.org/abs/2410.17812) - Polynomial rescaling and modulated input similarity
- [FLUX.1](https://github.com/black-forest-labs/flux) - State-of-the-art DiT model

## 📧 Contact

For questions or issues, please open an issue on GitHub.
