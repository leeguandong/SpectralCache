# SpectralCache Project Structure

```
SpectralCache/
├── README.md                   # Main project documentation
├── LICENSE                     # Apache 2.0 license
├── CONTRIBUTING.md             # Contribution guidelines
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── benchmark/                  # Benchmark and evaluation scripts
│   ├── spectral_sweep.py              # Threshold sensitivity analysis (Table 4)
│   ├── spectralcache_ablation.py      # Ablation study (Table 3)
│   ├── main_comparison.py             # Main results comparison (Table 1 & 2)
│   ├── quality_eval.py                # Quality metrics evaluation
│   ├── cache_error_analysis.py        # Error analysis experiments
│   └── ablation_study.py              # Additional ablation experiments
│
├── examples/                   # Example usage scripts
│   └── basic_usage.py                 # Basic SpectralCache usage example
│
├── paper/                      # Paper and figures
│   ├── main.tex                       # Full paper (LaTeX)
│   ├── appendix.tex                   # Appendix with theorem proofs
│   ├── references.bib                 # Bibliography
│   ├── temporal_sensitivity.pdf       # Figure: Temporal sensitivity curve
│   ├── error_cascade.pdf              # Figure: Error cascade analysis
│   ├── spectral_analysis.pdf          # Figure: Spectral heterogeneity
│   └── qual_comparison/               # Qualitative comparison images
│       ├── none_*.png                 # Baseline images
│       ├── tea_*.png                  # TeaCache images
│       └── spectralcache_*.png        # SpectralCache images
│
└── docs/                       # Documentation
    ├── IMPLEMENTATION.md              # Technical implementation details
    └── spectralcache.md               # Method documentation

```

## Key Files

### Root Level

- **README.md**: Main entry point with overview, installation, usage, and results
- **LICENSE**: Apache 2.0 license
- **CONTRIBUTING.md**: Guidelines for contributors
- **requirements.txt**: Python package dependencies
- **.gitignore**: Files and directories to exclude from git

### Benchmark Scripts

All benchmark scripts support command-line arguments and save results to structured output directories.

**Main Results**:
- `main_comparison.py`: Reproduces Table 1 (latency) and Table 2 (quality metrics)
  - Compares SpectralCache, TeaCache, FBCache, FastCache, and baseline
  - Outputs: JSON results, comparison plots

**Ablation Study**:
- `spectralcache_ablation.py`: Reproduces Table 3 (ablation study)
  - Tests all 7 combinations of TADS, CEB, FDC
  - Outputs: Ablation results table, speedup/quality plots

**Threshold Sensitivity**:
- `spectral_sweep.py`: Reproduces Table 4 (threshold sensitivity)
  - Sweeps tau from 0.3 to 0.8
  - Outputs: Quality-speed tradeoff curves

**Quality Evaluation**:
- `quality_eval.py`: Computes LPIPS, SSIM, PSNR metrics
  - Pairwise comparison against uncached baseline
  - Supports multiple cache methods

**Error Analysis**:
- `cache_error_analysis.py`: Analyzes error accumulation patterns
  - Consecutive vs. random caching comparison
  - Per-timestep error tracking

### Examples

- `basic_usage.py`: Minimal example showing how to use SpectralCache
  - Loads FLUX.1-schnell
  - Runs baseline and SpectralCache
  - Reports speedup and cache statistics

### Paper

- `main.tex`: Complete paper in single LaTeX file (606 lines)
  - All sections: Abstract, Introduction, Related Work, Motivation, Method, Experiments, Conclusion
  - Includes TikZ figures and algorithm pseudocode
- `appendix.tex`: Supplementary material with theorem proofs
- `references.bib`: Bibliography with 13 references
- `*.pdf`: Three main figures (temporal sensitivity, error cascade, spectral analysis)
- `qual_comparison/`: 12 qualitative comparison images (4 prompts × 3 methods)

### Documentation

- `IMPLEMENTATION.md`: Technical deep-dive
  - Component-by-component implementation details
  - Code examples for TADS, CEB, FDC
  - Hyperparameter tuning guidelines
  - Integration with xFuser
  - Performance characteristics
- `spectralcache.md`: Method overview and usage guide

## Reproducing Paper Results

### Table 1 & 2: Main Results

```bash
python benchmark/main_comparison.py \
    --model_type flux \
    --num_inference_steps 20 \
    --height 512 --width 512 \
    --output_dir results/main
```

### Table 3: Ablation Study

```bash
python benchmark/spectralcache_ablation.py \
    --model_type flux \
    --num_inference_steps 20 \
    --output_dir results/ablation
```

### Table 4: Threshold Sensitivity

```bash
python benchmark/spectral_sweep.py \
    --model_type flux \
    --thresholds 0.3 0.4 0.5 0.6 0.8 \
    --output_dir results/threshold
```

### Figure: Qualitative Comparison

```bash
python benchmark/quality_eval.py \
    --model_type flux \
    --cache_methods spectralcache teacache \
    --num_samples 4 \
    --output_dir results/qualitative
```

## Development Workflow

1. **Setup**: Install dependencies from `requirements.txt`
2. **Implement**: Add features following `CONTRIBUTING.md` guidelines
3. **Test**: Run benchmark scripts to verify correctness
4. **Document**: Update relevant documentation files
5. **Submit**: Open PR with clear description

## Notes

- All benchmark scripts save results to JSON for reproducibility
- Paper figures are generated from benchmark results
- Example scripts demonstrate basic usage patterns
- Documentation provides both high-level overview and implementation details
