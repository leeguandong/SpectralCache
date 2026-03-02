"""
SpectralCache Example: Basic Usage

This script demonstrates how to use SpectralCache for accelerating
FLUX.1-schnell image generation.
"""

import torch
from diffusers import FluxPipeline
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
import time


def main():
    # Configuration
    model_id = "black-forest-labs/FLUX.1-schnell"
    prompt = "a photo of an astronaut riding a horse on the moon"
    num_inference_steps = 20
    height = 512
    width = 512
    seed = 42

    # SpectralCache hyperparameters
    spectralcache_config = {
        'tau': 0.6,              # Base cache threshold
        's_min': 0.5,            # TADS min scale
        's_max': 1.5,            # TADS max scale
        'C_max': 2,              # CEB consecutive limit
        'gamma_low': 0.8,        # FDC low-band scale
        'gamma_high': 1.5,       # FDC high-band scale
        'r': 0.5                 # FDC frequency split ratio
    }

    print("=" * 60)
    print("SpectralCache Example")
    print("=" * 60)
    print(f"Model: {model_id}")
    print(f"Prompt: {prompt}")
    print(f"Steps: {num_inference_steps}")
    print(f"Resolution: {height}x{width}")
    print(f"Seed: {seed}")
    print()

    # Load model
    print("Loading model...")
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16
    ).to("cuda")

    # Set seed for reproducibility
    generator = torch.Generator("cuda").manual_seed(seed)

    # Baseline: No caching
    print("\n" + "-" * 60)
    print("Running baseline (no caching)...")
    print("-" * 60)
    start_time = time.time()
    baseline_image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    baseline_time = time.time() - start_time
    print(f"Baseline time: {baseline_time:.2f}s")
    baseline_image.save("output_baseline.png")
    print("Saved: output_baseline.png")

    # SpectralCache
    print("\n" + "-" * 60)
    print("Running with SpectralCache...")
    print("-" * 60)
    print(f"Config: {spectralcache_config}")

    # Wrap pipeline with SpectralCache
    wrapper = xFuserFastCachePipelineWrapper(pipe)
    wrapper.enable_spectralcache(**spectralcache_config)

    # Reset seed
    generator = torch.Generator("cuda").manual_seed(seed)

    start_time = time.time()
    spectralcache_image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        height=height,
        width=width,
        generator=generator,
    ).images[0]
    spectralcache_time = time.time() - start_time
    print(f"SpectralCache time: {spectralcache_time:.2f}s")
    spectralcache_image.save("output_spectralcache.png")
    print("Saved: output_spectralcache.png")

    # Get cache statistics
    stats = wrapper.get_cache_statistics()
    print(f"\nCache statistics:")
    print(f"  Cache hit rate: {stats['cache_hit_rate']:.1%}")
    print(f"  Total timesteps: {stats['total_timesteps']}")
    print(f"  Cached timesteps: {stats['cached_timesteps']}")

    # Summary
    speedup = baseline_time / spectralcache_time
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Baseline time:      {baseline_time:.2f}s")
    print(f"SpectralCache time: {spectralcache_time:.2f}s")
    print(f"Speedup:            {speedup:.2f}×")
    print(f"Cache hit rate:     {stats['cache_hit_rate']:.1%}")
    print("=" * 60)


if __name__ == "__main__":
    main()
