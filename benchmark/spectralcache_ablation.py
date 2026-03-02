#!/usr/bin/env python3
"""
SpectralCache 消融实验脚本
复现论文表 4：TADS/CEB/FDC 的 7 种组合

运行方式：
python benchmark/spectralcache_ablation.py --model_type flux --num_inference_steps 20
"""

import argparse
import torch
import time
import os
from pathlib import Path
from diffusers import FluxPipeline, PixArtSigmaPipeline
from xfuser.model_executor.pipelines.fastcache_pipeline import xFuserFastCachePipelineWrapper
import numpy as np
from PIL import Image
import json

# 导入质量评估指标
try:
    import lpips
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: lpips or scikit-image not installed. Quality metrics will be skipped.")
    METRICS_AVAILABLE = False


def load_model(model_type: str, model_path: str = None):
    """加载模型"""
    if model_type == "flux":
        model_path = model_path or "black-forest-labs/FLUX.1-schnell"
        pipe = FluxPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        )
    elif model_type == "pixart":
        model_path = model_path or "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
        pipe = PixArtSigmaPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    pipe = pipe.to("cuda")
    return pipe


def compute_quality_metrics(img1: np.ndarray, img2: np.ndarray):
    """计算图像质量指标"""
    if not METRICS_AVAILABLE:
        return {"lpips": 0.0, "ssim": 0.0, "psnr": 0.0}

    # LPIPS
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    img1_t = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    img2_t = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float() / 255.0 * 2 - 1
    img1_t = img1_t.cuda()
    img2_t = img2_t.cuda()

    with torch.no_grad():
        lpips_val = lpips_fn(img1_t, img2_t).item()

    # SSIM and PSNR
    ssim_val = ssim(img1, img2, channel_axis=2, data_range=255)
    psnr_val = psnr(img1, img2, data_range=255)

    return {
        "lpips": lpips_val,
        "ssim": ssim_val,
        "psnr": psnr_val,
    }


def run_ablation_config(
    pipe,
    prompt: str,
    num_inference_steps: int,
    seed: int,
    enable_tads: bool,
    enable_ceb: bool,
    enable_fdc: bool,
    baseline_image: np.ndarray = None,
    **kwargs
):
    """运行单个消融配置"""
    from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer

    # 应用 SpectralCache 到 transformer
    apply_cache_on_transformer(
        pipe.transformer,
        rel_l1_thresh=0.6,  # 论文中的 τ = 0.6
        return_hidden_states_first=False,
        num_steps=num_inference_steps,
        use_cache="SpectralCache",
        # TADS 参数
        tads_s_min=0.5,
        tads_s_max=1.5,
        enable_tads=enable_tads,
        # CEB 参数
        ceb_c_max=2,  # 论文中的 C_max = 2
        enable_ceb=enable_ceb,
        # FDC 参数
        fdc_freq_ratio=0.5,
        gamma_low=0.8,
        gamma_high=1.5,
        enable_fdc=enable_fdc,
    )

    # 运行推理并测量时间
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Warmup
    _ = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **kwargs
    )

    # 实际测量
    torch.cuda.synchronize()
    start_time = time.time()

    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        generator=generator,
        **kwargs
    )

    torch.cuda.synchronize()
    elapsed_time = time.time() - start_time

    # 获取缓存统计
    # 从 transformer_blocks 中获取统计信息
    cache_hits = 0
    total_steps = 0
    if hasattr(pipe.transformer, 'transformer_blocks'):
        for block in pipe.transformer.transformer_blocks:
            if hasattr(block, 'get_cache_statistics'):
                stats = block.get_cache_statistics()
                cache_hits += stats.get("cache_hits", 0)
                total_steps += stats.get("total_steps", 0)

    cache_hit_rate = cache_hits / total_steps if total_steps > 0 else 0.0

    # 转换图像为 numpy
    image = result.images[0]
    image_np = np.array(image)

    # 计算质量指标（与 baseline 对比）
    quality_metrics = {}
    if baseline_image is not None:
        quality_metrics = compute_quality_metrics(image_np, baseline_image)

    return {
        "time": elapsed_time,
        "cache_hit_rate": cache_hit_rate,
        "image": image,
        "image_np": image_np,
        **quality_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="SpectralCache Ablation Study")
    parser.add_argument("--model_type", type=str, default="flux", choices=["flux", "pixart"])
    parser.add_argument("--model", type=str, default=None, help="Model path")
    parser.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on the moon")
    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="spectralcache_ablation_results")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples for quality evaluation")

    args = parser.parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载模型
    print(f"Loading {args.model_type} model...")
    pipe = load_model(args.model_type, args.model)

    # 定义消融配置（论文表 4 的 8 种配置）
    ablation_configs = [
        {"name": "Baseline (No Cache)", "tads": False, "ceb": False, "fdc": False, "use_cache": False},
        {"name": "None", "tads": False, "ceb": False, "fdc": False, "use_cache": True},
        {"name": "TADS", "tads": True, "ceb": False, "fdc": False, "use_cache": True},
        {"name": "CEB", "tads": False, "ceb": True, "fdc": False, "use_cache": True},
        {"name": "FDC", "tads": False, "ceb": False, "fdc": True, "use_cache": True},
        {"name": "TADS+CEB", "tads": True, "ceb": True, "fdc": False, "use_cache": True},
        {"name": "TADS+FDC", "tads": True, "ceb": False, "fdc": True, "use_cache": True},
        {"name": "CEB+FDC", "tads": False, "ceb": True, "fdc": True, "use_cache": True},
        {"name": "TADS+CEB+FDC (Full)", "tads": True, "ceb": True, "fdc": True, "use_cache": True},
    ]

    # 运行实验
    results = []
    baseline_image = None

    kwargs = {"height": args.height, "width": args.width}
    if args.model_type == "flux":
        kwargs["guidance_scale"] = 0.0  # FLUX.1-schnell 不使用 CFG

    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Running: {config['name']}")
        print(f"{'='*60}")

        if not config["use_cache"]:
            # Baseline: 不使用缓存
            generator = torch.Generator(device="cuda").manual_seed(args.seed)

            # Warmup
            _ = pipe(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                **kwargs
            )

            # 实际测量
            torch.cuda.synchronize()
            start_time = time.time()

            result = pipe(
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                generator=generator,
                **kwargs
            )

            torch.cuda.synchronize()
            elapsed_time = time.time() - start_time

            baseline_image = np.array(result.images[0])

            result_dict = {
                "config": config["name"],
                "time": elapsed_time,
                "speedup": 1.0,
                "cache_hit_rate": 0.0,
                "lpips": 0.0,
                "ssim": 1.0,
                "psnr": float('inf'),
            }

            # 保存图像
            result.images[0].save(output_dir / f"{config['name'].replace(' ', '_').replace('+', '_')}.png")
        else:
            # 使用缓存
            result_dict = run_ablation_config(
                pipe=pipe,
                prompt=args.prompt,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                enable_tads=config["tads"],
                enable_ceb=config["ceb"],
                enable_fdc=config["fdc"],
                baseline_image=baseline_image,
                **kwargs
            )

            result_dict["config"] = config["name"]
            result_dict["speedup"] = results[0]["time"] / result_dict["time"] if results else 1.0

            # 保存图像
            result_dict["image"].save(output_dir / f"{config['name'].replace(' ', '_').replace('+', '_')}.png")

        results.append(result_dict)

        print(f"Time: {result_dict['time']:.2f}s")
        print(f"Speedup: {result_dict['speedup']:.2f}x")
        if result_dict.get("cache_hit_rate"):
            print(f"Cache Hit Rate: {result_dict['cache_hit_rate']:.2%}")
        if result_dict.get("lpips"):
            print(f"LPIPS: {result_dict['lpips']:.3f}")
            print(f"SSIM: {result_dict['ssim']:.3f}")
            print(f"PSNR: {result_dict['psnr']:.2f}")

    # 保存结果
    results_json = []
    for r in results:
        results_json.append({
            "config": r["config"],
            "time": r["time"],
            "speedup": r["speedup"],
            "cache_hit_rate": r.get("cache_hit_rate", 0.0),
            "lpips": r.get("lpips", 0.0),
            "ssim": r.get("ssim", 1.0),
            "psnr": r.get("psnr", float('inf')),
        })

    with open(output_dir / "results.json", "w") as f:
        json.dump(results_json, f, indent=2)

    # 打印表格
    print(f"\n{'='*80}")
    print("Ablation Study Results (Table 4)")
    print(f"{'='*80}")
    print(f"{'Config':<25} {'Time (s)':<10} {'Speedup':<10} {'LPIPS↓':<10} {'SSIM↑':<10}")
    print(f"{'-'*80}")
    for r in results_json:
        print(f"{r['config']:<25} {r['time']:<10.2f} {r['speedup']:<10.2f} "
              f"{r['lpips']:<10.3f} {r['ssim']:<10.3f}")

    print(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
