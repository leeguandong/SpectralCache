"""Ablation study: test all 8 combinations of TADS/CEB/FDC."""
import argparse, json, os, sys, time, itertools, types, importlib.util

# Direct-load only the modules we need, bypassing xfuser's heavy import tree.
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Stub xfuser.core.distributed (used by cache/utils.py)
_stub_dist = types.ModuleType('xfuser.core.distributed')
_stub_dist.get_sp_group = lambda: None
_stub_dist.get_sequence_parallel_world_size = lambda: 1
sys.modules['xfuser'] = types.ModuleType('xfuser')
sys.modules['xfuser.core'] = types.ModuleType('xfuser.core')
sys.modules['xfuser.core.distributed'] = _stub_dist

# Load cache/utils.py
_load_module('xfuser.model_executor', os.path.join(_project_root, 'xfuser', 'model_executor', '__init__.py'))
_load_module('xfuser.model_executor.cache', os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', '__init__.py'))
cache_utils = _load_module('xfuser.model_executor.cache.utils', os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', 'utils.py'))

# Load registry (stub out the model wrapper it tries to import)
_stub_models = types.ModuleType('xfuser.model_executor.models')
sys.modules['xfuser.model_executor.models'] = _stub_models
_stub_transformers = types.ModuleType('xfuser.model_executor.models.transformers')
sys.modules['xfuser.model_executor.models.transformers'] = _stub_transformers
_stub_flux_wrapper = types.ModuleType('xfuser.model_executor.models.transformers.transformer_flux')
_stub_flux_wrapper.xFuserFluxTransformer2DWrapper = None
sys.modules['xfuser.model_executor.models.transformers.transformer_flux'] = _stub_flux_wrapper

sys.modules['xfuser.model_executor.cache.diffusers_adapters'] = types.ModuleType('xfuser.model_executor.cache.diffusers_adapters')
registry = _load_module('xfuser.model_executor.cache.diffusers_adapters.registry',
                        os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', 'diffusers_adapters', 'registry.py'))
flux_adapter = _load_module('xfuser.model_executor.cache.diffusers_adapters.flux',
                            os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', 'diffusers_adapters', 'flux.py'))

apply_cache_on_transformer = flux_adapter.apply_cache_on_transformer

import torch
import numpy as np
from PIL import Image
import lpips
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from diffusers import FluxPipeline

PROMPTS = [
    "a photo of an astronaut riding a horse on the moon",
    "a cute cat sitting on a stack of books in a cozy library",
    "a futuristic cityscape at sunset with flying cars",
    "a detailed oil painting of a medieval castle on a cliff",
    "a close-up portrait of an elderly man with deep wrinkles",
    "a tropical beach with crystal clear water and palm trees",
    "a steampunk robot playing violin in a concert hall",
    "a bowl of ramen with steam rising, food photography",
    "a snowy mountain landscape with northern lights",
    "a colorful street market in Morocco with spices and textiles",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/dev_share/gdli7/models/checkpoints/black-forest-labs/FLUX___1-schnell")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--output_dir", type=str, default="ablation_results")
    return p.parse_args()

def generate_images(model, args):
    images = []
    for prompt in PROMPTS:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.no_grad():
            img = model(prompt=prompt, num_inference_steps=args.steps,
                       height=args.height, width=args.width, generator=gen).images[0]
        images.append(img)
    return images

def compute_metrics(baseline_imgs, test_imgs, lpips_fn):
    lpips_scores, ssim_scores, psnr_scores = [], [], []
    for ref, gen in zip(baseline_imgs, test_imgs):
        ref_np = np.array(ref).astype(np.float32) / 255.0
        gen_np = np.array(gen).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_np).permute(2,0,1).unsqueeze(0).cuda() * 2 - 1
        gen_t = torch.from_numpy(gen_np).permute(2,0,1).unsqueeze(0).cuda() * 2 - 1
        with torch.no_grad():
            lpips_scores.append(lpips_fn(ref_t, gen_t).item())
        ssim_scores.append(ssim(ref_np, gen_np, channel_axis=2, data_range=1.0))
        psnr_scores.append(psnr(ref_np, gen_np, data_range=1.0))
    return np.mean(lpips_scores), np.mean(ssim_scores), np.mean(psnr_scores)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    # Load model
    print("Loading model...")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    # Generate baseline (no cache)
    print("Generating baseline (no cache)...")
    t0 = time.time()
    baseline_imgs = generate_images(pipe, args)
    baseline_time = (time.time() - t0) / len(PROMPTS)
    print(f"Baseline avg time per image: {baseline_time:.2f}s")

    # All 8 ablation configs: (enable_tads, enable_ceb, enable_fdc)
    configs = list(itertools.product([False, True], repeat=3))
    results = {"baseline_time": baseline_time}

    for tads, ceb, fdc in configs:
        label = f"T{'1' if tads else '0'}_C{'1' if ceb else '0'}_F{'1' if fdc else '0'}"
        desc = []
        if tads: desc.append("TADS")
        if ceb: desc.append("CEB")
        if fdc: desc.append("FDC")
        desc_str = "+".join(desc) if desc else "Base"
        print(f"\n{'='*50}")
        print(f"Config: {desc_str} (TADS={tads}, CEB={ceb}, FDC={fdc})")
        print(f"{'='*50}")

        # Apply SpectralCache with toggles
        apply_cache_on_transformer(
            pipe.transformer,
            use_cache="SpectralCache",
            cache_threshold=args.threshold,
            num_steps=args.steps,
            enable_tads=tads,
            enable_ceb=ceb,
            enable_fdc=fdc,
        )

        # Generate and time
        t0 = time.time()
        test_imgs = generate_images(pipe, args)
        elapsed = time.time() - t0
        avg_time = elapsed / len(PROMPTS)

        # Compute metrics
        lp, ss, ps = compute_metrics(baseline_imgs, test_imgs, lpips_fn)

        speedup = baseline_time / avg_time if avg_time > 0 else 1.0

        results[label] = {
            "tads": tads, "ceb": ceb, "fdc": fdc,
            "desc": desc_str,
            "avg_time": float(avg_time),
            "speedup": float(speedup),
            "lpips": float(lp), "ssim": float(ss), "psnr": float(ps),
        }
        print(f"  Time: {avg_time:.2f}s, Speedup: {speedup:.2f}x")
        print(f"  LPIPS: {lp:.3f}, SSIM: {ss:.3f}, PSNR: {ps:.2f}")

        # Remove cache for next config
        pipe.transformer._original_forward = None
        # Re-load to clear cache state
        del pipe
        torch.cuda.empty_cache()
        pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")

    # Save results
    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Config':<15} {'Speedup':>8} {'LPIPS':>8} {'SSIM':>8} {'PSNR':>8}")
    print(f"{'-'*70}")
    for key, v in results.items():
        if key == "baseline_time":
            continue
        print(f"{v['desc']:<15} {v['speedup']:>7.2f}x {v['lpips']:>8.3f} {v['ssim']:>8.3f} {v['psnr']:>8.2f}")

if __name__ == "__main__":
    main()
