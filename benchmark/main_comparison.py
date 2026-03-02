"""Main comparison: SpectralCache vs TeaCache vs FBCache vs No Cache, threshold=0.5."""
import argparse, json, os, sys, time, types, importlib.util, gc

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

_stub_dist = types.ModuleType('xfuser.core.distributed')
_stub_dist.get_sp_group = lambda: None
_stub_dist.get_sequence_parallel_world_size = lambda: 1
sys.modules['xfuser'] = types.ModuleType('xfuser')
sys.modules['xfuser.core'] = types.ModuleType('xfuser.core')
sys.modules['xfuser.core.distributed'] = _stub_dist

_load_module('xfuser.model_executor', os.path.join(_project_root, 'xfuser', 'model_executor', '__init__.py'))
_load_module('xfuser.model_executor.cache', os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', '__init__.py'))
_load_module('xfuser.model_executor.cache.utils', os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', 'utils.py'))

_stub_models = types.ModuleType('xfuser.model_executor.models')
sys.modules['xfuser.model_executor.models'] = _stub_models
_stub_transformers = types.ModuleType('xfuser.model_executor.models.transformers')
sys.modules['xfuser.model_executor.models.transformers'] = _stub_transformers
_stub_flux_wrapper = types.ModuleType('xfuser.model_executor.models.transformers.transformer_flux')
_stub_flux_wrapper.xFuserFluxTransformer2DWrapper = None
sys.modules['xfuser.model_executor.models.transformers.transformer_flux'] = _stub_flux_wrapper

sys.modules['xfuser.model_executor.cache.diffusers_adapters'] = types.ModuleType('xfuser.model_executor.cache.diffusers_adapters')
_load_module('xfuser.model_executor.cache.diffusers_adapters.registry',
             os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', 'diffusers_adapters', 'registry.py'))
flux_adapter = _load_module('xfuser.model_executor.cache.diffusers_adapters.flux',
             os.path.join(_project_root, 'xfuser', 'model_executor', 'cache', 'diffusers_adapters', 'flux.py'))
apply_cache_on_transformer = flux_adapter.apply_cache_on_transformer

import torch
import numpy as np
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

METHODS = {
    "Tea": {"use_cache": "Tea", "rel_l1_thresh": 0.5},
    "Fb": {"use_cache": "Fb", "rel_l1_thresh": 0.5},
    "SpectralCache": {"use_cache": "SpectralCache", "rel_l1_thresh": 0.5},
}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/dev_share/gdli7/models/checkpoints/black-forest-labs/FLUX___1-schnell")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="main_comparison_results")
    return p.parse_args()

def generate_images(pipe, args):
    images = []
    for prompt in PROMPTS:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.no_grad():
            img = pipe(prompt=prompt, num_inference_steps=args.steps,
                       height=512, width=512, generator=gen).images[0]
        images.append(img)
    return images

def compute_metrics(baseline_imgs, test_imgs, lpips_fn):
    lp_scores, ss_scores, ps_scores = [], [], []
    for ref, gen in zip(baseline_imgs, test_imgs):
        ref_np = np.array(ref).astype(np.float32) / 255.0
        gen_np = np.array(gen).astype(np.float32) / 255.0
        ref_t = torch.from_numpy(ref_np).permute(2,0,1).unsqueeze(0).cuda() * 2 - 1
        gen_t = torch.from_numpy(gen_np).permute(2,0,1).unsqueeze(0).cuda() * 2 - 1
        with torch.no_grad():
            lp_scores.append(lpips_fn(ref_t, gen_t).item())
        ss_scores.append(ssim(ref_np, gen_np, channel_axis=2, data_range=1.0))
        ps_scores.append(psnr(ref_np, gen_np, data_range=1.0))
    return float(np.mean(lp_scores)), float(np.mean(ss_scores)), float(np.mean(ps_scores))

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    lpips_fn = lpips.LPIPS(net='alex').cuda()

    # Baseline
    print("Loading model...")
    pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    print("Generating baseline (No Cache)...")
    t0 = time.time()
    baseline_imgs = generate_images(pipe, args)
    baseline_time = (time.time() - t0) / len(PROMPTS)
    print(f"  Baseline: {baseline_time:.2f}s/img")
    del pipe; gc.collect(); torch.cuda.empty_cache()

    results = {"baseline_time": baseline_time}

    for name, cfg in METHODS.items():
        print(f"\n===== {name} =====")
        pipe = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
        apply_cache_on_transformer(
            pipe.transformer,
            use_cache=cfg["use_cache"],
            rel_l1_thresh=cfg["rel_l1_thresh"],
            return_hidden_states_first=False,
            num_steps=args.steps,
        )
        t0 = time.time()
        test_imgs = generate_images(pipe, args)
        elapsed = (time.time() - t0) / len(PROMPTS)
        lp, ss, ps = compute_metrics(baseline_imgs, test_imgs, lpips_fn)
        speedup = baseline_time / elapsed if elapsed > 0 else 1.0
        results[name] = {"avg_time": elapsed, "speedup": speedup, "lpips": lp, "ssim": ss, "psnr": ps}
        print(f"  Time: {elapsed:.2f}s, Speedup: {speedup:.2f}x")
        print(f"  LPIPS: {lp:.3f}, SSIM: {ss:.3f}, PSNR: {ps:.2f}")
        del pipe; gc.collect(); torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, "main_comparison.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

if __name__ == "__main__":
    main()
