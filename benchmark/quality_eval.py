"""
Quality evaluation: 10-image pairwise comparison against uncached baseline.
Computes LPIPS, SSIM, PSNR for each cache method vs No Cache.
"""
import os, sys, time, json, gc, argparse
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import lpips
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

from diffusers import FluxPipeline
from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer

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

METHODS = ["None", "Fast", "SpectralCache", "Fb", "Tea"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/dev_share/gdli7/models/checkpoints/black-forest-labs/FLUX___1-schnell")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", type=str, default="quality_eval_results")
    p.add_argument("--cache_threshold", type=float, default=0.15)
    return p.parse_args()


def load_model(args):
    return FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")


def get_threshold(method, default_thresh):
    """TeaCache and SpectralCache use polynomial rescaling, need higher threshold."""
    if method in ("Tea", "SpectralCache"):
        return 0.6
    return default_thresh


def apply_cache(model, method, args):
    if method == "None":
        return model
    thresh = get_threshold(method, args.cache_threshold)
    apply_cache_on_transformer(
        model.transformer,
        rel_l1_thresh=thresh,
        return_hidden_states_first=False,
        num_steps=args.steps,
        use_cache=method,
        motion_threshold=0.1,
    )
    return model


def generate_image(model, prompt, args):
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    with torch.no_grad():
        result = model(prompt=prompt, num_inference_steps=args.steps,
                       height=args.height, width=args.width, generator=gen)
    return result.images[0]


def img_to_tensor(img):
    """PIL Image -> [1, 3, H, W] float tensor in [-1, 1] for LPIPS."""
    t = T.ToTensor()(img).unsqueeze(0).cuda()  # [0,1]
    return t * 2 - 1  # [-1,1]


def img_to_tensor_01(img):
    """PIL Image -> [1, 3, H, W] float tensor in [0, 1] for SSIM/PSNR."""
    return T.ToTensor()(img).unsqueeze(0).cuda()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Init metrics
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()

    all_results = {}

    for method in METHODS:
        print(f"\n===== Generating with {method} =====")
        model = load_model(args)
        model = apply_cache(model, method, args)

        images = []
        t0 = time.time()
        for i, prompt in enumerate(PROMPTS):
            img = generate_image(model, prompt, args)
            img.save(os.path.join(args.output_dir, f"{method.lower()}_{i:02d}.png"))
            images.append(img)
            print(f"  [{i+1}/{len(PROMPTS)}] done")
        elapsed = time.time() - t0
        print(f"  Total: {elapsed:.1f}s, avg: {elapsed/len(PROMPTS):.1f}s/img")

        all_results[method] = {"images": images, "time": elapsed}
        del model; gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

    # Compute metrics: each cached method vs baseline
    baseline_imgs = all_results["None"]["images"]
    metrics = {}

    for method in METHODS:
        if method == "None":
            continue
        lpips_scores, ssim_scores, psnr_scores = [], [], []
        for i in range(len(PROMPTS)):
            ref = baseline_imgs[i]
            gen = all_results[method]["images"][i]

            ref_t = img_to_tensor(ref)
            gen_t = img_to_tensor(gen)
            ref_01 = img_to_tensor_01(ref)
            gen_01 = img_to_tensor_01(gen)

            with torch.no_grad():
                lp = lpips_fn(ref_t, gen_t).item()
                ss = ssim_fn(gen_01, ref_01).item()
                ps = psnr_fn(gen_01, ref_01).item()

            lpips_scores.append(lp)
            ssim_scores.append(ss)
            psnr_scores.append(ps)

            # Reset metric states
            ssim_fn.reset()
            psnr_fn.reset()

        metrics[method] = {
            "lpips_mean": np.mean(lpips_scores),
            "lpips_std": np.std(lpips_scores),
            "ssim_mean": np.mean(ssim_scores),
            "ssim_std": np.std(ssim_scores),
            "psnr_mean": np.mean(psnr_scores),
            "psnr_std": np.std(psnr_scores),
            "avg_time": all_results[method]["time"] / len(PROMPTS),
        }

    # Print results
    baseline_avg = all_results["None"]["time"] / len(PROMPTS)
    print(f"\n{'='*70}")
    print(f"Quality Evaluation Results ({args.height}x{args.width}, {args.steps} steps)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'Speedup':>8} {'LPIPS↓':>10} {'SSIM↑':>10} {'PSNR↑':>10}")
    print("-" * 52)
    print(f"{'No Cache':<12} {'1.00x':>8} {'---':>10} {'---':>10} {'---':>10}")
    for method in ["Fast", "SpectralCache", "Fb", "Tea"]:
        m = metrics[method]
        spd = baseline_avg / m["avg_time"]
        print(f"{method+'Cache':<12} {spd:>7.2f}x "
              f"{m['lpips_mean']:>9.4f} {m['ssim_mean']:>9.4f} {m['psnr_mean']:>9.2f}")

    # Save JSON
    save_data = {"config": vars(args), "metrics": metrics,
                 "baseline_avg_time": baseline_avg}
    with open(os.path.join(args.output_dir, "quality_metrics.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/quality_metrics.json")


if __name__ == "__main__":
    main()

