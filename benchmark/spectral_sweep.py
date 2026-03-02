"""
SpectralCache parameter sweep: test different rel_l1_thresh values.
Reuses baseline images from quality_eval_results_v2/.
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

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.8]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/dev_share/gdli7/models/checkpoints/black-forest-labs/FLUX___1-schnell")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline_dir", type=str, default="quality_eval_results_v2")
    p.add_argument("--output_dir", type=str, default="spectral_sweep_results")
    return p.parse_args()


def img_to_tensor(img):
    t = T.ToTensor()(img).unsqueeze(0).cuda()
    return t * 2 - 1

def img_to_tensor_01(img):
    return T.ToTensor()(img).unsqueeze(0).cuda()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load baseline images
    baseline_imgs = []
    for i in range(len(PROMPTS)):
        path = os.path.join(args.baseline_dir, f"none_{i:02d}.png")
        baseline_imgs.append(Image.open(path))
    print(f"Loaded {len(baseline_imgs)} baseline images from {args.baseline_dir}")

    # Init metrics
    lpips_fn = lpips.LPIPS(net='alex').cuda()
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    psnr_fn = PeakSignalNoiseRatio(data_range=1.0).cuda()

    results = {}

    for thresh in THRESHOLDS:
        print(f"\n===== SpectralCache thresh={thresh} =====")
        model = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
        apply_cache_on_transformer(
            model.transformer,
            rel_l1_thresh=thresh,
            return_hidden_states_first=False,
            num_steps=args.steps,
            use_cache="SpectralCache",
            motion_threshold=0.1,
        )

        images = []
        t0 = time.time()
        for i, prompt in enumerate(PROMPTS):
            gen = torch.Generator(device="cuda").manual_seed(args.seed)
            with torch.no_grad():
                result = model(prompt=prompt, num_inference_steps=args.steps,
                               height=args.height, width=args.width, generator=gen)
            img = result.images[0]
            img.save(os.path.join(args.output_dir, f"spectral_{thresh}_{i:02d}.png"))
            images.append(img)
        elapsed = time.time() - t0
        avg_time = elapsed / len(PROMPTS)

        # Compute metrics
        lp_scores, ss_scores, ps_scores = [], [], []
        for i in range(len(PROMPTS)):
            ref_t = img_to_tensor(baseline_imgs[i])
            gen_t = img_to_tensor(images[i])
            ref_01 = img_to_tensor_01(baseline_imgs[i])
            gen_01 = img_to_tensor_01(images[i])
            with torch.no_grad():
                lp_scores.append(lpips_fn(ref_t, gen_t).item())
                ss_scores.append(ssim_fn(gen_01, ref_01).item())
                ps_scores.append(psnr_fn(gen_01, ref_01).item())
            ssim_fn.reset(); psnr_fn.reset()

        baseline_avg = 4.87  # from previous run
        spd = baseline_avg / avg_time

        results[str(thresh)] = {
            "speedup": spd, "avg_time": avg_time,
            "lpips": float(np.mean(lp_scores)), "lpips_std": float(np.std(lp_scores)),
            "ssim": float(np.mean(ss_scores)), "ssim_std": float(np.std(ss_scores)),
            "psnr": float(np.mean(ps_scores)), "psnr_std": float(np.std(ps_scores)),
        }
        print(f"  Speedup: {spd:.2f}x, LPIPS: {np.mean(lp_scores):.4f}, "
              f"SSIM: {np.mean(ss_scores):.4f}, PSNR: {np.mean(ps_scores):.2f}")

        del model; gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SpectralCache Parameter Sweep ({args.height}x{args.width}, {args.steps} steps)")
    print(f"{'='*70}")
    print(f"{'Threshold':<12} {'Speedup':>8} {'LPIPS↓':>10} {'SSIM↑':>10} {'PSNR↑':>10}")
    print("-" * 52)
    for thresh in THRESHOLDS:
        r = results[str(thresh)]
        print(f"{thresh:<12} {r['speedup']:>7.2f}x {r['lpips']:>9.4f} "
              f"{r['ssim']:>9.4f} {r['psnr']:>9.2f}")

    # Save
    with open(os.path.join(args.output_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output_dir}/sweep_results.json")


if __name__ == "__main__":
    main()
