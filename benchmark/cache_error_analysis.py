"""
Per-timestep cache error analysis.
Instruments SpectralCache to measure actual approximation error at each timestep.
Runs both cached and full computation paths to compare.
"""
import os, sys, json, time, argparse
import torch
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from diffusers import FluxPipeline
from xfuser.model_executor.cache.diffusers_adapters.flux import apply_cache_on_transformer
from xfuser.model_executor.cache.utils import (
    FastCachePPTransformerBlocks, FastCachedTransformerBlocks,
    FBCachedTransformerBlocks, TeaCachedTransformerBlocks,
)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str,
                   default="/dev_share/gdli7/models/checkpoints/black-forest-labs/FLUX___1-schnell")
    p.add_argument("--height", type=int, default=512)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", type=str, default="a photo of an astronaut riding a horse on the moon")
    p.add_argument("--output_dir", type=str, default="error_analysis_results")
    p.add_argument("--cache_threshold", type=float, default=0.15)
    return p.parse_args()


CACHE_CLASSES = {
    "SpectralCache": FastCachePPTransformerBlocks,
    "Fast": FastCachedTransformerBlocks,
    "Fb": FBCachedTransformerBlocks,
    "Tea": TeaCachedTransformerBlocks,
}


def find_cached_blocks_in_closure(transformer, target_cls):
    """Extract cached blocks from the patched forward method's closure.
    apply_cache_on_transformer stores them in a closure, not as a direct attribute."""
    forward_fn = getattr(transformer.forward, '__func__', transformer.forward)
    closure = getattr(forward_fn, '__closure__', None)
    if closure is None:
        return None
    for cell in closure:
        try:
            contents = cell.cell_contents
            if isinstance(contents, torch.nn.ModuleList):
                for mod in contents:
                    if isinstance(mod, target_cls):
                        return mod
        except ValueError:
            continue
    return None


def instrument_cached_blocks(model, method_name):
    """Find the CachedTransformerBlocks in the model and monkey-patch forward
    to log per-timestep cache decisions and errors."""
    target_cls = CACHE_CLASSES.get(method_name)
    if target_cls is None:
        return None, []

    log = []  # list of per-timestep dicts

    # Find the cached blocks module (stored in closure by apply_cache_on_transformer)
    cached_blocks = find_cached_blocks_in_closure(model.transformer, target_cls)

    if cached_blocks is None:
        print(f"Warning: could not find {target_cls.__name__} in model closure")
        return None, log

    original_forward = cached_blocks.forward.__func__

    def instrumented_forward(self, hidden_states, encoder_hidden_states, *args, **kwargs):
        # Run the normal forward (may use cache)
        result = original_forward(self, hidden_states, encoder_hidden_states, *args, **kwargs)
        cached_hidden = result[0] if self.return_hidden_states_first else result[1]

        # Also run full computation to get ground truth
        orig_h = self.cache_context.original_hidden_states
        orig_e = self.cache_context.original_encoder_hidden_states
        if orig_h is not None:
            with torch.no_grad():
                # Run all blocks from scratch
                h, e = orig_h, orig_e
                for block in self.transformer_blocks:
                    h, e = block(h, e, *args, **kwargs)
                    h, e = (h, e) if self.return_hidden_states_first else (e, h)
                if self.single_transformer_blocks:
                    for block in self.single_transformer_blocks:
                        e, h = block(h, e, *args, **kwargs)
                full_hidden = h

                # Compute error metrics
                abs_err = torch.mean(torch.abs(cached_hidden - full_hidden)).item()
                rel_err = abs_err / (torch.mean(torch.abs(full_hidden)).item() + 1e-10)
                max_err = torch.max(torch.abs(cached_hidden - full_hidden)).item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    cached_hidden.flatten().unsqueeze(0),
                    full_hidden.flatten().unsqueeze(0)
                ).item()

                step_info = {
                    "step": len(log),
                    "used_cache": bool(self.use_cache),
                    "abs_error": abs_err,
                    "rel_error": rel_err,
                    "max_error": max_err,
                    "cosine_sim": cos_sim,
                }

                # SpectralCache-specific info
                if hasattr(self, '_consecutive_cached'):
                    step_info["consecutive_cached"] = self._consecutive_cached
                if hasattr(self, '_current_step'):
                    step_info["tads_scale"] = self.tads_scale(self._current_step)

                log.append(step_info)

        return result

    import types
    cached_blocks.forward = types.MethodType(instrumented_forward, cached_blocks)
    return cached_blocks, log


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_logs = {}

    for method in ["Fast", "SpectralCache", "Fb", "Tea"]:
        print(f"\n===== Analyzing {method}Cache =====")
        model = FluxPipeline.from_pretrained(args.model, torch_dtype=torch.float16).to("cuda")
        thresh = 0.6 if method in ("Tea", "SpectralCache") else args.cache_threshold
        apply_cache_on_transformer(
            model.transformer,
            rel_l1_thresh=thresh,
            return_hidden_states_first=False,
            num_steps=args.steps,
            use_cache=method,
            motion_threshold=0.1,
        )

        cached_blocks, log = instrument_cached_blocks(model, method)
        if cached_blocks is None:
            print(f"  Skipping {method} - could not instrument")
            continue

        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        t0 = time.time()
        with torch.no_grad():
            result = model(prompt=args.prompt, num_inference_steps=args.steps,
                           height=args.height, width=args.width, generator=gen)
        elapsed = time.time() - t0
        result.images[0].save(os.path.join(args.output_dir, f"{method.lower()}_output.png"))

        # Summarize
        cache_hits = sum(1 for s in log if s["used_cache"])
        total = len(log)
        cached_errors = [s["rel_error"] for s in log if s["used_cache"]]
        all_errors = [s["rel_error"] for s in log]

        print(f"  Time: {elapsed:.2f}s")
        print(f"  Cache hits: {cache_hits}/{total} ({100*cache_hits/max(total,1):.0f}%)")
        if cached_errors:
            print(f"  Cached steps rel error: mean={np.mean(cached_errors):.6f}, "
                  f"max={np.max(cached_errors):.6f}")
        print(f"  All steps rel error: mean={np.mean(all_errors):.6f}")
        print(f"  Cosine sim (all): mean={np.mean([s['cosine_sim'] for s in log]):.6f}")

        all_logs[method] = {
            "steps": log,
            "summary": {
                "time": elapsed,
                "cache_hits": cache_hits,
                "total_steps": total,
                "hit_rate": cache_hits / max(total, 1),
                "mean_rel_error": float(np.mean(all_errors)),
                "mean_cached_rel_error": float(np.mean(cached_errors)) if cached_errors else 0,
                "mean_cosine_sim": float(np.mean([s["cosine_sim"] for s in log])),
            }
        }

        del model; import gc; gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Per-Timestep Error Analysis ({args.height}x{args.width}, {args.steps} steps)")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'HitRate':>8} {'RelErr↓':>10} {'CachedErr↓':>12} {'CosSim↑':>10}")
    print("-" * 54)
    for method in ["Fast", "SpectralCache", "Fb", "Tea"]:
        if method not in all_logs:
            continue
        s = all_logs[method]["summary"]
        print(f"{method+'Cache':<12} {s['hit_rate']:>7.0%} "
              f"{s['mean_rel_error']:>10.6f} {s['mean_cached_rel_error']:>12.6f} "
              f"{s['mean_cosine_sim']:>10.6f}")

    # Save
    with open(os.path.join(args.output_dir, "error_analysis.json"), "w") as f:
        json.dump(all_logs, f, indent=2)
    print(f"\nDetailed results saved to {args.output_dir}/error_analysis.json")


if __name__ == "__main__":
    main()

