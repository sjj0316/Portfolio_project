from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on PYTHONPATH for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


import argparse
from datetime import datetime
from pathlib import Path

from image_gen_starter.config import load_config, ROOT
from image_gen_starter.generator import txt2img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="local|colab (or set PROFILE env)")
    ap.add_argument("--prompt", required=True, help="Text prompt for generation")
    ap.add_argument("--out", default=None, help="Output path (default: outputs/<timestamp>.png)")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    model_id = cfg.get("model_id", "hf-internal-testing/tiny-stable-diffusion-pipe")
    device = cfg.get("device", "auto")
    steps = int(cfg.get("num_inference_steps", 4))
    gs = float(cfg.get("guidance_scale", 0.0))
    seed = int(cfg.get("seed", 42))

    if args.out:
        out_path = Path(args.out)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "outputs" / f"txt2img_{ts}.png"

    saved = txt2img(
        model_id=model_id,
        prompt=args.prompt,
        out_path=out_path,
        device_pref=device,
        num_inference_steps=steps,
        guidance_scale=gs,
        seed=seed,
    )
    print(f"Saved: {saved}")

if __name__ == "__main__":
    main()
