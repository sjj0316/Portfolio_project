from __future__ import annotations

from pathlib import Path

def pick_device(device_pref: str) -> str:
    device_pref = (device_pref or "auto").lower()
    if device_pref in ("cpu", "cuda"):
        return device_pref

    # auto
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def txt2img(
    model_id: str,
    prompt: str,
    out_path: Path,
    device_pref: str = "auto",
    num_inference_steps: int = 4,
    guidance_scale: float = 0.0,
    seed: int = 42,
) -> Path:
    """Generate a single image from text prompt using Diffusers."""
    import torch
    from diffusers import DiffusionPipeline

    device = pick_device(device_pref)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    pipe = DiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    result = pipe(
        prompt,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
        generator=generator,
    )
    image = result.images[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return out_path
