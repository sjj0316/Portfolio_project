"""cli.py

This module defines the **English command contract** exposed via `pyproject.toml` console scripts.

Design goals (portfolio + Codex workflow):
- Commands are stable and simple: `smoke`, `train`, `eval`, `predict`, etc.
- Environment switching is done via **profiles** (local/colab) so the same code runs everywhere.
- Heavy dependencies are **lazy-imported** inside subcommands to keep `uv run smoke` fast
  and to avoid forcing YOLO/Diffusers installs unless you actually use them.

Tip:
- Use `uv sync` once per project folder.
- Then run commands with: `uv run <command> --profile local|colab`
"""


from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


# -----------------------------
# Paths / config
# -----------------------------
@dataclass(frozen=True)
class Cfg:
    profile: str
    data_dir: Path
    models_dir: Path
    outputs_dir: Path
    reports_dir: Path


def project_root() -> Path:
    # src/<pkg>/cli.py -> <root>
    return Path(__file__).resolve().parents[2]


def load_cfg(profile: str | None) -> Cfg:
    prof = profile or os.getenv("PROFILE", "local")
    cfg_file = project_root() / "configs" / f"{prof}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"Missing config for profile='{prof}'. Expected: {cfg_file}"
        )
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}
    def p(key: str, default_rel: str) -> Path:
        val = raw.get(key, default_rel)
        # Allow absolute paths or relative-to-project paths
        path = Path(val)
        return path if path.is_absolute() else (project_root() / path).resolve()

    return Cfg(
        profile=prof,
        data_dir=p("data_dir", "data"),
        models_dir=p("models_dir", "models"),
        outputs_dir=p("outputs_dir", "outputs"),
        reports_dir=p("reports_dir", "reports"),
    )


# -----------------------------
# Small utilities
# -----------------------------
def _ensure_dirs(cfg: Cfg) -> None:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "figures").mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> int:
    print("$", " ".join(cmd))
    return subprocess.call(cmd)


def _maybe_import(module: str) -> Any:
    try:
        return __import__(module)
    except Exception as e:
        raise RuntimeError(
            f"Missing optional dependency: '{module}'.\n"
            f"Install the appropriate extra (see README) and retry.\n"
            f"Original error: {e}"
        ) from e


# -----------------------------
# CLI entrypoints (console scripts)
# -----------------------------
def _parse_profile(argv: list[str]) -> tuple[str | None, list[str]]:
    # Minimal parsing: accept `--profile <name>` anywhere
    if "--profile" in argv:
        i = argv.index("--profile")
        if i + 1 >= len(argv):
            raise SystemExit("Expected value after --profile")
        prof = argv[i + 1]
        rest = argv[:i] + argv[i + 2 :]
        return prof, rest
    return None, argv


def setup_cmd() -> None:
    """Print the recommended setup commands (uv)."""
    print("Recommended setup:")
    print("  uv sync")
    print("Optional extras:")
    print("  uv sync --extra dev")
    
    print("  uv sync --extra diffusers      # optional (Diffusers + Torch + Pillow)")


def lint_cmd() -> None:
    """Run ruff lint (requires dev extra)."""
    _run(["python", "-m", "ruff", "check", "."])


def format_cmd() -> None:
    """Run ruff format (requires dev extra)."""
    _run(["python", "-m", "ruff", "format", "."])


def test_cmd() -> None:
    """Run pytest (requires dev extra)."""
    _run(["python", "-m", "pytest"])


def smoke_cmd() -> None:
    """Fast smoke test: config loads, folders exist, CUDA visibility printed."""
    import sys
    prof, rest = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)
    print(f"[ok] profile={cfg.profile}")
    print(f"[ok] data_dir={cfg.data_dir}")
    print(f"[ok] models_dir={cfg.models_dir}")
    print(f"[ok] outputs_dir={cfg.outputs_dir}")
    print(f"[ok] reports_dir={cfg.reports_dir}")
    # Torch is optional; only check if installed.
    try:
        import torch  # type: ignore
        print(f"[ok] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"[ok] cuda_device={torch.cuda.get_device_name(0)}")
    except Exception:
        print("[info] torch not installed (this is OK for smoke).")
    print("[done] smoke")


def clean_cmd() -> None:
    """Remove outputs/runs (safe cleanup for repeated experiments)."""
    root = project_root()
    for d in ["outputs", "runs"]:
        p = root / d
        if p.exists():
            shutil.rmtree(p)
            print(f"[ok] removed {p}")
    print("[done] clean")


def train_cmd() -> None:
    raise NotImplementedError("Train is project-specific. See scripts/ and README.")


def eval_cmd() -> None:
    raise NotImplementedError("Eval is project-specific. See scripts/ and README.")


def predict_cmd() -> None:
    raise NotImplementedError("Predict is project-specific. See scripts/ and README.")


# -----------------------------
# Image generation specific commands
# -----------------------------
def train_cmd() -> None:
    raise NotImplementedError(
        "Training (fine-tuning) is out of scope for this portfolio starter. "
        "If needed, add a LoRA fine-tune script later."
    )


def eval_cmd() -> None:
    raise NotImplementedError(
        "Evaluation is project-specific. You can add FID/CLIPScore or human eval later."
    )


def predict_cmd() -> None:
    """
    Text-to-image using Diffusers (optional extra).
    Default model is a tiny pipeline for quick smoke-like runs.
    Requires: `uv sync --extra diffusers`
    """
    import sys
    from argparse import ArgumentParser

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="predict")
    ap.add_argument("--prompt", required=True, help="Text prompt.")
    ap.add_argument("--model", default="hf-internal-testing/tiny-stable-diffusion-pipe", help="HF model id.")
    ap.add_argument("--steps", type=int, default=4, help="Number of inference steps.")
    ap.add_argument("--seed", type=int, default=0, help="Seed (0 = random).")
    ap.add_argument("--out", default="", help="Output file name (png).")
    args = ap.parse_args(argv)

    # Lazy imports
    _maybe_import("torch")
    import torch  # type: ignore
    _maybe_import("diffusers")
    from diffusers import DiffusionPipeline  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[ok] device={device} model={args.model}")

    # Load pipeline (will download if not cached)
    pipe = DiffusionPipeline.from_pretrained(args.model)
    pipe = pipe.to(device)

    gen = None
    if args.seed != 0:
        gen = torch.Generator(device=device).manual_seed(args.seed)

    image = pipe(prompt=args.prompt, num_inference_steps=args.steps, generator=gen).images[0]

    out_dir = cfg.outputs_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_name = args.out.strip() or "sample.png"
    out_path = out_dir / out_name
    image.save(out_path)
    print(f"[ok] saved -> {out_path}")
    print("[done] predict")
