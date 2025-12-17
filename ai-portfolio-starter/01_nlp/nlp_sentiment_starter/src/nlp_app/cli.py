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
    
    print("  uv sync --extra transformers   # optional (Transformers + Torch)")


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
# NLP-specific commands
# -----------------------------
def _try_import_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def train_cmd() -> None:
    """
    Train a tiny baseline sentiment model.
    - If scikit-learn is installed (extra: ml), trains a simple LogisticRegression.
    - Otherwise, writes a tiny lexicon model.
    """
    import sys
    import json
    import pickle
    from argparse import ArgumentParser

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="train")
    ap.add_argument("--model-name", default="sentiment.pkl", help="Output model file name.")
    args = ap.parse_args(argv)

    model_path = cfg.models_dir / args.model_name

    # Tiny training data (replace with your own)
    samples = [
        ("i love this", 1),
        ("this is great", 1),
        ("amazing work", 1),
        ("i hate this", 0),
        ("this is terrible", 0),
        ("worst experience", 0),
        ("not bad", 1),
        ("not good", 0),
    ]

    if _try_import_sklearn():
        from sklearn.feature_extraction.text import CountVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        X = [t for t, y in samples]
        y = [y for t, y in samples]

        pipe = Pipeline(
            steps=[
                ("vec", CountVectorizer(ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        pipe.fit(X, y)
        with open(model_path, "wb") as f:
            pickle.dump({"type": "sklearn", "pipeline": pipe}, f)
        print(f"[ok] trained sklearn baseline -> {model_path}")
    else:
        # Very small lexicon fallback (works without extra deps)
        lexicon = {
            "good": 1.0, "great": 1.2, "love": 1.4, "amazing": 1.6,
            "bad": -1.0, "terrible": -1.4, "hate": -1.6, "worst": -1.8
        }
        model = {"type": "lexicon", "lexicon": lexicon}
        model_path = cfg.models_dir / "sentiment_lexicon.json"
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print("[warn] scikit-learn not installed; wrote lexicon model instead.")
        print(f"[ok] model -> {model_path}")

    print("[done] train")


def eval_cmd() -> None:
    """Evaluate the latest model on a tiny sample and write a JSON report."""
    import sys
    import json
    import pickle
    from argparse import ArgumentParser

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="eval")
    ap.add_argument("--model", default="", help="Path to model. If empty, auto-detect.")
    args = ap.parse_args(argv)

    # Auto-detect model
    model_path = Path(args.model) if args.model else None
    if not model_path:
        pkl = sorted(cfg.models_dir.glob("*.pkl"))
        if pkl:
            model_path = pkl[-1]
        else:
            js = cfg.models_dir / "sentiment_lexicon.json"
            model_path = js if js.exists() else None

    if not model_path or not model_path.exists():
        raise SystemExit("No model found. Run: uv run train --profile local")

    # Tiny eval set
    evalset = [
        ("i love this product", 1),
        ("this is the worst", 0),
        ("not bad at all", 1),
        ("not good", 0),
        ("great and amazing", 1),
        ("terrible experience", 0),
    ]

    preds = []
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        pipe = obj["pipeline"]
        X = [t for t, y in evalset]
        y_true = [y for t, y in evalset]
        y_pred = pipe.predict(X).tolist()
        preds = list(zip(X, y_true, y_pred))
    else:
        import json as _json
        model = _json.loads(model_path.read_text(encoding="utf-8"))
        lex = model["lexicon"]

        def score(text: str) -> float:
            s = 0.0
            toks = [w.strip(".,!?").lower() for w in text.split()]
            neg = False
            for w in toks:
                if w == "not":
                    neg = True
                    continue
                v = float(lex.get(w, 0.0))
                s += (-v if neg else v)
                neg = False
            return s

        for t, y in evalset:
            yhat = 1 if score(t) >= 0 else 0
            preds.append((t, y, yhat))

    correct = sum(1 for _, y, yhat in preds if int(y) == int(yhat))
    acc = correct / max(1, len(preds))

    report = {
        "model_path": str(model_path),
        "n": len(preds),
        "accuracy": acc,
        "examples": [{"text": t, "y_true": y, "y_pred": yhat} for t, y, yhat in preds],
    }
    out = cfg.reports_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] wrote report -> {out}")
    print(f"[done] eval acc={acc:.3f}")


def predict_cmd() -> None:
    """Predict sentiment for a single text input."""
    import sys
    import pickle
    from argparse import ArgumentParser
    import json

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="predict")
    ap.add_argument("--text", required=True, help="Text to analyze.")
    ap.add_argument("--model", default="", help="Optional model path. If empty, auto-detect.")
    args = ap.parse_args(argv)

    model_path = Path(args.model) if args.model else None
    if not model_path:
        pkl = sorted(cfg.models_dir.glob("*.pkl"))
        if pkl:
            model_path = pkl[-1]
        else:
            js = cfg.models_dir / "sentiment_lexicon.json"
            model_path = js if js.exists() else None

    if not model_path or not model_path.exists():
        raise SystemExit("No model found. Run: uv run train --profile local")

    text = args.text

    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as f:
            obj = pickle.load(f)
        pipe = obj["pipeline"]
        pred = int(pipe.predict([text])[0])
        prob = None
        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba([text])[0][1])
        print({"label": "positive" if pred == 1 else "negative", "p_positive": prob})
    else:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        lex = model["lexicon"]
        toks = [w.strip(".,!?").lower() for w in text.split()]
        s = 0.0
        neg = False
        for w in toks:
            if w == "not":
                neg = True
                continue
            v = float(lex.get(w, 0.0))
            s += (-v if neg else v)
            neg = False
        label = "positive" if s >= 0 else "negative"
        print({"label": label, "score": s})

    print("[done] predict")
