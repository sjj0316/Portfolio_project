"""cli.py

Tabular classification starter with an English command contract for Codex and local runs.
명령은 uv console script로 노출되며, 프로필(local/colab) 기반 설정을 사용한다.
스모크는 가벼우며(scikit-learn 없음), 무거운 의존성은 지연 임포트된다.
"""

from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

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
    # 프로젝트 루트 경로를 일관되게 계산
    return Path(__file__).resolve().parents[2]


def load_cfg(profile: str | None) -> Cfg:
    # 프로필별 설정 파일을 읽어 절대경로로 정규화
    prof = profile or os.getenv("PROFILE", "local")
    cfg_file = project_root() / "configs" / f"{prof}.yaml"
    if not cfg_file.exists():
        raise FileNotFoundError(
            f"Missing config for profile='{prof}'. Expected: {cfg_file}"
        )
    raw = yaml.safe_load(cfg_file.read_text(encoding="utf-8")) or {}

    def p(key: str, default_rel: str) -> Path:
        val = raw.get(key, default_rel)
        path = Path(val)
        # 절대경로는 그대로, 상대경로는 프로젝트 루트 기준
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
    # 실행 전 필요한 디렉터리를 생성하여 I/O 오류 예방
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    cfg.outputs_dir.mkdir(parents=True, exist_ok=True)
    cfg.reports_dir.mkdir(parents=True, exist_ok=True)
    (cfg.reports_dir / "figures").mkdir(parents=True, exist_ok=True)


def _run(cmd: list[str]) -> int:
    # 명령 실행을 로그로 보여주고 반환 코드 전달
    print("$", " ".join(cmd))
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    return subprocess.call(cmd)


def _maybe_import(module: str) -> Any:
    # 선택적 의존성(sklearn 등) 누락 시 명확한 안내 메시지
    try:
        return __import__(module)
    except Exception as exc:  # pragma: no cover - 메시지 보존 목적
        raise RuntimeError(
            f"Missing optional dependency: '{module}'.\n"
            f"Install the appropriate extra (see README) and retry.\n"
            f"Original error: {exc}"
        ) from exc


# -----------------------------
# CLI entrypoints (console scripts)
# -----------------------------
def _parse_profile(argv: list[str]) -> tuple[str | None, list[str]]:
    # --profile 파라미터를 간단히 추출
    if "--profile" in argv:
        idx = argv.index("--profile")
        if idx + 1 >= len(argv):
            raise SystemExit("Expected value after --profile")
        prof = argv[idx + 1]
        rest = argv[:idx] + argv[idx + 2 :]
        return prof, rest
    return None, argv


def setup_cmd() -> None:
    """Print the recommended setup commands (uv)."""
    print("Recommended setup:")
    print("  uv sync")
    print("Optional extras:")
    print("  uv sync --extra dev")
    print("  uv sync --extra ml            # optional (scikit-learn baseline)")


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
    """Fast smoke test: config loads, folders exist, dependency hint."""
    import sys

    prof, _ = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)
    print(f"[ok] profile={cfg.profile}")
    print(f"[ok] data_dir={cfg.data_dir}")
    print(f"[ok] models_dir={cfg.models_dir}")
    print(f"[ok] outputs_dir={cfg.outputs_dir}")
    print(f"[ok] reports_dir={cfg.reports_dir}")
    try:
        import sklearn  # type: ignore

        print(f"[info] sklearn detected (version={getattr(sklearn, '__version__', 'unknown')})")
    except Exception:
        print("[info] sklearn not installed (this is OK for smoke).")
    print("[done] smoke")


def clean_cmd() -> None:
    """Remove outputs/runs (safe cleanup for repeated experiments)."""
    root = project_root()
    for name in ["outputs", "runs"]:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
            print(f"[ok] removed {path}")
    print("[done] clean")


# -----------------------------
# Tabular-specific helpers
# -----------------------------
def _has_sklearn() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def _default_train_data() -> list[tuple[list[float], int]]:
    # 작은 예제 데이터 (피처 3개, 이진 라벨)
    return [
        ([0.1, 0.5, 0.2], 0),
        ([0.2, 0.6, 0.1], 0),
        ([0.9, 0.3, 0.8], 1),
        ([0.8, 0.2, 0.9], 1),
        ([0.4, 0.7, 0.2], 0),
        ([0.7, 0.1, 0.6], 1),
        ([0.3, 0.6, 0.3], 0),
        ([0.85, 0.15, 0.75], 1),
    ]


def _train_fallback(samples: list[tuple[list[float], int]]) -> dict[str, Any]:
    # sklearn이 없을 때를 위한 단순 선형 점수 모델 (평균 차이를 가중치로 사용)
    pos = [feat for feat, y in samples if y == 1]
    neg = [feat for feat, y in samples if y == 0]
    if not pos or not neg:
        raise RuntimeError("Need both positive and negative samples for fallback model.")
    dim = len(pos[0])
    pos_mean = [sum(feat[i] for feat in pos) / len(pos) for i in range(dim)]
    neg_mean = [sum(feat[i] for feat in neg) / len(neg) for i in range(dim)]
    weights = [p - n for p, n in zip(pos_mean, neg_mean)]
    bias = -0.5 * sum(weights)  # 약식 바이어스
    return {"type": "linear", "weights": weights, "bias": bias}


def _score_fallback(model: dict[str, Any], features: Sequence[float]) -> float:
    # 선형 가중치와 바이어스로 점수를 계산
    w = model["weights"]
    b = float(model["bias"])
    return sum(f * wi for f, wi in zip(features, w)) + b


def _parse_features(raw: str) -> list[float]:
    # "0.1,0.2,0.3" 형태 문자열을 float 리스트로 변환
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise SystemExit("No features provided. Use comma-separated numbers, e.g., --features \"0.1,0.2,0.3\"")
    try:
        return [float(x) for x in parts]
    except ValueError as exc:  # pragma: no cover - 입력 오류 안내
        raise SystemExit(f"Could not parse features: {raw}") from exc


# -----------------------------
# Train / Eval / Predict
# -----------------------------
def train_cmd() -> None:
    """
    Train a tiny tabular classifier.
    - If scikit-learn is installed (extra: ml), trains LogisticRegression with StandardScaler.
    - Otherwise, writes a small linear fallback model.
    """
    import sys

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="train")
    ap.add_argument("--model-name", default="tabular.pkl", help="Output model file name.")
    args = ap.parse_args(argv)

    samples = _default_train_data()
    model_path = cfg.models_dir / args.model_name

    if _has_sklearn():
        # sklearn 경로: 표준화 + 로지스틱 회귀 파이프라인
        _maybe_import("sklearn")
        from sklearn.linear_model import LogisticRegression  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import StandardScaler  # type: ignore

        X = [feat for feat, _ in samples]
        y = [label for _, label in samples]

        pipe = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        pipe.fit(X, y)
        with open(model_path, "wb") as fh:
            pickle.dump({"type": "sklearn", "pipeline": pipe}, fh)
        print(f"[ok] trained sklearn baseline -> {model_path}")
    else:
        # fallback: 간단한 선형 가중치 모델을 JSON으로 저장
        model = _train_fallback(samples)
        model_path = cfg.models_dir / "tabular_linear.json"
        model_path.write_text(json.dumps(model, indent=2), encoding="utf-8")
        print("[warn] scikit-learn not installed; wrote linear fallback model instead.")
        print(f"[ok] model -> {model_path}")

    print("[done] train")


def eval_cmd() -> None:
    """Evaluate the latest model on a tiny sample and write a JSON report."""
    import sys

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="eval")
    ap.add_argument("--model", default="", help="Path to model. If empty, auto-detect.")
    args = ap.parse_args(argv)

    # 모델 자동 검색 (pkl 우선)
    model_path = Path(args.model) if args.model else None
    if not model_path:
        pkls = sorted(cfg.models_dir.glob("*.pkl"))
        if pkls:
            model_path = pkls[-1]
        else:
            linear = cfg.models_dir / "tabular_linear.json"
            model_path = linear if linear.exists() else None
    if not model_path or not model_path.exists():
        raise SystemExit("No model found. Run: uv run train --profile local")

    evalset = [
        ([0.15, 0.55, 0.15], 0),
        ([0.75, 0.1, 0.7], 1),
        ([0.25, 0.65, 0.25], 0),
        ([0.9, 0.2, 0.85], 1),
        ([0.35, 0.6, 0.2], 0),
        ([0.8, 0.15, 0.65], 1),
    ]

    preds: list[tuple[list[float], int, int]] = []
    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as fh:
            obj = pickle.load(fh)
        pipe = obj["pipeline"]
        X = [feat for feat, _ in evalset]
        y_true = [label for _, label in evalset]
        y_pred = pipe.predict(X).tolist()
        preds = list(zip(X, y_true, y_pred))
    else:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        for feats, label in evalset:
            score = _score_fallback(model, feats)
            yhat = 1 if score >= 0 else 0
            preds.append((feats, label, yhat))

    correct = sum(1 for _, y, yhat in preds if int(y) == int(yhat))
    acc = correct / max(1, len(preds))

    report = {
        "model_path": str(model_path),
        "n": len(preds),
        "accuracy": acc,
        "examples": [
            {"features": feats, "y_true": y, "y_pred": yhat} for feats, y, yhat in preds
        ],
    }
    out = cfg.reports_dir / "eval_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[ok] wrote report -> {out}")
    print(f"[done] eval acc={acc:.3f}")


def predict_cmd() -> None:
    """Predict class for a single comma-separated feature vector."""
    import sys

    prof, argv = _parse_profile(sys.argv[1:])
    cfg = load_cfg(prof)
    _ensure_dirs(cfg)

    ap = ArgumentParser(prog="predict")
    ap.add_argument(
        "--features",
        required=True,
        help='Comma-separated numeric features, e.g., "0.5,0.1,0.3".',
    )
    ap.add_argument("--model", default="", help="Optional model path. If empty, auto-detect.")
    args = ap.parse_args(argv)

    feats = _parse_features(args.features)
    model_path = Path(args.model) if args.model else None
    if not model_path:
        pkls = sorted(cfg.models_dir.glob("*.pkl"))
        if pkls:
            model_path = pkls[-1]
        else:
            linear = cfg.models_dir / "tabular_linear.json"
            model_path = linear if linear.exists() else None

    if not model_path or not model_path.exists():
        raise SystemExit("No model found. Run: uv run train --profile local")

    if model_path.suffix == ".pkl":
        with open(model_path, "rb") as fh:
            obj = pickle.load(fh)
        pipe = obj["pipeline"]
        pred = int(pipe.predict([feats])[0])
        prob = None
        if hasattr(pipe, "predict_proba"):
            prob = float(pipe.predict_proba([feats])[0][1])
        print({"label": pred, "p_positive": prob})
    else:
        model = json.loads(model_path.read_text(encoding="utf-8"))
        score = _score_fallback(model, feats)
        label = 1 if score >= 0 else 0
        print({"label": label, "score": score})

    print("[done] predict")
