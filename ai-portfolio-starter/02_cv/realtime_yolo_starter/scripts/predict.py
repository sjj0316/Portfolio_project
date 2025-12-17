from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on PYTHONPATH for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


import argparse
from pathlib import Path

from realtime_yolo_starter.config import load_config, ROOT
from realtime_yolo_starter.predictor import predict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="local|colab (or set PROFILE env)")
    ap.add_argument("--source", default=None, help="0 for webcam, or path to image/video/stream")
    ap.add_argument("--model", default=None, help="Override model weights (e.g., yolo11n.pt)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--show", type=str, default=None, help="true/false override")
    ap.add_argument("--save", type=str, default=None, help="true/false override")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    model_name = args.model or cfg.get("model", "yolo11n.pt")
    conf = float(args.conf if args.conf is not None else cfg.get("conf", 0.25))
    imgsz = int(args.imgsz if args.imgsz is not None else cfg.get("imgsz", 640))

    show_cfg = cfg.get("show", True)
    save_cfg = cfg.get("save", True)

    def to_bool(v, default):
        if v is None:
            return bool(default)
        return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

    show = to_bool(args.show, show_cfg)
    save = to_bool(args.save, save_cfg)

    # default source
    source = args.source
    if source is None:
        source = "0" if str(show).lower() != "false" else "data/sample.mp4"

    runs_dir = ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)

    predict(
        model_name=model_name,
        source=source,
        conf=conf,
        imgsz=imgsz,
        show=show,
        save=save,
        project_dir=runs_dir,
    )

if __name__ == "__main__":
    main()
