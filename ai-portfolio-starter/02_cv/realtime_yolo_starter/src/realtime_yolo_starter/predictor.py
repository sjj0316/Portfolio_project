from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

def predict(
    model_name: str,
    source: str,
    conf: float = 0.25,
    imgsz: int = 640,
    show: bool = True,
    save: bool = True,
    project_dir: Path | None = None,
) -> None:
    """Run YOLO prediction on webcam/video/image source."""
    from ultralytics import YOLO

    model = YOLO(model_name)
    kwargs: Dict[str, Any] = dict(conf=conf, imgsz=imgsz, show=show, save=save)
    if project_dir is not None:
        kwargs["project"] = str(project_dir)
        kwargs["name"] = "predict"

    # Ultralytics accepts webcam index like "0" or int 0, and paths/URLs/streams.
    model.predict(source=source, **kwargs)
