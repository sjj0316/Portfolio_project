from __future__ import annotations

from typing import Iterable, List, Dict, Any

def run_sentiment(model_id: str, device: int, texts: Iterable[str]) -> List[Dict[str, Any]]:
    """Run sentiment analysis using transformers pipeline."""
    from transformers import pipeline

    pipe = pipeline("sentiment-analysis", model=model_id, device=device)
    return pipe(list(texts))
