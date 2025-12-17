from __future__ import annotations

import sys
from pathlib import Path

# Ensure `src/` is on PYTHONPATH for direct script execution
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


import argparse
from nlp_sentiment_starter.config import load_config
from nlp_sentiment_starter.sentiment import run_sentiment

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None, help="local|colab (or set PROFILE env)")
    ap.add_argument("--text", action="append", default=None, help="Override sample text (repeatable)")
    args = ap.parse_args()

    cfg = load_config(args.profile)
    model_id = cfg.get("model_id", "distilbert-base-uncased-finetuned-sst-2-english")
    device = int(cfg.get("device", -1))

    if args.text:
        texts = args.text
    else:
        texts = cfg.get("sample_texts", ["Hello world!", "This is a test."])

    results = run_sentiment(model_id=model_id, device=device, texts=texts)

    print(f"model_id={model_id} device={device}")
    for t, r in zip(texts, results):
        print("-"*60)
        print("TEXT :", t)
        print("LABEL:", r.get("label"))
        print("SCORE:", r.get("score"))

if __name__ == "__main__":
    main()
