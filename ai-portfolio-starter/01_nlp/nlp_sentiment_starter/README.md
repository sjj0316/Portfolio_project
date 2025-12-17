# NLP: Sentiment Starter (Transformers)

최소 실행 목표: Hugging Face Transformers의 사전학습 감성분석 모델로 텍스트를 입력하면 결과를 출력합니다.

## 로컬(Windows) - uv로 바로 실행
1) 프로젝트 폴더로 이동:
   ```powershell
   cd nlp_sentiment_starter
   ```
2) (처음 1회) 의존성 설치 + 실행:
   ```powershell
   uv run python scripts\demo_sentiment.py --profile local
   ```

## Colab - 빠른 실행(권장)
Colab은 기본 Python 버전이 달라서, venv까지 고집하면 번거롭습니다.
아래처럼 그냥 pip 설치 후 실행이 가장 단순합니다.
```python
!pip -q install transformers torch pyyaml
!python scripts/demo_sentiment.py --profile colab
```

## GPU로 돌리고 싶다면(선택)
- Windows에서 PyTorch CUDA 휠은 별도 인덱스가 필요한 경우가 많습니다.
- 일단 smoke test는 CPU로 성공시키고, 이후 `configs/local.yaml`에서 `device: 0`으로 바꾼 뒤 GPU 설치를 진행하는 흐름을 추천합니다.

## 설정 파일
- `configs/local.yaml` / `configs/colab.yaml`
- 선택: `PROFILE=local` 또는 `--profile local`
