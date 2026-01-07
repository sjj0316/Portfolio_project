# AI Portfolio (Codex + Local) Starter

이 저장소는 AI 포트폴리오를 위한 멀티 프로젝트 스타터입니다.  
Codex에서 작은 단위의 작업을 수행하고, 로컬에서는 **uv**로 동일하게 실행할 수 있도록 구성했습니다.

## 프로젝트 구성
- `01_nlp/nlp_sentiment_starter`
- `02_cv/realtime_yolo_starter`
- `03_image_gen/image_gen_starter`

각 프로젝트는 아래 구성을 공통으로 가집니다.
- `AGENTS.md` (Codex 작업 규칙)
- `pyproject.toml` 기반 English command contract
- `configs/local.yaml` + `configs/colab.yaml` (로컬/코랩 프로파일)

## 빠른 시작
프로젝트 폴더로 이동 후 실행:

```bash
uv sync
uv run smoke --profile local
```

환경/SSL/네트워크 사전 점검(권장):
```bash
uv run check --profile local
```

무거운 기능은 프로젝트별 extra 설치가 필요합니다.
- CV: `uv sync --extra yolo`
- Image Gen: `uv sync --extra diffusers`

## 현재까지 진행된 내용
- 3개 프로젝트 모두 필수 구성 파일 및 디렉터리 확인 완료.
- `uv run smoke --profile local` 통과 확인.
- `uv run test` 실행 경로를 정리하여 테스트 통과 상태로 정비.
  - venv 파이썬을 사용하도록 통일
  - 테스트 실행 시 `PYTHONPATH`에 `src` 추가

## 앞으로 진행할 내용
- 로컬 GPU 사용 환경 검증 (필요 extra 설치 후 CUDA 인식 확인)
- Colab 프로파일 경로 설정 및 동작 검증
- 각 프로젝트별 기능 예제 실행
  - NLP: `train/eval/predict`
  - CV: YOLO `predict`
  - Image Gen: Diffusers `predict`
- 문서 보강: 실행 결과 캡처/로그 및 체크리스트 정리

## 우선순위와 실행 절차
P0. 기본 안정성 확인 (완료)
- `uv run smoke --profile local`
- `uv run test` (필요 시 `uv sync --extra dev`)

P1. 로컬 GPU 환경 검증 (권장 1순위)
- GPU 인식: `nvidia-smi`
- NLP: `uv sync --extra transformers` 후 `uv run smoke --profile local`
- CV: `uv sync --extra yolo` 후 `uv run smoke --profile local`
- Image Gen: `uv sync --extra diffusers` 후 `uv run smoke --profile local`

P2. Colab 프로파일 설정/검증 (권장 2순위)
- `configs/colab.yaml`에 Drive 경로 반영
  - 기준 경로: `/content/drive/MyDrive/ai-data`
  - 프로젝트별 하위 폴더: `nlp`, `cv`, `image_gen`
- 실행 시 `--profile colab` 사용 (또는 환경변수 `PROFILE=colab`)

P3. 기능 예제 실행 및 결과 기록 (권장 3순위)
- NLP: `uv run train --profile local` → `uv run eval --profile local` → `uv run predict --profile local --text "..."`.
- CV: `uv run predict --profile local --source 0 --model yolo11n.pt`
- Image Gen: `uv run predict --profile local --prompt "..." --steps 2`

P4. 루트 통합 실행 편의 개선 (권장 4순위)
- 루트에서 실행하는 래퍼 스크립트/문서 추가 (프로젝트 분리는 유지)
- 예: `all:smoke`, `all:test`, `all:predict` 같은 일괄 실행 절차 정리
- 이유: 의존성 충돌을 줄이면서도 사용성을 높이기 위함

## 확장 방향 제안 (이번 작업의 기준)
권장 순서: GPU/Colab 운영 표준화 → 공통 실행 편의 → 프로젝트별 기능 심화 → 데모/배포 강화  
이 순서를 기준으로 다음 작업을 진행합니다.

## 현재 환경 검증 결과 (로컬)
- GPU: `nvidia-smi` 미인식
- NLP/CV/Image Gen: `torch` 설치 후 `cuda_available=False` (CPU 동작)

## 예제 실행 결과 (로컬, P3 완료)
- NLP: `train/eval/predict` 실행 완료
  - 모델: `models/sentiment_lexicon.json` (scikit-learn 미설치로 lexicon fallback)
  - 리포트: `reports/eval_report.json` (acc=1.000)
- CV: YOLO 추론 실행 완료
  - 로컬 샘플 영상이 없어 URL 이미지로 대체 실행
  - 실행 예: `--source https://ultralytics.com/images/bus.jpg`
  - 출력: `outputs/yolo`
- Image Gen: Diffusers 예제 실행 완료
  - 모델: `hf-internal-testing/tiny-stable-diffusion-pipe`
  - 안전 검사 비활성화: `--no-safety-checker`
  - 출력: `outputs/images/sample.png`

## 사전 점검 결과 (로컬, check)
- Python: 3.13.7 (Windows)
- 네트워크/SSL: 정상 (pypi.org 접근 OK)
- Torch: 미설치 상태에서도 `check`는 정상 동작

## Codex 작업 흐름 (권장)
1) Codex에 작은 단위 작업을 요청
2) `AGENTS.md` 규칙 준수, 아래 커맨드 사용
   - `uv run smoke --profile local`
   - `uv run test` (필요 시 `--extra dev` 설치)
3) PR 생성 후 로컬에서 동일한 커맨드로 검증

PR은 작게 유지하고, 큰 산출물(`data/models/outputs`)은 커밋하지 않습니다.
