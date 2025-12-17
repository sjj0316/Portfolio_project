# AI Portfolio Starter (3 mini-projects)

이 ZIP은 **카테고리 폴더 아래에 “프로젝트 루트”를 두는 방식**입니다.
- 각 프로젝트 폴더가 독립 루트(각자 `pyproject.toml`, `configs/`, `src/`, `scripts/`)입니다.
- 따라서 “카테고리로 보기 좋게 정리” + “프로젝트별 가상환경/의존성 격리”를 같이 가져갑니다.

## 폴더 구성
- `01_nlp/nlp_sentiment_starter/` : NLP 감성분석 (Transformers)
- `02_cv/realtime_yolo_starter/` : 실시간 객체탐지 (Ultralytics YOLO)
- `03_image_gen/image_gen_starter/` : 이미지 생성(txt2img) (Diffusers, tiny 모델로 smoke test)

## 왜 이런 구조인가?
- “카테고리(자연어처리/실시간 영상/이미지 생성)”로 포트폴리오를 묶되,
  의존성 충돌을 피하려면 프로젝트 단위로 분리하는 편이 일반적으로 편합니다.
- 데이터/리포트/모델 아티팩트 폴더 분리는 (예: Cookiecutter Data Science류) 관행을 참고해
  `data/raw|interim|processed`, `reports/`, `models/`, `outputs/` 등을 기본으로 깔아뒀습니다.

## 용량(가상환경이 여러 개인 문제) 완화 팁
- uv 전역 캐시 + 링크 설치 덕분에 “env가 여러 개”여도 체감 디스크 중복이 줄어드는 편입니다.
- 그래도 디스크가 빡빡하면:
  1) YOLO/NLP/이미지생성을 한 프로젝트로 합쳐 env 1개로 운영하거나
  2) “torch 포함 프로젝트”만 묶어서 env 수를 줄이는 방식이 현실적입니다.

## 빠른 시작
원하는 프로젝트 폴더로 들어가서 해당 README의 명령을 그대로 실행하면 됩니다.
