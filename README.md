
# AI Portfolio Monorepo (uv + Codex-friendly)

이 레포는 **포트폴리오용 AI 실험 프로젝트 모음(모노레포)** 입니다.  
코드는 `.py` 중심으로 유지하고, 실행/검증은 `uv` + `pyproject.toml`의 **커맨드 계약(command contract)** 으로 통일합니다.  
또한 **Codex(Web) 기반 PR 워크플로우**(리뷰/CI 루프)와 **Colab GPU 실행용 런처 노트북**을 포함합니다.

---

## 핵심 특징
- **프로젝트별 독립 실행**: 각 프로젝트 폴더에서 `uv sync` 후 `uv run smoke/train/eval/predict` 실행
- **선택 설치(extras)**: 무거운 의존성(torch/yolo/diffusers 등)은 필요할 때만 `--extra`로 설치
- **Codex(Web) 워크플로우**: PR 단위로 작업 → 리뷰/CI로 검증(옵션)
- **Colab GPU 사용**: 코드가 `.ipynb`가 아니어도, 런처 노트북 1개로 Colab에서 실행 가능

---

## 폴더 구조(요약)
> 실제 폴더명은 레포 상태에 따라 다를 수 있습니다. 아래는 권장/기본 구조입니다.

```

.
├─ ai-portfolio/
│  ├─ 01_nlp/
│  │  └─ nlp_sentiment_starter/
│  ├─ 02_cv/
│  │  └─ realtime_yolo_starter/
│  └─ 03_image_gen/
│     └─ image_gen_starter/
├─ notebooks/
│  └─ colab_runner.ipynb
├─ AGENTS.md
└─ .github/
├─ pull_request_template.md
└─ workflows/
├─ ci.yml
└─ codex-review.yml

````

---

## 요구사항
- Python **3.12+** 권장
- Windows PowerShell 기준 명령 예시
- 패키지/가상환경: **uv 사용**

---

## 빠른 시작(로컬)
### 1) 프로젝트 폴더로 이동
예: NLP 프로젝트
```powershell
cd .\ai-portfolio\01_nlp\nlp_sentiment_starter
````

### 2) 설치 및 기본 점검

```powershell
uv sync
uv run smoke --profile local
```

### 3) 학습/평가/예측(프로젝트 제공 커맨드 기준)

```powershell
uv run train --profile local
uv run eval --profile local
uv run predict --profile local --text "this is surprisingly good"
```

---

## extras(선택 설치) 사용법

기본은 가볍게 유지하고, 필요할 때만 extra를 설치합니다.

```powershell
# 개발 도구(테스트/린트 등)
uv sync --extra dev
uv run lint
uv run test

# (예) 전통 ML(scikit-learn) 경로가 있는 경우
uv sync --extra ml
uv run train --profile local
uv run eval --profile local
```

> 어떤 extra가 존재하는지는 각 프로젝트의 `pyproject.toml`에서 `[project.optional-dependencies]`를 확인하세요.

---

## 프로젝트별 안내(요약)

### 01_nlp / nlp_sentiment_starter

* 목표: 간단한 감성 분류 파이프라인(포트폴리오용)
* 커맨드: `smoke / train / eval / predict`
* 특징: 무거운 라이브러리 없이도 `smoke`가 통과하도록 설계(경량 점검)

예시:

```powershell
cd .\ai-portfolio\01_nlp\nlp_sentiment_starter
uv sync
uv run smoke --profile local
uv run train --profile local
uv run eval --profile local
uv run predict --profile local --text "this is surprisingly good"
```

### 02_cv / realtime_yolo_starter

* 목표: YOLO 기반 실시간/영상 입력 추론(포트폴리오용)
* 권장: YOLO 관련 의존성은 extra로 설치

예시(프로젝트의 extra 이름에 맞춰 조정):

```powershell
cd .\ai-portfolio\02_cv\realtime_yolo_starter
uv sync --extra yolo
uv run smoke --profile local
uv run predict --profile local --source 0
```

### 03_image_gen / image_gen_starter

* 목표: 텍스트 기반 이미지 생성(포트폴리오용)
* 권장: diffusers/torch 등은 extra로 설치

예시(프로젝트의 extra 이름에 맞춰 조정):

```powershell
cd .\ai-portfolio\03_image_gen\image_gen_starter
uv sync --extra diffusers
uv run smoke --profile local
uv run predict --profile local --prompt "a cute robot on a desk" --steps 2
```

---

## Colab GPU 사용(코드가 .ipynb가 아니어도 가능)

이 레포는 코드가 `.py` 중심이지만, **Colab에서 실행하기 위한 런처 노트북**을 제공합니다.

### 사용 방법(VS Code + Colab 확장)

1. VS Code에서 `notebooks/colab_runner.ipynb` 열기
2. 커널/런타임을 **Colab**로 연결
3. 노트북 상단 셀에서 아래 값만 수정:

* `REPO_URL`, `REPO_DIR`
* (필요 시) `BRANCH`
* `PROJECT_PATH` (NLP/CV/Gen 중 선택)

4. 셀을 위에서 아래로 실행

> 주의: Colab 런타임은 휘발성이므로, 결과물을 남기려면 Drive 마운트 + `configs/colab.yaml`로 경로를 Drive로 지정하는 방식을 권장합니다.

---

## Codex(Web) 운영 방식(PR 기반)

> 이 기능들은 “PR이 있어야” 동작합니다. (main에 직접 push만 하면 PR 탭은 비어 있습니다.)

### 기본 흐름

1. 브랜치 생성 → 커밋 → push
2. GitHub에서 PR 생성
3. PR 코멘트로 Codex 호출

#### PR 코멘트 예시(복붙)

* 리뷰 요청:

```
@codex review for correctness, Windows portability, uv usage, and CI readiness.
```

* 수정 요청(실패/개선 루프):

```
@codex Please fix the issues in this PR.

Constraints:
- Use uv only.
- Keep smoke lightweight.
Acceptance:
- uv sync --extra dev
- uv run smoke --profile local
- uv run lint
- uv run test
```

### CI/자동 코멘트가 실패할 때(자주 발생)

* 403 `Resource not accessible by integration`:

  * 레포 Settings → Actions → Workflow permissions가 **Read-only**면 코멘트 작성이 막힙니다.
  * fork PR은 보안 정책 때문에 write가 제한될 수 있습니다.

---

## 자주 겪는 문제 해결

### 1) `uv run smoke`가 `program not found`

* 원인: `project.scripts` 엔트리포인트가 설치되지 않음
* 해결: 해당 프로젝트 `pyproject.toml`에 아래가 있는지 확인 후 다시 `uv sync`

```toml
[tool.uv]
package = true
```

### 2) 경로/프로필 문제

* `configs/colab.yaml`이 없으면 Colab 런처가 자동으로 `local` 프로필로 fallback하도록 구성 가능
* 프로젝트별로 `configs/local.yaml`, `configs/colab.yaml`를 운영하면 환경 전환이 쉬워집니다.

---

## 라이선스 / 기여

* 포트폴리오 목적이므로 배포/운영은 목표가 아닙니다.
* PR은 환영하지만, 큰 바이너리/모델 파일은 커밋하지 마세요.

```
