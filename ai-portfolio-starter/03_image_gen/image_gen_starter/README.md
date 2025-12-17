# Image Generation: Tiny Diffusers Starter

최소 실행 목표: Diffusers로 가벼운 테스트용 모델을 받아서 txt2img 1장을 생성합니다.
- 포트폴리오용으로는 구조/파이프라인/실험 재현성(설정 분리)을 보여주는 데 초점을 둡니다.

## 로컬(Windows) - uv로 바로 실행
```powershell
cd image_gen_starter
uv run python scripts\txt2img.py --profile local --prompt "a cute robot on a desk"
```

## Colab - 빠른 실행(권장)
```python
!pip -q install diffusers transformers accelerate safetensors torch pyyaml
!python scripts/txt2img.py --profile colab --prompt "a cozy room, warm lighting"
```

## 모델 교체(실사용 모델로)
기본 모델은 테스트용 tiny 모델입니다(다운로드가 가벼움).
`configs/*.yaml`의 `model_id`를 원하는 모델로 바꾸면 됩니다.

## 출력
- 생성 이미지는 `outputs/`에 저장됩니다.
