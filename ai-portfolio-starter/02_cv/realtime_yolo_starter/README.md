# CV: Realtime YOLO Starter (Ultralytics)

최소 실행 목표: Ultralytics YOLO로 웹캠(로컬) 또는 영상 파일(Colab) 추론을 돌립니다.

## 로컬(Windows) - 웹캠 바로 실행
```powershell
cd realtime_yolo_starter
uv run python scripts\predict.py --profile local --source 0
```
- `source 0`은 기본 웹캠을 의미합니다.

## Colab - 영상 파일로 실행
Colab은 웹캠이 제한적이라 보통 파일로 돌립니다.
```python
!pip -q install ultralytics opencv-python pyyaml
!python scripts/predict.py --profile colab --source /content/sample.mp4
```
- `/content/sample.mp4`는 직접 업로드하거나 Drive에서 가져오세요.

## 모델
기본값은 `yolo11n.pt`입니다. 필요하면 `configs/*.yaml`에서 바꾸세요.

## 참고(파이썬 버전)
최신 Python이 곧바로 모든 딥러닝 휠과 호환되지는 않습니다(특히 Windows).
이 템플릿은 3.13을 기본 추천합니다.
