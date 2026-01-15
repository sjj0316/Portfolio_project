# Portfolio Launcher UI

Gradio-based web UI to launch the portfolio projects with clicks.

## Quickstart
```powershell
cd .\ai-portfolio\00_ui\portfolio_launcher
uv sync --extra web
uv run predict --profile local
```

## 사용법
- 브라우저에서 표시된 로컬 주소(예: `http://127.0.0.1:7860`)로 접속합니다.
- 좌측에서 프로젝트/프로파일/옵션을 선택합니다.
- `필수 extra 설치`는 프로젝트에 필요한 extra를 자동 선택해 설치합니다.
- `추가 동기화`는 수동으로 extra를 입력해 설치할 때 사용합니다.
- `템플릿`을 선택하고 `템플릿 적용`으로 예시 입력을 자동 채울 수 있습니다.
- `최근값 불러오기`로 마지막 실행값을 복원할 수 있습니다.
- 우측에서 필요한 입력만 채우고 `실행`을 누릅니다.
- `명령 미리보기`에서 실행될 커맨드를 확인할 수 있습니다.
- 실행 로그는 `출력` 영역에 표시됩니다.
- `로그 저장`으로 실행 로그를 파일로 저장할 수 있습니다.

## Notes
- Use the "uv sync" buttons inside the UI to install dependencies per project.
- Commands are executed in the selected project's folder.
- Heavy extras (YOLO/Diffusers/etc.) are still optional and must be installed
  via `uv sync --extra <name>`.
