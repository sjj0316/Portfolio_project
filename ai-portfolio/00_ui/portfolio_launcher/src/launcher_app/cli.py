"""CLI entrypoints for the portfolio launcher UI."""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CommandSpec:
    label: str
    args: tuple[str, ...]
    use_profile: bool = True
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProjectSpec:
    name: str
    rel_path: Path
    commands: tuple[CommandSpec, ...]
    hint: str
    description: str
    usage: str
    required_extras: tuple[str, ...]
    templates: dict[str, dict[str, dict[str, Any]]]


FIELD_FLAGS: dict[str, str] = {
    "text": "--text",
    "source": "--source",
    "model_yolo": "--model",
    "imgsz": "--imgsz",
    "conf": "--conf",
    "prompt": "--prompt",
    "steps": "--steps",
    "model_img": "--model",
    "features": "--features",
    "transport": "--transport",
    "host": "--host",
    "port": "--port",
}


NUMERIC_FIELDS: dict[str, tuple[type, float | None, float | None]] = {
    "imgsz": (int, 1, None),
    "steps": (int, 1, None),
    "conf": (float, 0.0, 1.0),
    "port": (int, 1, 65535),
}


PROJECT_SPECS: tuple[ProjectSpec, ...] = (
    ProjectSpec(
        name="감성 분석 (NLP)",
        rel_path=Path("ai-portfolio/01_nlp/nlp_sentiment_starter"),
        commands=(
            CommandSpec("연결 점검", ("smoke",)),
            CommandSpec("학습", ("train",)),
            CommandSpec("평가", ("eval",)),
            CommandSpec("예측", ("predict",), required=("text",)),
        ),
        hint=(
            "예측은 --text 입력이 필요합니다. 실행 로그에 예시가 보이면 그대로 따라 해 보세요.\n"
            "추가 모델(Transformers)은 `uv sync --extra transformers`로 설치합니다."
        ),
        description=(
            "간단한 감성 분석 데모입니다. 기본은 가벼운 규칙/모델로 동작하고, "
            "선택적으로 scikit-learn이나 Transformers로 확장할 수 있습니다."
        ),
        usage=(
            "예시 명령어:\n"
            "  uv run predict --profile local --text \"this is surprisingly good\"\n\n"
            "예시 출력:\n"
            "  label: positive\n"
            "  score: 0.86"
        ),
        required_extras=(),
        templates={
            "예측": {
                "기본": {"text": "this is surprisingly good"},
                "부정 예시": {"text": "this is terrible"},
            }
        },
    ),
    ProjectSpec(
        name="실시간 YOLO (CV)",
        rel_path=Path("ai-portfolio/02_cv/realtime_yolo_starter"),
        commands=(
            CommandSpec("연결 점검", ("smoke",)),
            CommandSpec(
                "실행",
                ("predict",),
                required=("source",),
                optional=("model_yolo", "imgsz", "conf"),
                flags=("save",),
            ),
        ),
        hint=(
            "실행 전 `uv sync --extra yolo`가 필요합니다.\n"
            "웹캠은 --source 0, 파일은 --source path 로 지정하세요."
        ),
        description=(
            "웹캠/영상/이미지에서 실시간 객체 감지를 수행합니다. "
            "Ultralytics YOLO와 OpenCV가 필요합니다."
        ),
        usage=(
            "예시 명령어:\n"
            "  uv run predict --profile local --source 0 --model yolo11n.pt --save\n\n"
            "예시 출력:\n"
            "  [ok] model=yolo11n.pt source=0 imgsz=640 conf=0.25\n"
            "  [done] predict items=1 outputs=...\\outputs\\yolo"
        ),
        required_extras=("yolo",),
        templates={
            "실행": {
                "웹캠": {
                    "source": "0",
                    "model_yolo": "yolo11n.pt",
                    "imgsz": "640",
                    "conf": "0.25",
                    "save": True,
                },
                "샘플 파일": {
                    "source": "data/sample.mp4",
                    "model_yolo": "yolo11n.pt",
                    "imgsz": "640",
                    "conf": "0.25",
                    "save": True,
                },
            }
        },
    ),
    ProjectSpec(
        name="텍스트 -> 이미지",
        rel_path=Path("ai-portfolio/03_image_gen/image_gen_starter"),
        commands=(
            CommandSpec("연결 점검", ("smoke",)),
            CommandSpec(
                "생성",
                ("predict",),
                required=("prompt",),
                optional=("steps", "model_img"),
            ),
        ),
        hint=(
            "실행 전 `uv sync --extra diffusers`가 필요합니다.\n"
            "--prompt로 프롬프트를 지정하고 --steps로 속도를 조절합니다."
        ),
        description=(
            "텍스트 프롬프트를 이미지로 생성하는 데모입니다. "
            "기본 모델은 가볍게 테스트 가능한 tiny pipeline입니다."
        ),
        usage=(
            "예시 명령어:\n"
            "  uv run predict --profile local --prompt \"a cute robot on a desk\" --steps 2\n\n"
            "예시 출력:\n"
            "  [ok] saved outputs/images/robot.png"
        ),
        required_extras=("diffusers",),
        templates={
            "생성": {
                "로봇": {"prompt": "a cute robot on a desk", "steps": "2", "model_img": ""},
                "풍경": {"prompt": "sunset over a calm lake", "steps": "4", "model_img": ""},
            }
        },
    ),
    ProjectSpec(
        name="탭형 분류",
        rel_path=Path("ai-portfolio/04_tabular/tabular_classification_starter"),
        commands=(
            CommandSpec("연결 점검", ("smoke",)),
            CommandSpec("학습", ("train",)),
            CommandSpec("평가", ("eval",)),
            CommandSpec("예측", ("predict",), required=("features",)),
        ),
        hint=(
            "기본은 경량 샘플로 동작합니다. sklearn 기반은 `uv sync --extra ml`로 설치합니다.\n"
            "--features \"0.5,0.1,0.3\" 형태로 입력하세요."
        ),
        description=(
            "작은 탭형 데이터 분류 데모입니다. 기본 로직은 가볍게 동작하고, "
            "scikit-learn이 설치되면 베이스라인 모델을 사용합니다."
        ),
        usage=(
            "예시 명령어:\n"
            "  uv run predict --profile local --features \"0.5,0.1,0.3\"\n\n"
            "예시 출력:\n"
            "  label: class_1\n"
            "  score: 0.72"
        ),
        required_extras=(),
        templates={
            "예측": {
                "기본": {"features": "0.5,0.1,0.3"},
                "다른 샘플": {"features": "0.2,0.4,0.7"},
            }
        },
    ),
    ProjectSpec(
        name="MCP 감성 서버",
        rel_path=Path("ai-portfolio/04_agents_mcp/mcp_sentiment_server"),
        commands=(
            CommandSpec("연결 점검", ("smoke",), use_profile=False),
            CommandSpec(
                "서버 시작",
                ("serve",),
                use_profile=False,
                required=("transport",),
                optional=("host", "port"),
            ),
        ),
        hint=(
            "serve는 기본으로 http 8000 포트를 사용합니다.\n"
            "--transport stdio 또는 --port로 포트를 지정할 수 있습니다."
        ),
        description=(
            "MCP 프로토콜로 감성 분석 기능을 제공하는 서버입니다. "
            "클라이언트가 도구 호출을 통해 감성 분석을 요청할 수 있습니다."
        ),
        usage=(
            "예시 명령어:\n"
            "  uv run serve --transport streamable-http --port 8000\n\n"
            "예시 출력:\n"
            "  [ok] server listening on http://127.0.0.1:8000"
        ),
        required_extras=(),
        templates={
            "서버 시작": {
                "HTTP": {"transport": "streamable-http", "port": "8000", "host": ""},
                "STDIO": {"transport": "stdio", "port": "", "host": ""},
            }
        },
    ),
    ProjectSpec(
        name="MCP YOLO 서버",
        rel_path=Path("ai-portfolio/04_agents_mcp/mcp_yolo_server"),
        commands=(
            CommandSpec("연결 점검", ("smoke",), use_profile=False),
            CommandSpec(
                "서버 시작",
                ("serve",),
                use_profile=False,
                required=("transport",),
                optional=("host", "port"),
            ),
        ),
        hint=(
            "YOLO 의존성은 별도 설치가 필요할 수 있습니다.\n"
            "서버가 뜨면 MCP 클라이언트에서 호출하세요."
        ),
        description=(
            "MCP 기반으로 YOLO 추론을 제공하는 서버입니다. "
            "영상/이미지를 입력받아 객체 감지 결과를 반환합니다."
        ),
        usage=(
            "예시 명령어:\n"
            "  uv run serve --transport streamable-http --port 8001\n\n"
            "예시 출력:\n"
            "  [ok] server listening on http://127.0.0.1:8001"
        ),
        required_extras=(),
        templates={
            "서버 시작": {
                "HTTP": {"transport": "streamable-http", "port": "8001", "host": ""},
                "STDIO": {"transport": "stdio", "port": "", "host": ""},
            }
        },
    ),
)


@dataclass(frozen=True)
class Project:
    name: str
    path: Path
    commands: dict[str, CommandSpec]
    hint: str
    description: str
    usage: str
    required_extras: tuple[str, ...]
    templates: dict[str, dict[str, dict[str, Any]]]


_PROCESS_LOCK = threading.Lock()
_ACTIVE_PROCESS: subprocess.Popen[str] | None = None


def find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for parent in (current, *current.parents):
        if (parent / "ai-portfolio").is_dir() and (parent / "AGENTS.md").is_file():
            return parent
    raise FileNotFoundError("Unable to locate repo root (expected AGENTS.md + ai-portfolio/).")


def build_projects(repo_root: Path) -> dict[str, Project]:
    projects: dict[str, Project] = {}
    for spec in PROJECT_SPECS:
        commands = {cmd.label: cmd for cmd in spec.commands}
        projects[spec.name] = Project(
            spec.name,
            repo_root / spec.rel_path,
            commands,
            spec.hint,
            spec.description,
            spec.usage,
            spec.required_extras,
            spec.templates,
        )
    return projects


def setup_cmd() -> None:
    print("Recommended setup:")
    print("  uv sync")
    print("Optional extras:")
    print("  uv sync --extra dev")


def lint_cmd() -> None:
    subprocess.call(["python", "-m", "ruff", "check", "."])


def format_cmd() -> None:
    subprocess.call(["python", "-m", "ruff", "format", "."])


def test_cmd() -> None:
    subprocess.call(["python", "-m", "pytest"])


def clean_cmd() -> None:
    print("[info] clean is a no-op for the launcher.")


def smoke_cmd() -> None:
    repo_root = find_repo_root(Path.cwd())
    projects = build_projects(repo_root)
    missing = [p for p in projects.values() if not p.path.exists()]
    if missing:
        for project in missing:
            print(f"[error] missing project path: {project.path}")
        raise SystemExit(1)
    print(f"[ok] repo_root={repo_root}")
    for project in projects.values():
        print(f"[ok] project={project.name} path={project.path}")
    print("[done] smoke")


def predict_cmd() -> None:
    run_web_ui()


def run_web_ui() -> None:
    try:
        import gradio as gr
    except Exception as exc:  # pragma: no cover - user-facing import error
        raise RuntimeError(
            "Missing optional dependency: 'gradio'.\n"
            "Install it with: uv sync --extra web"
        ) from exc

    repo_root = find_repo_root(Path.cwd())
    projects = build_projects(repo_root)
    project_names = list(projects.keys())
    state_path = Path.cwd() / "outputs" / "launcher_state.json"

    def _run_subprocess_stream(cmd: list[str], cwd: Path):
        global _ACTIVE_PROCESS
        output_lines: list[str] = []
        with _PROCESS_LOCK:
            if _ACTIVE_PROCESS and _ACTIVE_PROCESS.poll() is None:
                output_lines.append("[warn] a command is already running.")
                yield "\n".join(output_lines)
                return
            try:
                _ACTIVE_PROCESS = subprocess.Popen(
                    cmd,
                    cwd=str(cwd),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                output_lines.append("[error] uv not found in PATH.")
                yield "\n".join(output_lines)
                return

        output_lines.append("> " + subprocess.list2cmdline(cmd))
        yield "\n".join(output_lines)

        assert _ACTIVE_PROCESS and _ACTIVE_PROCESS.stdout
        for line in _ACTIVE_PROCESS.stdout:
            output_lines.append(line.rstrip("\n"))
            yield "\n".join(output_lines)
        code = _ACTIVE_PROCESS.wait()
        if code != 0:
            output_lines.append(f"[error] exit_code={code}")
        else:
            output_lines.append(f"[done] exit_code={code}")
        with _PROCESS_LOCK:
            _ACTIVE_PROCESS = None
        yield "\n".join(output_lines)

    def _ascii_slug(value: str) -> str:
        cleaned = "".join(ch if ch.isascii() and ch.isalnum() else "_" for ch in value)
        cleaned = cleaned.strip("_")
        return cleaned or "project"

    def load_state() -> dict[str, Any]:
        if not state_path.exists():
            return {}
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def save_state(state: dict[str, Any]) -> None:
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(json.dumps(state, ensure_ascii=True, indent=2), encoding="utf-8")

    def save_recent(project_name: str, command_label: str, profile: str, values: dict[str, Any]) -> None:
        state = load_state()
        project_state = state.setdefault(project_name, {})
        cmd_state = project_state.setdefault(command_label, {})
        cmd_state["profile"] = profile
        cmd_state["values"] = values
        save_state(state)

    def load_recent(project_name: str, command_label: str) -> dict[str, Any]:
        state = load_state()
        return state.get(project_name, {}).get(command_label, {})

    def save_output(project_name: str, output_text: str) -> str:
        if not output_text.strip():
            gr.Warning("출력 로그가 없습니다.")
            return "[error] 출력 로그가 없습니다."
        logs_dir = Path.cwd() / "outputs" / "launcher_logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _ascii_slug(project_name)
        log_path = logs_dir / f"{slug}_{ts}.log"
        log_path.write_text(output_text, encoding="utf-8")
        return output_text + f"\n[ok] log_saved={log_path}"

    def run_sync(project: Project):
        return _run_subprocess_stream(["uv", "sync"], project.path)

    def run_sync_extra(project: Project, extra: str):
        extra = extra.strip()
        if not extra:
            gr.Warning("추가 옵션 이름을 입력하세요.")
            return "[error] 추가 옵션 이름을 입력하세요."
        return _run_subprocess_stream(["uv", "sync", "--extra", extra], project.path)

    def build_command(
        project_name: str,
        command_label: str,
        profile: str,
        values: dict[str, Any],
    ) -> tuple[list[str] | None, str | None]:
        project = projects[project_name]
        if command_label not in project.commands:
            return None, "[error] 명령을 찾을 수 없습니다."
        cmd_spec = project.commands[command_label]
        args = list(cmd_spec.args)

        for key in cmd_spec.required:
            if not str(values.get(key, "")).strip():
                return None, f"[error] '{key}' 값을 입력하세요."
            args.extend([FIELD_FLAGS[key], str(values[key]).strip()])

        for key in cmd_spec.optional:
            value = str(values.get(key, "")).strip()
            if value:
                args.extend([FIELD_FLAGS[key], value])

        for key in cmd_spec.flags:
            if bool(values.get(key, False)):
                args.append(f"--{key}")

        for key, (num_type, min_val, max_val) in NUMERIC_FIELDS.items():
            value = str(values.get(key, "")).strip()
            if not value:
                continue
            if key not in cmd_spec.required and key not in cmd_spec.optional:
                continue
            try:
                parsed = num_type(value)
            except ValueError:
                return None, f"[error] '{key}' 값이 숫자가 아닙니다."
            if min_val is not None and parsed < min_val:
                return None, f"[error] '{key}' 값은 {min_val} 이상이어야 합니다."
            if max_val is not None and parsed > max_val:
                return None, f"[error] '{key}' 값은 {max_val} 이하여야 합니다."

        cmd = ["uv", "run", *args]
        if cmd_spec.use_profile:
            cmd.extend(["--profile", profile])
        return cmd, None

    def visible_fields(project_name: str, command_label: str) -> dict[str, bool]:
        project = projects[project_name]
        cmd = project.commands.get(command_label)
        if not cmd:
            return {k: False for k in FIELD_FLAGS.keys()} | {"save": False, "profile": True}
        needed = set(cmd.required) | set(cmd.optional) | set(cmd.flags)
        vis = {k: k in needed for k in FIELD_FLAGS.keys()}
        vis["save"] = "save" in needed
        vis["profile"] = cmd.use_profile
        return vis

    def template_options(project_name: str, command_label: str):
        templates = projects[project_name].templates.get(command_label, {})
        names = list(templates.keys())
        default = names[0] if names else ""
        return names, default, bool(names)

    def update_project(project_name: str):
        project = projects[project_name]
        commands = list(project.commands.keys())
        default_cmd = commands[0] if commands else ""
        vis = visible_fields(project_name, default_cmd)
        extras = list(project.required_extras)
        default_extra = extras[0] if extras else ""
        template_names, template_default, template_visible = template_options(project_name, default_cmd)
        return (
            gr.update(choices=commands, value=default_cmd),
            project.description,
            project.hint,
            project.usage,
            gr.update(visible=vis["profile"]),
            gr.update(visible=vis["text"]),
            gr.update(visible=vis["source"]),
            gr.update(visible=vis["model_yolo"]),
            gr.update(visible=vis["imgsz"]),
            gr.update(visible=vis["conf"]),
            gr.update(visible=vis["save"]),
            gr.update(visible=vis["prompt"]),
            gr.update(visible=vis["steps"]),
            gr.update(visible=vis["model_img"]),
            gr.update(visible=vis["features"]),
            gr.update(visible=vis["transport"]),
            gr.update(visible=vis["host"]),
            gr.update(visible=vis["port"]),
            gr.update(choices=extras, value=default_extra),
            gr.update(visible=bool(extras)),
            gr.update(choices=template_names, value=template_default, visible=template_visible),
        )

    def update_command(project_name: str, command_label: str):
        vis = visible_fields(project_name, command_label)
        template_names, template_default, template_visible = template_options(project_name, command_label)
        return (
            gr.update(visible=vis["profile"]),
            gr.update(visible=vis["text"]),
            gr.update(visible=vis["source"]),
            gr.update(visible=vis["model_yolo"]),
            gr.update(visible=vis["imgsz"]),
            gr.update(visible=vis["conf"]),
            gr.update(visible=vis["save"]),
            gr.update(visible=vis["prompt"]),
            gr.update(visible=vis["steps"]),
            gr.update(visible=vis["model_img"]),
            gr.update(visible=vis["features"]),
            gr.update(visible=vis["transport"]),
            gr.update(visible=vis["host"]),
            gr.update(visible=vis["port"]),
            gr.update(choices=template_names, value=template_default, visible=template_visible),
        )

    def update_preview(
        project_name: str,
        command_label: str,
        profile: str,
        text: str,
        source: str,
        model_yolo: str,
        imgsz: str,
        conf: str,
        save: bool,
        prompt: str,
        steps: str,
        model_img: str,
        features: str,
        transport: str,
        host: str,
        port: str,
    ) -> str:
        values = {
            "text": text,
            "source": source,
            "model_yolo": model_yolo,
            "imgsz": imgsz,
            "conf": conf,
            "save": save,
            "prompt": prompt,
            "steps": steps,
            "model_img": model_img,
            "features": features,
            "transport": transport,
            "host": host,
            "port": port,
        }
        cmd, error = build_command(project_name, command_label, profile, values)
        if error:
            return error
        if not cmd:
            return "[error] 명령을 만들 수 없습니다."
        return subprocess.list2cmdline(cmd)

    def run_selected(
        project_name: str,
        command_label: str,
        profile: str,
        text: str,
        source: str,
        model_yolo: str,
        imgsz: str,
        conf: str,
        save: bool,
        prompt: str,
        steps: str,
        model_img: str,
        features: str,
        transport: str,
        host: str,
        port: str,
    ):
        values = {
            "text": text,
            "source": source,
            "model_yolo": model_yolo,
            "imgsz": imgsz,
            "conf": conf,
            "save": save,
            "prompt": prompt,
            "steps": steps,
            "model_img": model_img,
            "features": features,
            "transport": transport,
            "host": host,
            "port": port,
        }
        cmd, error = build_command(project_name, command_label, profile, values)
        if error:
            gr.Warning(error)
            return error
        if not cmd:
            gr.Warning("명령을 만들 수 없습니다.")
            return "[error] 명령을 만들 수 없습니다."
        save_recent(project_name, command_label, profile, values)
        project = projects[project_name]
        return _run_subprocess_stream(cmd, project.path)

    def apply_template(project_name: str, command_label: str, template_name: str):
        templates = projects[project_name].templates.get(command_label, {})
        values = templates.get(template_name, {})
        return (
            values.get("text", ""),
            values.get("source", "0"),
            values.get("model_yolo", "yolo11n.pt"),
            values.get("imgsz", "640"),
            values.get("conf", "0.25"),
            values.get("save", True),
            values.get("prompt", ""),
            values.get("steps", "2"),
            values.get("model_img", ""),
            values.get("features", ""),
            values.get("transport", "streamable-http"),
            values.get("host", ""),
            values.get("port", "8000"),
        )

    def load_recent_values(project_name: str, command_label: str):
        recent = load_recent(project_name, command_label)
        values = recent.get("values", {}) if isinstance(recent, dict) else {}
        return (
            values.get("text", ""),
            values.get("source", "0"),
            values.get("model_yolo", "yolo11n.pt"),
            values.get("imgsz", "640"),
            values.get("conf", "0.25"),
            values.get("save", True),
            values.get("prompt", ""),
            values.get("steps", "2"),
            values.get("model_img", ""),
            values.get("features", ""),
            values.get("transport", "streamable-http"),
            values.get("host", ""),
            values.get("port", "8000"),
        )

    header = (
        "# AI Portfolio Launcher\n"
        "좌측에서 프로젝트/프로파일/옵션을 고르고, 우측에서 필요한 입력만 채워 실행하세요.\n"
        "서버 주소는 실행 로그에 표시됩니다."
    )

    with gr.Blocks(title="AI Portfolio Launcher") as demo:
        gr.Markdown(header)
        with gr.Row():
            with gr.Column(scale=1):
                project = gr.Dropdown(project_names, value=project_names[0], label="프로젝트")
                profile = gr.Radio(["local", "colab"], value="local", label="프로파일")
                extra_choice = gr.Dropdown(label="필수 extra", choices=[], value="")
                extra = gr.Textbox(label="추가 옵션", placeholder="yolo / diffusers / ml")
                sync_btn = gr.Button("동기화 (uv sync)")
                sync_extra_btn = gr.Button("추가 동기화 (uv sync --extra)")
                sync_required_btn = gr.Button("필수 extra 설치")
                command = gr.Dropdown(label="명령")
                template = gr.Dropdown(label="템플릿")
                preview = gr.Textbox(label="명령 미리보기", interactive=False)
            with gr.Column(scale=2):
                description = gr.Textbox(label="프로젝트 설명", lines=3, interactive=False)
                hint = gr.Textbox(label="힌트", lines=3, interactive=False)
                usage = gr.Textbox(label="사용법 (예시)", lines=6, interactive=False)
                text = gr.Textbox(label="text", placeholder="this is surprisingly good")
                source = gr.Textbox(label="source", value="0")
                model_yolo = gr.Textbox(label="model", value="yolo11n.pt")
                imgsz = gr.Textbox(label="imgsz", value="640")
                conf = gr.Textbox(label="conf", value="0.25")
                save = gr.Checkbox(label="save", value=True)
                prompt = gr.Textbox(label="prompt", placeholder="a cute robot on a desk")
                steps = gr.Textbox(label="steps", value="2")
                model_img = gr.Textbox(label="model (optional)", value="")
                features = gr.Textbox(label="features", placeholder="0.5,0.1,0.3")
                transport = gr.Dropdown(["streamable-http", "stdio"], value="streamable-http", label="transport")
                host = gr.Textbox(label="host (optional)", value="")
                port = gr.Textbox(label="port (optional)", value="8000")
                apply_template_btn = gr.Button("템플릿 적용")
                load_recent_btn = gr.Button("최근값 불러오기")
                run_btn = gr.Button("실행")
                save_log_btn = gr.Button("로그 저장")
                output = gr.Textbox(label="출력", lines=18, interactive=False)

        project.change(
            update_project,
            inputs=project,
            outputs=[
                command,
                description,
                hint,
                usage,
                profile,
                text,
                source,
                model_yolo,
                imgsz,
                conf,
                save,
                prompt,
                steps,
                model_img,
                features,
                transport,
                host,
                port,
                extra_choice,
                sync_required_btn,
                template,
            ],
        )

        command.change(
            update_command,
            inputs=[project, command],
            outputs=[
                profile,
                text,
                source,
                model_yolo,
                imgsz,
                conf,
                save,
                prompt,
                steps,
                model_img,
                features,
                transport,
                host,
                port,
                template,
            ],
        )

        inputs_for_preview = [
            project,
            command,
            profile,
            text,
            source,
            model_yolo,
            imgsz,
            conf,
            save,
            prompt,
            steps,
            model_img,
            features,
            transport,
            host,
            port,
        ]
        for component in inputs_for_preview:
            component.change(update_preview, inputs=inputs_for_preview, outputs=preview)

        sync_btn.click(lambda p: run_sync(projects[p]), inputs=project, outputs=output)
        sync_extra_btn.click(lambda p, e: run_sync_extra(projects[p], e), inputs=[project, extra], outputs=output)
        sync_required_btn.click(
            lambda p, e: run_sync_extra(projects[p], e),
            inputs=[project, extra_choice],
            outputs=output,
        )
        apply_template_btn.click(
            apply_template,
            inputs=[project, command, template],
            outputs=[
                text,
                source,
                model_yolo,
                imgsz,
                conf,
                save,
                prompt,
                steps,
                model_img,
                features,
                transport,
                host,
                port,
            ],
        )
        load_recent_btn.click(
            load_recent_values,
            inputs=[project, command],
            outputs=[
                text,
                source,
                model_yolo,
                imgsz,
                conf,
                save,
                prompt,
                steps,
                model_img,
                features,
                transport,
                host,
                port,
            ],
        )
        run_btn.click(run_selected, inputs=inputs_for_preview, outputs=output)
        save_log_btn.click(save_output, inputs=[project, output], outputs=output)

        demo.load(
            lambda: update_project(project_names[0]),
            outputs=[
                command,
                description,
                hint,
                usage,
                profile,
                text,
                source,
                model_yolo,
                imgsz,
                conf,
                save,
                prompt,
                steps,
                model_img,
                features,
                transport,
                host,
                port,
                extra_choice,
                sync_required_btn,
                template,
            ],
        )
        demo.load(
            lambda: update_preview(
                project_names[0],
                list(projects[project_names[0]].commands.keys())[0],
                "local",
                "",
                "0",
                "yolo11n.pt",
                "640",
                "0.25",
                True,
                "",
                "2",
                "",
                "",
                "streamable-http",
                "",
                "8000",
            ),
            outputs=preview,
        )

    demo.launch()


if __name__ == "__main__":
    predict_cmd()
