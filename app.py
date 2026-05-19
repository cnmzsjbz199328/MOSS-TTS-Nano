from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.parse
from pathlib import Path
from typing import Optional, Sequence

import uvicorn
from fastapi import FastAPI

from moss_tts_nano_runtime import (
    DEFAULT_AUDIO_TOKENIZER_PATH,
    DEFAULT_CHECKPOINT_PATH,
    DEFAULT_OUTPUT_DIR,
    NanoTTSService,
)
from text_normalization_pipeline import (
    WeTextProcessingManager as SharedWeTextProcessingManager,
)

from core.models import DemoEntry
from core.warmup import WarmupManager
from core.stream_manager import RequestRuntimeManager, StreamingJobManager
from api import build_router
from api.routes import _render_index_html  # re-exported for app_onnx.py compatibility
APP_DIR = Path(__file__).resolve().parent
DEMO_METADATA_PATH = APP_DIR / "assets" / "demo.jsonl"
PROMPT_UPLOAD_DIR = APP_DIR / ".app_prompt_uploads"



def _load_demo_entries() -> list[DemoEntry]:
    if not DEMO_METADATA_PATH.is_file():
        logging.warning("demo metadata file not found: %s", DEMO_METADATA_PATH)
        return []

    demo_entries: list[DemoEntry] = []
    for line_index, raw_line in enumerate(DEMO_METADATA_PATH.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            logging.warning("failed to parse demo metadata line=%s path=%s", line_index, DEMO_METADATA_PATH, exc_info=True)
            continue

        prompt_audio_relative_path = str(payload.get("role", "")).strip()
        text = str(payload.get("text", "")).strip()
        if not prompt_audio_relative_path or not text:
            logging.warning("skip invalid demo metadata line=%s role/text missing", line_index)
            continue

        prompt_audio_path = (APP_DIR / prompt_audio_relative_path).resolve()
        if not prompt_audio_path.is_file():
            logging.warning(
                "skip demo metadata line=%s prompt speech missing: %s",
                line_index,
                prompt_audio_path,
            )
            continue

        try:
            prompt_audio_relative_path = str(prompt_audio_path.relative_to(APP_DIR))
        except ValueError:
            logging.warning(
                "skip demo metadata line=%s prompt speech escaped app dir: %s",
                line_index,
                prompt_audio_path,
            )
            continue

        demo_index = len(demo_entries) + 1
        name = str(payload.get("name", "")).strip() or f"Demo {demo_index}: {prompt_audio_path.stem}"
        demo_entries.append(
            DemoEntry(
                demo_id=f"demo-{demo_index}",
                name=name,
                prompt_audio_path=prompt_audio_path,
                prompt_audio_relative_path=prompt_audio_relative_path,
                text=text,
            )
        )
    return demo_entries


def _resolve_vscode_root_path(vscode_proxy_uri: Optional[str], server_port: int) -> Optional[str]:
    if not vscode_proxy_uri:
        return None
    raw = vscode_proxy_uri.strip()
    if not raw or raw == "/":
        return None

    port_str = str(server_port)
    replacements = (
        "{{port}}",
        "{port}",
        "%7B%7Bport%7D%7D",
        "%7b%7bport%7d%7d",
        "%7Bport%7D",
        "%7bport%7d",
    )
    resolved = raw
    for token in replacements:
        resolved = resolved.replace(token, port_str)

    parsed = urllib.parse.urlsplit(resolved)
    if parsed.scheme and parsed.netloc:
        path = parsed.path or "/"
    else:
        path = resolved

    if not path.startswith("/"):
        path = "/" + path
    normalized = path.rstrip("/")
    return normalized or None




def _build_app(
    runtime: NanoTTSService,
    warmup_manager: WarmupManager,
    text_normalizer_manager: "SharedWeTextProcessingManager | None",
    root_path: str | None,
) -> FastAPI:
    app = FastAPI(title="MOSS-TTS-Nano Demo", root_path=root_path or "")
    stream_jobs = StreamingJobManager()
    runtime_manager = RequestRuntimeManager(runtime)
    demo_entries = _load_demo_entries()
    demo_entries_by_id = {demo_entry.demo_id: demo_entry for demo_entry in demo_entries}
    router = build_router(
        runtime=runtime,
        warmup_manager=warmup_manager,
        text_normalizer_manager=text_normalizer_manager,
        demo_entries=demo_entries,
        demo_entries_by_id=demo_entries_by_id,
        stream_jobs=stream_jobs,
        runtime_manager=runtime_manager,
        prompt_upload_dir=PROMPT_UPLOAD_DIR,
        root_path=root_path or "",
    )
    app.include_router(router)
    return app


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="MOSS-TTS-Nano web demo")
    parser.add_argument("--checkpoint-path", "--checkpoint_path", dest="checkpoint_path", type=str, default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument(
        "--audio-tokenizer-path",
        "--audio_tokenizer_path",
        dest="audio_tokenizer_path",
        type=str,
        default=str(DEFAULT_AUDIO_TOKENIZER_PATH),
    )
    parser.add_argument("--output-dir", "--output_dir", dest="output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "auto"])
    parser.add_argument("--dtype", type=str, default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument(
        "--attn-implementation",
        "--attn_implementation",
        dest="attn_implementation",
        type=str,
        default="auto",
        choices=["auto", "sdpa", "eager"],
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=18083)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        level=logging.INFO,
    )

    resolved_runtime_device = "cpu"
    if args.device != "cpu":
        logging.info("CPU-only app mode: ignoring --device=%s and forcing cpu.", args.device)

    runtime = NanoTTSService(
        checkpoint_path=args.checkpoint_path,
        audio_tokenizer_path=args.audio_tokenizer_path,
        device=resolved_runtime_device,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        output_dir=args.output_dir,
    )
    text_normalizer_manager = SharedWeTextProcessingManager()
    text_normalizer_manager.start()
    warmup_manager = WarmupManager(runtime, text_normalizer_manager=text_normalizer_manager)
    warmup_manager.start()

    vscode_proxy_uri = os.getenv("VSCODE_PROXY_URI", "")
    root_path = _resolve_vscode_root_path(vscode_proxy_uri, args.port)
    logging.info("root_path=%s", root_path)
    if args.share:
        logging.warning("--share is ignored by the FastAPI-based Nano-TTS app.")

    app = _build_app(runtime, warmup_manager, text_normalizer_manager, root_path)
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        root_path=root_path or "",
    )


if __name__ == "__main__":
    main()
