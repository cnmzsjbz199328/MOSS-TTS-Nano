from __future__ import annotations

import base64
import json
import logging
import os
import queue
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse

from moss_tts_nano_runtime import NanoTTSService
from text_normalization_pipeline import (
    TextNormalizationSnapshot as SharedTextNormalizationSnapshot,
    WeTextProcessingManager as SharedWeTextProcessingManager,
    prepare_tts_request_texts as shared_prepare_tts_request_texts,
)
from core.models import DemoEntry, StreamingJob, WarmupSnapshot
from core.audio import (
    _audio_to_wav_bytes,
    _audio_to_pcm16le_bytes,
    _read_audio_file_base64,
    _maybe_delete_file,
)
from core.warmup import WarmupManager
from core.stream_manager import RequestRuntimeManager, StreamingJobManager

def _warmup_status_text(snapshot: WarmupSnapshot) -> str:
    progress_pct = int(round(snapshot.progress * 100.0))
    if snapshot.failed:
        return f"Warmup failed: {snapshot.error or snapshot.message}"
    if snapshot.ready:
        return snapshot.message
    return f"Warmup in progress ({progress_pct}%): {snapshot.message}"


def _format_run_status(result: dict[str, object]) -> str:
    waveform_numpy = np.asarray(result["waveform_numpy"])
    sample_count = int(waveform_numpy.shape[0]) if waveform_numpy.ndim >= 1 else 0
    sample_rate = int(result["sample_rate"])
    audio_seconds = sample_count / sample_rate if sample_rate > 0 else 0.0
    global_attn = str(result.get("effective_global_attn_implementation", "unknown"))
    local_attn = str(result.get("effective_local_attn_implementation", global_attn))
    attn_summary = global_attn if global_attn == local_attn else f"{global_attn}/{local_attn}"
    tts_batch_size = result.get("voice_clone_chunk_batch_size")
    codec_batch_size = result.get("voice_clone_codec_batch_size")
    batch_summary = ""
    if tts_batch_size is not None or codec_batch_size is not None:
        batch_summary = f" | tts_batch={int(tts_batch_size or 1)} | codec_batch={int(codec_batch_size or 1)}"
    execution_summary = ""
    execution_device = result.get("execution_device")
    cpu_threads = result.get("cpu_threads")
    if execution_device:
        execution_summary = f" | exec={execution_device}"
        if cpu_threads is not None:
            execution_summary += f" | cpu_threads={int(cpu_threads)}"
    prompt_audio_display_path = str(result.get("prompt_audio_display_path") or "").strip()
    prompt_audio_path = str(result.get("prompt_audio_path") or "").strip()
    speaker_summary = f"voice={result['voice']}"
    if prompt_audio_display_path:
        if prompt_audio_display_path.lower().startswith("uploaded:"):
            speaker_summary = f"prompt={prompt_audio_display_path.split(':', 1)[1].strip()}"
        else:
            speaker_summary = f"prompt={Path(prompt_audio_display_path).stem}"
    elif prompt_audio_path:
        speaker_summary = f"prompt={Path(prompt_audio_path).stem}"
    return (
        f"Done | mode={result['mode']} | {speaker_summary} | "
        f"attn={attn_summary}{batch_summary}{execution_summary} | audio={audio_seconds:.2f}s | elapsed={float(result['elapsed_seconds']):.2f}s"
    )


def _format_stream_status(snapshot: dict[str, object]) -> str:
    if bool(snapshot.get("failed")):
        return f"Stream failed: {snapshot.get('error') or snapshot.get('run_status') or 'Unknown error'}"
    if bool(snapshot.get("ready")):
        return str(snapshot.get("run_status") or "Stream complete.")
    if bool(snapshot.get("closed")):
        return "Stream closed."
    return str(snapshot.get("run_status") or "Streaming...")


def _normalize_stream_chunk_index(
    raw_chunk_index: object,
    *,
    chunk_count: int,
    current_base: int | None,
) -> tuple[int | None, int | None]:
    try:
        numeric_chunk_index = int(raw_chunk_index)
    except Exception:
        return None, current_base

    if chunk_count <= 0:
        return max(0, numeric_chunk_index), current_base

    normalized_base = current_base
    if normalized_base is None:
        if numeric_chunk_index == 0:
            normalized_base = 0
        elif numeric_chunk_index == chunk_count:
            normalized_base = 1
        elif numeric_chunk_index == 1:
            normalized_base = 1
        else:
            normalized_base = 0

    normalized_chunk_index = numeric_chunk_index - normalized_base
    if 0 <= normalized_chunk_index < chunk_count:
        return normalized_chunk_index, normalized_base
    if 0 <= numeric_chunk_index < chunk_count:
        return numeric_chunk_index, 0
    if 1 <= numeric_chunk_index <= chunk_count:
        return numeric_chunk_index - 1, 1
    return None, normalized_base


def _coerce_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _sanitize_uploaded_prompt_filename(filename: str | None) -> str:
    base_name = Path(str(filename or "")).name.strip()
    if not base_name:
        return "prompt_speech.wav"
    return base_name


def _format_uploaded_prompt_display_name(filename: str | None) -> str:
    return f"Uploaded: {_sanitize_uploaded_prompt_filename(filename)}"


async def _persist_uploaded_prompt_audio(upload: UploadFile | None, *, prompt_upload_dir: Path) -> tuple[str | None, str | None]:
    if upload is None:
        return None, None

    original_filename = _sanitize_uploaded_prompt_filename(upload.filename)
    suffix = Path(original_filename).suffix
    if not suffix or len(suffix) > 16:
        suffix = ".wav"

    prompt_upload_dir.mkdir(parents=True, exist_ok=True)
    temp_path: str | None = None
    bytes_written = 0
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            delete=False,
            prefix="prompt-speech-",
            suffix=suffix,
            dir=str(prompt_upload_dir),
        ) as handle:
            temp_path = handle.name
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                bytes_written += len(chunk)
    finally:
        await upload.close()

    if not temp_path or bytes_written <= 0:
        _maybe_delete_file(temp_path)
        raise ValueError("Uploaded prompt speech is empty.")

    return temp_path, _format_uploaded_prompt_display_name(original_filename)


def _render_index_html(
    *,
    request: Request,
    runtime: NanoTTSService,
    demo_entries: list[DemoEntry],
    warmup_status: str,
    text_normalization_status: str,
) -> str:
    base_path = request.scope.get("root_path", "").rstrip("/")
    template_path = Path(__file__).resolve().parent.parent / "ui" / "templates" / "index.html"
    template = template_path.read_text(encoding="utf-8")
    demos_payload = [
        {
            "id": demo_entry.demo_id,
            "name": demo_entry.name,
            "prompt_speech": demo_entry.prompt_audio_relative_path,
            "text": demo_entry.text,
        }
        for demo_entry in demo_entries
    ]
    replacements = {
        "__APP_BASE__": json.dumps(base_path),
        "__DEMOS__": json.dumps(demos_payload, ensure_ascii=False),
        "__DEFAULT_DEMO_ID__": json.dumps(demo_entries[0].demo_id if demo_entries else ""),
        "__DEFAULT_ATTN_IMPLEMENTATION__": json.dumps(runtime.attn_implementation or "model_default"),
        "__DEFAULT_CPU_THREADS__": json.dumps(max(1, int(os.cpu_count() or 1))),
        "__WARMUP_STATUS__": warmup_status,
        "__TEXT_NORMALIZATION_STATUS__": text_normalization_status,
        "__CHECKPOINT__": str(runtime.checkpoint_path),
        "__AUDIO_TOKENIZER__": str(runtime.audio_tokenizer_path),
    }
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


def build_router(
    *,
    runtime: NanoTTSService,
    warmup_manager: WarmupManager,
    text_normalizer_manager: "SharedWeTextProcessingManager | None",
    demo_entries: list[DemoEntry],
    demo_entries_by_id: dict[str, DemoEntry],
    stream_jobs: StreamingJobManager,
    runtime_manager: RequestRuntimeManager,
    prompt_upload_dir: Path,
    root_path: str,
) -> APIRouter:
    """Return a configured APIRouter with all TTS API routes."""
    router = APIRouter()

    def _resolve_voice_clone_text_chunks(
        *,
        text: str,
        voice_clone_max_text_tokens: int,
        cpu_threads: int,
    ) -> list[str]:
        normalized_text = str(text or "").strip()
        if not normalized_text:
            return []
    
        try:
            chunks, _, _ = runtime_manager.call_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                callback=lambda selected_runtime: selected_runtime.split_voice_clone_text(
                    text=normalized_text,
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                ),
            )
        except Exception:
            logging.warning("failed to resolve playback text chunks", exc_info=True)
            return [normalized_text]
    
        normalized_chunks = [str(chunk).strip() for chunk in chunks if str(chunk).strip()]
        return normalized_chunks or [normalized_text]
    
    def _resolve_demo_entry(demo_id: str) -> DemoEntry:
        normalized_demo_id = str(demo_id or "").strip()
        if not normalized_demo_id:
            raise ValueError("demo_id is required.")
        demo_entry = demo_entries_by_id.get(normalized_demo_id)
        if demo_entry is None:
            raise ValueError(f"Unknown demo_id: {normalized_demo_id}")
        return demo_entry
    
    async def _resolve_prompt_audio_request(
        *,
        demo_id: str,
        prompt_audio: UploadFile | None,
    ) -> tuple[DemoEntry | None, str, str, str | None]:
        normalized_demo_id = str(demo_id or "").strip()
        demo_entry = _resolve_demo_entry(normalized_demo_id) if normalized_demo_id else None
    
        uploaded_prompt_audio_path, uploaded_prompt_audio_display_path = await _persist_uploaded_prompt_audio(prompt_audio, prompt_upload_dir=prompt_upload_dir)
        if uploaded_prompt_audio_path is not None and uploaded_prompt_audio_display_path is not None:
            return (
                demo_entry,
                uploaded_prompt_audio_path,
                uploaded_prompt_audio_display_path,
                uploaded_prompt_audio_path,
            )
    
        if demo_entry is None:
            raise ValueError("demo_id is required unless prompt speech is uploaded.")
    
        return (
            demo_entry,
            str(demo_entry.prompt_audio_path),
            demo_entry.prompt_audio_relative_path,
            None,
        )
    
    def _stream_metrics_text(snapshot: dict[str, object]) -> str:
        metrics = [
            f"state={snapshot['state']}",
            f"emitted={float(snapshot['emitted_audio_seconds']):.2f}s",
            f"lead={float(snapshot['lead_seconds']):.2f}s",
        ]
        first_audio_latency = snapshot.get("first_audio_latency_seconds")
        if first_audio_latency is not None:
            metrics.append(f"first_audio={float(first_audio_latency):.2f}s")
        return " | ".join(metrics)
    
    def _text_normalization_status_text(snapshot: SharedTextNormalizationSnapshot | None) -> str:
        if snapshot is None:
            return "WeTextProcessing disabled."
        if snapshot.failed:
            return f"{snapshot.message} error={snapshot.error}"
        return snapshot.message
    
    def _resolve_attn_for_runtime(selected_runtime: NanoTTSService, requested_attn: str) -> str:
        normalized = str(requested_attn or "model_default").strip().lower()
        if selected_runtime.device.type != "cpu":
            return requested_attn
        if normalized in {"", "auto", "default", "model_default", "flash_attention_2"}:
            return "eager"
        return requested_attn
    
    def _put_stream_audio(job: StreamingJob, pcm_bytes: bytes) -> None:
        while True:
            with job.lock:
                if job.is_closed:
                    return
            try:
                job.audio_queue.put(pcm_bytes, timeout=0.1)
                return
            except queue.Full:
                continue
    
    def _run_streaming_job(
        job: StreamingJob,
        *,
        text: str,
        prompt_audio_path: str,
        prompt_audio_display_path: str,
        prompt_audio_cleanup_path: str | None,
        max_new_frames: int,
        voice_clone_max_text_tokens: int,
        tts_max_batch_size: int,
        codec_max_batch_size: int,
        cpu_threads: int,
        attn_implementation: str,
        do_sample: bool,
        text_temperature: float,
        text_top_p: float,
        text_top_k: int,
        audio_temperature: float,
        audio_top_p: float,
        audio_top_k: int,
        audio_repetition_penalty: float,
        seed: int | None,
    ) -> None:
        try:
            initial_execution_label = "cpu"
            with job.lock:
                job.started_at = time.monotonic()
                job.state = "running"
                job.run_status = f"Streaming realtime audio... exec={initial_execution_label}"
    
            def _stream_factory(selected_runtime: NanoTTSService):
                return selected_runtime.synthesize_stream(
                    text=text,
                    mode="voice_clone",
                    voice=None,
                    prompt_audio_path=prompt_audio_path,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    attn_implementation=_resolve_attn_for_runtime(selected_runtime, attn_implementation),
                    do_sample=bool(do_sample),
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                    seed=seed,
                )
    
            for event, resolved_execution_device, resolved_cpu_threads in runtime_manager.iter_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                factory=_stream_factory,
            ):
                event_type = str(event.get("type", ""))
                with job.lock:
                    if job.is_closed:
                        break
    
                if event_type == "audio":
                    waveform_numpy = np.asarray(event["waveform_numpy"], dtype=np.float32)
                    pcm_bytes = _audio_to_pcm16le_bytes(waveform_numpy)
                    if not pcm_bytes:
                        continue
                    sample_rate = int(event["sample_rate"])
                    channels = 1 if waveform_numpy.ndim == 1 else int(waveform_numpy.shape[1])
                    is_pause = bool(event.get("is_pause", False))
                    event_duration_seconds = (
                        float(waveform_numpy.shape[0]) / float(sample_rate)
                        if sample_rate > 0 and waveform_numpy.ndim >= 1
                        else 0.0
                    )
                    with job.lock:
                        job.sample_rate = sample_rate
                        job.channels = channels
                        job.emitted_audio_seconds = float(event.get("emitted_audio_seconds", 0.0))
                        job.lead_seconds = float(event.get("lead_seconds", 0.0))
                        normalized_chunk_index, job.chunk_index_base = _normalize_stream_chunk_index(
                            event.get("chunk_index"),
                            chunk_count=len(job.text_chunks),
                            current_base=job.chunk_index_base,
                        )
                        if normalized_chunk_index is not None:
                            job.current_chunk_index = normalized_chunk_index
                            if not is_pause and event_duration_seconds > 0.0:
                                chunk_end_seconds = job.emitted_audio_seconds
                                chunk_start_seconds = max(0.0, chunk_end_seconds - event_duration_seconds)
                                job.audio_chunk_ranges.append(
                                    (chunk_start_seconds, chunk_end_seconds, normalized_chunk_index)
                                )
                        if job.first_audio_at is None and not is_pause:
                            job.first_audio_at = time.monotonic()
                        job.run_status = (
                            f"Streaming | emitted={job.emitted_audio_seconds:.2f}s | lead={job.lead_seconds:.2f}s"
                        )
                    _put_stream_audio(job, pcm_bytes)
                    continue
    
                if event_type == "result":
                    formatted_result = dict(event)
                    formatted_result["execution_device"] = resolved_execution_device
                    formatted_result["prompt_audio_display_path"] = prompt_audio_display_path
                    if resolved_cpu_threads is not None:
                        formatted_result["cpu_threads"] = resolved_cpu_threads
                    formatted_run_status = _format_run_status(formatted_result)
                    with job.lock:
                        job.final_result = {
                            "audio_path": event.get("audio_path"),
                            "prompt_audio_path": prompt_audio_display_path,
                            "run_status": formatted_run_status,
                            "text_chunks": list(job.text_chunks),
                        }
                        job.prompt_audio_path = prompt_audio_display_path
                        job.state = "done"
                        job.completed_at = time.monotonic()
                        job.run_status = formatted_run_status
        except Exception as exc:
            logging.exception("Nano-TTS realtime streaming job failed")
            with job.lock:
                job.state = "failed"
                job.error = str(exc)
                job.completed_at = time.monotonic()
                job.run_status = f"Stream failed: {exc}"
        finally:
            _maybe_delete_file(prompt_audio_cleanup_path)
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
    
    @router.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        return HTMLResponse(
            _render_index_html(
                request=request,
                runtime=runtime,
                demo_entries=demo_entries,
                warmup_status=_warmup_status_text(warmup_manager.snapshot()),
                text_normalization_status=_text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
            )
        )
    
    @router.get("/health")
    async def health():
        return {
            "status": "ok",
            "device": str(runtime.device),
            "dtype": str(runtime.dtype),
            "cpu_runtime_loaded": runtime_manager.is_cpu_runtime_loaded(),
            "default_cpu_threads": runtime_manager.default_cpu_threads,
            "attn_implementation": runtime.attn_implementation or "model_default",
            "checkpoint_default_attn_implementation": runtime._checkpoint_global_attn_implementation or "unknown",
            "checkpoint_default_local_attn_implementation": runtime._checkpoint_local_attn_implementation or "unknown",
            "configured_attn_implementation": runtime._configured_global_attn_implementation or "unknown",
            "configured_local_attn_implementation": runtime._configured_local_attn_implementation or "unknown",
            "checkpoint_path": str(runtime.checkpoint_path),
            "audio_tokenizer_path": str(runtime.audio_tokenizer_path),
            "text_normalization_status": _text_normalization_status_text(
                text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
            ),
        }
    
    @router.get("/api/warmup-status")
    async def warmup_status():
        snapshot = warmup_manager.snapshot()
        return {
            "state": snapshot.state,
            "progress": snapshot.progress,
            "message": snapshot.message,
            "error": snapshot.error,
            "ready": snapshot.ready,
            "failed": snapshot.failed,
            "status_text": _warmup_status_text(snapshot),
        }
    
    @router.get("/api/text-normalization-status")
    async def text_normalization_status():
        snapshot = text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
        if snapshot is None:
            return {
                "state": "disabled",
                "message": "WeTextProcessing disabled.",
                "error": None,
                "ready": False,
                "failed": False,
                "available": False,
                "status_text": "WeTextProcessing disabled.",
            }
        return {
            "state": snapshot.state,
            "message": snapshot.message,
            "error": snapshot.error,
            "ready": snapshot.ready,
            "failed": snapshot.failed,
            "available": snapshot.available,
            "status_text": _text_normalization_status_text(snapshot),
        }
    
    @router.get("/api/demo-prompt-audio/{demo_id}")
    async def demo_prompt_audio(demo_id: str):
        try:
            demo_entry = _resolve_demo_entry(demo_id)
        except ValueError as exc:
            return JSONResponse(status_code=404, content={"error": str(exc)})
    
        media_type = "audio/wav" if demo_entry.prompt_audio_path.suffix.lower() == ".wav" else "application/octet-stream"
        return FileResponse(
            path=str(demo_entry.prompt_audio_path),
            media_type=media_type,
            filename=demo_entry.prompt_audio_path.name,
        )
    
    @router.post("/api/generate-stream/start")
    async def generate_stream_start(
        text: str = Form(...),
        demo_id: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(75),
        tts_max_batch_size: int = Form(0),
        codec_max_batch_size: int = Form(0),
        enable_text_normalization: str = Form("1"),
        enable_normalize_tts_text: str = Form("1"),
        cpu_threads: int = Form(0),
        attn_implementation: str = Form("model_default"),
        do_sample: str = Form("1"),
        text_temperature: float = Form(1.0),
        text_top_p: float = Form(1.0),
        text_top_k: int = Form(50),
        audio_temperature: float = Form(0.8),
        audio_top_p: float = Form(0.95),
        audio_top_k: int = Form(25),
        audio_repetition_penalty: float = Form(1.2),
        seed: str = Form("0"),
    ):
        try:
            demo_entry, prompt_audio_path, prompt_audio_display_path, prompt_audio_cleanup_path = (
                await _resolve_prompt_audio_request(demo_id=demo_id, prompt_audio=prompt_audio)
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
    
        resolved_text = str(text or "").strip() or (demo_entry.text if demo_entry is not None else "")
        if not resolved_text:
            _maybe_delete_file(prompt_audio_cleanup_path)
            return JSONResponse(status_code=400, content={"error": "text is required."})
    
        try:
            prepared_texts = shared_prepare_tts_request_texts(
                text=resolved_text,
                enable_wetext=_coerce_bool(enable_text_normalization, False),
                enable_normalize_tts_text=_coerce_bool(enable_normalize_tts_text, True),
                text_normalizer_manager=text_normalizer_manager,
            )
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
        warmup_snapshot = warmup_manager.snapshot()
        if not warmup_snapshot.ready:
            warmup_snapshot = warmup_manager.ensure_ready()
            if not warmup_snapshot.ready:
                _maybe_delete_file(prompt_audio_cleanup_path)
                return JSONResponse(
                    status_code=500,
                    content={"error": _warmup_status_text(warmup_snapshot)},
                )
    
        try:
            normalized_seed = None if seed in {"", "0"} else int(seed)
            text_chunks = _resolve_voice_clone_text_chunks(
                text=str(prepared_texts["text"]),
                voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                cpu_threads=int(cpu_threads),
            )
            job = stream_jobs.create()
            with job.lock:
                job.prompt_audio_path = prompt_audio_display_path
                job.text_chunks = list(text_chunks)
            thread = threading.Thread(
                target=_run_streaming_job,
                kwargs={
                    "job": job,
                    "text": str(prepared_texts["text"]),
                    "prompt_audio_path": prompt_audio_path,
                    "prompt_audio_display_path": prompt_audio_display_path,
                    "prompt_audio_cleanup_path": prompt_audio_cleanup_path,
                    "max_new_frames": int(max_new_frames),
                    "voice_clone_max_text_tokens": int(voice_clone_max_text_tokens),
                    "tts_max_batch_size": int(tts_max_batch_size),
                    "codec_max_batch_size": int(codec_max_batch_size),
                    "cpu_threads": int(cpu_threads),
                    "attn_implementation": attn_implementation,
                    "do_sample": _coerce_bool(do_sample, True),
                    "text_temperature": float(text_temperature),
                    "text_top_p": float(text_top_p),
                    "text_top_k": int(text_top_k),
                    "audio_temperature": float(audio_temperature),
                    "audio_top_p": float(audio_top_p),
                    "audio_top_k": int(audio_top_k),
                    "audio_repetition_penalty": float(audio_repetition_penalty),
                    "seed": normalized_seed,
                },
                name=f"nano-tts-stream-{job.stream_id}",
                daemon=True,
            )
            thread.start()
            prompt_audio_cleanup_path = None
    
            initial_execution_label = "cpu"
    
            return {
                "stream_id": job.stream_id,
                "audio_url": f"{root_path}/api/generate-stream/{job.stream_id}/audio",
                "status_url": f"{root_path}/api/generate-stream/{job.stream_id}/status",
                "result_url": f"{root_path}/api/generate-stream/{job.stream_id}/result",
                "sample_rate": job.sample_rate,
                "channels": job.channels,
                "run_status": f"Streaming realtime audio... exec={initial_execution_label}",
                "prompt_audio_path": prompt_audio_display_path,
                "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
                "text_normalization_status_text": _text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
                "text_chunks": text_chunks,
                "normalized_text": str(prepared_texts["normalized_text"]),
                "normalization_method": str(prepared_texts["normalization_method"]),
                "text_normalization_language": str(prepared_texts["text_normalization_language"]),
            }
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
    
    @router.get("/api/generate-stream/{stream_id}/status")
    async def generate_stream_status(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        snapshot["stream_metrics"] = _stream_metrics_text(snapshot)
        return snapshot
    
    @router.get("/api/generate-stream/{stream_id}/audio")
    async def generate_stream_audio(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
    
        def _iter_audio():
            while True:
                item = job.audio_queue.get()
                if item is None:
                    break
                yield item
    
        return StreamingResponse(
            _iter_audio(),
            media_type="application/octet-stream",
            headers={
                "X-Audio-Codec": "pcm_s16le",
                "X-Audio-Sample-Rate": str(job.sample_rate),
                "X-Audio-Channels": str(job.channels),
                "X-Stream-Id": stream_id,
            },
        )
    
    @router.get("/api/generate-stream/{stream_id}/result")
    async def generate_stream_result(stream_id: str):
        job = stream_jobs.get(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        snapshot = job.snapshot()
        if snapshot["failed"]:
            return JSONResponse(status_code=500, content={"error": snapshot["error"], **snapshot})
        if not snapshot["ready"] or job.final_result is None:
            return JSONResponse(status_code=202, content=snapshot)
    
        result = dict(job.final_result)
        audio_chunk_ranges: list[list[float | int]] = []
        with job.lock:
            audio_chunk_ranges = [
                [float(start_seconds), float(end_seconds), int(chunk_index)]
                for start_seconds, end_seconds, chunk_index in job.audio_chunk_ranges
            ]
        audio_base64_payload = str(result.get("audio_base64") or "")
        audio_path_for_response = str(result.get("audio_path") or "").strip()
        if not audio_base64_payload and audio_path_for_response:
            audio_base64_payload = _read_audio_file_base64(audio_path_for_response)
            if audio_base64_payload:
                with job.lock:
                    if job.final_result is not None:
                        job.final_result["audio_base64"] = audio_base64_payload
                        job.final_result["audio_path"] = ""
                _maybe_delete_file(audio_path_for_response)
    
        return {
            "stream_id": stream_id,
            "ready": True,
            "state": snapshot["state"],
            "prompt_audio_path": result.get("prompt_audio_path") or snapshot.get("prompt_audio_path") or "",
            "run_status": result.get("run_status") or snapshot["run_status"],
            "stream_metrics": _stream_metrics_text(snapshot),
            "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
            "text_chunks": result.get("text_chunks") or snapshot.get("text_chunks") or [],
            "audio_chunk_ranges": audio_chunk_ranges,
            "audio_base64": audio_base64_payload,
        }
    
    @router.post("/api/generate-stream/{stream_id}/close")
    async def generate_stream_close(stream_id: str):
        job = stream_jobs.close(stream_id)
        if job is None:
            return JSONResponse(status_code=404, content={"error": "stream not found"})
        audio_cleanup_path = ""
        with job.lock:
            if job.final_result is not None:
                audio_cleanup_path = str(job.final_result.get("audio_path") or "").strip()
        snapshot = job.snapshot()
        snapshot["status_text"] = _format_stream_status(snapshot)
        stream_jobs.delete(stream_id)
        _maybe_delete_file(audio_cleanup_path)
        return snapshot
    
    @router.post("/api/generate")
    async def generate(
        text: str = Form(...),
        demo_id: str = Form(""),
        prompt_audio: UploadFile | None = File(None),
        max_new_frames: int = Form(375),
        voice_clone_max_text_tokens: int = Form(75),
        tts_max_batch_size: int = Form(0),
        codec_max_batch_size: int = Form(0),
        enable_text_normalization: str = Form("1"),
        enable_normalize_tts_text: str = Form("1"),
        cpu_threads: int = Form(0),
        attn_implementation: str = Form("model_default"),
        do_sample: str = Form("1"),
        text_temperature: float = Form(1.0),
        text_top_p: float = Form(1.0),
        text_top_k: int = Form(50),
        audio_temperature: float = Form(0.8),
        audio_top_p: float = Form(0.95),
        audio_top_k: int = Form(25),
        audio_repetition_penalty: float = Form(1.2),
        seed: str = Form("0"),
    ):
        try:
            demo_entry, prompt_audio_path, prompt_audio_display_path, prompt_audio_cleanup_path = (
                await _resolve_prompt_audio_request(demo_id=demo_id, prompt_audio=prompt_audio)
            )
        except ValueError as exc:
            return JSONResponse(status_code=400, content={"error": str(exc)})
    
        resolved_text = str(text or "").strip() or (demo_entry.text if demo_entry is not None else "")
        if not resolved_text:
            _maybe_delete_file(prompt_audio_cleanup_path)
            return JSONResponse(status_code=400, content={"error": "text is required."})
    
        try:
            prepared_texts = shared_prepare_tts_request_texts(
                text=resolved_text,
                enable_wetext=_coerce_bool(enable_text_normalization, False),
                enable_normalize_tts_text=_coerce_bool(enable_normalize_tts_text, True),
                text_normalizer_manager=text_normalizer_manager,
            )
        except Exception:
            _maybe_delete_file(prompt_audio_cleanup_path)
            raise
        warmup_snapshot = warmup_manager.snapshot()
        if not warmup_snapshot.ready:
            warmup_snapshot = warmup_manager.ensure_ready()
            if not warmup_snapshot.ready:
                _maybe_delete_file(prompt_audio_cleanup_path)
                return JSONResponse(
                    status_code=500,
                    content={"error": _warmup_status_text(warmup_snapshot)},
                )
    
        generated_audio_path: str | None = None
        try:
            normalized_seed = None if seed in {"", "0"} else int(seed)
    
            def _synthesize(selected_runtime: NanoTTSService):
                return selected_runtime.synthesize(
                    text=str(prepared_texts["text"]),
                    mode="voice_clone",
                    voice=None,
                    prompt_audio_path=prompt_audio_path,
                    max_new_frames=int(max_new_frames),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    tts_max_batch_size=int(tts_max_batch_size),
                    codec_max_batch_size=int(codec_max_batch_size),
                    attn_implementation=_resolve_attn_for_runtime(selected_runtime, attn_implementation),
                    do_sample=_coerce_bool(do_sample, True),
                    text_temperature=float(text_temperature),
                    text_top_p=float(text_top_p),
                    text_top_k=int(text_top_k),
                    audio_temperature=float(audio_temperature),
                    audio_top_p=float(audio_top_p),
                    audio_top_k=int(audio_top_k),
                    audio_repetition_penalty=float(audio_repetition_penalty),
                    seed=normalized_seed,
                )
    
            result, resolved_execution_device, resolved_cpu_threads = runtime_manager.call_with_runtime(
                requested_execution_device="cpu",
                cpu_threads=cpu_threads,
                callback=_synthesize,
            )
            result["execution_device"] = resolved_execution_device
            result["prompt_audio_display_path"] = prompt_audio_display_path
            if resolved_cpu_threads is not None:
                result["cpu_threads"] = resolved_cpu_threads
            text_chunks = [
                str(chunk).strip()
                for chunk in (result.get("voice_clone_text_chunks") or [])
                if str(chunk).strip()
            ]
            if not text_chunks:
                text_chunks = _resolve_voice_clone_text_chunks(
                    text=str(prepared_texts["text"]),
                    voice_clone_max_text_tokens=int(voice_clone_max_text_tokens),
                    cpu_threads=int(cpu_threads),
                )
            generated_audio_path = str(result["audio_path"])
            wav_bytes = _audio_to_wav_bytes(result["waveform_numpy"], int(result["sample_rate"]))
            return {
                "audio_base64": base64.b64encode(wav_bytes).decode("ascii"),
                "sample_rate": int(result["sample_rate"]),
                "run_status": _format_run_status(result),
                "prompt_audio_path": prompt_audio_display_path,
                "warmup_status_text": _warmup_status_text(warmup_manager.snapshot()),
                "text_normalization_status_text": _text_normalization_status_text(
                    text_normalizer_manager.snapshot() if text_normalizer_manager is not None else None
                ),
                "text_chunks": text_chunks,
                "normalized_text": str(prepared_texts["normalized_text"]),
                "normalization_method": str(prepared_texts["normalization_method"]),
                "text_normalization_language": str(prepared_texts["text_normalization_language"]),
            }
        except Exception as exc:
            logging.exception("Nano-TTS generation failed")
            return JSONResponse(status_code=500, content={"error": str(exc)})
        finally:
            _maybe_delete_file(generated_audio_path)
            _maybe_delete_file(prompt_audio_cleanup_path)
    

    return router

