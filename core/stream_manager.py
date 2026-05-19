from __future__ import annotations

import os
import queue
import threading
import time
import uuid
from typing import Callable, Iterator, TypeVar

import torch
from core.models import StreamingJob
from moss_tts_nano_runtime import NanoTTSService

T = TypeVar("T")

class RequestRuntimeManager:
    def __init__(self, default_runtime: NanoTTSService) -> None:
        self.default_runtime = default_runtime
        self.default_cpu_threads = max(1, int(os.cpu_count() or 1))
        self._lock = threading.Lock()
        self._cpu_execution_lock = threading.Lock()
        self._cpu_runtime: NanoTTSService | None = None

    @staticmethod
    def normalize_requested_execution_device(requested: str | None) -> str:
        normalized = str(requested or "default").strip().lower()
        if normalized not in {"default", "cpu"}:
            return "default"
        return normalized

    def is_dedicated_cpu_request(self, requested: str | None) -> bool:
        normalized = self.normalize_requested_execution_device(requested)
        return normalized == "cpu" and self.default_runtime.device.type != "cpu"

    def is_cpu_runtime_loaded(self) -> bool:
        with self._lock:
            return self._cpu_runtime is not None

    def _build_cpu_runtime_locked(self) -> NanoTTSService:
        if self._cpu_runtime is not None:
            return self._cpu_runtime
        self._cpu_runtime = NanoTTSService(
            checkpoint_path=self.default_runtime.checkpoint_path,
            audio_tokenizer_path=self.default_runtime.audio_tokenizer_path,
            device="cpu",
            dtype="float32",
            attn_implementation=self.default_runtime.attn_implementation or "auto",
            output_dir=self.default_runtime.output_dir,
            voice_presets=self.default_runtime.voice_presets,
        )
        return self._cpu_runtime

    def resolve_runtime(self, requested: str | None) -> tuple[NanoTTSService, str]:
        normalized = self.normalize_requested_execution_device(requested)
        if normalized != "cpu":
            return self.default_runtime, str(self.default_runtime.device.type)
        if self.default_runtime.device.type == "cpu":
            return self.default_runtime, "cpu"
        with self._lock:
            return self._build_cpu_runtime_locked(), "cpu"

    def _resolve_cpu_threads(self, cpu_threads: int | None) -> int:
        if cpu_threads is None:
            return self.default_cpu_threads
        try:
            normalized_threads = int(cpu_threads)
        except Exception:
            return self.default_cpu_threads
        if normalized_threads <= 0:
            return self.default_cpu_threads
        return max(1, normalized_threads)

    def call_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        callback: Callable[[NanoTTSService], T],
    ) -> tuple[T, str, int | None]:
        runtime, execution_device = self.resolve_runtime(requested_execution_device)
        if runtime.device.type != "cpu":
            return callback(runtime), execution_device, None

        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._cpu_execution_lock:
            previous_threads = torch.get_num_threads()
            threads_changed = previous_threads != resolved_cpu_threads
            if threads_changed:
                torch.set_num_threads(resolved_cpu_threads)
            try:
                return callback(runtime), execution_device, resolved_cpu_threads
            finally:
                if threads_changed:
                    torch.set_num_threads(previous_threads)

    def iter_with_runtime(
        self,
        *,
        requested_execution_device: str | None,
        cpu_threads: int | None,
        factory: Callable[[NanoTTSService], Iterator[T]],
    ) -> Iterator[tuple[T, str, int | None]]:
        runtime, execution_device = self.resolve_runtime(requested_execution_device)
        if runtime.device.type != "cpu":
            for item in factory(runtime):
                yield item, execution_device, None
            return

        resolved_cpu_threads = self._resolve_cpu_threads(cpu_threads)
        with self._cpu_execution_lock:
            previous_threads = torch.get_num_threads()
            threads_changed = previous_threads != resolved_cpu_threads
            if threads_changed:
                torch.set_num_threads(resolved_cpu_threads)
            try:
                for item in factory(runtime):
                    yield item, execution_device, resolved_cpu_threads
            finally:
                if threads_changed:
                    torch.set_num_threads(previous_threads)



class StreamingJobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, StreamingJob] = {}

    def create(self) -> StreamingJob:
        stream_id = f"stream-{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
        job = StreamingJob(stream_id=stream_id)
        with self._lock:
            self._jobs[stream_id] = job
        return job

    def get(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            return self._jobs.get(stream_id)

    def close(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            job = self._jobs.get(stream_id)
        if job is None:
            return None
        with job.lock:
            job.is_closed = True
            job.state = "closed" if job.state not in {"done", "failed"} else job.state
            try:
                job.audio_queue.put_nowait(None)
            except queue.Full:
                pass
        return job

    def delete(self, stream_id: str) -> StreamingJob | None:
        with self._lock:
            return self._jobs.pop(stream_id, None)


