from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class DemoEntry:
    demo_id: str
    name: str
    prompt_audio_path: Path
    prompt_audio_relative_path: str
    text: str


@dataclass(frozen=True)
class WarmupSnapshot:
    state: str
    progress: float
    message: str
    error: str | None = None

    @property
    def ready(self) -> bool:
        return self.state == "ready"

    @property
    def failed(self) -> bool:
        return self.state == "failed"


@dataclass
class StreamingJob:
    stream_id: str
    audio_queue: "queue.Queue[bytes | None]" = field(default_factory=lambda: queue.Queue(maxsize=64))
    created_at: float = field(default_factory=time.monotonic)
    started_at: float | None = None
    first_audio_at: float | None = None
    completed_at: float | None = None
    state: str = "starting"
    run_status: str = "Starting realtime synthesis..."
    error: str | None = None
    prompt_audio_path: str | None = None
    sample_rate: int = 48000
    channels: int = 2
    emitted_audio_seconds: float = 0.0
    lead_seconds: float = 0.0
    current_chunk_index: int | None = None
    text_chunks: list[str] = field(default_factory=list)
    chunk_index_base: int | None = None
    audio_chunk_ranges: list[tuple[float, float, int]] = field(default_factory=list)
    is_closed: bool = False
    final_result: dict[str, object] | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)

    def _resolve_playback_chunk_index_locked(self) -> int | None:
        if not self.audio_chunk_ranges:
            return self.current_chunk_index

        playback_audio_seconds = max(0.0, float(self.emitted_audio_seconds) - float(self.lead_seconds))
        for start_seconds, end_seconds, chunk_index in self.audio_chunk_ranges:
            if playback_audio_seconds <= end_seconds + 1e-6:
                return chunk_index
        return self.audio_chunk_ranges[-1][2]

    def snapshot(self) -> dict[str, object]:
        with self.lock:
            return {
                "stream_id": self.stream_id,
                "state": self.state,
                "run_status": self.run_status,
                "error": self.error,
                "prompt_audio_path": self.prompt_audio_path,
                "sample_rate": self.sample_rate,
                "channels": self.channels,
                "emitted_audio_seconds": self.emitted_audio_seconds,
                "lead_seconds": self.lead_seconds,
                "current_chunk_index": self.current_chunk_index,
                "playback_chunk_index": self._resolve_playback_chunk_index_locked(),
                "text_chunks": list(self.text_chunks),
                "first_audio_latency_seconds": (
                    None
                    if self.started_at is None or self.first_audio_at is None
                    else max(0.0, self.first_audio_at - self.started_at)
                ),
                "completed_at": self.completed_at,
                "ready": self.state == "done",
                "failed": self.state == "failed",
                "closed": self.is_closed,
            }
