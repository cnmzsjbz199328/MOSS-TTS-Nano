from __future__ import annotations

import logging
import threading

from core.models import WarmupSnapshot
from core.audio import _maybe_delete_file
from moss_tts_nano_runtime import NanoTTSService
from text_normalization_pipeline import WeTextProcessingManager as SharedWeTextProcessingManager

class WarmupManager:
    def __init__(self, runtime: NanoTTSService, text_normalizer_manager: "SharedWeTextProcessingManager | None" = None) -> None:
        self.runtime = runtime
        self.text_normalizer_manager = text_normalizer_manager
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._started = False
        self._state = "pending"
        self._progress = 0.0
        self._message = "Waiting for startup warmup."
        self._error: str | None = None

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._started = True
            self._thread = threading.Thread(target=self._run, name="nano-tts-warmup", daemon=True)
            self._thread.start()

    def snapshot(self) -> WarmupSnapshot:
        with self._lock:
            return WarmupSnapshot(
                state=self._state,
                progress=self._progress,
                message=self._message,
                error=self._error,
            )

    def ensure_ready(self) -> WarmupSnapshot:
        with self._lock:
            if not self._started:
                self._started = True
                self._thread = threading.Thread(target=self._run, name="nano-tts-warmup", daemon=True)
                self._thread.start()
            thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join()
        return self.snapshot()

    def _set_state(
        self,
        *,
        state: str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock:
            if state is not None:
                self._state = state
            if progress is not None:
                self._progress = max(0.0, min(1.0, float(progress)))
            if message is not None:
                self._message = message
            self._error = error

    def _run(self) -> None:
        try:
            self._set_state(state="running", progress=0.1, message="Loading Nano-TTS model.", error=None)
            self.runtime.get_model()
            self._set_state(state="running", progress=0.6, message="Running startup warmup synthesis.", error=None)
            result = self.runtime.warmup()
            _maybe_delete_file(result["audio_path"])
            if self.text_normalizer_manager is not None:
                self._set_state(
                    state="running",
                    progress=0.85,
                    message="Loading WeTextProcessing text normalization.",
                    error=None,
                )
                normalization_snapshot = self.text_normalizer_manager.ensure_ready()
                if normalization_snapshot.failed:
                    raise RuntimeError(normalization_snapshot.error or normalization_snapshot.message)
            self._set_state(
                state="ready",
                progress=1.0,
                message=(
                    f"Warmup complete. device={self.runtime.device} "
                    f"elapsed={result['elapsed_seconds']:.2f}s"
                    + (" | WeTextProcessing ready." if self.text_normalizer_manager is not None else "")
                ),
                error=None,
            )
        except Exception as exc:
            logging.exception("Nano-TTS warmup failed")
            self._set_state(state="failed", progress=1.0, message="Warmup failed.", error=str(exc))


