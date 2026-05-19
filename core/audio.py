from __future__ import annotations

import base64
import io
import logging
import wave
from pathlib import Path

import numpy as np

def _audio_to_wav_bytes(audio_array, sample_rate: int) -> bytes:
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(int(audio_int16.shape[1]))
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())

    buffer.seek(0)
    return buffer.read()



def _audio_to_pcm16le_bytes(audio_array) -> bytes:
    audio_np = np.asarray(audio_array, dtype=np.float32)
    if audio_np.ndim == 1:
        audio_np = audio_np[:, None]
    elif audio_np.ndim == 2 and audio_np.shape[0] <= 8 and audio_np.shape[0] < audio_np.shape[1]:
        audio_np = audio_np.T
    elif audio_np.ndim != 2:
        raise ValueError(f"Unsupported audio array shape: {audio_np.shape}")

    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767.0).astype(np.int16)
    return audio_int16.tobytes()



def _read_audio_file_base64(path_value: str | None) -> str:
    path_text = str(path_value or "").strip()
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.is_file():
        return ""
    try:
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except Exception:
        logging.warning("failed to read audio file for base64 response: %s", path, exc_info=True)
        return ""



def _maybe_delete_file(path_value: str | None) -> None:
    if not path_value:
        return
    try:
        Path(path_value).unlink(missing_ok=True)
    except Exception:
        logging.warning("failed to remove temporary file: %s", path_value, exc_info=True)


