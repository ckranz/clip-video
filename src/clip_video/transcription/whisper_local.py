"""Local Whisper transcription provider.

Supports local whisper implementations like faster-whisper or whisper.cpp.
This is a stub implementation that can be extended for actual local transcription.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from clip_video.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


class WhisperLocalProvider(TranscriptionProvider):
    """Transcription provider using local Whisper implementations.

    Supports faster-whisper (recommended) or whisper.cpp via command line.
    This provider has no API costs but requires local GPU or CPU resources.

    Note: This is a partial implementation. The faster-whisper integration
    is functional if faster-whisper is installed. The whisper.cpp support
    is stubbed out for future implementation.
    """

    # Default model for faster-whisper
    DEFAULT_MODEL = "medium"

    # Available models (ordered by size)
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "auto",
        compute_type: str = "auto",
        whisper_cpp_path: str | None = None,
    ):
        """Initialize the local Whisper provider.

        Args:
            model: Model size to use (tiny, base, small, medium, large, large-v2, large-v3)
            device: Device to use ("auto", "cpu", "cuda")
            compute_type: Compute type ("auto", "int8", "float16", "float32")
            whisper_cpp_path: Path to whisper.cpp executable (optional)
        """
        self._model_name = model
        self._device = device
        self._compute_type = compute_type
        self._whisper_cpp_path = whisper_cpp_path
        self._faster_whisper_model: Any = None
        self._use_faster_whisper: bool | None = None

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "whisper_local"

    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps."""
        # faster-whisper supports word timestamps
        return True

    def is_available(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if faster-whisper is installed or whisper.cpp is configured
        """
        # Check faster-whisper
        try:
            import faster_whisper
            return True
        except ImportError:
            pass

        # Check whisper.cpp
        if self._whisper_cpp_path:
            cpp_path = Path(self._whisper_cpp_path)
            if cpp_path.exists() and cpp_path.is_file():
                return True

        return False

    def _check_faster_whisper(self) -> bool:
        """Check if faster-whisper is available.

        Returns:
            True if faster-whisper can be imported
        """
        if self._use_faster_whisper is not None:
            return self._use_faster_whisper

        try:
            import faster_whisper
            self._use_faster_whisper = True
        except ImportError:
            self._use_faster_whisper = False

        return self._use_faster_whisper

    def _get_faster_whisper_model(self) -> Any:
        """Get or create the faster-whisper model.

        Returns:
            WhisperModel instance

        Raises:
            ImportError: If faster-whisper is not installed
        """
        if self._faster_whisper_model is not None:
            return self._faster_whisper_model

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise ImportError(
                "faster-whisper package is required for local transcription. "
                "Install it with: pip install faster-whisper"
            )

        # Determine device
        device = self._device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # Determine compute type
        compute_type = self._compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        self._faster_whisper_model = WhisperModel(
            self._model_name,
            device=device,
            compute_type=compute_type,
        )

        return self._faster_whisper_model

    def estimate_cost(self, duration_seconds: float) -> float | None:
        """Estimate cost for transcribing audio of given duration.

        Args:
            duration_seconds: Duration of audio in seconds

        Returns:
            None (local transcription is free)
        """
        return None

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "en",
        prompt: str = "",
    ) -> TranscriptionResult:
        """Transcribe an audio/video file using local Whisper.

        Args:
            audio_path: Path to audio or video file
            language: Language code (e.g., "en", "es")
            prompt: Optional prompt for conditioning

        Returns:
            TranscriptionResult with transcribed content

        Raises:
            FileNotFoundError: If audio file doesn't exist
            RuntimeError: If no local whisper implementation is available
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        # Try faster-whisper first
        if self._check_faster_whisper():
            return self._transcribe_faster_whisper(audio_path, language, prompt)

        # Try whisper.cpp
        if self._whisper_cpp_path:
            return self._transcribe_whisper_cpp(audio_path, language, prompt)

        raise RuntimeError(
            "No local Whisper implementation available. "
            "Install faster-whisper: pip install faster-whisper"
        )

    def _transcribe_faster_whisper(
        self,
        audio_path: Path,
        language: str,
        prompt: str,
    ) -> TranscriptionResult:
        """Transcribe using faster-whisper.

        Args:
            audio_path: Path to audio file
            language: Language code
            prompt: Conditioning prompt

        Returns:
            TranscriptionResult
        """
        model = self._get_faster_whisper_model()

        # Transcribe with word timestamps
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt if prompt else None,
            word_timestamps=True,
            vad_filter=True,
        )

        # Collect segments
        segments = []
        full_text_parts = []

        for segment in segments_iter:
            segment_words = []

            if segment.words:
                for word in segment.words:
                    segment_words.append(TranscriptionWord(
                        word=word.word.strip(),
                        start=word.start,
                        end=word.end,
                        confidence=word.probability if hasattr(word, "probability") else 1.0,
                    ))

            segments.append(TranscriptionSegment(
                text=segment.text.strip(),
                start=segment.start,
                end=segment.end,
                words=segment_words,
                confidence=1.0,
            ))
            full_text_parts.append(segment.text.strip())

        full_text = " ".join(full_text_parts)

        return TranscriptionResult(
            video_path=str(audio_path),
            text=full_text,
            segments=segments,
            language=info.language if info.language else language,
            duration=info.duration if hasattr(info, "duration") else 0.0,
            provider=self.name,
            model=self._model_name,
            timestamp=datetime.now(),
            cost_usd=None,
        )

    def _transcribe_whisper_cpp(
        self,
        audio_path: Path,
        language: str,
        prompt: str,
    ) -> TranscriptionResult:
        """Transcribe using whisper.cpp (stub implementation).

        Note: This is a stub for future implementation. whisper.cpp
        requires additional setup and model conversion.

        Args:
            audio_path: Path to audio file
            language: Language code
            prompt: Conditioning prompt

        Returns:
            TranscriptionResult

        Raises:
            NotImplementedError: whisper.cpp support not fully implemented
        """
        # TODO: Implement whisper.cpp support
        # This would involve:
        # 1. Converting audio to 16kHz WAV (required by whisper.cpp)
        # 2. Running the whisper.cpp executable
        # 3. Parsing the JSON output
        #
        # Example command:
        # ./main -m models/ggml-base.bin -f audio.wav -ojf --language en
        raise NotImplementedError(
            "whisper.cpp support is not yet fully implemented. "
            "Please use faster-whisper instead: pip install faster-whisper"
        )

    def get_model_info(self) -> dict[str, Any]:
        """Get information about the current model configuration.

        Returns:
            Dictionary with model information
        """
        return {
            "model": self._model_name,
            "device": self._device,
            "compute_type": self._compute_type,
            "faster_whisper_available": self._check_faster_whisper(),
            "whisper_cpp_path": self._whisper_cpp_path,
        }
