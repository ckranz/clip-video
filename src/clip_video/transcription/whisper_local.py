"""Local Whisper transcription provider.

Supports multiple local Whisper backends:
- openai-whisper: Original OpenAI Whisper (PyTorch) - uses standard model cache
- faster-whisper: CTranslate2-based implementation - faster but separate model cache
- whisper.cpp: C++ implementation (stub, not fully implemented)
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from clip_video.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


# Backend type alias
WhisperBackend = Literal["openai-whisper", "faster-whisper", "auto"]


class WhisperLocalProvider(TranscriptionProvider):
    """Transcription provider using local Whisper implementations.

    Supports multiple backends:
    - openai-whisper: Original OpenAI Whisper package. Uses PyTorch models that
      may already be cached from other projects. Install: pip install openai-whisper
    - faster-whisper: CTranslate2-based, typically 4x faster and uses less memory.
      Uses separate model cache. Install: pip install faster-whisper
    - auto: Automatically selects based on what's installed (prefers openai-whisper
      if both are available, to reuse cached models)

    This provider has no API costs but requires local GPU or CPU resources.
    """

    # Default model
    DEFAULT_MODEL = "medium"

    # Available models (ordered by size)
    AVAILABLE_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]

    # Available backends
    AVAILABLE_BACKENDS = ["auto", "openai-whisper", "faster-whisper"]

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        backend: WhisperBackend = "auto",
        device: str = "auto",
        compute_type: str = "auto",
        whisper_cpp_path: str | None = None,
    ):
        """Initialize the local Whisper provider.

        Args:
            model: Model size to use (tiny, base, small, medium, large, large-v2, large-v3)
            backend: Which backend to use ("auto", "openai-whisper", "faster-whisper")
            device: Device to use ("auto", "cpu", "cuda")
            compute_type: Compute type for faster-whisper ("auto", "int8", "float16", "float32")
            whisper_cpp_path: Path to whisper.cpp executable (optional)
        """
        self._model_name = model
        self._backend = backend
        self._device = device
        self._compute_type = compute_type
        self._whisper_cpp_path = whisper_cpp_path
        self._faster_whisper_model: Any = None
        self._openai_whisper_model: Any = None
        self._resolved_backend: str | None = None

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "whisper_local"

    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps."""
        # Both openai-whisper and faster-whisper support word timestamps
        return True

    def is_available(self) -> bool:
        """Check if the provider is available.

        Returns:
            True if any whisper backend is installed
        """
        return (
            self._check_openai_whisper() or
            self._check_faster_whisper() or
            self._check_whisper_cpp()
        )

    def _check_openai_whisper(self) -> bool:
        """Check if openai-whisper is available.

        Returns:
            True if whisper (openai-whisper) can be imported
        """
        try:
            import whisper
            return True
        except ImportError:
            return False

    def _check_faster_whisper(self) -> bool:
        """Check if faster-whisper is available.

        Returns:
            True if faster-whisper can be imported
        """
        try:
            import faster_whisper
            return True
        except ImportError:
            return False

    def _check_whisper_cpp(self) -> bool:
        """Check if whisper.cpp is available.

        Returns:
            True if whisper.cpp executable exists
        """
        if self._whisper_cpp_path:
            cpp_path = Path(self._whisper_cpp_path)
            return cpp_path.exists() and cpp_path.is_file()
        return False

    def _resolve_backend(self) -> str:
        """Resolve which backend to use based on config and availability.

        Returns:
            The backend to use: "openai-whisper", "faster-whisper", or "whisper-cpp"

        Raises:
            RuntimeError: If no backend is available
        """
        if self._resolved_backend is not None:
            return self._resolved_backend

        if self._backend == "openai-whisper":
            if self._check_openai_whisper():
                self._resolved_backend = "openai-whisper"
                return self._resolved_backend
            raise RuntimeError(
                "openai-whisper backend requested but not installed. "
                "Install with: pip install openai-whisper"
            )

        if self._backend == "faster-whisper":
            if self._check_faster_whisper():
                self._resolved_backend = "faster-whisper"
                return self._resolved_backend
            raise RuntimeError(
                "faster-whisper backend requested but not installed. "
                "Install with: pip install faster-whisper"
            )

        # Auto mode: prefer openai-whisper (to reuse cached models), then faster-whisper
        if self._backend == "auto":
            if self._check_openai_whisper():
                self._resolved_backend = "openai-whisper"
                return self._resolved_backend
            if self._check_faster_whisper():
                self._resolved_backend = "faster-whisper"
                return self._resolved_backend
            if self._check_whisper_cpp():
                self._resolved_backend = "whisper-cpp"
                return self._resolved_backend

        raise RuntimeError(
            "No local Whisper backend available. Install one of:\n"
            "  pip install openai-whisper  (recommended if you have models cached)\n"
            "  pip install faster-whisper  (faster, but downloads separate models)"
        )

    def get_active_backend(self) -> str:
        """Get the backend that will be used for transcription.

        Returns:
            The active backend name
        """
        return self._resolve_backend()

    def _get_device(self) -> str:
        """Resolve the device to use.

        Returns:
            Device string: "cuda" or "cpu"
        """
        device = self._device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        return device

    def _get_openai_whisper_model(self) -> Any:
        """Get or create the openai-whisper model.

        Returns:
            Whisper model instance

        Raises:
            ImportError: If openai-whisper is not installed
        """
        if self._openai_whisper_model is not None:
            return self._openai_whisper_model

        try:
            import whisper
        except ImportError:
            raise ImportError(
                "openai-whisper package is required. "
                "Install it with: pip install openai-whisper"
            )

        device = self._get_device()
        self._openai_whisper_model = whisper.load_model(self._model_name, device=device)
        return self._openai_whisper_model

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
                "faster-whisper package is required. "
                "Install it with: pip install faster-whisper"
            )

        device = self._get_device()

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

    def _transcribe_openai_whisper(
        self,
        audio_path: Path,
        language: str,
        prompt: str,
    ) -> TranscriptionResult:
        """Transcribe using openai-whisper (original Whisper).

        Args:
            audio_path: Path to audio file
            language: Language code
            prompt: Conditioning prompt

        Returns:
            TranscriptionResult
        """
        model = self._get_openai_whisper_model()

        # Transcribe with word timestamps
        result = model.transcribe(
            str(audio_path),
            language=language,
            initial_prompt=prompt if prompt else None,
            word_timestamps=True,
            verbose=False,
        )

        # Collect segments
        segments = []
        full_text_parts = []

        for segment in result.get("segments", []):
            segment_words = []

            # Extract word-level timestamps if available
            for word_data in segment.get("words", []):
                segment_words.append(TranscriptionWord(
                    word=word_data.get("word", "").strip(),
                    start=word_data.get("start", 0.0),
                    end=word_data.get("end", 0.0),
                    confidence=word_data.get("probability", 1.0),
                ))

            segments.append(TranscriptionSegment(
                text=segment.get("text", "").strip(),
                start=segment.get("start", 0.0),
                end=segment.get("end", 0.0),
                words=segment_words,
                confidence=1.0,
            ))
            full_text_parts.append(segment.get("text", "").strip())

        full_text = " ".join(full_text_parts)

        return TranscriptionResult(
            video_path=str(audio_path),
            text=full_text,
            segments=segments,
            language=result.get("language", language),
            duration=segments[-1].end if segments else 0.0,
            provider=self.name,
            model=self._model_name,
            timestamp=datetime.now(),
            cost_usd=None,
        )

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

        backend = self._resolve_backend()

        if backend == "openai-whisper":
            return self._transcribe_openai_whisper(audio_path, language, prompt)
        elif backend == "faster-whisper":
            return self._transcribe_faster_whisper(audio_path, language, prompt)
        elif backend == "whisper-cpp":
            return self._transcribe_whisper_cpp(audio_path, language, prompt)
        else:
            raise RuntimeError(f"Unknown backend: {backend}")

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
            "backend_requested": self._backend,
            "backend_active": self._resolved_backend or "(not resolved yet)",
            "device": self._device,
            "compute_type": self._compute_type,
            "openai_whisper_available": self._check_openai_whisper(),
            "faster_whisper_available": self._check_faster_whisper(),
            "whisper_cpp_available": self._check_whisper_cpp(),
        }
