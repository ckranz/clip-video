"""OpenAI Whisper API transcription provider.

Uses the OpenAI API for cloud-based transcription with word-level timestamps.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from clip_video.transcription.base import (
    TranscriptionProvider,
    TranscriptionResult,
    TranscriptionSegment,
    TranscriptionWord,
)


class WhisperAPIProvider(TranscriptionProvider):
    """Transcription provider using OpenAI's Whisper API.

    Requires OPENAI_API_KEY environment variable to be set.
    Pricing: $0.006 per minute of audio (as of 2024).
    """

    # Whisper API pricing per minute
    COST_PER_MINUTE_USD = 0.006

    # Maximum file size for Whisper API (25 MB)
    MAX_FILE_SIZE_MB = 25

    # Supported audio formats
    SUPPORTED_FORMATS = {"mp3", "mp4", "mpeg", "mpga", "m4a", "wav", "webm"}

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "whisper-1",
    ):
        """Initialize the Whisper API provider.

        Args:
            api_key: OpenAI API key (defaults to OPENAI_API_KEY env var)
            model: Model to use (currently only "whisper-1" available)
        """
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._model = model
        self._client: Any = None

    @property
    def name(self) -> str:
        """Return the provider name."""
        return "whisper_api"

    @property
    def supports_word_timestamps(self) -> bool:
        """Return True if provider supports word-level timestamps."""
        return True

    def _get_client(self) -> Any:
        """Get or create the OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not configured
        """
        if self._client is not None:
            return self._client

        if not self._api_key:
            raise ValueError(
                "OpenAI API key not configured. "
                "Set OPENAI_API_KEY environment variable or pass api_key to constructor."
            )

        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for Whisper API. "
                "Install it with: pip install openai>=1.0"
            )

        self._client = OpenAI(api_key=self._api_key)
        return self._client

    def is_available(self) -> bool:
        """Check if the provider is available and configured.

        Returns:
            True if API key is configured and openai package is available
        """
        if not self._api_key:
            return False

        try:
            from openai import OpenAI
            return True
        except ImportError:
            return False

    def estimate_cost(self, duration_seconds: float) -> float | None:
        """Estimate cost for transcribing audio of given duration.

        Args:
            duration_seconds: Duration of audio in seconds

        Returns:
            Estimated cost in USD
        """
        duration_minutes = duration_seconds / 60.0
        return duration_minutes * self.COST_PER_MINUTE_USD

    def _get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio/video file in seconds.

        Uses ffprobe to determine duration.

        Args:
            audio_path: Path to audio/video file

        Returns:
            Duration in seconds
        """
        try:
            from clip_video.ffmpeg_binary import get_ffprobe_path

            ffprobe_path = get_ffprobe_path()
            if not ffprobe_path:
                return 0.0

            result = subprocess.run(
                [
                    ffprobe_path,
                    "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception:
            pass

        return 0.0

    def _extract_audio(self, video_path: Path, output_path: Path) -> bool:
        """Extract audio from video file as MP3.

        Args:
            video_path: Path to video file
            output_path: Path to output audio file

        Returns:
            True if extraction succeeded
        """
        try:
            from clip_video.ffmpeg_binary import get_ffmpeg_path

            ffmpeg_path = get_ffmpeg_path()
            if not ffmpeg_path:
                return False

            result = subprocess.run(
                [
                    ffmpeg_path,
                    "-i", str(video_path),
                    "-vn",  # No video
                    "-acodec", "libmp3lame",
                    "-ar", "16000",  # 16kHz sample rate (good for speech)
                    "-ac", "1",  # Mono
                    "-b:a", "64k",  # Low bitrate for smaller file
                    "-y",  # Overwrite output
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )

            return result.returncode == 0 and output_path.exists()
        except Exception:
            return False

    def transcribe(
        self,
        audio_path: Path | str,
        language: str = "en",
        prompt: str = "",
    ) -> TranscriptionResult:
        """Transcribe an audio/video file using OpenAI Whisper API.

        Args:
            audio_path: Path to audio or video file
            language: Language code (e.g., "en", "es")
            prompt: Optional prompt for conditioning (vocabulary terms)

        Returns:
            TranscriptionResult with transcribed content

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If API key not configured
            RuntimeError: If transcription fails
        """
        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        client = self._get_client()

        # Get duration for cost tracking
        duration = self._get_audio_duration(audio_path)

        # Determine if we need to extract audio
        suffix = audio_path.suffix.lower().lstrip(".")
        needs_extraction = suffix not in self.SUPPORTED_FORMATS

        # Check file size
        file_size_mb = audio_path.stat().st_size / (1024 * 1024)
        needs_extraction = needs_extraction or file_size_mb > self.MAX_FILE_SIZE_MB

        try:
            if needs_extraction:
                # Extract audio to temporary MP3
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                    tmp_path = Path(tmp.name)

                try:
                    if not self._extract_audio(audio_path, tmp_path):
                        raise RuntimeError(f"Failed to extract audio from {audio_path}")

                    # Transcribe the extracted audio
                    result = self._call_api(client, tmp_path, language, prompt)
                finally:
                    # Clean up temp file
                    if tmp_path.exists():
                        tmp_path.unlink()
            else:
                # Transcribe directly
                result = self._call_api(client, audio_path, language, prompt)

            # Build transcription result
            segments = self._parse_segments(result)
            full_text = result.text

            return TranscriptionResult(
                video_path=str(audio_path),
                text=full_text,
                segments=segments,
                language=language,
                duration=duration,
                provider=self.name,
                model=self._model,
                timestamp=datetime.now(),
                cost_usd=self.estimate_cost(duration),
            )

        except Exception as e:
            raise RuntimeError(f"Transcription failed: {e}") from e

    def _call_api(
        self,
        client: Any,
        audio_path: Path,
        language: str,
        prompt: str,
    ) -> Any:
        """Call the Whisper API.

        Args:
            client: OpenAI client
            audio_path: Path to audio file
            language: Language code
            prompt: Conditioning prompt

        Returns:
            API response object
        """
        with open(audio_path, "rb") as audio_file:
            response = client.audio.transcriptions.create(
                model=self._model,
                file=audio_file,
                language=language,
                response_format="verbose_json",
                timestamp_granularities=["word", "segment"],
                prompt=prompt if prompt else None,
            )

        return response

    def _parse_segments(self, response: Any) -> list[TranscriptionSegment]:
        """Parse API response into TranscriptionSegment objects.

        Args:
            response: API response object

        Returns:
            List of TranscriptionSegment objects
        """
        segments = []

        # Get segments from response
        api_segments = getattr(response, "segments", []) or []
        api_words = getattr(response, "words", []) or []

        # Build word lookup by time range
        word_objects = []
        for w in api_words:
            word_obj = TranscriptionWord(
                word=w.word.strip() if hasattr(w, "word") else str(w.get("word", "")).strip(),
                start=float(w.start if hasattr(w, "start") else w.get("start", 0)),
                end=float(w.end if hasattr(w, "end") else w.get("end", 0)),
                confidence=1.0,
            )
            word_objects.append(word_obj)

        # Assign words to segments
        word_index = 0
        for seg in api_segments:
            seg_start = float(seg.start if hasattr(seg, "start") else seg.get("start", 0))
            seg_end = float(seg.end if hasattr(seg, "end") else seg.get("end", 0))
            seg_text = (seg.text if hasattr(seg, "text") else seg.get("text", "")).strip()

            # Find words that belong to this segment
            segment_words = []
            while word_index < len(word_objects):
                word = word_objects[word_index]
                # Word belongs to segment if it starts within segment bounds
                if word.start >= seg_start and word.start < seg_end:
                    segment_words.append(word)
                    word_index += 1
                elif word.start >= seg_end:
                    break
                else:
                    word_index += 1

            segments.append(TranscriptionSegment(
                text=seg_text,
                start=seg_start,
                end=seg_end,
                words=segment_words,
                confidence=1.0,
            ))

        # If no segments but we have words, create one segment
        if not segments and word_objects:
            segments.append(TranscriptionSegment(
                text=response.text,
                start=word_objects[0].start if word_objects else 0.0,
                end=word_objects[-1].end if word_objects else 0.0,
                words=word_objects,
                confidence=1.0,
            ))

        return segments
