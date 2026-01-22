"""Ollama LLM provider implementation.

Provides integration with local Ollama server for transcript analysis
and highlight detection. Free, runs locally with no API costs.
"""

from __future__ import annotations

import json
import time
import urllib.request
import urllib.error
from typing import Any

from clip_video.llm.base import (
    LLMProvider,
    LLMConfig,
    LLMProviderType,
    HighlightAnalysis,
    HighlightSegment,
    ClipValidationRequest,
    ClipValidationResponse,
)
from clip_video.llm.prompts import HighlightPromptBuilder, CONFERENCE_PROMPT


class OllamaAPIError(Exception):
    """Error from Ollama API."""

    pass


class OllamaConnectionError(OllamaAPIError):
    """Cannot connect to Ollama server."""

    pass


class OllamaLLM(LLMProvider):
    """Ollama LLM provider for local inference.

    Uses a local Ollama server to analyze transcripts and identify
    highlight-worthy segments. Free, no API costs.

    Requires Ollama to be installed and running:
    - Install: https://ollama.ai
    - Run: ollama serve
    - Pull a model: ollama pull llama3.2

    Recommended models for this task:
    - llama3.2 (default, good balance of speed/quality)
    - mistral (fast, good for simpler tasks)
    - llama3.1:70b (higher quality, needs more RAM)
    - mixtral (good quality, moderate resources)
    """

    # Default Ollama API endpoint
    DEFAULT_BASE_URL = "http://localhost:11434"

    # Recommended models (ordered by capability)
    RECOMMENDED_MODELS = [
        "llama3.2",
        "llama3.1",
        "mistral",
        "mixtral",
        "gemma2",
        "qwen2.5",
    ]

    def __init__(
        self,
        config: LLMConfig | None = None,
        prompt_builder: HighlightPromptBuilder | None = None,
        base_url: str | None = None,
    ):
        """Initialize Ollama provider.

        Args:
            config: LLM configuration
            prompt_builder: Optional custom prompt builder
            base_url: Ollama API base URL (default: http://localhost:11434)
        """
        if config is None:
            config = LLMConfig(provider=LLMProviderType.OLLAMA)

        super().__init__(config)
        self.prompt_builder = prompt_builder or CONFERENCE_PROMPT
        self.base_url = base_url or self.DEFAULT_BASE_URL

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "Ollama (Local)"

    def is_available(self) -> bool:
        """Check if Ollama server is available.

        Returns:
            True if Ollama is running and responding
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5) as response:
                return response.status == 200
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return False

    def get_available_models(self) -> list[str]:
        """Get list of models available in Ollama.

        Returns:
            List of model names installed locally
        """
        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json.loads(response.read().decode("utf-8"))
                return [m["name"] for m in data.get("models", [])]
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
            return []

    def _make_request(
        self,
        messages: list[dict[str, str]],
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        """Make a request to the Ollama API.

        Args:
            messages: List of message dicts with role and content
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature

        Returns:
            Response data from Ollama

        Raises:
            OllamaAPIError: If request fails
        """
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.URLError as e:
                if attempt < max_retries - 1:
                    time.sleep((attempt + 1) * 2)
                    continue
                raise OllamaConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    "Make sure Ollama is running: ollama serve"
                ) from e
            except urllib.error.HTTPError as e:
                raise OllamaAPIError(f"Ollama API error: {e.code} {e.reason}") from e
            except json.JSONDecodeError as e:
                raise OllamaAPIError(f"Invalid response from Ollama: {e}") from e

    def _extract_json_from_response(self, response_text: str) -> dict:
        """Extract JSON from LLM response text.

        Args:
            response_text: Raw response text from LLM

        Returns:
            Parsed JSON data

        Raises:
            OllamaAPIError: If JSON cannot be extracted
        """
        # Handle potential markdown code blocks
        text = response_text.strip()

        if "```json" in text:
            json_start = text.index("```json") + 7
            json_end = text.index("```", json_start)
            text = text[json_start:json_end].strip()
        elif "```" in text:
            json_start = text.index("```") + 3
            json_end = text.index("```", json_start)
            text = text[json_start:json_end].strip()

        # Try to find JSON object in response
        if not text.startswith("{"):
            # Look for first { and last }
            start = text.find("{")
            end = text.rfind("}") + 1
            if start != -1 and end > start:
                text = text[start:end]

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise OllamaAPIError(
                f"Failed to parse Ollama response as JSON: {e}\n"
                f"Response was: {response_text[:500]}..."
            )

    def analyze_transcript(
        self,
        transcript_text: str,
        session_description: str | None = None,
        target_clips: int = 3,
        video_id: str = "",
    ) -> HighlightAnalysis:
        """Analyze transcript to identify highlight segments.

        Args:
            transcript_text: Full transcript text with timestamps
            session_description: Optional description of the session
            target_clips: Target number of clips to identify
            video_id: Identifier for the video

        Returns:
            HighlightAnalysis with identified segments
        """
        system_prompt = self.prompt_builder.build_system_prompt()
        analysis_prompt = self.prompt_builder.build_analysis_prompt(
            transcript_text=transcript_text,
            session_description=session_description,
            target_clips=target_clips,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": analysis_prompt},
        ]

        response = self._make_request(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        response_text = response.get("message", {}).get("content", "")
        data = self._extract_json_from_response(response_text)

        # Build HighlightAnalysis from response
        segments = []
        for seg_data in data.get("segments", []):
            segments.append(HighlightSegment(
                start_time=float(seg_data["start_time"]),
                end_time=float(seg_data["end_time"]),
                summary=seg_data.get("summary", ""),
                hook_text=seg_data.get("hook_text", ""),
                reason=seg_data.get("reason", ""),
                topics=seg_data.get("topics", []),
                quality_score=float(seg_data.get("quality_score", 0.8)),
            ))

        # Ollama doesn't provide token counts in the same way, estimate from response
        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        total_tokens = eval_count + prompt_eval_count

        return HighlightAnalysis(
            video_id=video_id,
            segments=segments,
            session_summary=data.get("session_summary", ""),
            main_topics=data.get("main_topics", []),
            recommended_count=target_clips,
            model_used=self.config.model,
            tokens_used=total_tokens,
            cost_estimate=0.0,  # Local = free
        )

    def estimate_cost(self, transcript_text: str) -> float:
        """Estimate cost of analyzing a transcript.

        Args:
            transcript_text: Transcript to estimate for

        Returns:
            0.0 (local inference is free)
        """
        return 0.0

    def validate_clip(
        self,
        request: ClipValidationRequest,
    ) -> ClipValidationResponse:
        """Validate a clip segment against quality criteria.

        Args:
            request: ClipValidationRequest with segment details

        Returns:
            ClipValidationResponse with pass/fail for each criterion
        """
        system_prompt = self.prompt_builder.build_validation_system_prompt()
        validation_prompt = self.prompt_builder.build_validation_prompt(
            transcript_segment=request.transcript_segment,
            start_time=request.start_time,
            end_time=request.end_time,
            clip_summary=request.clip_summary,
            brand_context=request.brand_context if request.brand_context else None,
            full_transcript=request.full_transcript_context,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": validation_prompt},
        ]

        response = self._make_request(
            messages=messages,
            max_tokens=1024,
            temperature=0.1,
        )

        response_text = response.get("message", {}).get("content", "")
        data = self._extract_json_from_response(response_text)

        eval_count = response.get("eval_count", 0)
        prompt_eval_count = response.get("prompt_eval_count", 0)
        tokens_used = eval_count + prompt_eval_count

        return ClipValidationResponse(
            clip_id=request.clip_id,
            is_valid=data.get("is_valid", False),
            sentence_boundaries_ok=data.get("sentence_boundaries_ok", True),
            topic_complete=data.get("topic_complete", True),
            has_hook=data.get("has_hook", False),
            standalone_valid=data.get("standalone_valid", True),
            brand_relevant=data.get("brand_relevant", True),
            transcript_aligned=data.get("transcript_aligned", True),
            issues=data.get("issues", []),
            suggestions=data.get("suggestions", []),
            confidence=data.get("confidence", 0.8),
            tokens_used=tokens_used,
        )

    def find_replacement_clips(
        self,
        rejected_clips: list[ClipValidationResponse],
        transcript_text: str,
        used_segments: list[tuple[float, float]],
        target_count: int = 1,
    ) -> list[HighlightSegment]:
        """Find replacement clips for rejected segments.

        Args:
            rejected_clips: List of clips that failed validation
            transcript_text: Full transcript
            used_segments: Time ranges to avoid
            target_count: Number of replacements needed

        Returns:
            List of new HighlightSegment suggestions
        """
        # Convert rejected clips to dict format for prompt
        rejected_dicts = [
            {
                "start_time": 0.0,
                "end_time": 0.0,
                "issues": clip.issues,
            }
            for clip in rejected_clips
        ]

        system_prompt = self.prompt_builder.build_replacement_system_prompt()
        replacement_prompt = self.prompt_builder.build_replacement_prompt(
            transcript_text=transcript_text,
            rejected_clips=rejected_dicts,
            used_segments=used_segments,
            target_count=target_count,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": replacement_prompt},
        ]

        response = self._make_request(
            messages=messages,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        response_text = response.get("message", {}).get("content", "")
        data = self._extract_json_from_response(response_text)

        # Build HighlightSegment list
        segments = []
        for seg_data in data.get("segments", []):
            segments.append(HighlightSegment(
                start_time=float(seg_data["start_time"]),
                end_time=float(seg_data["end_time"]),
                summary=seg_data.get("summary", ""),
                hook_text=seg_data.get("hook_text", ""),
                reason=seg_data.get("reason", ""),
                topics=seg_data.get("topics", []),
                quality_score=float(seg_data.get("quality_score", 0.8)),
            ))

        return segments
