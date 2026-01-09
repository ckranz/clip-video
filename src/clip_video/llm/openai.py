"""OpenAI LLM provider implementation.

Provides integration with OpenAI API for transcript analysis
and highlight detection.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

from clip_video.llm.base import (
    LLMProvider,
    LLMConfig,
    LLMProviderType,
    HighlightAnalysis,
    HighlightSegment,
)
from clip_video.llm.prompts import HighlightPromptBuilder, CONFERENCE_PROMPT


class OpenAIAPIError(Exception):
    """Error from OpenAI API."""

    pass


class OpenAIRateLimitError(OpenAIAPIError):
    """Rate limit exceeded."""

    pass


class OpenAILLM(LLMProvider):
    """OpenAI LLM provider.

    Uses the OpenAI API to analyze transcripts and identify
    highlight-worthy segments.

    Requires OPENAI_API_KEY environment variable or api_key in config.
    """

    # Pricing per million tokens (as of early 2025)
    PRICING = {
        "gpt-4o": {"input": 2.5, "output": 10.0},
        "gpt-4o-mini": {"input": 0.15, "output": 0.6},
        "gpt-4-turbo": {"input": 10.0, "output": 30.0},
        "gpt-4": {"input": 30.0, "output": 60.0},
        "gpt-3.5-turbo": {"input": 0.5, "output": 1.5},
    }

    def __init__(
        self,
        config: LLMConfig | None = None,
        prompt_builder: HighlightPromptBuilder | None = None,
    ):
        """Initialize OpenAI provider.

        Args:
            config: LLM configuration
            prompt_builder: Optional custom prompt builder
        """
        if config is None:
            config = LLMConfig(provider=LLMProviderType.OPENAI)

        super().__init__(config)
        self.prompt_builder = prompt_builder or CONFERENCE_PROMPT
        self._client = None

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "OpenAI"

    def _get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        return self.config.api_key or os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        """Check if OpenAI API is available.

        Returns:
            True if API key is set
        """
        return self._get_api_key() is not None

    def _get_client(self):
        """Get or create OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package required for OpenAI provider. "
                    "Install with: pip install openai"
                )

            api_key = self._get_api_key()
            if not api_key:
                raise OpenAIAPIError(
                    "OPENAI_API_KEY not set. Set environment variable or "
                    "provide api_key in LLMConfig."
                )

            self._client = OpenAI(api_key=api_key)

        return self._client

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
        client = self._get_client()

        system_prompt = self.prompt_builder.build_system_prompt()
        analysis_prompt = self.prompt_builder.build_analysis_prompt(
            transcript_text=transcript_text,
            session_description=session_description,
            target_clips=target_clips,
        )

        # Make API call with retry for rate limits
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": analysis_prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    continue
                raise OpenAIAPIError(f"OpenAI API error: {e}") from e

        # Parse response
        response_text = response.choices[0].message.content

        try:
            data = json.loads(response_text)
        except json.JSONDecodeError as e:
            raise OpenAIAPIError(f"Failed to parse OpenAI response as JSON: {e}")

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

        # Calculate token usage and cost
        usage = response.usage
        input_tokens = usage.prompt_tokens
        output_tokens = usage.completion_tokens
        total_tokens = usage.total_tokens

        pricing = self.PRICING.get(self.config.model, {"input": 2.5, "output": 10.0})
        cost = (
            (input_tokens / 1_000_000) * pricing["input"]
            + (output_tokens / 1_000_000) * pricing["output"]
        )

        return HighlightAnalysis(
            video_id=video_id,
            segments=segments,
            session_summary=data.get("session_summary", ""),
            main_topics=data.get("main_topics", []),
            recommended_count=target_clips,
            model_used=self.config.model,
            tokens_used=total_tokens,
            cost_estimate=cost,
        )

    def estimate_cost(self, transcript_text: str) -> float:
        """Estimate cost of analyzing a transcript.

        Args:
            transcript_text: Transcript to estimate for

        Returns:
            Estimated cost in USD
        """
        # Rough token estimation: ~4 chars per token for English
        estimated_input_tokens = len(transcript_text) // 4
        # Add overhead for prompts
        estimated_input_tokens += 1000

        # Assume output is about 10% of input
        estimated_output_tokens = estimated_input_tokens // 10

        pricing = self.PRICING.get(self.config.model, {"input": 2.5, "output": 10.0})
        cost = (
            (estimated_input_tokens / 1_000_000) * pricing["input"]
            + (estimated_output_tokens / 1_000_000) * pricing["output"]
        )

        return cost


def get_llm_provider(config: LLMConfig | None = None) -> LLMProvider:
    """Factory function to get the appropriate LLM provider.

    Args:
        config: LLM configuration

    Returns:
        LLMProvider instance based on config
    """
    if config is None:
        config = LLMConfig()

    if config.provider == LLMProviderType.CLAUDE:
        from clip_video.llm.claude import ClaudeLLM
        return ClaudeLLM(config)
    elif config.provider == LLMProviderType.OPENAI:
        return OpenAILLM(config)
    else:
        raise ValueError(f"Unsupported LLM provider: {config.provider}")
