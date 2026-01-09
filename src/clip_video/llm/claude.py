"""Claude (Anthropic) LLM provider implementation.

Provides integration with Claude API for transcript analysis
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


class ClaudeAPIError(Exception):
    """Error from Claude API."""

    pass


class ClaudeRateLimitError(ClaudeAPIError):
    """Rate limit exceeded."""

    pass


class ClaudeLLM(LLMProvider):
    """Claude (Anthropic) LLM provider.

    Uses the Anthropic API to analyze transcripts and identify
    highlight-worthy segments.

    Requires ANTHROPIC_API_KEY environment variable or api_key in config.
    """

    # Pricing per million tokens (as of early 2025)
    PRICING = {
        "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
        "claude-3-5-sonnet-20241022": {"input": 3.0, "output": 15.0},
        "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
        "claude-3-opus-20240229": {"input": 15.0, "output": 75.0},
    }

    def __init__(
        self,
        config: LLMConfig | None = None,
        prompt_builder: HighlightPromptBuilder | None = None,
    ):
        """Initialize Claude provider.

        Args:
            config: LLM configuration
            prompt_builder: Optional custom prompt builder
        """
        if config is None:
            config = LLMConfig(provider=LLMProviderType.CLAUDE)

        super().__init__(config)
        self.prompt_builder = prompt_builder or CONFERENCE_PROMPT
        self._client = None

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "Claude (Anthropic)"

    def _get_api_key(self) -> str | None:
        """Get API key from config or environment."""
        return self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")

    def is_available(self) -> bool:
        """Check if Claude API is available.

        Returns:
            True if API key is set
        """
        return self._get_api_key() is not None

    def _get_client(self):
        """Get or create Anthropic client."""
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package required for Claude provider. "
                    "Install with: pip install anthropic"
                )

            api_key = self._get_api_key()
            if not api_key:
                raise ClaudeAPIError(
                    "ANTHROPIC_API_KEY not set. Set environment variable or "
                    "provide api_key in LLMConfig."
                )

            self._client = Anthropic(api_key=api_key)

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
                response = client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": analysis_prompt},
                    ],
                )
                break
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5
                    time.sleep(wait_time)
                    continue
                raise ClaudeAPIError(f"Claude API error: {e}") from e

        # Parse response
        response_text = response.content[0].text

        # Extract JSON from response
        try:
            # Handle potential markdown code blocks
            if "```json" in response_text:
                json_start = response_text.index("```json") + 7
                json_end = response_text.index("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.index("```") + 3
                json_end = response_text.index("```", json_start)
                response_text = response_text[json_start:json_end].strip()

            data = json.loads(response_text)
        except (json.JSONDecodeError, ValueError) as e:
            raise ClaudeAPIError(f"Failed to parse Claude response as JSON: {e}")

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
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        total_tokens = input_tokens + output_tokens

        pricing = self.PRICING.get(self.config.model, {"input": 3.0, "output": 15.0})
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

        pricing = self.PRICING.get(self.config.model, {"input": 3.0, "output": 15.0})
        cost = (
            (estimated_input_tokens / 1_000_000) * pricing["input"]
            + (estimated_output_tokens / 1_000_000) * pricing["output"]
        )

        return cost
