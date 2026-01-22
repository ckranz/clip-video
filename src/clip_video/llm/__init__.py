"""LLM integration for highlight detection.

Provides abstraction layer for Claude, OpenAI, and Ollama to analyze
transcripts and identify highlight-worthy segments for social media clips.
"""

from clip_video.llm.base import (
    LLMProvider,
    LLMConfig,
    LLMProviderType,
    HighlightSegment,
    HighlightAnalysis,
    ClipValidationRequest,
    ClipValidationResponse,
)
from clip_video.llm.claude import ClaudeLLM
from clip_video.llm.openai import OpenAILLM
from clip_video.llm.ollama import OllamaLLM
from clip_video.llm.prompts import HighlightPromptBuilder

__all__ = [
    "LLMProvider",
    "LLMConfig",
    "LLMProviderType",
    "HighlightSegment",
    "HighlightAnalysis",
    "ClipValidationRequest",
    "ClipValidationResponse",
    "ClaudeLLM",
    "OpenAILLM",
    "OllamaLLM",
    "HighlightPromptBuilder",
]
