"""Shared LLM caller for provider-agnostic prompt dispatch.

Extracts the provider-dispatch logic from TranscriptRefiner into a reusable
class. Supports Claude (Anthropic SDK), OpenAI (OpenAI SDK), and Ollama
(urllib) providers.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from clip_video.llm.base import LLMConfig, LLMProviderType

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore[assignment,misc]

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore[assignment,misc]


class LLMCaller:
    """Dispatches LLM calls to the configured provider.

    Wraps the provider-specific API details (Anthropic SDK, OpenAI SDK,
    Ollama via urllib) behind a single call(system_prompt, user_prompt) -> str
    interface.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Send prompts to the configured LLM provider and return the response text.

        Raises:
            ValueError: If the provider is not supported.
        """
        if self.config.provider == LLMProviderType.CLAUDE:
            return self._call_claude(system_prompt, user_prompt)
        elif self.config.provider == LLMProviderType.OPENAI:
            return self._call_openai(system_prompt, user_prompt)
        elif self.config.provider == LLMProviderType.OLLAMA:
            return self._call_ollama(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.provider}")

    def _call_claude(self, system_prompt: str, user_prompt: str) -> str:
        if Anthropic is None:
            raise ImportError(
                "anthropic package required for Claude provider. "
                "Install with: pip install anthropic"
            )

        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=0.1,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        if OpenAI is None:
            raise ImportError(
                "openai package required for OpenAI provider. "
                "Install with: pip install openai"
            )

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=0.1,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content or ""

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "options": {
                "temperature": 0.1,
            },
        }

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=self.config.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
            return result["message"]["content"]
