"""Tests for shared LLMCaller."""

import json
from unittest.mock import patch, MagicMock

import pytest

from clip_video.llm.base import LLMConfig, LLMProviderType
from clip_video.llm.caller import LLMCaller


class TestLLMCallerInit:
    def test_init_stores_config(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE)
        caller = LLMCaller(config)
        assert caller.config is config

    def test_init_with_each_provider(self):
        for provider in LLMProviderType:
            config = LLMConfig(provider=provider)
            caller = LLMCaller(config)
            assert caller.config.provider == provider


class TestLLMCallerDispatch:
    def test_call_dispatches_to_claude(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE)
        caller = LLMCaller(config)

        with patch.object(caller, "_call_claude", return_value="claude response") as mock:
            result = caller.call("system", "user")

        assert result == "claude response"
        mock.assert_called_once_with("system", "user")

    def test_call_dispatches_to_openai(self):
        config = LLMConfig(provider=LLMProviderType.OPENAI)
        caller = LLMCaller(config)

        with patch.object(caller, "_call_openai", return_value="openai response") as mock:
            result = caller.call("system", "user")

        assert result == "openai response"
        mock.assert_called_once_with("system", "user")

    def test_call_dispatches_to_ollama(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        caller = LLMCaller(config)

        with patch.object(caller, "_call_ollama", return_value="ollama response") as mock:
            result = caller.call("system", "user")

        assert result == "ollama response"
        mock.assert_called_once_with("system", "user")


class TestLLMCallerUnsupportedProvider:
    def test_unsupported_provider_raises_value_error(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE)
        caller = LLMCaller(config)
        # Force an invalid provider value
        caller.config.provider = "bogus"

        with pytest.raises(ValueError, match="Unsupported LLM provider"):
            caller.call("system", "user")


class TestLLMCallerClaude:
    def test_call_claude_uses_anthropic_sdk(self):
        config = LLMConfig(
            provider=LLMProviderType.CLAUDE,
            api_key="test-key",
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
        )
        caller = LLMCaller(config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="hello from claude")]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("clip_video.llm.caller.Anthropic", return_value=mock_client) as mock_cls:
            result = caller._call_claude("be helpful", "what is 2+2")

        mock_cls.assert_called_once_with(api_key="test-key")
        mock_client.messages.create.assert_called_once_with(
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            temperature=0.3,
            system="be helpful",
            messages=[{"role": "user", "content": "what is 2+2"}],
        )
        assert result == "hello from claude"

    def test_call_claude_falls_back_to_env_key(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE, api_key=None)
        caller = LLMCaller(config)

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="ok")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response

        with patch("clip_video.llm.caller.Anthropic", return_value=mock_client) as mock_cls:
            with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "env-key"}):
                caller._call_claude("sys", "usr")

        mock_cls.assert_called_once_with(api_key="env-key")

    def test_call_claude_raises_when_anthropic_not_installed(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE, api_key="k")
        caller = LLMCaller(config)

        with patch("clip_video.llm.caller.Anthropic", None):
            with pytest.raises(ImportError, match="anthropic package required"):
                caller._call_claude("sys", "usr")


class TestLLMCallerOpenAI:
    def test_call_openai_uses_openai_sdk(self):
        config = LLMConfig(
            provider=LLMProviderType.OPENAI,
            api_key="test-key",
            model="gpt-4.1",
            max_tokens=2048,
        )
        caller = LLMCaller(config)

        mock_message = MagicMock()
        mock_message.content = "hello from openai"
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("clip_video.llm.caller.OpenAI", return_value=mock_client) as mock_cls:
            result = caller._call_openai("be helpful", "what is 2+2")

        mock_cls.assert_called_once_with(api_key="test-key")
        mock_client.chat.completions.create.assert_called_once_with(
            model="gpt-4.1",
            max_tokens=2048,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "be helpful"},
                {"role": "user", "content": "what is 2+2"},
            ],
        )
        assert result == "hello from openai"

    def test_call_openai_returns_empty_string_on_none_content(self):
        config = LLMConfig(provider=LLMProviderType.OPENAI, api_key="k")
        caller = LLMCaller(config)

        mock_message = MagicMock()
        mock_message.content = None
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response

        with patch("clip_video.llm.caller.OpenAI", return_value=mock_client):
            result = caller._call_openai("sys", "usr")

        assert result == ""

    def test_call_openai_raises_when_openai_not_installed(self):
        config = LLMConfig(provider=LLMProviderType.OPENAI, api_key="k")
        caller = LLMCaller(config)

        with patch("clip_video.llm.caller.OpenAI", None):
            with pytest.raises(ImportError, match="openai package required"):
                caller._call_openai("sys", "usr")


class TestLLMCallerOllama:
    def test_call_ollama_uses_urllib(self):
        config = LLMConfig(
            provider=LLMProviderType.OLLAMA,
            model="llama3.2",
            timeout=120,
        )
        caller = LLMCaller(config)

        response_data = {"message": {"content": "hello from ollama"}}
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(response_data).encode("utf-8")
        mock_response.__enter__ = lambda s: s
        mock_response.__exit__ = MagicMock(return_value=False)

        with patch("clip_video.llm.caller.urllib.request.urlopen", return_value=mock_response) as mock_urlopen:
            result = caller._call_ollama("be helpful", "what is 2+2")

        assert result == "hello from ollama"
        # Verify the request was made to the right URL
        call_args = mock_urlopen.call_args
        request_obj = call_args[0][0]
        assert request_obj.full_url == "http://localhost:11434/api/chat"
        assert call_args[1]["timeout"] == 120
