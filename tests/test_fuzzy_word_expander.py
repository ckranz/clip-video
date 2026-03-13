"""Tests for FuzzyWordExpander and parse_fuzzy_response."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from clip_video.llm.base import LLMConfig, LLMProviderType
from clip_video.lyrics.fuzzy import FuzzyWordExpander, parse_fuzzy_response


class TestParseFuzzyResponse:
    def test_valid_json(self):
        response = '{"gonna": ["going to", "connor"], "ain\'t": ["is not"]}'
        result = parse_fuzzy_response(response)
        assert result == {"gonna": ["going to", "connor"], "ain't": ["is not"]}

    def test_json_in_code_block(self):
        response = '```json\n{"gonna": ["going to"]}\n```'
        result = parse_fuzzy_response(response)
        assert result == {"gonna": ["going to"]}

    def test_invalid_json_returns_empty(self):
        result = parse_fuzzy_response("this is not json at all")
        assert result == {}

    def test_empty_object(self):
        result = parse_fuzzy_response("{}")
        assert result == {}

    def test_filters_non_list_values(self):
        response = '{"good": ["fine"], "bad": "string", "also_bad": 42}'
        result = parse_fuzzy_response(response)
        assert result == {"good": ["fine"]}

    def test_filters_non_string_items_in_lists(self):
        response = '{"word": ["valid", 123, null, "also valid"]}'
        result = parse_fuzzy_response(response)
        assert result == {"word": ["valid", "also valid"]}


class TestFuzzyWordExpander:
    def _make_expander(self) -> FuzzyWordExpander:
        config = LLMConfig(provider=LLMProviderType.OLLAMA, model="test-model")
        return FuzzyWordExpander(config)

    def test_expand_returns_alternatives(self):
        expander = self._make_expander()
        expander._caller = MagicMock()
        expander._caller.call.return_value = json.dumps(
            {"gonna": ["going to", "connor"], "wanna": ["want to"]}
        )
        result = expander.expand(["gonna", "wanna"])
        assert result == {"gonna": ["going to", "connor"], "wanna": ["want to"]}
        expander._caller.call.assert_called_once()

    def test_empty_list_skips_llm(self):
        expander = self._make_expander()
        expander._caller = MagicMock()
        result = expander.expand([])
        assert result == {}
        expander._caller.call.assert_not_called()

    def test_llm_error_returns_empty_dict(self):
        expander = self._make_expander()
        expander._caller = MagicMock()
        expander._caller.call.side_effect = Exception("LLM unavailable")
        result = expander.expand(["gonna"])
        assert result == {}

    def test_malformed_response_returns_empty_dict(self):
        expander = self._make_expander()
        expander._caller = MagicMock()
        expander._caller.call.return_value = "Sorry, I can't do that."
        result = expander.expand(["gonna"])
        assert result == {}

    def test_prompt_contains_all_words(self):
        expander = self._make_expander()
        expander._caller = MagicMock()
        expander._caller.call.return_value = "{}"
        expander.expand(["gonna", "ain't", "wanna"])
        call_args = expander._caller.call.call_args
        user_prompt = call_args[0][1] if call_args[0] else call_args[1]["user_prompt"]
        assert "gonna" in user_prompt
        assert "ain't" in user_prompt
        assert "wanna" in user_prompt
