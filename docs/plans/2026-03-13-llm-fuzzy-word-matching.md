# LLM Fuzzy Word Matching Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Automatically generate fuzzy alternatives for missing lyric words via LLM, so words like "gonna" match against "going to" or "connor" in transcripts.

**Architecture:** Extract the LLM provider-dispatch logic from `TranscriptRefiner` into a shared `LLMCaller` class. Build a `FuzzyWordExpander` that uses it to generate alternatives. Integrate into `LyricMatchProcessor.search_all()` as an automatic second pass after initial search.

**Tech Stack:** Python, existing LLM infrastructure (Ollama/Claude/OpenAI), Typer CLI

---

### Task 1: Extract shared LLM caller from TranscriptRefiner

**Files:**
- Create: `src/clip_video/llm/caller.py`
- Modify: `src/clip_video/transcription/llm_refine.py:280-472`
- Test: `tests/test_llm_caller.py`

**Step 1: Write the failing test for LLMCaller**

```python
"""Tests for shared LLM caller."""

import json
from unittest.mock import patch, MagicMock

import pytest

from clip_video.llm.base import LLMConfig, LLMProviderType
from clip_video.llm.caller import LLMCaller


class TestLLMCaller:
    def test_init_with_config(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        caller = LLMCaller(config)
        assert caller.config is config

    def test_call_ollama_dispatches_correctly(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        caller = LLMCaller(config)

        with patch.object(caller, "_call_ollama", return_value="response") as mock:
            result = caller.call("system", "user")

        mock.assert_called_once_with("system", "user")
        assert result == "response"

    def test_call_claude_dispatches_correctly(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE, api_key="test-key")
        caller = LLMCaller(config)

        with patch.object(caller, "_call_claude", return_value="response") as mock:
            result = caller.call("system", "user")

        mock.assert_called_once_with("system", "user")
        assert result == "response"

    def test_call_openai_dispatches_correctly(self):
        config = LLMConfig(provider=LLMProviderType.OPENAI, api_key="test-key")
        caller = LLMCaller(config)

        with patch.object(caller, "_call_openai", return_value="response") as mock:
            result = caller.call("system", "user")

        mock.assert_called_once_with("system", "user")
        assert result == "response"

    def test_call_unsupported_provider_raises(self):
        config = LLMConfig(provider=LLMProviderType.CLAUDE)
        config.provider = "unsupported"
        caller = LLMCaller(config)

        with pytest.raises(ValueError, match="Unsupported"):
            caller.call("system", "user")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_llm_caller.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clip_video.llm.caller'`

**Step 3: Write LLMCaller implementation**

Create `src/clip_video/llm/caller.py` — move `_call_llm`, `_call_claude`, `_call_openai`, `_call_ollama` methods from `TranscriptRefiner` (lines 357-471 of `llm_refine.py`) into a new `LLMCaller` class. The public method is `call(system_prompt, user_prompt) -> str` instead of `_call_llm`.

```python
"""Shared LLM caller with provider dispatch.

Provides a single call(system_prompt, user_prompt) method that dispatches
to the configured LLM provider (Claude, OpenAI, or Ollama).
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from clip_video.llm.base import LLMConfig, LLMProviderType


class LLMCaller:
    """Dispatches LLM calls to the configured provider.

    Extracted from TranscriptRefiner to be reusable across features
    that need simple system+user prompt LLM calls.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def call(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM provider.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt.

        Returns:
            Raw response text from the LLM.

        Raises:
            Exception: On any API or connection error.
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
        try:
            from anthropic import Anthropic
        except ImportError:
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
        try:
            from openai import OpenAI
        except ImportError:
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
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_llm_caller.py -v`
Expected: PASS (all 5 tests)

**Step 5: Commit**

```bash
git add src/clip_video/llm/caller.py tests/test_llm_caller.py
git commit -m "feat: extract shared LLMCaller from TranscriptRefiner"
```

---

### Task 2: Wire TranscriptRefiner to use LLMCaller

**Files:**
- Modify: `src/clip_video/transcription/llm_refine.py:280-472`
- Test: `tests/test_llm_refine.py` (existing — must still pass)

**Step 1: Run existing TranscriptRefiner tests to confirm baseline**

Run: `uv run pytest tests/test_llm_refine.py -v`
Expected: PASS (all tests)

**Step 2: Refactor TranscriptRefiner to delegate to LLMCaller**

Replace the four methods (`_call_llm`, `_call_claude`, `_call_openai`, `_call_ollama`) in `TranscriptRefiner` (lines 357-471) with delegation to `LLMCaller`. Keep the `_call_llm` method as a thin wrapper so existing tests that mock it continue to work:

```python
class TranscriptRefiner:
    def __init__(self, config: LLMConfig, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.config = config
        self.chunk_size = chunk_size
        self._caller = LLMCaller(config)

    # ... is_available() and refine() unchanged ...

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch LLM call to the configured provider.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt with transcript data.

        Returns:
            Raw response text from the LLM.

        Raises:
            Exception: On any API or connection error.
        """
        return self._caller.call(system_prompt, user_prompt)
```

Remove `_call_claude`, `_call_openai`, `_call_ollama` from `TranscriptRefiner`. Add `from clip_video.llm.caller import LLMCaller` to imports. Remove now-unused `os`, `urllib.request`, `urllib.error` imports if they're only used by the removed methods (check — `is_available` and `_call_llm` via `refine` still need them... actually `is_available` uses `urllib` and `os` directly, so keep those).

**Step 3: Run existing tests to verify no breakage**

Run: `uv run pytest tests/test_llm_refine.py -v`
Expected: PASS (all tests unchanged — they mock `_call_llm` which still exists)

**Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS (no regressions)

**Step 5: Commit**

```bash
git add src/clip_video/transcription/llm_refine.py
git commit -m "refactor: delegate TranscriptRefiner LLM calls to shared LLMCaller"
```

---

### Task 3: Build FuzzyWordExpander

**Files:**
- Create: `src/clip_video/lyrics/fuzzy.py`
- Test: `tests/test_fuzzy_word_expander.py`

**Step 1: Write failing tests for FuzzyWordExpander**

```python
"""Tests for LLM fuzzy word expansion."""

import json
from unittest.mock import patch, MagicMock

import pytest

from clip_video.llm.base import LLMConfig, LLMProviderType
from clip_video.lyrics.fuzzy import FuzzyWordExpander, parse_fuzzy_response


class TestParseFuzzyResponse:
    def test_parse_valid_json(self):
        response = json.dumps({
            "gonna": ["going to", "connor"],
            "ain't": ["isn't", "aren't"],
        })
        result = parse_fuzzy_response(response)
        assert result == {
            "gonna": ["going to", "connor"],
            "ain't": ["isn't", "aren't"],
        }

    def test_parse_json_in_code_block(self):
        response = '```json\n{"gonna": ["going to"]}\n```'
        result = parse_fuzzy_response(response)
        assert result == {"gonna": ["going to"]}

    def test_parse_invalid_json_returns_empty(self):
        result = parse_fuzzy_response("this is not json")
        assert result == {}

    def test_parse_empty_object(self):
        result = parse_fuzzy_response("{}")
        assert result == {}

    def test_parse_filters_non_list_values(self):
        response = json.dumps({
            "good": ["fine", "okay"],
            "bad": "not a list",
            "also_bad": 42,
        })
        result = parse_fuzzy_response(response)
        assert result == {"good": ["fine", "okay"]}

    def test_parse_filters_non_string_items(self):
        response = json.dumps({
            "word": ["valid", 123, "also valid"],
        })
        result = parse_fuzzy_response(response)
        assert result == {"word": ["valid", "also valid"]}


class TestFuzzyWordExpander:
    def test_expand_returns_alternatives(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        expander = FuzzyWordExpander(config)

        mock_response = json.dumps({
            "gonna": ["going to", "connor"],
            "ain't": ["isn't"],
        })

        with patch.object(expander._caller, "call", return_value=mock_response):
            result = expander.expand(["gonna", "ain't"])

        assert result == {
            "gonna": ["going to", "connor"],
            "ain't": ["isn't"],
        }

    def test_expand_empty_list_skips_llm(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        expander = FuzzyWordExpander(config)

        with patch.object(expander._caller, "call") as mock_call:
            result = expander.expand([])

        mock_call.assert_not_called()
        assert result == {}

    def test_expand_handles_llm_error_gracefully(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        expander = FuzzyWordExpander(config)

        with patch.object(expander._caller, "call", side_effect=Exception("API error")):
            result = expander.expand(["gonna"])

        assert result == {}

    def test_expand_handles_malformed_response(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        expander = FuzzyWordExpander(config)

        with patch.object(expander._caller, "call", return_value="not json at all"):
            result = expander.expand(["gonna"])

        assert result == {}

    def test_expand_prompt_contains_all_words(self):
        config = LLMConfig(provider=LLMProviderType.OLLAMA)
        expander = FuzzyWordExpander(config)

        with patch.object(expander._caller, "call", return_value="{}") as mock_call:
            expander.expand(["gonna", "ain't", "wanna"])

        call_args = mock_call.call_args
        user_prompt = call_args[0][1]
        assert "gonna" in user_prompt
        assert "ain't" in user_prompt
        assert "wanna" in user_prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_fuzzy_word_expander.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'clip_video.lyrics.fuzzy'`

**Step 3: Write FuzzyWordExpander implementation**

```python
"""LLM-powered fuzzy word expansion for lyric matching.

Generates alternative search terms for missing lyric words by asking
an LLM for phonetic matches, contractions/expansions, and casual
speech equivalents.
"""

from __future__ import annotations

import json
import re

from clip_video.llm.base import LLMConfig
from clip_video.llm.caller import LLMCaller


FUZZY_SYSTEM_PROMPT = """\
You suggest alternative words for lyric matching against video transcripts.

For each word, suggest 1-3 alternatives that:
- Sound similar (phonetic matches): "gonna" -> "connor"
- Are contractions or expansions: "gonna" -> "going to", "ain't" -> "is not"
- Are casual/formal equivalents: "ain't" -> "isn't", "wanna" -> "want to"

Return ONLY a JSON object mapping each input word to a list of alternatives.
Do not include the original word in its own alternatives list.
If you cannot think of any alternatives for a word, map it to an empty list.

Example:
{"gonna": ["going to", "connor", "going"], "ain't": ["isn't", "aren't", "is not"]}\
"""


def parse_fuzzy_response(response_text: str) -> dict[str, list[str]]:
    """Parse LLM response into word -> alternatives mapping.

    Handles JSON wrapped in markdown code blocks. Filters out
    non-string values and non-list entries.

    Args:
        response_text: Raw text from the LLM.

    Returns:
        Dict mapping words to lists of alternative strings.
        Returns empty dict on parse failure.
    """
    text = response_text.strip()

    # Strip markdown code block if present
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        parsed = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    result: dict[str, list[str]] = {}
    for word, alternatives in parsed.items():
        if not isinstance(alternatives, list):
            continue
        filtered = [a for a in alternatives if isinstance(a, str)]
        if filtered:
            result[word] = filtered

    return result


class FuzzyWordExpander:
    """Generates fuzzy alternatives for missing lyric words via LLM.

    Sends a batch of missing words to the LLM and gets back
    phonetic, contraction, and casual speech alternatives.
    Non-fatal: LLM errors return empty results.
    """

    def __init__(self, config: LLMConfig):
        self._caller = LLMCaller(config)

    def expand(self, missing_words: list[str]) -> dict[str, list[str]]:
        """Generate alternatives for missing words.

        Args:
            missing_words: List of words that had no exact match.

        Returns:
            Dict mapping each word to a list of alternative strings.
            Returns empty dict if LLM call fails or no words provided.
        """
        if not missing_words:
            return {}

        user_prompt = (
            "Suggest alternatives for these words:\n"
            + json.dumps(missing_words)
        )

        try:
            response = self._caller.call(FUZZY_SYSTEM_PROMPT, user_prompt)
            return parse_fuzzy_response(response)
        except Exception:
            return {}
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_fuzzy_word_expander.py -v`
Expected: PASS (all 11 tests)

**Step 5: Commit**

```bash
git add src/clip_video/lyrics/fuzzy.py tests/test_fuzzy_word_expander.py
git commit -m "feat: add FuzzyWordExpander for LLM-powered lyric alternatives"
```

---

### Task 4: Add fuzzy_matching flag to LyricMatchConfig

**Files:**
- Modify: `src/clip_video/modes/lyric_match.py:30-53`
- Test: `tests/test_lyric_match.py`

**Step 1: Write the failing test**

Add to the existing `TestLyricMatchConfig` class in `tests/test_lyric_match.py`:

```python
def test_default_fuzzy_matching_enabled(self):
    config = LyricMatchConfig()
    assert config.fuzzy_matching is True

def test_fuzzy_matching_can_be_disabled(self):
    config = LyricMatchConfig(fuzzy_matching=False)
    assert config.fuzzy_matching is False
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_lyric_match.py::TestLyricMatchConfig::test_default_fuzzy_matching_enabled -v`
Expected: FAIL — `TypeError: unexpected keyword argument 'fuzzy_matching'`

**Step 3: Add the field to LyricMatchConfig**

In `src/clip_video/modes/lyric_match.py`, add to the `LyricMatchConfig` dataclass (after line 53):

```python
    fuzzy_matching: bool = True  # LLM-powered fuzzy alternatives for missing words
```

**Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_lyric_match.py::TestLyricMatchConfig -v`
Expected: PASS

**Step 5: Commit**

```bash
git add src/clip_video/modes/lyric_match.py tests/test_lyric_match.py
git commit -m "feat: add fuzzy_matching config flag (default enabled)"
```

---

### Task 5: Integrate fuzzy expansion into search_all

**Files:**
- Modify: `src/clip_video/modes/lyric_match.py:546-605`
- Test: `tests/test_lyric_match.py`

**Step 1: Write the failing tests**

Add a new test class to `tests/test_lyric_match.py`:

```python
class TestSearchAllFuzzyMatching:
    """Tests for fuzzy matching integration in search_all."""

    def _make_project_with_missing_words(self, tmp_path):
        """Create a project where some words will be missing."""
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("gonna ain't hello")

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
            config=LyricMatchConfig(extract_phrases=False),
        )

        project = processor.create_project("my_song", lyrics_file)
        return processor, project

    @patch("clip_video.modes.lyric_match.FuzzyWordExpander")
    def test_fuzzy_expands_missing_words(self, MockExpander, tmp_path):
        """Fuzzy expansion is called for words with no search results."""
        processor, project = self._make_project_with_missing_words(tmp_path)

        # Mock the expander to return alternatives
        mock_instance = MockExpander.return_value
        mock_instance.expand.return_value = {
            "gonna": ["going to", "connor"],
            "ain't": ["isn't"],
        }

        results = processor.search_all(project)

        # All 3 words should have no results (no transcripts indexed)
        # So fuzzy should be called with all 3 missing words
        mock_instance.expand.assert_called_once()
        called_words = mock_instance.expand.call_args[0][0]
        assert "gonna" in called_words
        assert "ain't" in called_words
        assert "hello" in called_words

    @patch("clip_video.modes.lyric_match.FuzzyWordExpander")
    def test_fuzzy_skipped_when_disabled(self, MockExpander, tmp_path):
        """Fuzzy expansion is skipped when fuzzy_matching=False."""
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("gonna ain't")

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
            config=LyricMatchConfig(fuzzy_matching=False, extract_phrases=False),
        )

        project = processor.create_project("my_song", lyrics_file)
        results = processor.search_all(project)

        MockExpander.assert_not_called()

    @patch("clip_video.modes.lyric_match.FuzzyWordExpander")
    def test_fuzzy_adds_alternatives_to_targets(self, MockExpander, tmp_path):
        """Fuzzy alternatives are stored on ExtractionTarget objects."""
        processor, project = self._make_project_with_missing_words(tmp_path)

        mock_instance = MockExpander.return_value
        mock_instance.expand.return_value = {
            "gonna": ["going to"],
        }

        processor.search_all(project)

        # Find the "gonna" target and check it got alternatives
        for lcs in project.line_clip_sets:
            for target in lcs.targets:
                if target.text == "gonna":
                    assert "going to" in target.alternatives

    @patch("clip_video.modes.lyric_match.FuzzyWordExpander")
    def test_fuzzy_skipped_when_no_missing_words(self, MockExpander, tmp_path):
        """Fuzzy expansion is skipped when all words have matches."""
        brand_path = tmp_path / "test_brand"
        brand_path.mkdir(parents=True)

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text("hello")

        processor = LyricMatchProcessor(
            brand_name="test_brand",
            brands_root=tmp_path,
            config=LyricMatchConfig(extract_phrases=False),
        )

        project = processor.create_project("my_song", lyrics_file)

        # Pre-populate the index with the word "hello" so it has results
        from clip_video.transcript.index import TranscriptIndex, WordOccurrence
        index = TranscriptIndex(brand_name="test_brand")
        index.words["hello"] = [
            WordOccurrence(
                word="hello", start=1.0, end=1.5, confidence=0.9,
                project_name="p", video_id="v", segment_index=0, word_index=0,
            )
        ]
        processor.searcher._index = index

        results = processor.search_all(project)

        # No missing words, so fuzzy should not be called
        MockExpander.assert_not_called()

    @patch("clip_video.modes.lyric_match.FuzzyWordExpander")
    def test_fuzzy_skips_words_with_existing_alternatives(self, MockExpander, tmp_path):
        """Words that already have alternatives are not sent to the LLM."""
        processor, project = self._make_project_with_missing_words(tmp_path)

        # Pre-set alternatives on "gonna" target
        for lcs in project.line_clip_sets:
            for target in lcs.targets:
                if target.text == "gonna":
                    target.alternatives = ["going to"]

        mock_instance = MockExpander.return_value
        mock_instance.expand.return_value = {}

        processor.search_all(project)

        # "gonna" should NOT be in the missing words sent to expand
        if mock_instance.expand.called:
            called_words = mock_instance.expand.call_args[0][0]
            assert "gonna" not in called_words
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_lyric_match.py::TestSearchAllFuzzyMatching -v`
Expected: FAIL — `ImportError` or `AttributeError` (FuzzyWordExpander not imported in lyric_match.py)

**Step 3: Integrate fuzzy expansion into search_all**

In `src/clip_video/modes/lyric_match.py`, modify `search_all()` to add a fuzzy expansion pass after the initial search. Add the import and modify the method:

At top of the method (or at module level with lazy import inside the method to avoid circular imports), add:

```python
from clip_video.lyrics.fuzzy import FuzzyWordExpander
```

After the existing search loop (after line 604, before `return all_results`), add:

```python
        # Fuzzy matching: generate alternatives for missing single-word targets
        if self.config.fuzzy_matching:
            # Find missing single-word targets that don't already have alternatives
            missing_words = []
            for target in targets:
                if target.is_phrase:
                    continue
                if target.alternatives:
                    continue
                # Check if this target or any of its alternatives got results
                has_results = (
                    target.text in all_results
                    and all_results[target.text].results
                )
                if not has_results:
                    missing_words.append(target.text)

            # Deduplicate
            missing_words = list(dict.fromkeys(missing_words))

            if missing_words:
                from clip_video.config import load_brand_config
                brand_config = load_brand_config(self.brand_name)
                llm_config = LLMConfig(
                    provider=LLMProviderType(brand_config.llm_provider),
                    model=brand_config.llm_model,
                )
                expander = FuzzyWordExpander(llm_config)
                alternatives = expander.expand(missing_words)

                # Add alternatives to targets and re-search
                for target in targets:
                    if target.text in alternatives:
                        for alt in alternatives[target.text]:
                            if alt not in target.alternatives:
                                target.alternatives.append(alt)

                        # Search the new alternatives
                        for alt in target.alternatives:
                            if alt not in all_results:
                                alt_results = self.searcher.search(
                                    alt,
                                    max_results=self.config.max_candidates_per_target * 5,
                                )
                                if self.config.shuffle_candidates and alt_results.results:
                                    shuffled = list(alt_results.results)
                                    random.shuffle(shuffled)
                                    alt_results.results = shuffled[:self.config.max_candidates_per_target]
                                    alt_results.total_count = len(alt_results.results)
                                all_results[alt] = alt_results

                # Save project to persist new alternatives
                project.save()
```

Also add the necessary imports near the top of the method or at module level:

```python
from clip_video.llm.base import LLMConfig, LLMProviderType
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_lyric_match.py::TestSearchAllFuzzyMatching -v`
Expected: PASS (all 5 tests)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS (no regressions)

**Step 6: Commit**

```bash
git add src/clip_video/modes/lyric_match.py tests/test_lyric_match.py
git commit -m "feat: integrate fuzzy word expansion into search_all"
```

---

### Task 6: Add --no-fuzzy CLI flag

**Files:**
- Modify: `src/clip_video/cli.py:1442-1523`
- Test: `tests/test_cli_lyric_match.py` (or add to existing CLI tests)

**Step 1: Check for existing CLI tests**

Run: `uv run pytest tests/ -k lyric --collect-only 2>&1 | head -30`

If there are CLI-level tests, add to them. Otherwise this is a straightforward wiring change.

**Step 2: Add the --no-fuzzy flag**

In `src/clip_video/cli.py`, add a new parameter to the `lyric_match` function (after the `yes` parameter around line 1473):

```python
    no_fuzzy: Annotated[
        bool,
        typer.Option("--no-fuzzy", help="Disable LLM fuzzy matching for missing words"),
    ] = False,
```

Then in the config creation (around line 1516), add it:

```python
    config = LyricMatchConfig(
        max_candidates_per_target=max_candidates,
        extract_words=not no_words,
        extract_phrases=not no_phrases,
        fuzzy_matching=not no_fuzzy,
    )
```

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/clip_video/cli.py
git commit -m "feat: add --no-fuzzy CLI flag to disable LLM fuzzy matching"
```

---

### Task 7: Add fuzzy matching console output

**Files:**
- Modify: `src/clip_video/modes/lyric_match.py` (search_all progress callback or return value)
- Modify: `src/clip_video/cli.py` (display output)

**Step 1: Add a fuzzy_callback parameter to search_all**

In `search_all()`, add an optional `fuzzy_callback: Callable[[int], None] | None = None` parameter. Call it with the number of missing words before calling the expander, so the CLI can display progress:

```python
    def search_all(
        self,
        project: LyricMatchProject,
        progress_callback: Callable[[str, int, int], None] | None = None,
        fuzzy_callback: Callable[[int], None] | None = None,
    ) -> dict[str, SearchResults]:
```

Inside the fuzzy matching block, before calling `expander.expand()`:

```python
                if fuzzy_callback:
                    fuzzy_callback(len(missing_words))
```

**Step 2: Add console output in CLI**

In `src/clip_video/cli.py`, after the search progress bar, define a fuzzy callback and pass it:

```python
        def fuzzy_progress(count: int):
            console.print(f"\n[cyan]Generating fuzzy alternatives for {count} missing words...[/cyan]")

        search_results = processor.search_all(
            project,
            progress_callback=search_progress,
            fuzzy_callback=fuzzy_progress,
        )
```

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS

**Step 4: Commit**

```bash
git add src/clip_video/modes/lyric_match.py src/clip_video/cli.py
git commit -m "feat: add console output for fuzzy matching progress"
```

---

### Task 8: Update CLAUDE.md and README

**Files:**
- Modify: `CLAUDE.md`
- Modify: `README.md` (if it documents CLI flags)

**Step 1: Update CLAUDE.md**

Add fuzzy matching to the relevant sections:
- Under "Key Concepts" or "Data Flow" for Lyric Match Mode, mention fuzzy expansion
- Under "Common Tasks" or CLI reference, document `--no-fuzzy`
- Under "LLM Analysis" section, note that fuzzy matching also uses the same LLM provider config

**Step 2: Check if README needs updating**

If README documents the lyric-match CLI flags, add `--no-fuzzy`.

**Step 3: Commit**

```bash
git add CLAUDE.md README.md
git commit -m "docs: document LLM fuzzy word matching feature"
```

---

### Task 9: Final verification

**Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: PASS (all tests)

**Step 2: Run type checking**

Run: `uv run mypy src/`
Expected: PASS (no new errors)

**Step 3: Run linting**

Run: `uv run ruff check src/`
Expected: PASS (no new errors)

**Step 4: Verify CLI help**

Run: `uv run clip-video lyric-match --help`
Expected: Shows `--no-fuzzy` flag with description
