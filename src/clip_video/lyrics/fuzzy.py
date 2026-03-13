"""Fuzzy word expansion via LLM for finding phonetic and semantic alternatives."""

from __future__ import annotations

import json
import re

from clip_video.llm.base import LLMConfig
from clip_video.llm.caller import LLMCaller

FUZZY_SYSTEM_PROMPT = """\
You suggest alternative spellings and phrasings for words that might appear \
differently in speech transcripts.

For each word, suggest 1-3 alternatives from these categories:
- Sound-alike (phonetic): words that sound similar, e.g. "gonna" -> "connor"
- Contractions/expansions: e.g. "gonna" -> "going to", "ain't" -> "is not"
- Casual/formal equivalents: e.g. "ain't" -> "isn't"

Return a JSON object mapping each input word to a list of alternative strings.
Do not include the original word in its own alternatives list.
If no good alternatives exist for a word, map it to an empty list.

Example input: ["gonna", "yeah"]
Example output: {"gonna": ["going to", "gunna", "gotta"], "yeah": ["yes", "yea"]}\
"""


def parse_fuzzy_response(response_text: str) -> dict[str, list[str]]:
    """Parse LLM response text into a word -> alternatives mapping.

    Handles JSON wrapped in markdown code blocks and extracts the JSON object
    from surrounding text. Filters out non-list values and non-string list items.
    """
    text = response_text.strip()

    # Strip markdown code block wrapper
    code_block = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    try:
        raw = json.loads(text[start : end + 1])
    except (json.JSONDecodeError, ValueError):
        return {}

    if not isinstance(raw, dict):
        return {}

    result: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            continue
        filtered = [item for item in value if isinstance(item, str)]
        result[key] = filtered

    return result


class FuzzyWordExpander:
    """Generates fuzzy search alternatives for missing lyric words via LLM."""

    def __init__(self, config: LLMConfig):
        self._caller = LLMCaller(config)

    def expand(self, missing_words: list[str]) -> dict[str, list[str]]:
        """Ask the LLM for phonetic/semantic alternatives for each missing word.

        Returns empty dict if the word list is empty or on any LLM error.
        """
        if not missing_words:
            return {}

        user_prompt = (
            "Suggest transcript-friendly alternatives for these words:\n"
            + json.dumps(missing_words)
        )

        try:
            response = self._caller.call(FUZZY_SYSTEM_PROMPT, user_prompt)
        except Exception:
            return {}

        return parse_fuzzy_response(response)
