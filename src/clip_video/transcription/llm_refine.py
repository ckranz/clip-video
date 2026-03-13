"""LLM-based transcript refinement.

Uses an LLM to fix domain-specific terms, acronyms, proper nouns, and grammar
in Whisper transcription output without rephrasing or restructuring.
"""

from __future__ import annotations

import copy
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass

from clip_video.llm.base import LLMConfig, LLMProviderType
from clip_video.llm.caller import LLMCaller
from clip_video.transcription.base import TranscriptionSegment, TranscriptionWord
from clip_video.vocabulary.correction import Correction, CorrectionLog

DEFAULT_CHUNK_SIZE = 15


@dataclass
class RefinementContext:
    """Optional context to guide LLM refinement of transcription.

    Attributes:
        talk_title: Title of the talk or session.
        talk_description: Description or abstract of the talk.
        speaker_name: Name of the speaker.
        domain: Subject domain (e.g., "kubernetes", "technology").
        vocabulary_terms: Known correct terms from brand vocabulary.
    """

    talk_title: str | None = None
    talk_description: str | None = None
    speaker_name: str | None = None
    domain: str = "technology"
    vocabulary_terms: list[str] | None = None


REFINEMENT_SYSTEM_PROMPT = """\
You are a transcription correction assistant. Your job is to fix errors in \
automated speech-to-text output.

Fix ONLY:
- Domain-specific terms and acronyms (e.g., "cube control" -> "kubectl")
- Proper nouns (people, products, companies)
- Grammar errors introduced by the transcriber
- Obvious mishearings

Do NOT:
- Rephrase or restructure sentences
- Add words that weren't spoken
- Remove filler words (um, uh, like)
- Change casual speech to formal speech

Return ONLY a JSON array of corrections. Each correction must have these keys:
- "original": the exact text to replace (as it appears in the transcript)
- "corrected": the corrected text
- "reason": brief explanation of why this is a correction

If no corrections are needed, return an empty array: []

Examples:
[
  {"original": "cooper netties", "corrected": "Kubernetes", "reason": "Misheard product name"},
  {"original": "go roo teens", "corrected": "goroutines", "reason": "Go concurrency term"},
  {"original": "Jim Starlings", "corrected": "Jim Starling", "reason": "Speaker name correction"}
]\
"""


def build_refinement_prompt(
    segments: list[TranscriptionSegment],
    context: RefinementContext | None = None,
) -> str:
    """Build a user prompt for LLM transcript refinement.

    Args:
        segments: Transcription segments with timestamps and text.
        context: Optional context about the talk to guide corrections.

    Returns:
        A formatted prompt string for the LLM.
    """
    parts: list[str] = []

    if context is not None:
        context_lines: list[str] = []
        if context.talk_title:
            context_lines.append(f"Title: {context.talk_title}")
        if context.talk_description:
            context_lines.append(f"Description: {context.talk_description}")
        if context.speaker_name:
            context_lines.append(f"Speaker: {context.speaker_name}")
        context_lines.append(f"Domain: {context.domain}")
        if context.vocabulary_terms:
            context_lines.append(
                f"Known vocabulary: {', '.join(context.vocabulary_terms)}"
            )
        parts.append("Context:\n" + "\n".join(context_lines))

    parts.append("Transcript:")
    for segment in segments:
        parts.append(f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text}")

    parts.append(
        "Identify any transcription errors and return corrections as a JSON array."
    )

    return "\n\n".join(parts)


def parse_llm_corrections(response_text: str) -> list[dict[str, str]]:
    """Parse raw LLM response text into a list of correction dicts.

    Handles JSON wrapped in markdown code blocks or raw JSON.
    Finds the first ``[`` to the last ``]`` to extract the array.

    Args:
        response_text: Raw text from the LLM.

    Returns:
        List of correction dicts with "original", "corrected", and "reason" keys.
        Returns empty list on parse failure (never raises).
    """
    text = response_text.strip()

    # Strip markdown code block if present
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if code_block_match:
        text = code_block_match.group(1).strip()

    # Find the JSON array boundaries
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []

    json_text = text[start : end + 1]

    try:
        parsed = json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return []

    if not isinstance(parsed, list):
        return []

    required_keys = {"original", "corrected", "reason"}
    results: list[dict[str, str]] = []
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        if not required_keys.issubset(entry.keys()):
            continue
        # Skip no-op corrections
        if entry["original"] == entry["corrected"]:
            continue
        results.append(entry)

    return results


def _update_segment_words(
    words: list[TranscriptionWord],
    original: str,
    corrected: str,
) -> list[TranscriptionWord]:
    """Update word-level data by replacing sequences matching the original phrase.

    Finds sequences of adjacent words whose concatenation matches the original
    phrase (case-insensitive) and replaces them with the corrected text,
    distributing timestamps evenly.

    Args:
        words: List of TranscriptionWord objects (will not be mutated).
        original: The original phrase to find.
        corrected: The corrected replacement text.

    Returns:
        New list of TranscriptionWord objects with replacements applied.
    """
    orig_tokens = original.lower().split()
    corr_tokens = corrected.split()
    n = len(orig_tokens)

    if not orig_tokens or not words:
        return list(words)

    result: list[TranscriptionWord] = []
    i = 0
    while i < len(words):
        # Check if words[i:i+n] matches the original tokens
        if i + n <= len(words):
            candidate = [w.word.lower() for w in words[i : i + n]]
            if candidate == orig_tokens:
                # Match found - merge timestamps
                span_start = words[i].start
                span_end = words[i + n - 1].end
                span_duration = span_end - span_start
                num_new = len(corr_tokens)

                for j, token in enumerate(corr_tokens):
                    t_start = span_start + (span_duration * j / num_new) if num_new > 0 else span_start
                    t_end = span_start + (span_duration * (j + 1) / num_new) if num_new > 0 else span_end
                    new_word = TranscriptionWord(
                        word=token,
                        start=t_start,
                        end=t_end,
                        confidence=words[i].confidence,
                        original_word=" ".join(w.word for w in words[i:i + n]) if j == 0 else None,
                    )
                    result.append(new_word)

                i += n
                continue

        result.append(words[i])
        i += 1

    return result


def apply_corrections(
    segments: list[TranscriptionSegment],
    corrections: list[dict[str, str]],
    source_file: str = "",
) -> tuple[list[TranscriptionSegment], CorrectionLog]:
    """Apply LLM corrections to transcription segments.

    Creates deep copies of segments before modifying -- never mutates originals.

    Args:
        segments: Transcription segments to correct.
        corrections: Parsed correction dicts from ``parse_llm_corrections``.
        source_file: Source file name for logging.

    Returns:
        Tuple of (corrected_segments, correction_log).
    """
    log = CorrectionLog(source_file=source_file)
    corrected_segments = copy.deepcopy(segments)

    for correction in corrections:
        original = correction["original"]
        corrected = correction["corrected"]
        reason = correction.get("reason", "")

        pattern = re.compile(r"\b" + re.escape(original) + r"\b", re.IGNORECASE)

        for seg in corrected_segments:
            matches = list(pattern.finditer(seg.text))
            if not matches:
                continue

            # Replace in segment text
            seg.text = pattern.sub(corrected, seg.text)

            # Update word-level data if present
            if seg.words:
                seg.words = _update_segment_words(seg.words, original, corrected)

            # Log each segment match
            for match in matches:
                log.add(Correction(
                    original=match.group(0),
                    corrected=corrected,
                    match_type="llm",
                    confidence=1.0,
                    position=match.start(),
                    context=reason,
                ))

    return corrected_segments, log


class TranscriptRefiner:
    """Refines transcription segments using an LLM to fix domain-specific errors.

    Processes segments in chunks, sends each chunk to the configured LLM provider,
    and applies corrections. On any LLM failure, returns original segments unchanged
    (never raises from refine()).

    Args:
        config: LLM provider configuration.
        chunk_size: Number of segments per LLM call.
    """

    def __init__(self, config: LLMConfig, chunk_size: int = DEFAULT_CHUNK_SIZE):
        self.config = config
        self.chunk_size = chunk_size
        self._caller = LLMCaller(config)

    def is_available(self) -> bool:
        """Check if the configured LLM provider is available.

        Returns:
            True if the provider can be used.
        """
        if self.config.provider == LLMProviderType.CLAUDE:
            return bool(self.config.api_key or os.environ.get("ANTHROPIC_API_KEY"))
        elif self.config.provider == LLMProviderType.OPENAI:
            return bool(self.config.api_key or os.environ.get("OPENAI_API_KEY"))
        elif self.config.provider == LLMProviderType.OLLAMA:
            try:
                req = urllib.request.Request(
                    "http://localhost:11434/api/tags",
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=5):
                    return True
            except Exception:
                return False
        return False

    def refine(
        self,
        segments: list[TranscriptionSegment],
        context: RefinementContext | None = None,
    ) -> tuple[list[TranscriptionSegment], CorrectionLog]:
        """Refine transcription segments using LLM corrections.

        Processes segments in chunks, collects corrections from the LLM,
        and applies them. On LLM failure, returns originals unchanged.

        Args:
            segments: Transcription segments to refine.
            context: Optional context to guide corrections.

        Returns:
            Tuple of (corrected_segments, correction_log).
        """
        all_corrections: list[dict[str, str]] = []

        for i in range(0, len(segments), self.chunk_size):
            chunk = segments[i : i + self.chunk_size]
            user_prompt = build_refinement_prompt(chunk, context=context)

            try:
                response = self._call_llm(REFINEMENT_SYSTEM_PROMPT, user_prompt)
                corrections = parse_llm_corrections(response)
                all_corrections.extend(corrections)
            except Exception:
                # On failure: if we have no corrections yet, return deep copy
                # of originals. If partial corrections exist, apply what we have.
                if not all_corrections:
                    return copy.deepcopy(segments), CorrectionLog()
                break

        if not all_corrections:
            return copy.deepcopy(segments), CorrectionLog()

        return apply_corrections(segments, all_corrections)

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Dispatch LLM call to the configured provider via LLMCaller.

        Kept as a thin wrapper so existing tests can mock this method.

        Args:
            system_prompt: System prompt for the LLM.
            user_prompt: User prompt with transcript data.

        Returns:
            Raw response text from the LLM.

        Raises:
            Exception: On any API or connection error.
        """
        return self._caller.call(system_prompt, user_prompt)
