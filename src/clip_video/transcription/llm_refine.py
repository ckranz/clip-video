"""LLM-based transcript refinement.

Uses an LLM to fix domain-specific terms, acronyms, proper nouns, and grammar
in Whisper transcription output without rephrasing or restructuring.
"""

from __future__ import annotations

from dataclasses import dataclass

from clip_video.transcription.base import TranscriptionSegment


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
