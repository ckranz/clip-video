"""Transcript correction using vocabulary matching.

Applies vocabulary corrections to transcripts, supporting both exact
matching and fuzzy phonetic matching.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from clip_video.vocabulary.phonetic import (
    levenshtein_distance,
    phrase_phonetic_similarity,
    phonetic_similarity,
)
from clip_video.vocabulary.terms import VocabularyTerms


@dataclass
class Correction:
    """Represents a single correction made to a transcript."""

    original: str
    corrected: str
    match_type: str  # "exact", "fuzzy", "phonetic"
    confidence: float  # 0.0 to 1.0
    position: int | None = None  # Character position in text
    context: str = ""  # Surrounding text for reference

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original": self.original,
            "corrected": self.corrected,
            "match_type": self.match_type,
            "confidence": self.confidence,
            "position": self.position,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Correction":
        """Create from dictionary."""
        return cls(
            original=data["original"],
            corrected=data["corrected"],
            match_type=data["match_type"],
            confidence=data["confidence"],
            position=data.get("position"),
            context=data.get("context", ""),
        )


@dataclass
class CorrectionLog:
    """Log of all corrections made during a correction pass."""

    corrections: list[Correction] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    source_file: str = ""
    vocabulary_terms: int = 0

    def add(self, correction: Correction) -> None:
        """Add a correction to the log."""
        self.corrections.append(correction)

    def __len__(self) -> int:
        """Return number of corrections."""
        return len(self.corrections)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "source_file": self.source_file,
            "vocabulary_terms": self.vocabulary_terms,
            "correction_count": len(self.corrections),
            "corrections": [c.to_dict() for c in self.corrections],
        }

    def save(self, path: Path | str) -> None:
        """Save log to JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)

    def append_to_log_file(self, path: Path | str) -> None:
        """Append corrections to a log file.

        Each correction is written on a separate line for easy review.

        Args:
            path: Path to log file
        """
        path = Path(path)

        with open(path, "a", encoding="utf-8") as f:
            f.write(f"\n--- Corrections at {self.timestamp.isoformat()} ---\n")
            f.write(f"Source: {self.source_file}\n")
            f.write(f"Terms: {self.vocabulary_terms}\n\n")

            for c in self.corrections:
                f.write(f"  [{c.match_type}:{c.confidence:.2f}] ")
                f.write(f'"{c.original}" -> "{c.corrected}"\n')
                if c.context:
                    f.write(f"    Context: ...{c.context}...\n")

            f.write(f"\nTotal: {len(self.corrections)} corrections\n")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CorrectionLog":
        """Create from dictionary."""
        log = cls(
            timestamp=datetime.fromisoformat(data["timestamp"]),
            source_file=data.get("source_file", ""),
            vocabulary_terms=data.get("vocabulary_terms", 0),
        )
        for c_data in data.get("corrections", []):
            log.corrections.append(Correction.from_dict(c_data))
        return log

    @classmethod
    def load(cls, path: Path | str) -> "CorrectionLog":
        """Load log from JSON file.

        Args:
            path: Path to load from

        Returns:
            CorrectionLog instance
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class CorrectionBlocklist:
    """Manages corrections that should not be applied.

    Allows users to mark corrections as incorrect so they won't
    be applied in future correction passes.
    """

    def __init__(self, blocklist: set[tuple[str, str]] | None = None):
        """Initialize blocklist.

        Args:
            blocklist: Set of (original, corrected) tuples to block
        """
        self._blocklist: set[tuple[str, str]] = blocklist or set()

    def add(self, original: str, corrected: str) -> None:
        """Add a correction to the blocklist.

        Args:
            original: The original text
            corrected: The correction that should be blocked
        """
        self._blocklist.add((original.lower(), corrected.lower()))

    def remove(self, original: str, corrected: str) -> bool:
        """Remove a correction from the blocklist.

        Args:
            original: The original text
            corrected: The correction to unblock

        Returns:
            True if found and removed
        """
        key = (original.lower(), corrected.lower())
        if key in self._blocklist:
            self._blocklist.remove(key)
            return True
        return False

    def is_blocked(self, original: str, corrected: str) -> bool:
        """Check if a correction is blocked.

        Args:
            original: The original text
            corrected: The proposed correction

        Returns:
            True if this correction is blocked
        """
        return (original.lower(), corrected.lower()) in self._blocklist

    def to_list(self) -> list[dict[str, str]]:
        """Export blocklist as list of dicts."""
        return [{"original": o, "corrected": c} for o, c in self._blocklist]

    def save(self, path: Path | str) -> None:
        """Save blocklist to JSON file."""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, indent=2)

    @classmethod
    def load(cls, path: Path | str) -> "CorrectionBlocklist":
        """Load blocklist from JSON file."""
        path = Path(path)
        if not path.exists():
            return cls()

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        blocklist = set()
        for item in data:
            blocklist.add((item["original"].lower(), item["corrected"].lower()))

        return cls(blocklist)

    def __len__(self) -> int:
        return len(self._blocklist)

    def __contains__(self, item: tuple[str, str]) -> bool:
        original, corrected = item
        return self.is_blocked(original, corrected)


class TranscriptCorrector:
    """Applies vocabulary corrections to transcripts.

    Supports exact matching against known variants and fuzzy matching
    using phonetic algorithms.
    """

    # Minimum confidence for fuzzy matches to be applied
    # Set high to avoid false positives (e.g., "container" -> "containerd")
    DEFAULT_FUZZY_THRESHOLD = 0.95

    # Minimum confidence for phonetic matches
    # Set high to avoid false positives from similar-sounding words
    DEFAULT_PHONETIC_THRESHOLD = 0.92

    def __init__(
        self,
        vocabulary: VocabularyTerms,
        blocklist: CorrectionBlocklist | None = None,
        fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
        phonetic_threshold: float = DEFAULT_PHONETIC_THRESHOLD,
    ):
        """Initialize corrector.

        Args:
            vocabulary: Vocabulary terms to use
            blocklist: Corrections to skip
            fuzzy_threshold: Minimum confidence for fuzzy matches
            phonetic_threshold: Minimum confidence for phonetic matches
        """
        self.vocabulary = vocabulary
        self.blocklist = blocklist or CorrectionBlocklist()
        self.fuzzy_threshold = fuzzy_threshold
        self.phonetic_threshold = phonetic_threshold

    def correct_text(
        self,
        text: str,
        source_file: str = "",
        enable_fuzzy: bool = True,
        enable_phonetic: bool = True,
    ) -> tuple[str, CorrectionLog]:
        """Apply corrections to text.

        Args:
            text: Text to correct
            source_file: Source file name for logging
            enable_fuzzy: Enable fuzzy Levenshtein matching
            enable_phonetic: Enable phonetic matching

        Returns:
            Tuple of (corrected_text, correction_log)
        """
        log = CorrectionLog(
            source_file=source_file,
            vocabulary_terms=len(self.vocabulary),
        )

        # First pass: exact multi-word phrase matching
        corrected = text
        for canonical in self.vocabulary.get_all_terms():
            for variant in self.vocabulary.get_variants(canonical):
                if self.blocklist.is_blocked(variant, canonical):
                    continue

                # Case-insensitive replacement preserving boundaries
                pattern = re.compile(
                    r"\b" + re.escape(variant) + r"\b",
                    re.IGNORECASE,
                )

                matches = list(pattern.finditer(corrected))
                for match in reversed(matches):  # Reverse to preserve positions
                    original = match.group(0)
                    start, end = match.start(), match.end()

                    # Get context
                    ctx_start = max(0, start - 20)
                    ctx_end = min(len(corrected), end + 20)
                    context = corrected[ctx_start:ctx_end]

                    # Preserve original case pattern if possible
                    replacement = self._match_case(original, canonical)

                    log.add(Correction(
                        original=original,
                        corrected=replacement,
                        match_type="exact",
                        confidence=1.0,
                        position=start,
                        context=context,
                    ))

                    corrected = corrected[:start] + replacement + corrected[end:]

        # Second pass: word-by-word fuzzy/phonetic matching
        if enable_fuzzy or enable_phonetic:
            words = re.findall(r"\b\w+\b", corrected)
            unique_words = set(w.lower() for w in words)

            # Check each unique word
            for word in unique_words:
                # Skip if already a canonical term
                if self.vocabulary.is_canonical(word):
                    continue

                # Skip if already a known variant (handled in exact pass)
                if self.vocabulary.is_variant(word):
                    continue

                # Find best match
                best_match = self._find_best_fuzzy_match(
                    word,
                    enable_fuzzy=enable_fuzzy,
                    enable_phonetic=enable_phonetic,
                )

                if best_match:
                    canonical, confidence, match_type = best_match

                    if self.blocklist.is_blocked(word, canonical):
                        continue

                    # Apply correction
                    pattern = re.compile(
                        r"\b" + re.escape(word) + r"\b",
                        re.IGNORECASE,
                    )

                    matches = list(pattern.finditer(corrected))
                    for match in reversed(matches):
                        original = match.group(0)
                        start, end = match.start(), match.end()

                        ctx_start = max(0, start - 20)
                        ctx_end = min(len(corrected), end + 20)
                        context = corrected[ctx_start:ctx_end]

                        replacement = self._match_case(original, canonical)

                        log.add(Correction(
                            original=original,
                            corrected=replacement,
                            match_type=match_type,
                            confidence=confidence,
                            position=start,
                            context=context,
                        ))

                        corrected = corrected[:start] + replacement + corrected[end:]

        return corrected, log

    def correct_words(
        self,
        words: list[dict[str, Any]],
        source_file: str = "",
        enable_fuzzy: bool = True,
        enable_phonetic: bool = True,
    ) -> tuple[list[dict[str, Any]], CorrectionLog]:
        """Apply corrections to a list of word objects.

        Useful for correcting transcript word-level data that includes
        timestamps and other metadata.

        Args:
            words: List of word dicts with "word" key
            source_file: Source file name for logging
            enable_fuzzy: Enable fuzzy matching
            enable_phonetic: Enable phonetic matching

        Returns:
            Tuple of (corrected_words, correction_log)
        """
        log = CorrectionLog(
            source_file=source_file,
            vocabulary_terms=len(self.vocabulary),
        )

        corrected_words = []

        for word_data in words:
            word_data = word_data.copy()  # Don't modify original
            original_word = word_data.get("word", "")

            if not original_word:
                corrected_words.append(word_data)
                continue

            # Try exact match first
            canonical = self.vocabulary.get_canonical(original_word)

            if canonical:
                if not self.blocklist.is_blocked(original_word, canonical):
                    replacement = self._match_case(original_word, canonical)
                    word_data["word"] = replacement
                    word_data["original_word"] = original_word

                    log.add(Correction(
                        original=original_word,
                        corrected=replacement,
                        match_type="exact",
                        confidence=1.0,
                    ))
            elif enable_fuzzy or enable_phonetic:
                # Try fuzzy/phonetic match
                if not self.vocabulary.is_canonical(original_word.lower()):
                    best_match = self._find_best_fuzzy_match(
                        original_word,
                        enable_fuzzy=enable_fuzzy,
                        enable_phonetic=enable_phonetic,
                    )

                    if best_match:
                        canonical, confidence, match_type = best_match

                        if not self.blocklist.is_blocked(original_word, canonical):
                            replacement = self._match_case(original_word, canonical)
                            word_data["word"] = replacement
                            word_data["original_word"] = original_word

                            log.add(Correction(
                                original=original_word,
                                corrected=replacement,
                                match_type=match_type,
                                confidence=confidence,
                            ))

            corrected_words.append(word_data)

        return corrected_words, log

    def _find_best_fuzzy_match(
        self,
        word: str,
        enable_fuzzy: bool = True,
        enable_phonetic: bool = True,
    ) -> tuple[str, float, str] | None:
        """Find the best fuzzy match for a word.

        Args:
            word: Word to match
            enable_fuzzy: Enable Levenshtein matching
            enable_phonetic: Enable phonetic matching

        Returns:
            Tuple of (canonical, confidence, match_type) or None
        """
        word_lower = word.lower()
        best_match = None
        best_confidence = 0.0
        best_type = ""

        for canonical in self.vocabulary.get_all_terms():
            # Check against canonical term
            if enable_fuzzy:
                distance = levenshtein_distance(word_lower, canonical)
                max_len = max(len(word_lower), len(canonical))
                confidence = 1.0 - (distance / max_len) if max_len > 0 else 0.0

                if confidence >= self.fuzzy_threshold and confidence > best_confidence:
                    best_match = canonical
                    best_confidence = confidence
                    best_type = "fuzzy"

            if enable_phonetic:
                confidence = phonetic_similarity(word, canonical)

                if confidence >= self.phonetic_threshold and confidence > best_confidence:
                    best_match = canonical
                    best_confidence = confidence
                    best_type = "phonetic"

            # Check against variants
            for variant in self.vocabulary.get_variants(canonical):
                # Skip multi-word variants for single-word matching
                if " " in variant:
                    continue

                if enable_fuzzy:
                    distance = levenshtein_distance(word_lower, variant)
                    max_len = max(len(word_lower), len(variant))
                    confidence = 1.0 - (distance / max_len) if max_len > 0 else 0.0

                    if confidence >= self.fuzzy_threshold and confidence > best_confidence:
                        best_match = canonical
                        best_confidence = confidence
                        best_type = "fuzzy"

                if enable_phonetic:
                    confidence = phonetic_similarity(word, variant)

                    if confidence >= self.phonetic_threshold and confidence > best_confidence:
                        best_match = canonical
                        best_confidence = confidence
                        best_type = "phonetic"

        if best_match:
            return (best_match, best_confidence, best_type)
        return None

    def _match_case(self, original: str, replacement: str) -> str:
        """Match the case pattern of the original to the replacement.

        Args:
            original: Original word with case to match
            replacement: Replacement word to adjust

        Returns:
            Replacement with matching case pattern
        """
        if not original or not replacement:
            return replacement

        if original.isupper():
            return replacement.upper()
        elif original.istitle():
            return replacement.capitalize()
        elif original.islower():
            return replacement.lower()
        else:
            # Mixed case - return as-is or lowercase
            return replacement.lower()

    def generate_whisper_prompt(self) -> str:
        """Generate Whisper prompt conditioning string.

        Returns:
            Prompt string for Whisper API
        """
        return self.vocabulary.generate_whisper_prompt()
