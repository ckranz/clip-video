"""Phrase extraction for lyric matching.

Generates word/phrase extraction lists from parsed lyrics that drive
dictionary building and video searching.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from clip_video.lyrics.parser import ParsedLyrics, LyricsParser


@dataclass
class ExtractionTarget:
    """A word or phrase to extract from video library.

    Attributes:
        text: The word or phrase text (normalized/lowercase)
        source_line: Line number in the lyrics file
        source_text: Original line text for context
        is_phrase: True if this is a multi-word phrase
        priority: Higher priority means more important to find
        alternatives: Alternative spellings/forms to also search for
    """

    text: str
    source_line: int
    source_text: str
    is_phrase: bool = False
    priority: int = 1
    alternatives: list[str] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        """Number of words in this target."""
        return len(self.text.split())

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "source_line": self.source_line,
            "source_text": self.source_text,
            "is_phrase": self.is_phrase,
            "priority": self.priority,
            "alternatives": self.alternatives,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionTarget":
        """Create from dictionary."""
        return cls(
            text=data["text"],
            source_line=data["source_line"],
            source_text=data["source_text"],
            is_phrase=data.get("is_phrase", False),
            priority=data.get("priority", 1),
            alternatives=data.get("alternatives", []),
        )


@dataclass
class ExtractionList:
    """Complete list of extraction targets for a lyrics file.

    Organizes targets by their source line to maintain song structure.

    Attributes:
        lyrics_file: Path to source lyrics file
        title: Song title
        artist: Artist name
        targets_by_line: Dict mapping line number to list of targets
        all_targets: Flat list of all targets
    """

    lyrics_file: Path | None
    title: str | None
    artist: str | None
    targets_by_line: dict[int, list[ExtractionTarget]] = field(default_factory=dict)

    @property
    def all_targets(self) -> list[ExtractionTarget]:
        """Get all targets as a flat list."""
        targets = []
        for line_targets in self.targets_by_line.values():
            targets.extend(line_targets)
        return targets

    @property
    def unique_words(self) -> set[str]:
        """Get all unique single words to extract."""
        return {
            t.text for t in self.all_targets
            if not t.is_phrase
        }

    @property
    def unique_phrases(self) -> set[str]:
        """Get all unique phrases to extract."""
        return {
            t.text for t in self.all_targets
            if t.is_phrase
        }

    @property
    def lines_in_order(self) -> Iterator[tuple[int, list[ExtractionTarget]]]:
        """Iterate over lines and their targets in order."""
        for line_num in sorted(self.targets_by_line.keys()):
            yield line_num, self.targets_by_line[line_num]

    def get_line_targets(self, line_number: int) -> list[ExtractionTarget]:
        """Get all targets for a specific line.

        Args:
            line_number: Line number in the lyrics

        Returns:
            List of ExtractionTarget objects
        """
        return self.targets_by_line.get(line_number, [])

    def add_alternative(self, text: str, alternative: str) -> bool:
        """Add an alternative spelling for a target.

        Args:
            text: Original target text
            alternative: Alternative spelling to add

        Returns:
            True if alternative was added
        """
        found = False
        for targets in self.targets_by_line.values():
            for target in targets:
                if target.text == text:
                    if alternative not in target.alternatives:
                        target.alternatives.append(alternative)
                    found = True
        return found

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "lyrics_file": str(self.lyrics_file) if self.lyrics_file else None,
            "title": self.title,
            "artist": self.artist,
            "targets_by_line": {
                str(line): [t.to_dict() for t in targets]
                for line, targets in self.targets_by_line.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ExtractionList":
        """Create from dictionary."""
        targets_by_line = {}
        for line_str, targets_data in data.get("targets_by_line", {}).items():
            line = int(line_str)
            targets_by_line[line] = [
                ExtractionTarget.from_dict(t) for t in targets_data
            ]

        return cls(
            lyrics_file=Path(data["lyrics_file"]) if data.get("lyrics_file") else None,
            title=data.get("title"),
            artist=data.get("artist"),
            targets_by_line=targets_by_line,
        )

    def save(self, path: Path) -> None:
        """Save extraction list to JSON file.

        Args:
            path: Path to save the file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ExtractionList":
        """Load extraction list from JSON file.

        Args:
            path: Path to the file

        Returns:
            ExtractionList object

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


class PhraseExtractor:
    """Extracts words and phrases from lyrics for video searching.

    Configurable extraction strategy that can:
    - Extract individual words only
    - Extract phrases only
    - Extract both words and phrases
    - Deduplicate across the entire lyrics
    - Prioritize certain words/phrases

    Example usage:
        parser = LyricsParser()
        lyrics = parser.parse_file("song.txt")

        extractor = PhraseExtractor(extract_words=True, extract_phrases=True)
        extraction_list = extractor.extract(lyrics)

        for line_num, targets in extraction_list.lines_in_order:
            for target in targets:
                print(f"Line {line_num}: {target.text}")
    """

    # Common words to exclude from extraction (stop words)
    DEFAULT_STOP_WORDS = frozenset([
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
        "be", "have", "has", "had", "do", "does", "did", "will", "would",
        "could", "should", "may", "might", "must", "shall", "can",
        "this", "that", "these", "those", "i", "you", "he", "she", "it",
        "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
        "its", "our", "their", "what", "which", "who", "whom", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "also", "now", "here", "there",
        "if", "then", "else", "because", "while", "although", "though",
        "oh", "ah", "yeah", "hey", "uh", "um", "ooh", "la", "na", "da",
    ])

    def __init__(
        self,
        extract_words: bool = True,
        extract_phrases: bool = True,
        deduplicate: bool = True,
        use_stop_words: bool = False,  # Default False for lyric matching
        stop_words: set[str] | None = None,
        min_word_length: int = 1,  # Default 1 to capture all words
        min_phrase_words: int = 2,
        max_phrase_words: int = 5,  # 2-5 word phrases work best
    ):
        """Initialize the extractor.

        Args:
            extract_words: Whether to extract individual words
            extract_phrases: Whether to extract phrases
            deduplicate: Whether to deduplicate across the entire lyrics
            use_stop_words: Whether to filter out stop words
            stop_words: Custom stop words set (defaults to DEFAULT_STOP_WORDS)
            min_word_length: Minimum word length to extract
            min_phrase_words: Minimum words in a phrase to extract
            max_phrase_words: Maximum words in a phrase to extract
        """
        self.extract_words = extract_words
        self.extract_phrases = extract_phrases
        self.deduplicate = deduplicate
        self.use_stop_words = use_stop_words
        self.stop_words = stop_words or self.DEFAULT_STOP_WORDS
        self.min_word_length = min_word_length
        self.min_phrase_words = min_phrase_words
        self.max_phrase_words = max_phrase_words

    def extract(self, lyrics: ParsedLyrics) -> ExtractionList:
        """Extract words and phrases from parsed lyrics.

        Args:
            lyrics: ParsedLyrics object

        Returns:
            ExtractionList with all extraction targets
        """
        extraction_list = ExtractionList(
            lyrics_file=lyrics.source_file,
            title=lyrics.title,
            artist=lyrics.artist,
        )

        # Track seen targets for deduplication
        seen_targets: set[str] = set()

        for line in lyrics.content_lines:
            line_targets = []

            # Extract individual words
            if self.extract_words:
                for word in line.words:
                    if self._should_extract_word(word, seen_targets):
                        target = ExtractionTarget(
                            text=word,
                            source_line=line.line_number,
                            source_text=line.raw_text.strip(),
                            is_phrase=False,
                            priority=self._calculate_word_priority(word),
                        )
                        line_targets.append(target)

                        if self.deduplicate:
                            seen_targets.add(word)

            # Extract phrases
            if self.extract_phrases:
                for phrase in line.phrases:
                    if self._should_extract_phrase(phrase, seen_targets):
                        target = ExtractionTarget(
                            text=phrase,
                            source_line=line.line_number,
                            source_text=line.raw_text.strip(),
                            is_phrase=True,
                            priority=self._calculate_phrase_priority(phrase),
                        )
                        line_targets.append(target)

                        if self.deduplicate:
                            seen_targets.add(phrase)

            if line_targets:
                extraction_list.targets_by_line[line.line_number] = line_targets

        return extraction_list

    def _should_extract_word(self, word: str, seen: set[str]) -> bool:
        """Check if a word should be extracted.

        Args:
            word: Word to check
            seen: Set of already seen targets

        Returns:
            True if word should be extracted
        """
        # Check length
        if len(word) < self.min_word_length:
            return False

        # Check stop words
        if self.use_stop_words and word in self.stop_words:
            return False

        # Check deduplication
        if self.deduplicate and word in seen:
            return False

        return True

    def _should_extract_phrase(self, phrase: str, seen: set[str]) -> bool:
        """Check if a phrase should be extracted.

        Args:
            phrase: Phrase to check
            seen: Set of already seen targets

        Returns:
            True if phrase should be extracted
        """
        words = phrase.split()

        # Check word count bounds
        if len(words) < self.min_phrase_words:
            return False
        if len(words) > self.max_phrase_words:
            return False

        # Check if phrase is all stop words
        if self.use_stop_words:
            non_stop_words = [w for w in words if w not in self.stop_words]
            if not non_stop_words:
                return False

        # Check deduplication
        if self.deduplicate and phrase in seen:
            return False

        return True

    def _calculate_word_priority(self, word: str) -> int:
        """Calculate extraction priority for a word.

        Higher priority words are more important to find matches for.

        Args:
            word: Word to prioritize

        Returns:
            Priority value (higher = more important)
        """
        priority = 1

        # Longer words are generally more distinctive
        if len(word) >= 6:
            priority += 1
        if len(word) >= 8:
            priority += 1

        return priority

    def _calculate_phrase_priority(self, phrase: str) -> int:
        """Calculate extraction priority for a phrase.

        Args:
            phrase: Phrase to prioritize

        Returns:
            Priority value (higher = more important)
        """
        priority = 2  # Phrases generally higher priority than words

        words = phrase.split()

        # More words = more specific = higher priority
        if len(words) >= 3:
            priority += 1
        if len(words) >= 4:
            priority += 1

        return priority

    @classmethod
    def from_lyrics_file(
        cls,
        path: Path | str,
        **kwargs,
    ) -> ExtractionList:
        """Convenience method to parse and extract from a file.

        Args:
            path: Path to lyrics file
            **kwargs: Arguments to pass to PhraseExtractor

        Returns:
            ExtractionList with all extraction targets
        """
        parser = LyricsParser()
        lyrics = parser.parse_file(path)
        extractor = cls(**kwargs)
        return extractor.extract(lyrics)
