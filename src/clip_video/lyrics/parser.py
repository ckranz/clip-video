"""Lyrics file parser.

Parses lyrics text files into structured data with support for:
- Plain text lyrics (one line per row)
- Verse/chorus markers ([Verse 1], [Chorus], etc.)
- Manual phrase boundary markers ([phrase]words[/phrase])
- Repeated section markers (x2, x3)
- Common formatting variations
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator


@dataclass
class LyricsLine:
    """A single line from the lyrics.

    Attributes:
        line_number: Original line number in the file (1-indexed)
        raw_text: Original text before any processing
        text: Cleaned text with markers removed
        words: List of individual words
        phrases: List of phrases (from [phrase] markers or auto-generated)
        section: Section name if this line belongs to a labeled section
        is_section_header: True if this line is a section header (e.g., [Verse 1])
        repeat_count: Number of times this line repeats (from x2, x3 markers)
    """

    line_number: int
    raw_text: str
    text: str
    words: list[str] = field(default_factory=list)
    phrases: list[str] = field(default_factory=list)
    section: str | None = None
    is_section_header: bool = False
    repeat_count: int = 1

    def __post_init__(self):
        """Parse words if not provided."""
        if not self.words and self.text:
            self.words = self._tokenize(self.text)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of words (lowercase, punctuation stripped)
        """
        # Split on whitespace
        tokens = text.split()
        words = []
        for token in tokens:
            # Remove leading/trailing punctuation
            word = token.strip(".,!?;:\"'()[]{}/-")
            # Lowercase
            word = word.lower()
            if word:
                words.append(word)
        return words


@dataclass
class ParsedLyrics:
    """Complete parsed lyrics file.

    Attributes:
        source_file: Path to the original lyrics file
        title: Song title (if detected from filename or header)
        artist: Artist name (if detected)
        lines: List of parsed lines
        sections: Dict mapping section names to line indices
    """

    source_file: Path | None
    title: str | None
    artist: str | None
    lines: list[LyricsLine] = field(default_factory=list)
    sections: dict[str, list[int]] = field(default_factory=dict)

    @property
    def all_words(self) -> list[str]:
        """Get all unique words from the lyrics, respecting repeat counts."""
        words = []
        seen = set()
        for line in self.lines:
            if line.is_section_header:
                continue
            for word in line.words:
                if word not in seen:
                    words.append(word)
                    seen.add(word)
        return words

    @property
    def all_phrases(self) -> list[str]:
        """Get all unique phrases from the lyrics."""
        phrases = []
        seen = set()
        for line in self.lines:
            if line.is_section_header:
                continue
            for phrase in line.phrases:
                if phrase not in seen:
                    phrases.append(phrase)
                    seen.add(phrase)
        return phrases

    @property
    def content_lines(self) -> Iterator[LyricsLine]:
        """Iterate over content lines (excluding section headers)."""
        for line in self.lines:
            if not line.is_section_header:
                yield line

    def get_section_lines(self, section_name: str) -> list[LyricsLine]:
        """Get all lines belonging to a section.

        Args:
            section_name: Name of the section

        Returns:
            List of LyricsLine objects in the section
        """
        if section_name not in self.sections:
            return []
        return [self.lines[i] for i in self.sections[section_name]]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "source_file": str(self.source_file) if self.source_file else None,
            "title": self.title,
            "artist": self.artist,
            "lines": [
                {
                    "line_number": line.line_number,
                    "raw_text": line.raw_text,
                    "text": line.text,
                    "words": line.words,
                    "phrases": line.phrases,
                    "section": line.section,
                    "is_section_header": line.is_section_header,
                    "repeat_count": line.repeat_count,
                }
                for line in self.lines
            ],
            "sections": self.sections,
        }


class LyricsParser:
    """Parser for lyrics text files.

    Supports various lyrics formats including:
    - Plain text (one line per row)
    - Section markers: [Verse 1], [Chorus], [Bridge], etc.
    - Phrase markers: [phrase]word word[/phrase]
    - Repeat markers: (x2), (x3), x2, x3
    - Metadata: Title: ..., Artist: ...

    Example usage:
        parser = LyricsParser()
        lyrics = parser.parse_file("song.txt")
        for word in lyrics.all_words:
            print(word)
    """

    # Pattern for section headers like [Verse 1], [Chorus], [Bridge]
    SECTION_PATTERN = re.compile(
        r"^\s*\[(?P<section>(?:verse|chorus|bridge|intro|outro|hook|pre-chorus|"
        r"post-chorus|refrain|interlude|breakdown|solo|instrumental|spoken|"
        r"ad[- ]?lib|repeat|x\d+)[^\]]*)\]\s*$",
        re.IGNORECASE
    )

    # Pattern for phrase markers
    PHRASE_START_PATTERN = re.compile(r"\[phrase\]", re.IGNORECASE)
    PHRASE_END_PATTERN = re.compile(r"\[/phrase\]", re.IGNORECASE)

    # Pattern for repeat markers like (x2), x3, etc.
    REPEAT_PATTERN = re.compile(r"\(?x(\d+)\)?$", re.IGNORECASE)

    # Pattern for metadata lines
    METADATA_PATTERNS = {
        "title": re.compile(r"^(?:title|song)\s*[:=]\s*(.+)$", re.IGNORECASE),
        "artist": re.compile(r"^(?:artist|by|singer)\s*[:=]\s*(.+)$", re.IGNORECASE),
    }

    def __init__(
        self,
        auto_phrase_lines: bool = True,
        min_phrase_words: int = 2,
    ):
        """Initialize the parser.

        Args:
            auto_phrase_lines: If True, treat each line as a phrase even without markers
            min_phrase_words: Minimum words for auto-generated phrases
        """
        self.auto_phrase_lines = auto_phrase_lines
        self.min_phrase_words = min_phrase_words

    def parse_file(self, path: Path | str) -> ParsedLyrics:
        """Parse a lyrics file.

        Args:
            path: Path to the lyrics file

        Returns:
            ParsedLyrics object

        Raises:
            FileNotFoundError: If file doesn't exist
            UnicodeDecodeError: If file encoding is not UTF-8
        """
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        lyrics = self.parse_text(text)
        lyrics.source_file = path

        # Try to extract title from filename if not in metadata
        if not lyrics.title:
            # Remove extension and common prefixes
            name = path.stem
            # Remove common patterns like "01 - " or "Artist - "
            name = re.sub(r"^\d+\s*[-_.]\s*", "", name)
            if " - " in name:
                parts = name.split(" - ", 1)
                if not lyrics.artist:
                    lyrics.artist = parts[0].strip()
                lyrics.title = parts[1].strip()
            else:
                lyrics.title = name

        return lyrics

    def parse_text(self, text: str) -> ParsedLyrics:
        """Parse lyrics from text.

        Args:
            text: Lyrics text content

        Returns:
            ParsedLyrics object
        """
        lyrics = ParsedLyrics(
            source_file=None,
            title=None,
            artist=None,
        )

        current_section = None
        raw_lines = text.splitlines()

        for line_num, raw_line in enumerate(raw_lines, start=1):
            stripped = raw_line.strip()

            # Skip empty lines
            if not stripped:
                continue

            # Check for metadata
            metadata_found = False
            for key, pattern in self.METADATA_PATTERNS.items():
                match = pattern.match(stripped)
                if match:
                    setattr(lyrics, key, match.group(1).strip())
                    metadata_found = True
                    break

            if metadata_found:
                continue

            # Check for section header
            section_match = self.SECTION_PATTERN.match(stripped)
            if section_match:
                current_section = section_match.group("section").strip()
                line = LyricsLine(
                    line_number=line_num,
                    raw_text=raw_line,
                    text="",
                    is_section_header=True,
                    section=current_section,
                )
                lyrics.lines.append(line)

                # Initialize section in sections dict
                if current_section not in lyrics.sections:
                    lyrics.sections[current_section] = []

                continue

            # Parse content line
            line = self._parse_line(line_num, raw_line, current_section)

            # Track section membership
            if current_section and current_section in lyrics.sections:
                lyrics.sections[current_section].append(len(lyrics.lines))

            lyrics.lines.append(line)

        return lyrics

    def _parse_line(
        self,
        line_number: int,
        raw_text: str,
        section: str | None,
    ) -> LyricsLine:
        """Parse a single content line.

        Args:
            line_number: Line number in the file
            raw_text: Raw line text
            section: Current section name

        Returns:
            LyricsLine object
        """
        text = raw_text.strip()
        phrases = []

        # Check for repeat marker at end of line
        repeat_count = 1
        repeat_match = self.REPEAT_PATTERN.search(text)
        if repeat_match:
            repeat_count = int(repeat_match.group(1))
            text = text[:repeat_match.start()].strip()

        # Extract manual phrase markers
        phrase_positions = []

        # Find all phrase markers
        starts = list(self.PHRASE_START_PATTERN.finditer(text))
        ends = list(self.PHRASE_END_PATTERN.finditer(text))

        if starts and ends:
            # Match starts with ends
            for start in starts:
                # Find next end after this start
                for end in ends:
                    if end.start() > start.end():
                        phrase_text = text[start.end():end.start()].strip()
                        if phrase_text:
                            phrases.append(phrase_text.lower())
                        phrase_positions.append((start.start(), end.end()))
                        break

        # Remove phrase markers from text
        clean_text = text
        for start, end in sorted(phrase_positions, reverse=True):
            clean_text = clean_text[:start] + clean_text[end:]

        # Remove any remaining markers
        clean_text = self.PHRASE_START_PATTERN.sub("", clean_text)
        clean_text = self.PHRASE_END_PATTERN.sub("", clean_text)
        clean_text = clean_text.strip()

        # Auto-generate phrase from line if enabled
        if self.auto_phrase_lines:
            words = LyricsLine._tokenize(clean_text)
            if len(words) >= self.min_phrase_words:
                line_phrase = " ".join(words)
                if line_phrase not in phrases:
                    phrases.append(line_phrase)

        return LyricsLine(
            line_number=line_number,
            raw_text=raw_text,
            text=clean_text,
            section=section,
            phrases=phrases,
            repeat_count=repeat_count,
        )
