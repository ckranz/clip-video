"""Inverted index for fast word and phrase lookup across transcripts.

Provides efficient searching for words and phrases across all transcripts
in a brand, returning occurrences with timestamps and source video information.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from clip_video.models.transcript import Transcript, TranscriptWord
from clip_video.storage import atomic_write_json, read_json, NotFoundError


@dataclass
class WordOccurrence:
    """A single occurrence of a word in a transcript.

    Contains the word, its timing information, and source location.
    """

    word: str
    start: float
    end: float
    confidence: float
    project_name: str
    video_id: str
    segment_index: int
    word_index: int  # Index within segment

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "word": self.word,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "project_name": self.project_name,
            "video_id": self.video_id,
            "segment_index": self.segment_index,
            "word_index": self.word_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WordOccurrence":
        """Create from dictionary."""
        return cls(
            word=data["word"],
            start=data["start"],
            end=data["end"],
            confidence=data["confidence"],
            project_name=data["project_name"],
            video_id=data["video_id"],
            segment_index=data["segment_index"],
            word_index=data["word_index"],
        )


@dataclass
class PhraseMatch:
    """A match for a multi-word phrase in a transcript.

    Contains the full phrase, timing spanning all words, and source location.
    """

    phrase: str
    words: list[WordOccurrence]
    start: float  # Start of first word
    end: float  # End of last word
    project_name: str
    video_id: str

    @property
    def duration(self) -> float:
        """Get the duration of the phrase in seconds."""
        return self.end - self.start

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phrase": self.phrase,
            "words": [w.to_dict() for w in self.words],
            "start": self.start,
            "end": self.end,
            "project_name": self.project_name,
            "video_id": self.video_id,
        }


@dataclass
class TranscriptIndex:
    """Inverted index for fast word/phrase lookup across transcripts.

    The index maps normalized words to their occurrences across all transcripts
    in a brand. This enables fast searching without scanning all transcript files.

    Index structure:
    - words: dict mapping normalized word -> list of WordOccurrence
    - metadata: dict with index creation info and statistics
    """

    brand_name: str
    words: dict[str, list[WordOccurrence]] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    # Internal tracking of indexed transcripts
    _indexed_transcripts: set[tuple[str, str]] = field(default_factory=set)

    def __post_init__(self):
        """Initialize internal data structures."""
        if not hasattr(self, "_indexed_transcripts") or self._indexed_transcripts is None:
            self._indexed_transcripts = set()

    @staticmethod
    def normalize_word(word: str) -> str:
        """Normalize a word for indexing.

        Converts to lowercase and removes punctuation for consistent matching.

        Args:
            word: Word to normalize

        Returns:
            Normalized word
        """
        # Convert to lowercase
        word = word.lower()
        # Remove leading/trailing punctuation
        word = word.strip(".,!?;:\"'()[]{}/-")
        # Normalize unicode
        word = unicodedata.normalize("NFKC", word)
        return word

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of normalized words
        """
        # Split on whitespace and punctuation
        tokens = re.split(r"[\s.,!?;:\"'()\[\]{}/\-]+", text)
        # Normalize and filter empty
        return [
            TranscriptIndex.normalize_word(t)
            for t in tokens
            if t and TranscriptIndex.normalize_word(t)
        ]

    def add_transcript(
        self,
        project_name: str,
        video_id: str,
        transcript: Transcript,
    ) -> int:
        """Add a transcript to the index.

        Indexes all words from the transcript for fast lookup.

        Args:
            project_name: Project name
            video_id: Video identifier
            transcript: Transcript to index

        Returns:
            Number of words indexed
        """
        word_count = 0

        for seg_idx, segment in enumerate(transcript.segments):
            for word_idx, word in enumerate(segment.words):
                normalized = self.normalize_word(word.word)
                if not normalized:
                    continue

                occurrence = WordOccurrence(
                    word=word.word,
                    start=word.start,
                    end=word.end,
                    confidence=word.confidence,
                    project_name=project_name,
                    video_id=video_id,
                    segment_index=seg_idx,
                    word_index=word_idx,
                )

                if normalized not in self.words:
                    self.words[normalized] = []
                self.words[normalized].append(occurrence)
                word_count += 1

        self._indexed_transcripts.add((project_name, video_id))
        return word_count

    def remove_transcript(self, project_name: str, video_id: str) -> int:
        """Remove a transcript from the index.

        Removes all word occurrences from the specified transcript.

        Args:
            project_name: Project name
            video_id: Video identifier

        Returns:
            Number of words removed
        """
        removed_count = 0

        # Filter out occurrences from this transcript
        for normalized_word in list(self.words.keys()):
            original_len = len(self.words[normalized_word])
            self.words[normalized_word] = [
                occ
                for occ in self.words[normalized_word]
                if not (occ.project_name == project_name and occ.video_id == video_id)
            ]
            removed_count += original_len - len(self.words[normalized_word])

            # Remove empty entries
            if not self.words[normalized_word]:
                del self.words[normalized_word]

        self._indexed_transcripts.discard((project_name, video_id))
        return removed_count

    def is_indexed(self, project_name: str, video_id: str) -> bool:
        """Check if a transcript is already indexed.

        Args:
            project_name: Project name
            video_id: Video identifier

        Returns:
            True if transcript is in the index
        """
        return (project_name, video_id) in self._indexed_transcripts

    def search_word(
        self,
        word: str,
        project_name: str | None = None,
        video_id: str | None = None,
    ) -> list[WordOccurrence]:
        """Search for occurrences of a word.

        Args:
            word: Word to search for
            project_name: Optional filter by project
            video_id: Optional filter by video

        Returns:
            List of WordOccurrence objects sorted by timestamp
        """
        normalized = self.normalize_word(word)
        if normalized not in self.words:
            return []

        occurrences = self.words[normalized]

        # Apply filters
        if project_name is not None:
            occurrences = [o for o in occurrences if o.project_name == project_name]
        if video_id is not None:
            occurrences = [o for o in occurrences if o.video_id == video_id]

        # Sort by project, video, then timestamp
        return sorted(
            occurrences,
            key=lambda o: (o.project_name, o.video_id, o.start),
        )

    def search_phrase(
        self,
        phrase: str,
        max_gap: float = 2.0,
        project_name: str | None = None,
        video_id: str | None = None,
    ) -> list[PhraseMatch]:
        """Search for occurrences of a multi-word phrase.

        Finds consecutive word matches where words appear in sequence.

        Args:
            phrase: Phrase to search for (space-separated words)
            max_gap: Maximum time gap between consecutive words (seconds)
            project_name: Optional filter by project
            video_id: Optional filter by video

        Returns:
            List of PhraseMatch objects sorted by timestamp
        """
        words = self.tokenize(phrase)
        if not words:
            return []

        if len(words) == 1:
            # Single word - convert to phrase matches
            occurrences = self.search_word(words[0], project_name, video_id)
            return [
                PhraseMatch(
                    phrase=occ.word,
                    words=[occ],
                    start=occ.start,
                    end=occ.end,
                    project_name=occ.project_name,
                    video_id=occ.video_id,
                )
                for occ in occurrences
            ]

        # Get occurrences of first word
        first_word_occurrences = self.search_word(words[0], project_name, video_id)

        matches = []

        for first_occ in first_word_occurrences:
            # Try to find consecutive words starting from this occurrence
            current_chain = [first_occ]
            current_video = first_occ.video_id
            current_project = first_occ.project_name

            for next_word in words[1:]:
                # Get occurrences of next word in same video
                next_occurrences = self.search_word(
                    next_word, current_project, current_video
                )

                # Find occurrence that follows current chain
                last_in_chain = current_chain[-1]
                found_next = False

                for next_occ in next_occurrences:
                    # Check if this occurrence follows the last one
                    # Must be in same segment OR adjacent segments
                    # and within max_gap time

                    # Check time proximity
                    time_gap = next_occ.start - last_in_chain.end
                    if time_gap < 0 or time_gap > max_gap:
                        continue

                    # Check sequential position (consecutive word index or next segment)
                    is_consecutive = (
                        # Same segment, consecutive word
                        (
                            next_occ.segment_index == last_in_chain.segment_index
                            and next_occ.word_index == last_in_chain.word_index + 1
                        )
                        or
                        # Adjacent segment, first word
                        (
                            next_occ.segment_index == last_in_chain.segment_index + 1
                            and next_occ.word_index == 0
                        )
                    )

                    if is_consecutive:
                        current_chain.append(next_occ)
                        found_next = True
                        break

                if not found_next:
                    break

            # Check if we found all words
            if len(current_chain) == len(words):
                matches.append(
                    PhraseMatch(
                        phrase=" ".join(occ.word for occ in current_chain),
                        words=current_chain,
                        start=current_chain[0].start,
                        end=current_chain[-1].end,
                        project_name=current_project,
                        video_id=current_video,
                    )
                )

        # Sort by project, video, then timestamp
        return sorted(
            matches,
            key=lambda m: (m.project_name, m.video_id, m.start),
        )

    def get_vocabulary(self) -> list[str]:
        """Get all unique normalized words in the index.

        Returns:
            Sorted list of unique words
        """
        return sorted(self.words.keys())

    def get_word_count(self, word: str) -> int:
        """Get the number of occurrences of a word.

        Args:
            word: Word to count

        Returns:
            Number of occurrences
        """
        normalized = self.normalize_word(word)
        return len(self.words.get(normalized, []))

    def get_statistics(self) -> dict:
        """Get index statistics.

        Returns:
            Dict with statistics (unique words, total occurrences, etc.)
        """
        total_occurrences = sum(len(occs) for occs in self.words.values())
        return {
            "brand_name": self.brand_name,
            "unique_words": len(self.words),
            "total_occurrences": total_occurrences,
            "indexed_transcripts": len(self._indexed_transcripts),
            "transcripts": [
                {"project": p, "video_id": v}
                for p, v in sorted(self._indexed_transcripts)
            ],
        }

    def clear(self) -> None:
        """Clear the entire index."""
        self.words.clear()
        self._indexed_transcripts.clear()
        self.metadata = {}

    def rebuild(
        self,
        transcripts: list[tuple[str, str, Transcript]],
    ) -> dict:
        """Rebuild the index from a list of transcripts.

        Clears existing index and re-indexes all provided transcripts.

        Args:
            transcripts: List of (project_name, video_id, transcript) tuples

        Returns:
            Statistics about the rebuild operation
        """
        self.clear()

        total_words = 0
        for project_name, video_id, transcript in transcripts:
            word_count = self.add_transcript(project_name, video_id, transcript)
            total_words += word_count

        return {
            "transcripts_indexed": len(transcripts),
            "total_words_indexed": total_words,
            "unique_words": len(self.words),
        }

    def to_dict(self) -> dict:
        """Convert index to dictionary for serialization.

        Returns:
            Dict representation of the index
        """
        return {
            "brand_name": self.brand_name,
            "metadata": self.metadata,
            "indexed_transcripts": [
                {"project": p, "video_id": v}
                for p, v in sorted(self._indexed_transcripts)
            ],
            "words": {
                word: [occ.to_dict() for occ in occurrences]
                for word, occurrences in self.words.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TranscriptIndex":
        """Create index from dictionary.

        Args:
            data: Dict representation of the index

        Returns:
            TranscriptIndex instance
        """
        index = cls(
            brand_name=data.get("brand_name", ""),
            metadata=data.get("metadata", {}),
        )

        # Restore indexed transcripts
        for item in data.get("indexed_transcripts", []):
            index._indexed_transcripts.add((item["project"], item["video_id"]))

        # Restore word index
        for word, occurrences in data.get("words", {}).items():
            index.words[word] = [
                WordOccurrence.from_dict(occ) for occ in occurrences
            ]

        return index

    def save(self, path: Path) -> None:
        """Save index to a JSON file.

        Args:
            path: Path to save the index
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "TranscriptIndex":
        """Load index from a JSON file.

        Args:
            path: Path to the index file

        Returns:
            TranscriptIndex instance

        Raises:
            NotFoundError: If file doesn't exist
        """
        data = read_json(path)
        return cls.from_dict(data)


class TranscriptIndexManager:
    """Manager for transcript indexes.

    Handles index creation, loading, saving, and rebuilding for brands.
    """

    def __init__(self, brands_root: Path | None = None):
        """Initialize the index manager.

        Args:
            brands_root: Root directory for all brands.
                        Defaults to ./brands in current directory.
        """
        self.brands_root = brands_root or Path.cwd() / "brands"

    def _index_path(self, brand_name: str) -> Path:
        """Get the path for a brand's transcript index."""
        return self.brands_root / brand_name / "transcript_index.json"

    def exists(self, brand_name: str) -> bool:
        """Check if an index exists for a brand.

        Args:
            brand_name: Brand name

        Returns:
            True if index exists
        """
        return self._index_path(brand_name).exists()

    def get(self, brand_name: str) -> TranscriptIndex:
        """Get the transcript index for a brand.

        Creates a new empty index if one doesn't exist.

        Args:
            brand_name: Brand name

        Returns:
            TranscriptIndex for the brand
        """
        path = self._index_path(brand_name)
        if path.exists():
            return TranscriptIndex.load(path)
        return TranscriptIndex(brand_name=brand_name)

    def save(self, index: TranscriptIndex) -> Path:
        """Save a transcript index.

        Args:
            index: Index to save

        Returns:
            Path to the saved index file
        """
        path = self._index_path(index.brand_name)
        index.save(path)
        return path

    def delete(self, brand_name: str) -> bool:
        """Delete a transcript index.

        Args:
            brand_name: Brand name

        Returns:
            True if deleted, False if didn't exist
        """
        path = self._index_path(brand_name)
        if not path.exists():
            return False
        path.unlink()
        return True

    def rebuild_from_store(
        self,
        brand_name: str,
        transcript_store: "TranscriptStore",
    ) -> TranscriptIndex:
        """Rebuild index from transcript store.

        Args:
            brand_name: Brand name
            transcript_store: TranscriptStore to read transcripts from

        Returns:
            Rebuilt TranscriptIndex
        """
        # Import here to avoid circular dependency
        from clip_video.transcript.storage import TranscriptStore

        index = TranscriptIndex(brand_name=brand_name)
        transcripts = transcript_store.get_all_for_brand(brand_name)
        index.rebuild(transcripts)
        self.save(index)
        return index
