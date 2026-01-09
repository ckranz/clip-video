"""Transcript models for clip-video.

Provides models for storing transcription data with word-level timestamps.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class TranscriptWord(BaseModel):
    """A single word with timing information.

    Stores word-level timestamps for precise caption synchronization.
    """

    word: str
    start: float  # Start time in seconds
    end: float  # End time in seconds
    confidence: float = 1.0  # Confidence score from transcription

    @property
    def duration(self) -> float:
        """Get the duration of this word in seconds."""
        return self.end - self.start

    def overlaps(self, start: float, end: float) -> bool:
        """Check if this word overlaps with a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds

        Returns:
            True if there is any overlap
        """
        return self.start < end and self.end > start

    def contains_time(self, time: float) -> bool:
        """Check if this word is being spoken at a given time.

        Args:
            time: Time in seconds

        Returns:
            True if time falls within word boundaries
        """
        return self.start <= time < self.end


class TranscriptSegment(BaseModel):
    """A segment of transcript containing multiple words.

    Typically represents a sentence or phrase that forms a logical unit.
    """

    id: int  # Segment index
    text: str  # Full text of the segment
    start: float  # Start time in seconds
    end: float  # End time in seconds
    words: list[TranscriptWord] = Field(default_factory=list)
    speaker: str | None = None  # Optional speaker identification

    @property
    def duration(self) -> float:
        """Get the duration of this segment in seconds."""
        return self.end - self.start

    @property
    def word_count(self) -> int:
        """Get the number of words in this segment."""
        return len(self.words)

    def get_words_in_range(self, start: float, end: float) -> list[TranscriptWord]:
        """Get all words that overlap with a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds

        Returns:
            List of words that overlap with the range
        """
        return [w for w in self.words if w.overlaps(start, end)]

    def get_word_at_time(self, time: float) -> TranscriptWord | None:
        """Get the word being spoken at a specific time.

        Args:
            time: Time in seconds

        Returns:
            The word at that time, or None if no word is being spoken
        """
        for word in self.words:
            if word.contains_time(time):
                return word
        return None


class Transcript(BaseModel):
    """Complete transcript for a video file.

    Contains all segments and words with timing information,
    plus metadata about the transcription process.
    """

    video_path: str  # Path to the source video
    language: str = "en"  # Detected or specified language
    created_at: datetime = Field(default_factory=datetime.now)

    # Transcription metadata
    provider: str = "whisper_api"  # Transcription service used
    model: str = "whisper-1"  # Model used for transcription
    duration: float = 0.0  # Total duration of the video in seconds

    # Transcript content
    segments: list[TranscriptSegment] = Field(default_factory=list)

    # Full text without timing (for reference)
    full_text: str = ""

    @property
    def segment_count(self) -> int:
        """Get the number of segments in the transcript."""
        return len(self.segments)

    @property
    def word_count(self) -> int:
        """Get the total number of words in the transcript."""
        return sum(seg.word_count for seg in self.segments)

    def get_all_words(self) -> list[TranscriptWord]:
        """Get all words from all segments in order.

        Returns:
            Flat list of all words
        """
        words = []
        for segment in self.segments:
            words.extend(segment.words)
        return words

    def get_text_in_range(self, start: float, end: float) -> str:
        """Get the text spoken within a time range.

        Args:
            start: Start time in seconds
            end: End time in seconds

        Returns:
            Text of all words within the range
        """
        words = []
        for segment in self.segments:
            words.extend(segment.get_words_in_range(start, end))
        return " ".join(w.word for w in words)

    def get_segment_at_time(self, time: float) -> TranscriptSegment | None:
        """Get the segment containing a specific time.

        Args:
            time: Time in seconds

        Returns:
            The segment at that time, or None
        """
        for segment in self.segments:
            if segment.start <= time < segment.end:
                return segment
        return None

    def get_word_at_time(self, time: float) -> TranscriptWord | None:
        """Get the word being spoken at a specific time.

        Args:
            time: Time in seconds

        Returns:
            The word at that time, or None
        """
        segment = self.get_segment_at_time(time)
        if segment:
            return segment.get_word_at_time(time)
        return None

    def search_word(self, word: str, case_sensitive: bool = False) -> list[TranscriptWord]:
        """Search for occurrences of a word in the transcript.

        Args:
            word: Word to search for
            case_sensitive: Whether to match case

        Returns:
            List of matching TranscriptWord objects
        """
        matches = []
        search_word = word if case_sensitive else word.lower()
        for segment in self.segments:
            for w in segment.words:
                w_text = w.word if case_sensitive else w.word.lower()
                if w_text == search_word:
                    matches.append(w)
        return matches

    def apply_vocabulary_corrections(self, vocabulary: dict[str, list[str]]) -> int:
        """Apply vocabulary corrections to the transcript.

        Args:
            vocabulary: Dict mapping canonical words to alternative spellings

        Returns:
            Number of corrections made
        """
        corrections = 0
        for segment in self.segments:
            for word in segment.words:
                word_lower = word.word.lower()
                for canonical, alternatives in vocabulary.items():
                    if word_lower in [alt.lower() for alt in alternatives]:
                        # Preserve original capitalization style
                        if word.word.isupper():
                            word.word = canonical.upper()
                        elif word.word[0].isupper():
                            word.word = canonical.capitalize()
                        else:
                            word.word = canonical.lower()
                        corrections += 1
                        break
        # Rebuild segment text
        for segment in self.segments:
            segment.text = " ".join(w.word for w in segment.words)
        # Rebuild full text
        self.full_text = " ".join(seg.text for seg in self.segments)
        return corrections
