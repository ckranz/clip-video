"""Tests for lyrics parsing and phrase extraction."""

import pytest
from pathlib import Path
import tempfile

from clip_video.lyrics.parser import LyricsParser, LyricsLine, ParsedLyrics
from clip_video.lyrics.phrases import PhraseExtractor, ExtractionTarget, ExtractionList


class TestLyricsLine:
    """Tests for LyricsLine class."""

    def test_tokenize_simple(self):
        """Test tokenizing simple text."""
        words = LyricsLine._tokenize("Hello World")
        assert words == ["hello", "world"]

    def test_tokenize_with_punctuation(self):
        """Test tokenizing text with punctuation."""
        words = LyricsLine._tokenize("Hello, world! How's it going?")
        assert words == ["hello", "world", "how's", "it", "going"]

    def test_tokenize_empty(self):
        """Test tokenizing empty text."""
        words = LyricsLine._tokenize("")
        assert words == []

    def test_auto_tokenize_on_init(self):
        """Test that words are auto-tokenized on init."""
        line = LyricsLine(
            line_number=1,
            raw_text="Hello World",
            text="Hello World",
        )
        assert line.words == ["hello", "world"]


class TestLyricsParser:
    """Tests for LyricsParser class."""

    def test_parse_simple_lyrics(self):
        """Test parsing simple lyrics without markers."""
        text = """First line of the song
Second line here
Third line too"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        assert len(lyrics.lines) == 3
        assert lyrics.lines[0].text == "First line of the song"
        assert lyrics.lines[1].text == "Second line here"
        assert lyrics.lines[2].text == "Third line too"

    def test_parse_with_section_markers(self):
        """Test parsing lyrics with section markers."""
        text = """[Verse 1]
First verse line one
First verse line two

[Chorus]
Chorus line one
Chorus line two"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        # Check section headers
        assert lyrics.lines[0].is_section_header
        assert lyrics.lines[0].section == "Verse 1"

        # Check content lines have correct section
        assert lyrics.lines[1].section == "Verse 1"
        assert lyrics.lines[2].section == "Verse 1"

        # Check chorus section
        assert lyrics.lines[3].is_section_header
        assert lyrics.lines[3].section == "Chorus"
        assert lyrics.lines[4].section == "Chorus"

        # Check sections dict
        assert "Verse 1" in lyrics.sections
        assert "Chorus" in lyrics.sections

    def test_parse_phrase_markers(self):
        """Test parsing lyrics with phrase markers."""
        text = """This is [phrase]a special phrase[/phrase] in the line
Another [phrase]marked phrase[/phrase] here"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        assert "a special phrase" in lyrics.lines[0].phrases
        assert "marked phrase" in lyrics.lines[1].phrases

    def test_parse_repeat_markers(self):
        """Test parsing lyrics with repeat markers."""
        text = """This line repeats x2
Another repeat (x3)"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        assert lyrics.lines[0].repeat_count == 2
        assert lyrics.lines[1].repeat_count == 3

    def test_parse_metadata(self):
        """Test parsing lyrics with metadata."""
        text = """Title: My Song
Artist: The Artist
First actual line"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        assert lyrics.title == "My Song"
        assert lyrics.artist == "The Artist"
        assert len(lyrics.lines) == 1
        assert lyrics.lines[0].text == "First actual line"

    def test_parse_file(self, tmp_path):
        """Test parsing lyrics from a file."""
        lyrics_file = tmp_path / "test_song.txt"
        lyrics_file.write_text("""First line
Second line
Third line""")

        parser = LyricsParser()
        lyrics = parser.parse_file(lyrics_file)

        assert len(lyrics.lines) == 3
        assert lyrics.source_file == lyrics_file
        assert lyrics.title == "test_song"

    def test_auto_phrase_lines(self):
        """Test auto-generating phrases from lines."""
        text = """Short
This is a longer line that becomes a phrase"""

        parser = LyricsParser(auto_phrase_lines=True, min_phrase_words=2)
        lyrics = parser.parse_text(text)

        # Short line (1 word) should not get auto-phrase
        assert lyrics.lines[0].phrases == []

        # Longer line should get auto-phrase
        assert len(lyrics.lines[1].phrases) == 1

    def test_all_words_unique(self):
        """Test that all_words returns unique words."""
        text = """Hello world hello
World world hello"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        words = lyrics.all_words
        assert len(words) == 2  # Only "hello" and "world"
        assert "hello" in words
        assert "world" in words

    def test_content_lines_excludes_headers(self):
        """Test that content_lines excludes section headers."""
        text = """[Verse 1]
Line one
Line two
[Chorus]
Chorus line"""

        parser = LyricsParser()
        lyrics = parser.parse_text(text)

        content_lines = list(lyrics.content_lines)
        assert len(content_lines) == 3
        assert all(not line.is_section_header for line in content_lines)


class TestPhraseExtractor:
    """Tests for PhraseExtractor class."""

    def test_extract_words_only(self):
        """Test extracting only words."""
        text = """Hello beautiful world"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=True,
            extract_phrases=False,
        )
        extraction = extractor.extract(lyrics)

        # Should have words but not phrases
        assert len(extraction.unique_words) > 0
        assert len(extraction.unique_phrases) == 0

    def test_extract_phrases_only(self):
        """Test extracting only phrases."""
        text = """[phrase]Hello world[/phrase] is here"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=False,
            extract_phrases=True,
        )
        extraction = extractor.extract(lyrics)

        # Should have phrases but limited words
        assert "hello world" in extraction.unique_phrases
        assert len(extraction.unique_words) == 0

    def test_stop_words_filtered(self):
        """Test that stop words are filtered."""
        text = """The quick brown fox"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=True,
            use_stop_words=True,
        )
        extraction = extractor.extract(lyrics)

        # "the" should be filtered
        assert "the" not in extraction.unique_words
        # "quick", "brown", "fox" should be present
        assert "quick" in extraction.unique_words
        assert "brown" in extraction.unique_words
        assert "fox" in extraction.unique_words

    def test_deduplication(self):
        """Test that deduplication works."""
        text = """Hello world
Hello again
World hello"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=True,
            deduplicate=True,
        )
        extraction = extractor.extract(lyrics)

        # Each word should appear only once
        all_words = [t.text for t in extraction.all_targets if not t.is_phrase]
        assert all_words.count("hello") == 1
        assert all_words.count("world") == 1

    def test_no_deduplication(self):
        """Test with deduplication disabled."""
        text = """Hello world
Hello again"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=True,
            deduplicate=False,
        )
        extraction = extractor.extract(lyrics)

        # "hello" should appear twice
        all_words = [t.text for t in extraction.all_targets if not t.is_phrase]
        assert all_words.count("hello") == 2

    def test_min_word_length(self):
        """Test minimum word length filter."""
        text = """I am a big boy"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=True,
            min_word_length=3,
            use_stop_words=False,
        )
        extraction = extractor.extract(lyrics)

        # "I", "am", "a" should be filtered (too short)
        assert "i" not in extraction.unique_words
        assert "am" not in extraction.unique_words
        assert "a" not in extraction.unique_words
        # "big", "boy" should be present
        assert "big" in extraction.unique_words
        assert "boy" in extraction.unique_words

    def test_target_source_info(self):
        """Test that targets have correct source info."""
        text = """First line here
Second line there"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(extract_words=True)
        extraction = extractor.extract(lyrics)

        # Check that targets have correct line numbers
        line_1_targets = extraction.get_line_targets(1)
        assert all(t.source_line == 1 for t in line_1_targets)
        assert all("First" in t.source_text for t in line_1_targets)

    def test_priority_calculation(self):
        """Test that priority is calculated correctly."""
        text = """Hi extraordinary"""

        parser = LyricsParser(auto_phrase_lines=False)
        lyrics = parser.parse_text(text)

        extractor = PhraseExtractor(
            extract_words=True,
            use_stop_words=False,
            min_word_length=1,
        )
        extraction = extractor.extract(lyrics)

        # Find the targets
        targets = {t.text: t for t in extraction.all_targets}

        # Longer word should have higher priority
        assert targets["extraordinary"].priority > targets["hi"].priority


class TestExtractionList:
    """Tests for ExtractionList class."""

    def test_serialization(self, tmp_path):
        """Test saving and loading extraction list."""
        extraction = ExtractionList(
            lyrics_file=Path("/path/to/lyrics.txt"),
            title="Test Song",
            artist="Test Artist",
            targets_by_line={
                1: [ExtractionTarget(
                    text="hello",
                    source_line=1,
                    source_text="Hello World",
                    is_phrase=False,
                    priority=1,
                )],
            },
        )

        # Save
        save_path = tmp_path / "extraction.json"
        extraction.save(save_path)

        # Load
        loaded = ExtractionList.load(save_path)

        assert loaded.title == "Test Song"
        assert loaded.artist == "Test Artist"
        assert len(loaded.targets_by_line) == 1
        assert loaded.targets_by_line[1][0].text == "hello"

    def test_add_alternative(self):
        """Test adding alternatives to targets."""
        extraction = ExtractionList(
            lyrics_file=None,
            title=None,
            artist=None,
            targets_by_line={
                1: [ExtractionTarget(
                    text="kubernetes",
                    source_line=1,
                    source_text="Using Kubernetes",
                )],
            },
        )

        # Add alternative
        result = extraction.add_alternative("kubernetes", "k8s")

        assert result is True
        assert "k8s" in extraction.targets_by_line[1][0].alternatives

    def test_lines_in_order(self):
        """Test that lines_in_order returns lines sorted."""
        extraction = ExtractionList(
            lyrics_file=None,
            title=None,
            artist=None,
            targets_by_line={
                5: [ExtractionTarget(text="fifth", source_line=5, source_text="")],
                1: [ExtractionTarget(text="first", source_line=1, source_text="")],
                3: [ExtractionTarget(text="third", source_line=3, source_text="")],
            },
        )

        lines = list(extraction.lines_in_order)

        assert lines[0][0] == 1
        assert lines[1][0] == 3
        assert lines[2][0] == 5


class TestIntegration:
    """Integration tests for the full parsing and extraction workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow from file to extraction list."""
        # Create a test lyrics file
        lyrics_content = """Title: Test Song
Artist: Test Artist

[Verse 1]
The quick brown fox jumps
Over the lazy dog

[Chorus]
[phrase]This is the chorus[/phrase]
Singing loud x2"""

        lyrics_file = tmp_path / "test_song.txt"
        lyrics_file.write_text(lyrics_content)

        # Parse and extract
        extraction = PhraseExtractor.from_lyrics_file(
            lyrics_file,
            extract_words=True,
            extract_phrases=True,
        )

        # Verify
        assert extraction.title == "Test Song"
        assert extraction.artist == "Test Artist"

        # Check that phrase was extracted
        assert "this is the chorus" in extraction.unique_phrases

        # Check that non-stop words were extracted
        assert "quick" in extraction.unique_words
        assert "brown" in extraction.unique_words
        assert "fox" in extraction.unique_words

        # Check that stop words were filtered
        assert "the" not in extraction.unique_words

    def test_extract_and_save(self, tmp_path):
        """Test extracting and saving results."""
        lyrics_content = """Hello world today
Beautiful sunshine here"""

        lyrics_file = tmp_path / "song.txt"
        lyrics_file.write_text(lyrics_content)

        extraction = PhraseExtractor.from_lyrics_file(lyrics_file)

        # Save extraction list
        output_path = tmp_path / "extraction.json"
        extraction.save(output_path)

        # Load and verify
        loaded = ExtractionList.load(output_path)
        assert len(loaded.all_targets) > 0
