"""Tests for brand caption enhancements."""

import pytest
from pathlib import Path

from clip_video.captions.enhancements import (
    EmojiTrigger,
    LogoOverlay,
    BrandEnhancements,
    EnhancedCaptionRenderer,
    create_cncf_enhancements,
    create_tech_enhancements,
)


class TestEmojiTrigger:
    """Tests for EmojiTrigger class."""

    def test_apply_after(self):
        """Test inserting emoji after word."""
        trigger = EmojiTrigger(word="hello", emoji="👋", position="after")
        result = trigger.apply("Say hello to everyone")

        assert "hello 👋" in result

    def test_apply_before(self):
        """Test inserting emoji before word."""
        trigger = EmojiTrigger(word="important", emoji="⚡", position="before")
        result = trigger.apply("This is important information")

        assert "⚡ important" in result

    def test_apply_replace(self):
        """Test replacing word with emoji."""
        trigger = EmojiTrigger(word="heart", emoji="❤️", position="replace")
        result = trigger.apply("I heart this")

        assert result == "I ❤️ this"
        assert "heart" not in result

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        trigger = EmojiTrigger(word="kubernetes", emoji="☸️")
        result = trigger.apply("Using KUBERNETES for orchestration")

        assert "☸️" in result

    def test_whole_word_match(self):
        """Test whole word matching."""
        trigger = EmojiTrigger(word="cloud", emoji="☁️", match_whole_word=True)
        result = trigger.apply("CloudFormation is not the same as cloud")

        # Should only match "cloud" not "Cloud" in "CloudFormation"
        assert result.count("☁️") == 1

    def test_partial_word_match(self):
        """Test partial word matching."""
        trigger = EmojiTrigger(word="cloud", emoji="☁️", match_whole_word=False)
        result = trigger.apply("CloudFormation and cloud computing")

        # Should match both
        assert result.count("☁️") == 2


class TestLogoOverlay:
    """Tests for LogoOverlay class."""

    def test_defaults(self):
        """Test default values."""
        overlay = LogoOverlay(image_path=Path("/logo.png"))

        assert overlay.x == "W-w-10"
        assert overlay.y == "10"
        assert overlay.scale == 1.0
        assert overlay.opacity == 1.0
        assert overlay.trigger_word is None

    def test_with_trigger(self):
        """Test overlay with trigger word."""
        overlay = LogoOverlay(
            image_path=Path("/logo.png"),
            trigger_word="kubernetes",
            duration=5.0,
        )

        assert overlay.trigger_word == "kubernetes"
        assert overlay.duration == 5.0


class TestBrandEnhancements:
    """Tests for BrandEnhancements class."""

    def test_add_emoji_trigger(self):
        """Test adding emoji triggers."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.add_emoji_trigger("hello", "👋")

        assert len(enhancements.emoji_triggers) == 1
        assert enhancements.emoji_triggers[0].word == "hello"

    def test_add_logo_overlay(self):
        """Test adding logo overlays."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.add_logo_overlay("/path/to/logo.png")

        assert len(enhancements.logo_overlays) == 1

    def test_enhance_text_emoji(self):
        """Test text enhancement with emojis."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.add_emoji_trigger("kubernetes", "☸️")
        enhancements.add_emoji_trigger("deploy", "🚀")

        result = enhancements.enhance_text("Deploy to Kubernetes today")

        assert "☸️" in result
        assert "🚀" in result

    def test_enhance_text_replacements(self):
        """Test text enhancement with word replacements."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.word_replacements = {
            "k8s": "Kubernetes",
            "js": "JavaScript",
        }

        result = enhancements.enhance_text("Using k8s and js")

        assert "Kubernetes" in result
        assert "JavaScript" in result
        assert "k8s" not in result

    def test_get_triggered_logos_static(self):
        """Test getting static logo overlays."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.add_logo_overlay("/logo.png")  # No trigger = static

        logos = enhancements.get_triggered_logos("any text", 0.0)

        assert len(logos) == 1
        assert logos[0][1] == 0.0  # Start time
        assert logos[0][2] == float("inf")  # End time (always shown)

    def test_get_triggered_logos_triggered(self):
        """Test getting triggered logo overlays."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.add_logo_overlay(
            "/k8s-logo.png",
            trigger_word="kubernetes",
        )

        # No match
        logos = enhancements.get_triggered_logos("hello world", 5.0)
        assert len(logos) == 0

        # Match
        logos = enhancements.get_triggered_logos("Using Kubernetes", 10.0)
        assert len(logos) == 1
        assert logos[0][1] == 10.0  # Start at current time

    def test_serialization(self):
        """Test to_dict and from_dict."""
        enhancements = BrandEnhancements(name="Test")
        enhancements.add_emoji_trigger("hello", "👋")
        enhancements.add_logo_overlay("/logo.png", trigger_word="brand")
        enhancements.word_replacements = {"k8s": "Kubernetes"}

        data = enhancements.to_dict()
        loaded = BrandEnhancements.from_dict(data)

        assert loaded.name == "Test"
        assert len(loaded.emoji_triggers) == 1
        assert loaded.emoji_triggers[0].emoji == "👋"
        assert len(loaded.logo_overlays) == 1
        assert loaded.logo_overlays[0].trigger_word == "brand"
        assert loaded.word_replacements["k8s"] == "Kubernetes"


class TestPreBuiltEnhancements:
    """Tests for pre-built enhancement sets."""

    def test_cncf_enhancements(self):
        """Test CNCF enhancement set."""
        enhancements = create_cncf_enhancements()

        assert enhancements.name == "CNCF"
        assert len(enhancements.emoji_triggers) > 0

        # Test kubernetes trigger
        result = enhancements.enhance_text("Running on Kubernetes")
        assert "☸️" in result

    def test_tech_enhancements(self):
        """Test generic tech enhancement set."""
        enhancements = create_tech_enhancements()

        assert enhancements.name == "Tech"
        assert len(enhancements.emoji_triggers) > 0

        # Test common triggers
        result = enhancements.enhance_text("This is an important tip")
        assert "⚡" in result or "💡" in result


class TestEnhancedCaptionRenderer:
    """Tests for EnhancedCaptionRenderer class."""

    def test_init(self):
        """Test renderer initialization."""
        renderer = EnhancedCaptionRenderer()
        assert renderer.ffmpeg_path is not None

    # Note: Full rendering tests would require actual video files
    # These are integration tests that would be done manually
