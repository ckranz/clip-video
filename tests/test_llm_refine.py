"""Tests for LLM transcript refinement prompt builder."""

import json

from clip_video.transcription.base import TranscriptionSegment, TranscriptionWord
from clip_video.transcription.llm_refine import (
    REFINEMENT_SYSTEM_PROMPT,
    RefinementContext,
    apply_corrections,
    build_refinement_prompt,
    parse_llm_corrections,
)


class TestRefinementContext:
    def test_default_context(self):
        ctx = RefinementContext()
        assert ctx.talk_title is None
        assert ctx.talk_description is None
        assert ctx.speaker_name is None
        assert ctx.domain == "technology"
        assert ctx.vocabulary_terms is None

    def test_context_with_all_fields(self):
        ctx = RefinementContext(
            talk_title="Scaling Kubernetes",
            talk_description="A deep dive into K8s scaling strategies",
            speaker_name="Jane Doe",
            domain="cloud infrastructure",
            vocabulary_terms=["Kubernetes", "kubectl", "etcd"],
        )
        assert ctx.talk_title == "Scaling Kubernetes"
        assert ctx.talk_description == "A deep dive into K8s scaling strategies"
        assert ctx.speaker_name == "Jane Doe"
        assert ctx.domain == "cloud infrastructure"
        assert ctx.vocabulary_terms == ["Kubernetes", "kubectl", "etcd"]


class TestBuildRefinementPrompt:
    def _make_segments(self) -> list[TranscriptionSegment]:
        return [
            TranscriptionSegment(text="Welcome to the talk on cooper netties.", start=0.0, end=3.0),
            TranscriptionSegment(text="Today we discuss go roo teens.", start=3.0, end=6.5),
        ]

    def test_prompt_includes_segment_text(self):
        segments = self._make_segments()
        prompt = build_refinement_prompt(segments)
        assert "cooper netties" in prompt
        assert "go roo teens" in prompt
        assert "[0.0s - 3.0s]" in prompt
        assert "[3.0s - 6.5s]" in prompt

    def test_prompt_includes_context_when_provided(self):
        segments = self._make_segments()
        ctx = RefinementContext(
            talk_title="Intro to K8s",
            talk_description="Kubernetes basics",
            speaker_name="Jane Doe",
            domain="cloud",
        )
        prompt = build_refinement_prompt(segments, context=ctx)
        assert "Title: Intro to K8s" in prompt
        assert "Description: Kubernetes basics" in prompt
        assert "Speaker: Jane Doe" in prompt
        assert "Domain: cloud" in prompt

    def test_prompt_includes_vocabulary_terms(self):
        segments = self._make_segments()
        ctx = RefinementContext(
            vocabulary_terms=["Kubernetes", "kubectl", "goroutines"],
        )
        prompt = build_refinement_prompt(segments, context=ctx)
        assert "Known vocabulary: Kubernetes, kubectl, goroutines" in prompt

    def test_prompt_requests_json_output(self):
        segments = self._make_segments()
        prompt = build_refinement_prompt(segments)
        assert "JSON" in prompt

    def test_prompt_without_context_has_no_context_section(self):
        segments = self._make_segments()
        prompt = build_refinement_prompt(segments, context=None)
        assert "Context:" not in prompt

    def test_system_prompt_mentions_json_array(self):
        assert "JSON array" in REFINEMENT_SYSTEM_PROMPT

    def test_system_prompt_forbids_rephrasing(self):
        assert "Rephrase" in REFINEMENT_SYSTEM_PROMPT or "rephrase" in REFINEMENT_SYSTEM_PROMPT.lower()


class TestParseLLMCorrections:
    def test_parse_valid_json(self):
        response = json.dumps([
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "Misheard"},
        ])
        result = parse_llm_corrections(response)
        assert len(result) == 1
        assert result[0]["original"] == "cooper netties"
        assert result[0]["corrected"] == "Kubernetes"

    def test_parse_json_in_code_block(self):
        response = '```json\n[{"original": "go roo teens", "corrected": "goroutines", "reason": "Go term"}]\n```'
        result = parse_llm_corrections(response)
        assert len(result) == 1
        assert result[0]["corrected"] == "goroutines"

    def test_parse_empty_array(self):
        result = parse_llm_corrections("[]")
        assert result == []

    def test_parse_invalid_json_returns_empty(self):
        result = parse_llm_corrections("this is not json at all")
        assert result == []

    def test_parse_filters_invalid_entries(self):
        response = json.dumps([
            {"original": "bad", "corrected": "good", "reason": "fix"},
            {"original": "missing reason", "corrected": "oops"},
            {"not_a_correction": True},
        ])
        result = parse_llm_corrections(response)
        assert len(result) == 1
        assert result[0]["original"] == "bad"

    def test_parse_skips_no_op_corrections(self):
        response = json.dumps([
            {"original": "same", "corrected": "same", "reason": "no change"},
            {"original": "diff", "corrected": "different", "reason": "real fix"},
        ])
        result = parse_llm_corrections(response)
        assert len(result) == 1
        assert result[0]["original"] == "diff"


class TestApplyCorrections:
    def _make_segments_with_words(self) -> list[TranscriptionSegment]:
        return [
            TranscriptionSegment(
                text="Welcome to the talk on cooper netties today.",
                start=0.0,
                end=4.0,
                words=[
                    TranscriptionWord(word="Welcome", start=0.0, end=0.5),
                    TranscriptionWord(word="to", start=0.5, end=0.7),
                    TranscriptionWord(word="the", start=0.7, end=0.9),
                    TranscriptionWord(word="talk", start=0.9, end=1.2),
                    TranscriptionWord(word="on", start=1.2, end=1.4),
                    TranscriptionWord(word="cooper", start=1.4, end=1.8),
                    TranscriptionWord(word="netties", start=1.8, end=2.2),
                    TranscriptionWord(word="today.", start=2.2, end=2.6),
                ],
            ),
        ]

    def test_apply_single_word_correction(self):
        """Multi-word 'cooper netties' -> single word 'Kubernetes'."""
        segments = self._make_segments_with_words()
        corrections = [
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "Misheard"},
        ]
        result, log = apply_corrections(segments, corrections)
        assert "Kubernetes" in result[0].text
        assert "cooper netties" not in result[0].text

        # Word-level: should have merged two words into one
        words = result[0].words
        kube_words = [w for w in words if w.word == "Kubernetes"]
        assert len(kube_words) == 1
        # Should span from 1.4 to 2.2 (the original two words' range)
        assert kube_words[0].start == 1.4
        assert kube_words[0].end == 2.2
        # Should have original_word set
        assert kube_words[0].original_word == "cooper netties"

    def test_apply_grammar_correction(self):
        segments = [
            TranscriptionSegment(
                text="He go to the store.",
                start=0.0,
                end=3.0,
                words=[
                    TranscriptionWord(word="He", start=0.0, end=0.5),
                    TranscriptionWord(word="go", start=0.5, end=1.0),
                    TranscriptionWord(word="to", start=1.0, end=1.5),
                    TranscriptionWord(word="the", start=1.5, end=2.0),
                    TranscriptionWord(word="store.", start=2.0, end=2.5),
                ],
            ),
        ]
        corrections = [
            {"original": "go", "corrected": "goes", "reason": "Grammar fix"},
        ]
        result, log = apply_corrections(segments, corrections)
        assert "goes" in result[0].text
        goes_words = [w for w in result[0].words if w.word == "goes"]
        assert len(goes_words) == 1
        assert goes_words[0].start == 0.5
        assert goes_words[0].end == 1.0

    def test_apply_no_corrections(self):
        segments = self._make_segments_with_words()
        result, log = apply_corrections(segments, [])
        assert len(result) == 1
        assert result[0].text == segments[0].text
        assert len(log) == 0

    def test_corrections_logged_as_llm_type(self):
        segments = self._make_segments_with_words()
        corrections = [
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "Misheard"},
        ]
        _, log = apply_corrections(segments, corrections, source_file="test.json")
        assert len(log) == 1
        assert log.corrections[0].match_type == "llm"
        assert log.corrections[0].context == "Misheard"
        assert log.source_file == "test.json"

    def test_does_not_modify_original_segments(self):
        segments = self._make_segments_with_words()
        original_text = segments[0].text
        original_word_count = len(segments[0].words)
        original_first_word = segments[0].words[0].word

        corrections = [
            {"original": "cooper netties", "corrected": "Kubernetes", "reason": "Misheard"},
        ]
        apply_corrections(segments, corrections)

        # Originals must be untouched
        assert segments[0].text == original_text
        assert len(segments[0].words) == original_word_count
        assert segments[0].words[0].word == original_first_word
        assert "cooper" in segments[0].text
