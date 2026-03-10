"""Tests for LLM transcript refinement prompt builder."""

from clip_video.transcription.base import TranscriptionSegment
from clip_video.transcription.llm_refine import (
    REFINEMENT_SYSTEM_PROMPT,
    RefinementContext,
    build_refinement_prompt,
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
