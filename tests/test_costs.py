"""Tests for cost tracking and reporting."""

import pytest
from pathlib import Path
from unittest.mock import Mock

from clip_video.costs import (
    CostCategory,
    CostEntry,
    CostSummary,
    CostTracker,
    CostEstimate,
    CostEstimator,
    Pricing,
    format_cost,
    format_cost_report,
    format_estimate,
)


class TestCostCategory:
    """Tests for CostCategory enum."""

    def test_categories(self):
        """Test category values."""
        assert CostCategory.TRANSCRIPTION.value == "transcription"
        assert CostCategory.LLM_ANALYSIS.value == "llm_analysis"
        assert CostCategory.OTHER.value == "other"


class TestCostEntry:
    """Tests for CostEntry dataclass."""

    def test_basic_creation(self):
        """Test basic entry creation."""
        entry = CostEntry(
            category=CostCategory.TRANSCRIPTION,
            service="whisper",
            amount=0.50,
        )

        assert entry.category == CostCategory.TRANSCRIPTION
        assert entry.service == "whisper"
        assert entry.amount == 0.50
        assert entry.timestamp != ""

    def test_full_creation(self):
        """Test entry with all fields."""
        entry = CostEntry(
            category=CostCategory.LLM_ANALYSIS,
            service="claude",
            amount=0.02,
            description="Highlight analysis",
            tokens_used=1500,
            metadata={"model": "sonnet"},
        )

        assert entry.tokens_used == 1500
        assert entry.metadata["model"] == "sonnet"

    def test_serialization(self):
        """Test to_dict and from_dict."""
        entry = CostEntry(
            category=CostCategory.TRANSCRIPTION,
            service="whisper",
            amount=0.50,
            duration_seconds=300,
        )

        data = entry.to_dict()
        loaded = CostEntry.from_dict(data)

        assert loaded.category == entry.category
        assert loaded.service == entry.service
        assert loaded.amount == entry.amount
        assert loaded.duration_seconds == entry.duration_seconds


class TestCostSummary:
    """Tests for CostSummary dataclass."""

    def test_creation(self):
        """Test summary creation."""
        summary = CostSummary(
            category="transcription",
            service="all",
            total_amount=1.50,
            entry_count=3,
            total_duration=1800,
        )

        assert summary.total_amount == 1.50
        assert summary.entry_count == 3

    def test_serialization(self):
        """Test to_dict."""
        summary = CostSummary(
            category="llm_analysis",
            service="claude",
            total_amount=0.05,
            entry_count=5,
            total_tokens=7500,
        )

        data = summary.to_dict()

        assert data["category"] == "llm_analysis"
        assert data["total_tokens"] == 7500


class TestCostTracker:
    """Tests for CostTracker class."""

    def test_init(self):
        """Test initialization."""
        tracker = CostTracker()

        assert len(tracker.entries) == 0
        assert tracker.get_total() == 0.0

    def test_add_cost(self):
        """Test adding a cost entry."""
        tracker = CostTracker()
        entry = CostEntry(
            category=CostCategory.TRANSCRIPTION,
            service="whisper",
            amount=0.50,
        )

        tracker.add_cost(entry)

        assert len(tracker.entries) == 1
        assert tracker.get_total() == 0.50

    def test_add_transcription_cost(self):
        """Test adding transcription cost."""
        tracker = CostTracker()
        entry = tracker.add_transcription_cost(
            service="whisper",
            amount=0.50,
            duration_seconds=300,
        )

        assert entry.category == CostCategory.TRANSCRIPTION
        assert entry.duration_seconds == 300
        assert tracker.get_total() == 0.50

    def test_add_llm_cost(self):
        """Test adding LLM cost."""
        tracker = CostTracker()
        entry = tracker.add_llm_cost(
            service="claude",
            amount=0.02,
            tokens_used=1500,
        )

        assert entry.category == CostCategory.LLM_ANALYSIS
        assert entry.tokens_used == 1500

    def test_add_other_cost(self):
        """Test adding other cost."""
        tracker = CostTracker()
        entry = tracker.add_other_cost(
            service="storage",
            amount=0.01,
            description="S3 storage",
        )

        assert entry.category == CostCategory.OTHER

    def test_get_total(self):
        """Test total calculation."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_llm_cost("claude", 0.02, 1500)
        tracker.add_llm_cost("claude", 0.03, 2000)

        assert tracker.get_total() == 0.55

    def test_get_total_by_category(self):
        """Test total by category."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_transcription_cost("whisper", 0.25, 150)
        tracker.add_llm_cost("claude", 0.02, 1500)

        transcription_total = tracker.get_total_by_category(CostCategory.TRANSCRIPTION)
        llm_total = tracker.get_total_by_category(CostCategory.LLM_ANALYSIS)

        assert transcription_total == 0.75
        assert llm_total == 0.02

    def test_get_total_by_service(self):
        """Test total by service."""
        tracker = CostTracker()
        tracker.add_llm_cost("claude", 0.02, 1500)
        tracker.add_llm_cost("claude", 0.03, 2000)
        tracker.add_llm_cost("openai", 0.05, 3000)

        claude_total = tracker.get_total_by_service("claude")
        openai_total = tracker.get_total_by_service("openai")

        assert claude_total == 0.05
        assert openai_total == 0.05

    def test_get_summary_by_category(self):
        """Test summary by category."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_transcription_cost("whisper", 0.25, 150)
        tracker.add_llm_cost("claude", 0.02, 1500)

        summaries = tracker.get_summary_by_category()

        assert "transcription" in summaries
        assert summaries["transcription"].total_amount == 0.75
        assert summaries["transcription"].entry_count == 2
        assert summaries["transcription"].total_duration == 450

    def test_get_summary_by_service(self):
        """Test summary by service."""
        tracker = CostTracker()
        tracker.add_llm_cost("claude", 0.02, 1500)
        tracker.add_llm_cost("claude", 0.03, 2000)
        tracker.add_llm_cost("openai", 0.05, 3000)

        summaries = tracker.get_summary_by_service()

        assert "claude" in summaries
        assert "openai" in summaries
        assert summaries["claude"].total_amount == 0.05
        assert summaries["claude"].total_tokens == 3500

    def test_get_full_report(self):
        """Test full report generation."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_llm_cost("claude", 0.02, 1500)

        report = tracker.get_full_report()

        assert "total_cost" in report
        assert report["total_cost"] == 0.52
        assert "by_category" in report
        assert "by_service" in report
        assert "entries" in report

    def test_cost_callback(self):
        """Test cost callback is called."""
        callback_calls = []

        def callback(amount: float, description: str):
            callback_calls.append((amount, description))

        tracker = CostTracker(cost_callback=callback)
        tracker.add_transcription_cost("whisper", 0.50, 300)

        assert len(callback_calls) == 1
        assert callback_calls[0][0] == 0.50

    def test_serialization(self):
        """Test to_dict and from_dict."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_llm_cost("claude", 0.02, 1500)

        data = tracker.to_dict()
        loaded = CostTracker.from_dict(data)

        assert len(loaded.entries) == 2
        assert loaded.get_total() == 0.52

    def test_save_and_load(self, tmp_path):
        """Test persistence."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_llm_cost("claude", 0.02, 1500)

        file_path = tmp_path / "costs.json"
        tracker.save(file_path)

        assert file_path.exists()

        loaded = CostTracker.load(file_path)

        assert len(loaded.entries) == 2
        assert loaded.get_total() == 0.52


class TestCostEstimate:
    """Tests for CostEstimate dataclass."""

    def test_creation(self):
        """Test estimate creation."""
        estimate = CostEstimate(
            min_cost=0.50,
            max_cost=1.00,
            expected_cost=0.75,
            breakdown={"transcription": 0.50, "llm": 0.25},
            assumptions=["30 minute video"],
        )

        assert estimate.min_cost == 0.50
        assert estimate.max_cost == 1.00
        assert estimate.expected_cost == 0.75

    def test_serialization(self):
        """Test to_dict."""
        estimate = CostEstimate(
            min_cost=0.50,
            max_cost=1.00,
            expected_cost=0.75,
        )

        data = estimate.to_dict()

        assert data["min_cost"] == 0.5
        assert data["expected_cost"] == 0.75


class TestCostEstimator:
    """Tests for CostEstimator class."""

    def test_estimate_transcription_cost(self):
        """Test transcription cost estimation."""
        estimator = CostEstimator()

        # 10 minutes = 600 seconds
        cost = estimator.estimate_transcription_cost(600, "whisper")

        # 10 minutes * $0.006/min = $0.06
        assert cost == pytest.approx(0.06, rel=0.01)

    def test_estimate_llm_cost_claude(self):
        """Test Claude LLM cost estimation."""
        estimator = CostEstimator()

        # 10000 input, 500 output
        cost = estimator.estimate_llm_cost(
            input_tokens=10000,
            output_tokens=500,
            service="claude",
        )

        # (10000/1M) * $3 + (500/1M) * $15 = $0.03 + $0.0075 = $0.0375
        assert cost == pytest.approx(0.0375, rel=0.01)

    def test_estimate_llm_cost_claude_haiku(self):
        """Test Claude Haiku cost estimation."""
        estimator = CostEstimator()

        cost = estimator.estimate_llm_cost(
            input_tokens=10000,
            output_tokens=500,
            service="claude",
            model="haiku",
        )

        # Much cheaper than Sonnet
        assert cost < 0.01

    def test_estimate_llm_cost_openai(self):
        """Test OpenAI cost estimation."""
        estimator = CostEstimator()

        cost = estimator.estimate_llm_cost(
            input_tokens=10000,
            output_tokens=500,
            service="openai",
        )

        assert cost > 0

    def test_estimate_highlight_processing(self):
        """Test highlight processing estimation."""
        estimator = CostEstimator()

        # 30 minute video = 1800 seconds
        estimate = estimator.estimate_highlight_processing(1800)

        assert estimate.min_cost > 0
        assert estimate.max_cost > estimate.min_cost
        assert estimate.expected_cost > 0
        assert "transcription" in estimate.breakdown
        assert "llm_analysis" in estimate.breakdown

    def test_estimate_batch_processing(self):
        """Test batch processing estimation."""
        estimator = CostEstimator()

        # Three 30-minute videos
        durations = [1800, 1800, 1800]
        estimate = estimator.estimate_batch_processing(durations)

        # Should be roughly 3x a single video
        single = estimator.estimate_highlight_processing(1800)
        assert estimate.expected_cost == pytest.approx(single.expected_cost * 3, rel=0.01)


class TestPricing:
    """Tests for Pricing constants."""

    def test_whisper_pricing(self):
        """Test Whisper pricing constant."""
        assert Pricing.WHISPER_API_PER_MINUTE == 0.006

    def test_claude_pricing(self):
        """Test Claude pricing constants."""
        assert Pricing.CLAUDE_INPUT_PER_1M == 3.00
        assert Pricing.CLAUDE_OUTPUT_PER_1M == 15.00
        assert Pricing.CLAUDE_HAIKU_INPUT_PER_1M == 0.25

    def test_openai_pricing(self):
        """Test OpenAI pricing constants."""
        assert Pricing.GPT4O_INPUT_PER_1M == 2.50
        assert Pricing.GPT4O_OUTPUT_PER_1M == 10.00


class TestFormatFunctions:
    """Tests for formatting functions."""

    def test_format_cost_small(self):
        """Test formatting small costs."""
        assert format_cost(0.001) == "$0.0010"
        assert format_cost(0.0056) == "$0.0056"

    def test_format_cost_medium(self):
        """Test formatting medium costs."""
        assert format_cost(0.05) == "$0.050"
        assert format_cost(0.525) == "$0.525"

    def test_format_cost_large(self):
        """Test formatting large costs."""
        assert format_cost(1.50) == "$1.50"
        assert format_cost(10.00) == "$10.00"

    def test_format_cost_report(self):
        """Test formatting a cost report."""
        tracker = CostTracker()
        tracker.add_transcription_cost("whisper", 0.50, 300)
        tracker.add_llm_cost("claude", 0.02, 1500)

        report = format_cost_report(tracker)

        assert "COST REPORT" in report
        assert "Total Cost" in report
        assert "transcription" in report
        assert "claude" in report

    def test_format_estimate(self):
        """Test formatting an estimate."""
        estimate = CostEstimate(
            min_cost=0.50,
            max_cost=1.00,
            expected_cost=0.75,
            breakdown={"transcription": 0.50, "llm_analysis": 0.25},
            assumptions=["30 minute video"],
        )

        formatted = format_estimate(estimate)

        assert "Expected" in formatted
        assert "Range" in formatted
        assert "Breakdown" in formatted
        assert "Assumptions" in formatted


class TestIntegration:
    """Integration tests for cost tracking."""

    def test_full_workflow(self, tmp_path):
        """Test a complete cost tracking workflow."""
        # Create tracker
        tracker = CostTracker()

        # Add various costs
        tracker.add_transcription_cost("whisper", 0.06, 600)  # 10 min
        tracker.add_llm_cost("claude", 0.05, 5000, model="sonnet")
        tracker.add_transcription_cost("whisper", 0.12, 1200)  # 20 min
        tracker.add_llm_cost("claude", 0.08, 8000, model="sonnet")

        # Check totals
        assert tracker.get_total() == 0.31
        assert tracker.get_total_by_category(CostCategory.TRANSCRIPTION) == 0.18
        assert tracker.get_total_by_category(CostCategory.LLM_ANALYSIS) == 0.13

        # Generate and check report
        report = tracker.get_full_report()
        assert report["total_cost"] == 0.31
        assert report["entry_count"] == 4

        # Save and reload
        tracker.save(tmp_path / "costs.json")
        loaded = CostTracker.load(tmp_path / "costs.json")
        assert loaded.get_total() == 0.31

    def test_estimation_then_actual(self):
        """Test estimating then tracking actual costs."""
        estimator = CostEstimator()
        tracker = CostTracker()

        # Estimate for 30 minute video
        estimate = estimator.estimate_highlight_processing(1800)

        # Simulate actual processing
        tracker.add_transcription_cost(
            "whisper",
            30 * Pricing.WHISPER_API_PER_MINUTE,  # 30 min
            1800,
        )
        tracker.add_llm_cost(
            "claude",
            0.04,
            4000,
        )

        actual = tracker.get_total()

        # Actual should be within estimate range
        assert estimate.min_cost <= actual <= estimate.max_cost
