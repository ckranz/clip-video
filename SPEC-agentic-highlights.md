# SPEC: Multi-Pass Agentic Highlights Workflow

**Discovery Session**: highlights-agentic-workflow-20260110
**Status**: Ready for Implementation
**Total Discoveries**: 23

---

## Overview

Enhance the existing highlights workflow with a multi-pass agentic system that:
1. Analyzes transcripts to identify highlight candidates
2. Validates each candidate against quality criteria
3. Replaces rejected candidates with alternatives
4. Creates clips with professional captions (white text, black stroke, lower thirds)
5. Maintains a human review queue for rejected clips

---

## Tasks

### Phase 1: Configuration Updates

#### Task 1.1: Update Duration Constraints
**File**: `src/clip_video/modes/highlights.py`
**Change**: Update `HighlightsConfig` defaults

```python
# Current (lines 173-174):
min_duration: float = 15.0
max_duration: float = 60.0

# New:
min_duration: float = 30.0
max_duration: float = 120.0
```

**Tests**:
- [ ] Unit test: `test_config_duration_defaults()` - verify new defaults
- [ ] Unit test: `test_config_duration_validation()` - reject min > max

---

#### Task 1.2: Add New Configuration Fields
**File**: `src/clip_video/modes/highlights.py`
**Change**: Extend `HighlightsConfig` with new fields

```python
@dataclass
class HighlightsConfig:
    # Existing fields...

    # New fields for agentic workflow
    target_clips: int = 5
    min_acceptable_clips: int = 3
    max_replacement_attempts: int = 3
    cost_ceiling_gbp: float = 5.0
    enable_validation_pass: bool = True
```

**Tests**:
- [ ] Unit test: `test_config_new_fields_defaults()` - verify all new defaults
- [ ] Unit test: `test_config_min_acceptable_less_than_target()` - validation

---

### Phase 2: Quality Validation System

#### Task 2.1: Create Validation Data Structures
**File**: `src/clip_video/validation/__init__.py` (new module)
**File**: `src/clip_video/validation/criteria.py`

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class ValidationResult(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"

@dataclass
class CriterionResult:
    criterion: str
    result: ValidationResult
    reason: str
    details: dict = None

@dataclass
class ClipValidation:
    clip_id: str
    start_time: float
    end_time: float
    overall_result: ValidationResult
    criteria_results: List[CriterionResult]
    transcript_segment: str

    @property
    def passed(self) -> bool:
        return self.overall_result == ValidationResult.PASS

    @property
    def failure_reasons(self) -> List[str]:
        return [r.reason for r in self.criteria_results
                if r.result == ValidationResult.FAIL]
```

**Tests**:
- [ ] Unit test: `test_clip_validation_passed_property()`
- [ ] Unit test: `test_clip_validation_failure_reasons()`
- [ ] Unit test: `test_criterion_result_serialization()`

---

#### Task 2.2: Implement Quality Criteria Checker
**File**: `src/clip_video/validation/checker.py`

Implement deterministic checks that can run before LLM validation:

```python
class QualityCriteriaChecker:
    """Checks clips against quality criteria."""

    def __init__(self, config: HighlightsConfig, brand_context: dict = None):
        self.config = config
        self.brand_context = brand_context or {}
        self.used_segments: List[Tuple[float, float]] = []

    def check_duration(self, start: float, end: float) -> CriterionResult:
        """D001: Duration must be 30-120 seconds."""
        duration = end - start
        if duration < self.config.min_duration:
            return CriterionResult(
                criterion="duration",
                result=ValidationResult.FAIL,
                reason=f"Too short: {duration:.1f}s < {self.config.min_duration}s"
            )
        if duration > self.config.max_duration:
            return CriterionResult(
                criterion="duration",
                result=ValidationResult.FAIL,
                reason=f"Too long: {duration:.1f}s > {self.config.max_duration}s"
            )
        return CriterionResult(
            criterion="duration",
            result=ValidationResult.PASS,
            reason=f"Duration OK: {duration:.1f}s"
        )

    def check_overlap(self, start: float, end: float) -> CriterionResult:
        """D008/D013: No overlap with already-approved clips."""
        for used_start, used_end in self.used_segments:
            if start < used_end and end > used_start:
                return CriterionResult(
                    criterion="overlap",
                    result=ValidationResult.FAIL,
                    reason=f"Overlaps with segment {used_start:.1f}-{used_end:.1f}s"
                )
        return CriterionResult(
            criterion="overlap",
            result=ValidationResult.PASS,
            reason="No overlap detected"
        )

    def mark_segment_used(self, start: float, end: float) -> None:
        """Track approved segment to prevent future overlaps."""
        self.used_segments.append((start, end))

    def run_deterministic_checks(
        self,
        start: float,
        end: float
    ) -> List[CriterionResult]:
        """Run all deterministic (non-LLM) checks."""
        return [
            self.check_duration(start, end),
            self.check_overlap(start, end),
        ]
```

**Tests**:
- [ ] Unit test: `test_check_duration_too_short()` - 20s clip fails
- [ ] Unit test: `test_check_duration_too_long()` - 150s clip fails
- [ ] Unit test: `test_check_duration_valid()` - 60s clip passes
- [ ] Unit test: `test_check_overlap_detected()` - overlapping segments fail
- [ ] Unit test: `test_check_overlap_adjacent_ok()` - adjacent segments pass
- [ ] Unit test: `test_mark_segment_used()` - tracking works correctly
- [ ] Integration test: `test_run_deterministic_checks_all_pass()`
- [ ] Integration test: `test_run_deterministic_checks_multiple_failures()`

---

#### Task 2.3: Add LLM Validation Method
**File**: `src/clip_video/llm/base.py`
**Change**: Add `validate_clip()` method to base class

```python
@dataclass
class ClipValidationRequest:
    transcript_segment: str
    start_time: float
    end_time: float
    clip_summary: str
    brand_context: dict
    quality_criteria: List[str]

@dataclass
class ClipValidationResponse:
    is_valid: bool
    sentence_boundaries_ok: bool  # D002
    topic_complete: bool  # D003
    has_hook: bool  # D004 (optional)
    standalone_valid: bool  # D005
    brand_relevant: bool  # D006
    transcript_aligned: bool  # D007
    issues: List[str]
    suggestions: List[str]
```

**File**: `src/clip_video/llm/claude.py` and `openai.py`
**Change**: Implement `validate_clip()` in both providers

**Tests**:
- [ ] Unit test: `test_validation_request_serialization()`
- [ ] Unit test: `test_validation_response_parsing()`
- [ ] Mock test: `test_claude_validate_clip_passing()`
- [ ] Mock test: `test_claude_validate_clip_failing()`
- [ ] Mock test: `test_openai_validate_clip_passing()`

---

### Phase 3: Review Queue System

#### Task 3.1: Create Review Queue Data Structures
**File**: `src/clip_video/review/__init__.py` (new module)
**File**: `src/clip_video/review/queue.py`

```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import json

@dataclass
class RejectedClip:
    """A clip that failed validation and needs human review."""

    clip_id: str
    video_path: str
    start_time: float
    end_time: float
    transcript_segment: str
    rejection_reasons: List[str]
    validation_details: dict
    rejected_at: str = ""
    replacement_attempts: int = 0

    def __post_init__(self):
        if not self.rejected_at:
            self.rejected_at = datetime.now().isoformat()

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def preview_command(self) -> str:
        """CLI command to preview this clip."""
        return (
            f'clip-video preview "{self.video_path}" '
            f'--start {self.start_time} --end {self.end_time}'
        )

    def to_dict(self) -> dict:
        return {
            "clip_id": self.clip_id,
            "video_path": self.video_path,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration": self.duration,
            "transcript_segment": self.transcript_segment,
            "rejection_reasons": self.rejection_reasons,
            "validation_details": self.validation_details,
            "rejected_at": self.rejected_at,
            "replacement_attempts": self.replacement_attempts,
            "preview_command": self.preview_command,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RejectedClip":
        return cls(
            clip_id=data["clip_id"],
            video_path=data["video_path"],
            start_time=data["start_time"],
            end_time=data["end_time"],
            transcript_segment=data["transcript_segment"],
            rejection_reasons=data["rejection_reasons"],
            validation_details=data.get("validation_details", {}),
            rejected_at=data.get("rejected_at", ""),
            replacement_attempts=data.get("replacement_attempts", 0),
        )


class ReviewQueue:
    """Manages the review queue for rejected clips."""

    def __init__(self, review_dir: Path):
        self.review_dir = Path(review_dir)
        self.review_dir.mkdir(parents=True, exist_ok=True)

    def add(self, clip: RejectedClip) -> Path:
        """Add a rejected clip to the queue."""
        filename = f"{clip.clip_id}_{clip.rejected_at.replace(':', '-')}.json"
        filepath = self.review_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(clip.to_dict(), f, indent=2, ensure_ascii=False)

        return filepath

    def list_all(self) -> List[RejectedClip]:
        """List all clips in the review queue."""
        clips = []
        for filepath in self.review_dir.glob("*.json"):
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
                clips.append(RejectedClip.from_dict(data))
        return sorted(clips, key=lambda c: c.rejected_at, reverse=True)

    def get_summary(self) -> dict:
        """Get summary of review queue."""
        clips = self.list_all()
        return {
            "total_clips": len(clips),
            "by_reason": self._group_by_reason(clips),
        }

    def _group_by_reason(self, clips: List[RejectedClip]) -> dict:
        reasons = {}
        for clip in clips:
            for reason in clip.rejection_reasons:
                reasons[reason] = reasons.get(reason, 0) + 1
        return reasons
```

**Tests**:
- [ ] Unit test: `test_rejected_clip_duration_property()`
- [ ] Unit test: `test_rejected_clip_preview_command()`
- [ ] Unit test: `test_rejected_clip_serialization_roundtrip()`
- [ ] Unit test: `test_review_queue_add()`
- [ ] Unit test: `test_review_queue_list_all()`
- [ ] Unit test: `test_review_queue_get_summary()`
- [ ] Integration test: `test_review_queue_filesystem_operations()`

---

### Phase 4: Cost Tracking

#### Task 4.1: Create Cost Tracker
**File**: `src/clip_video/costs/__init__.py` (new module)
**File**: `src/clip_video/costs/tracker.py`

```python
from dataclasses import dataclass, field
from typing import List
from datetime import datetime

# Approximate costs per 1K tokens (as of 2024)
LLM_COSTS = {
    "claude-3-5-sonnet": {"input": 0.003, "output": 0.015},
    "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

# GBP/USD exchange rate (configurable)
USD_TO_GBP = 0.79

@dataclass
class LLMCall:
    operation: str  # "analysis", "validation", "replacement"
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

@dataclass
class CostTracker:
    """Tracks LLM costs for a processing run."""

    calls: List[LLMCall] = field(default_factory=list)
    ceiling_gbp: float = 5.0

    @property
    def total_usd(self) -> float:
        return sum(c.cost_usd for c in self.calls)

    @property
    def total_gbp(self) -> float:
        return self.total_usd * USD_TO_GBP

    @property
    def remaining_gbp(self) -> float:
        return max(0, self.ceiling_gbp - self.total_gbp)

    @property
    def ceiling_reached(self) -> bool:
        return self.total_gbp >= self.ceiling_gbp

    def add_call(
        self,
        operation: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> LLMCall:
        """Record an LLM call and its cost."""
        costs = LLM_COSTS.get(model, {"input": 0.01, "output": 0.03})
        cost_usd = (
            (input_tokens / 1000) * costs["input"] +
            (output_tokens / 1000) * costs["output"]
        )

        call = LLMCall(
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
        self.calls.append(call)
        return call

    def get_summary(self) -> dict:
        return {
            "total_calls": len(self.calls),
            "total_usd": round(self.total_usd, 4),
            "total_gbp": round(self.total_gbp, 4),
            "ceiling_gbp": self.ceiling_gbp,
            "remaining_gbp": round(self.remaining_gbp, 4),
            "ceiling_reached": self.ceiling_reached,
            "by_operation": self._group_by_operation(),
        }

    def _group_by_operation(self) -> dict:
        ops = {}
        for call in self.calls:
            if call.operation not in ops:
                ops[call.operation] = {"count": 0, "cost_usd": 0}
            ops[call.operation]["count"] += 1
            ops[call.operation]["cost_usd"] += call.cost_usd
        return ops
```

**Tests**:
- [ ] Unit test: `test_cost_tracker_add_call()`
- [ ] Unit test: `test_cost_tracker_total_calculations()`
- [ ] Unit test: `test_cost_tracker_ceiling_reached()`
- [ ] Unit test: `test_cost_tracker_gbp_conversion()`
- [ ] Unit test: `test_cost_tracker_summary()`

---

### Phase 5: Agentic Workflow Orchestrator

#### Task 5.1: Create Validation Orchestrator
**File**: `src/clip_video/validation/orchestrator.py`

```python
from dataclasses import dataclass
from typing import List, Optional, Tuple
from pathlib import Path

from clip_video.validation.checker import QualityCriteriaChecker
from clip_video.validation.criteria import ClipValidation, ValidationResult
from clip_video.review.queue import ReviewQueue, RejectedClip
from clip_video.costs.tracker import CostTracker
from clip_video.llm.base import HighlightSegment

@dataclass
class ValidationPass:
    """Result of a validation pass on all clips."""

    approved: List[HighlightSegment]
    rejected: List[Tuple[HighlightSegment, ClipValidation]]
    exhausted: bool = False
    cost_ceiling_hit: bool = False

class AgenticValidator:
    """Orchestrates multi-pass validation workflow."""

    def __init__(
        self,
        config: HighlightsConfig,
        llm,  # LLM provider instance
        transcript_text: str,
        brand_context: dict,
        project_root: Path,
    ):
        self.config = config
        self.llm = llm
        self.transcript_text = transcript_text
        self.brand_context = brand_context

        self.checker = QualityCriteriaChecker(config, brand_context)
        self.review_queue = ReviewQueue(project_root / "review")
        self.cost_tracker = CostTracker(ceiling_gbp=config.cost_ceiling_gbp)

        self.approved_clips: List[HighlightSegment] = []
        self.used_segments: List[Tuple[float, float]] = []

    def validate_and_refine(
        self,
        initial_segments: List[HighlightSegment],
    ) -> Tuple[List[HighlightSegment], dict]:
        """
        Main entry point: validate clips and find replacements.

        Returns:
            Tuple of (approved_clips, run_summary)
        """
        candidates = list(initial_segments)
        iteration = 0

        while len(self.approved_clips) < self.config.target_clips:
            iteration += 1

            # Check termination conditions
            if self.cost_tracker.ceiling_reached:
                break

            if not candidates:
                # Try to find more candidates
                candidates = self._find_replacement_candidates(
                    count=self.config.target_clips - len(self.approved_clips)
                )
                if not candidates:
                    break  # Transcript exhausted

            # Validate next candidate
            candidate = candidates.pop(0)
            validation = self._validate_clip(candidate)

            if validation.passed:
                self._approve_clip(candidate)
            else:
                self._reject_clip(candidate, validation)

        return self.approved_clips, self._get_run_summary()

    def _validate_clip(self, segment: HighlightSegment) -> ClipValidation:
        """Run full validation on a clip."""
        # First: deterministic checks
        deterministic_results = self.checker.run_deterministic_checks(
            segment.start_time, segment.end_time
        )

        # Early exit if deterministic checks fail
        if any(r.result == ValidationResult.FAIL for r in deterministic_results):
            return ClipValidation(
                clip_id=f"clip_{segment.start_time}",
                start_time=segment.start_time,
                end_time=segment.end_time,
                overall_result=ValidationResult.FAIL,
                criteria_results=deterministic_results,
                transcript_segment=segment.text or "",
            )

        # Second: LLM validation
        llm_result = self.llm.validate_clip(
            transcript_segment=self._get_segment_text(segment),
            start_time=segment.start_time,
            end_time=segment.end_time,
            clip_summary=segment.summary,
            brand_context=self.brand_context,
        )

        # Track cost
        # (Assume LLM returns token counts)

        # Combine results
        all_results = deterministic_results + self._llm_to_criteria(llm_result)
        overall = (
            ValidationResult.PASS
            if llm_result.is_valid
            else ValidationResult.FAIL
        )

        return ClipValidation(
            clip_id=f"clip_{segment.start_time}",
            start_time=segment.start_time,
            end_time=segment.end_time,
            overall_result=overall,
            criteria_results=all_results,
            transcript_segment=self._get_segment_text(segment),
        )

    def _approve_clip(self, segment: HighlightSegment) -> None:
        """Mark a clip as approved."""
        self.approved_clips.append(segment)
        self.checker.mark_segment_used(segment.start_time, segment.end_time)
        self.used_segments.append((segment.start_time, segment.end_time))

    def _reject_clip(
        self,
        segment: HighlightSegment,
        validation: ClipValidation
    ) -> None:
        """Add rejected clip to review queue."""
        rejected = RejectedClip(
            clip_id=validation.clip_id,
            video_path="",  # Set by caller
            start_time=segment.start_time,
            end_time=segment.end_time,
            transcript_segment=validation.transcript_segment,
            rejection_reasons=validation.failure_reasons,
            validation_details={
                "criteria": [r.__dict__ for r in validation.criteria_results]
            },
        )
        self.review_queue.add(rejected)

    def _find_replacement_candidates(
        self,
        count: int
    ) -> List[HighlightSegment]:
        """Ask LLM to find replacement candidates."""
        if self.cost_tracker.ceiling_reached:
            return []

        # Build exclusion list
        exclusions = self.used_segments.copy()

        # Ask LLM for more candidates
        result = self.llm.find_additional_highlights(
            transcript_text=self.transcript_text,
            exclude_ranges=exclusions,
            count=count,
            brand_context=self.brand_context,
        )

        return result.segments if result else []

    def _get_segment_text(self, segment: HighlightSegment) -> str:
        """Extract transcript text for a segment."""
        # Implementation depends on transcript format
        return segment.text or ""

    def _llm_to_criteria(self, llm_result) -> List[CriterionResult]:
        """Convert LLM validation result to criterion results."""
        results = []

        checks = [
            ("sentence_boundaries", llm_result.sentence_boundaries_ok, "D002"),
            ("topic_complete", llm_result.topic_complete, "D003"),
            ("standalone_valid", llm_result.standalone_valid, "D005"),
            ("brand_relevant", llm_result.brand_relevant, "D006"),
            ("transcript_aligned", llm_result.transcript_aligned, "D007"),
        ]

        for name, passed, ref in checks:
            results.append(CriterionResult(
                criterion=name,
                result=ValidationResult.PASS if passed else ValidationResult.FAIL,
                reason=f"{ref}: {'OK' if passed else 'Failed'}",
            ))

        # Hook is optional (D004)
        results.append(CriterionResult(
            criterion="has_hook",
            result=ValidationResult.PASS,  # Never fails
            reason=f"D004: Hook {'present' if llm_result.has_hook else 'not present (optional)'}",
        ))

        return results

    def _get_run_summary(self) -> dict:
        """Get summary of the validation run."""
        return {
            "approved_clips": len(self.approved_clips),
            "target_clips": self.config.target_clips,
            "min_acceptable": self.config.min_acceptable_clips,
            "success": len(self.approved_clips) >= self.config.min_acceptable_clips,
            "review_queue": self.review_queue.get_summary(),
            "costs": self.cost_tracker.get_summary(),
            "used_segments": self.used_segments,
        }
```

**Tests**:
- [ ] Unit test: `test_validator_approve_clip()`
- [ ] Unit test: `test_validator_reject_clip_adds_to_queue()`
- [ ] Unit test: `test_validator_marks_segments_used()`
- [ ] Unit test: `test_validator_stops_at_cost_ceiling()`
- [ ] Unit test: `test_validator_stops_when_exhausted()`
- [ ] Integration test: `test_full_validation_pass_all_approved()`
- [ ] Integration test: `test_full_validation_with_rejections()`
- [ ] Integration test: `test_full_validation_with_replacements()`

---

### Phase 6: Integration with Existing Workflow

#### Task 6.1: Update HighlightsProcessor
**File**: `src/clip_video/modes/highlights.py`
**Change**: Add validation pass after analysis

```python
def process(
    self,
    project: HighlightsProject,
    transcript_segments: list[TranscriptionSegment] | None = None,
    skip_captions: bool = False,
) -> HighlightsProject:
    """Run the full highlights workflow."""
    # Step 1: Ensure we have a transcript
    self.transcribe(project, transcript_segments)

    # Step 2: Analyze for highlights
    analysis = self.analyze(project)

    # Step 3: NEW - Validate and refine clips
    if self.config.enable_validation_pass:
        approved_clips, run_summary = self._validate_clips(
            project, analysis.segments
        )
        # Update analysis with only approved segments
        analysis.segments = [
            seg for seg in analysis.segments
            if seg in approved_clips
        ]
        project.analysis = analysis
        project.save()

        # Log summary
        self._log_validation_summary(run_summary)

    # Step 4: Extract clips (only approved ones now)
    self.extract_clips(project)

    # Step 5: Convert to portrait
    self.convert_to_portrait(project)

    # Step 6: Burn captions (unless skipped)
    if not skip_captions:
        self.burn_captions(project, transcript_segments)

    # Step 7: Generate metadata
    self.generate_metadata(project)

    return project

def _validate_clips(
    self,
    project: HighlightsProject,
    segments: List[HighlightSegment],
) -> Tuple[List[HighlightSegment], dict]:
    """Run agentic validation on clip candidates."""
    from clip_video.validation.orchestrator import AgenticValidator

    validator = AgenticValidator(
        config=self.config,
        llm=self.llm,
        transcript_text=project.transcript_text,
        brand_context=self._get_brand_context(project),
        project_root=project.project_root,
    )

    return validator.validate_and_refine(segments)
```

**Tests**:
- [ ] Integration test: `test_process_with_validation_enabled()`
- [ ] Integration test: `test_process_with_validation_disabled()`
- [ ] Integration test: `test_process_respects_min_acceptable()`

---

#### Task 6.2: Add LLM Methods
**File**: `src/clip_video/llm/base.py`

Add abstract methods:
```python
@abstractmethod
def validate_clip(self, ...) -> ClipValidationResponse:
    """Validate a clip against quality criteria."""
    pass

@abstractmethod
def find_additional_highlights(self, ...) -> HighlightAnalysis:
    """Find additional highlight candidates, excluding given ranges."""
    pass
```

**File**: `src/clip_video/llm/claude.py`

Implement with appropriate prompts for:
- Validation prompt (checks D002-D007)
- Replacement search prompt (finds new candidates)

**Tests**:
- [ ] Mock test: `test_claude_validate_clip_prompt_structure()`
- [ ] Mock test: `test_claude_find_additional_prompt_includes_exclusions()`

---

### Phase 7: CLI Updates

#### Task 7.1: Add Review Queue Commands
**File**: `src/clip_video/cli.py`

```python
@app.command()
def review_list(
    brand: str = typer.Argument(..., help="Brand name"),
    project: str = typer.Argument(..., help="Project name"),
):
    """List clips in the review queue."""
    # Implementation

@app.command()
def review_approve(
    brand: str = typer.Argument(..., help="Brand name"),
    project: str = typer.Argument(..., help="Project name"),
    clip_id: str = typer.Argument(..., help="Clip ID to approve"),
):
    """Force-approve a clip from the review queue."""
    # Implementation

@app.command()
def preview(
    video: Path = typer.Argument(..., help="Video file"),
    start: float = typer.Option(..., help="Start time in seconds"),
    end: float = typer.Option(..., help="End time in seconds"),
):
    """Preview a clip segment (opens in default player)."""
    # Implementation
```

**Tests**:
- [ ] CLI test: `test_review_list_command()`
- [ ] CLI test: `test_review_approve_command()`
- [ ] CLI test: `test_preview_command()`

---

## Quality Assurance Checklist

### Unit Test Coverage
- [ ] All new dataclasses have serialization roundtrip tests
- [ ] All validation criteria have pass/fail test cases
- [ ] Cost calculations tested with known values
- [ ] Review queue file operations tested

### Integration Test Coverage
- [ ] Full workflow with all clips passing
- [ ] Full workflow with some rejections and replacements
- [ ] Workflow stops at cost ceiling
- [ ] Workflow handles transcript exhaustion
- [ ] Review queue persists and loads correctly

### Manual QA Scenarios
- [ ] Process a 5-minute lightning talk (expect early stop)
- [ ] Process a 45-minute talk (expect full 5 clips)
- [ ] Verify review queue JSON files are human-readable
- [ ] Verify preview command works on Windows
- [ ] Test cost tracking accuracy against actual API usage

---

## Acceptance Criteria Summary

| ID | Criterion | Test Type |
|----|-----------|-----------|
| D001 | Duration 30-120s | Unit |
| D002 | Complete sentences | LLM validation |
| D003 | Topic coverage | LLM validation |
| D004 | Optional hooks | LLM validation |
| D005 | Standalone validity | LLM validation |
| D006 | Brand relevance | LLM validation |
| D007 | Transcript alignment | LLM validation |
| D008 | Non-overlapping | Unit |
| D018 | Max 3 replacements | Integration |
| D019 | Target 5, min 3 | Integration |
| D020 | Early stop | Integration |
| D021 | Cost ceiling £5 | Integration |
| D022 | File-based review | Unit |
| D023 | Review contents | Unit |

---

## Dependencies

No new external dependencies required. Uses existing:
- `dataclasses` (stdlib)
- `json` (stdlib)
- `pathlib` (stdlib)
- Existing LLM integrations (Claude/OpenAI)

---

## File Structure

```
src/clip_video/
├── validation/
│   ├── __init__.py
│   ├── criteria.py      # Task 2.1
│   ├── checker.py       # Task 2.2
│   └── orchestrator.py  # Task 5.1
├── review/
│   ├── __init__.py
│   └── queue.py         # Task 3.1
├── costs/
│   ├── __init__.py
│   └── tracker.py       # Task 4.1
├── modes/
│   └── highlights.py    # Tasks 1.1, 1.2, 6.1
├── llm/
│   ├── base.py          # Task 2.3
│   ├── claude.py        # Task 6.2
│   └── openai.py        # Task 6.2
└── cli.py               # Task 7.1

tests/
├── test_validation_criteria.py
├── test_validation_checker.py
├── test_validation_orchestrator.py
├── test_review_queue.py
├── test_cost_tracker.py
└── test_highlights_integration.py
```
