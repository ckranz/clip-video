"""Validation orchestrator for multi-pass agentic workflow.

Coordinates the validate-and-refine loop:
1. Validate candidate clips against quality criteria
2. Approve passing clips, reject failing ones
3. Find replacement candidates for rejected clips
4. Repeat until target reached or resources exhausted

D009-D013: Termination conditions
D014-D016: Cost tracking
D017-D023: Review queue integration
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from clip_video.validation.checker import QualityCriteriaChecker
from clip_video.validation.criteria import (
    ClipValidation,
    CriterionResult,
    ValidationResult,
)
from clip_video.review.queue import ReviewQueue, RejectedClip
from clip_video.costs import CostTracker
from clip_video.llm.base import (
    HighlightSegment,
    ClipValidationRequest,
    ClipValidationResponse,
    LLMProvider,
)

if TYPE_CHECKING:
    from clip_video.modes.highlights import HighlightsConfig


@dataclass
class ValidationPass:
    """Result of a validation pass on clips.

    Attributes:
        approved: Clips that passed validation
        rejected: Clips that failed with their validation results
        exhausted: Whether transcript has been exhausted (no more candidates)
        cost_ceiling_hit: Whether cost ceiling stopped processing
        iterations: Number of validation iterations performed
    """

    approved: list[HighlightSegment]
    rejected: list[tuple[HighlightSegment, ClipValidation]]
    exhausted: bool = False
    cost_ceiling_hit: bool = False
    iterations: int = 0


@dataclass
class RunSummary:
    """Summary of a complete validation run.

    Attributes:
        approved_count: Number of approved clips
        rejected_count: Number of rejected clips
        target_clips: Target number of clips
        min_acceptable: Minimum acceptable clips
        target_met: Whether target was met
        minimum_met: Whether minimum was met
        iterations: Total validation iterations
        total_cost_usd: Total cost in USD
        total_cost_gbp: Total cost in GBP
        cost_ceiling_gbp: Cost ceiling in GBP
        cost_ceiling_hit: Whether ceiling was reached
        transcript_exhausted: Whether transcript was exhausted
        termination_reason: Why processing stopped
    """

    approved_count: int
    rejected_count: int
    target_clips: int
    min_acceptable: int
    target_met: bool
    minimum_met: bool
    iterations: int
    total_cost_usd: float
    total_cost_gbp: float
    cost_ceiling_gbp: float | None
    cost_ceiling_hit: bool
    transcript_exhausted: bool
    termination_reason: str


class AgenticValidator:
    """Orchestrates multi-pass validation workflow.

    Manages the iterative process of:
    1. Validating clips (deterministic + LLM checks)
    2. Approving valid clips
    3. Rejecting invalid clips to review queue
    4. Finding replacement candidates
    5. Terminating when conditions are met

    Attributes:
        config: Highlights configuration
        llm: LLM provider for validation and replacement search
        transcript_text: Full transcript text
        brand_context: Brand/channel context for relevance checks
        video_path: Path to the source video (for review queue)
    """

    def __init__(
        self,
        config: "HighlightsConfig",
        llm: LLMProvider,
        transcript_text: str,
        brand_context: dict[str, Any] | None = None,
        project_root: Path | None = None,
        video_path: str = "",
        cost_callback: Any = None,
    ):
        """Initialize the validator.

        Args:
            config: Highlights configuration
            llm: LLM provider instance
            transcript_text: Full transcript text
            brand_context: Optional brand/channel context
            project_root: Root directory for review queue
            video_path: Path to source video
            cost_callback: Optional callback for cost updates
        """
        self.config = config
        self.llm = llm
        self.transcript_text = transcript_text
        self.brand_context = brand_context or {}
        self.video_path = video_path

        # Initialize components
        self.checker = QualityCriteriaChecker(config, self.brand_context)

        review_dir = (project_root or Path.cwd()) / "review"
        self.review_queue = ReviewQueue(review_dir)

        self.cost_tracker = CostTracker(
            cost_callback=cost_callback,
            ceiling_gbp=config.cost_ceiling_gbp if hasattr(config, "cost_ceiling_gbp") else None,
        )

        # State tracking
        self.approved_clips: list[HighlightSegment] = []
        self.used_segments: list[tuple[float, float]] = []
        self.rejection_counts: dict[str, int] = {}  # clip_id -> attempts
        self.iterations = 0

    def validate_and_refine(
        self,
        initial_segments: list[HighlightSegment],
    ) -> tuple[list[HighlightSegment], RunSummary]:
        """Main entry point: validate clips and find replacements.

        Implements the multi-pass workflow:
        1. Validate initial candidates
        2. Approve passing clips
        3. Find replacements for rejected clips
        4. Repeat until target met or terminated

        Args:
            initial_segments: Initial highlight candidates from LLM

        Returns:
            Tuple of (approved_clips, run_summary)
        """
        candidates = list(initial_segments)
        rejected_segments: list[tuple[HighlightSegment, ClipValidation]] = []

        while len(self.approved_clips) < self.config.target_clips:
            self.iterations += 1

            # Check termination conditions
            if self.cost_tracker.ceiling_reached:
                break

            if not candidates:
                # Try to find more candidates
                needed = self.config.target_clips - len(self.approved_clips)
                candidates = self._find_replacement_candidates(count=needed)
                if not candidates:
                    break  # Transcript exhausted

            # Validate next candidate
            candidate = candidates.pop(0)
            validation = self._validate_clip(candidate)

            if validation.passed:
                self._approve_clip(candidate)
            else:
                rejected_segments.append((candidate, validation))
                self._reject_clip(candidate, validation)

                # Check max replacement attempts per clip
                clip_id = self._get_clip_id(candidate)
                if self.rejection_counts.get(clip_id, 0) >= self.config.max_replacement_attempts:
                    continue  # Don't try to replace this clip anymore

        return self.approved_clips, self._get_run_summary(rejected_segments)

    def _validate_clip(self, segment: HighlightSegment) -> ClipValidation:
        """Run full validation on a clip.

        First runs deterministic checks (duration, overlap).
        If those pass, runs LLM validation for semantic checks.

        Args:
            segment: The clip segment to validate

        Returns:
            ClipValidation with results of all checks
        """
        clip_id = self._get_clip_id(segment)
        transcript_segment = self._get_segment_text(segment)

        # First: deterministic checks (fast, no LLM cost)
        deterministic_results = self.checker.run_deterministic_checks(
            segment.start_time, segment.end_time
        )

        # Early exit if deterministic checks fail
        if any(r.result == ValidationResult.FAIL for r in deterministic_results):
            return ClipValidation(
                clip_id=clip_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                overall_result=ValidationResult.FAIL,
                criteria_results=deterministic_results,
                transcript_segment=transcript_segment,
            )

        # Check if we can afford LLM validation
        if not self.cost_tracker.can_afford(0.01):  # ~$0.01 per validation
            # Skip LLM validation if at ceiling, approve with warning
            return ClipValidation(
                clip_id=clip_id,
                start_time=segment.start_time,
                end_time=segment.end_time,
                overall_result=ValidationResult.WARN,
                criteria_results=deterministic_results + [
                    CriterionResult(
                        criterion="llm_validation",
                        result=ValidationResult.WARN,
                        reason="Cost ceiling reached, LLM validation skipped",
                    )
                ],
                transcript_segment=transcript_segment,
            )

        # Second: LLM validation (semantic checks)
        request = ClipValidationRequest(
            clip_id=clip_id,
            transcript_segment=transcript_segment,
            start_time=segment.start_time,
            end_time=segment.end_time,
            clip_summary=segment.summary,
            brand_context=self.brand_context,
            full_transcript_context=self.transcript_text[:2000] if self.transcript_text else None,
        )

        llm_response = self.llm.validate_clip(request)

        # Track cost (estimate based on tokens)
        estimated_cost = (llm_response.tokens_used / 1_000_000) * 3.0  # ~$3/M tokens
        self.cost_tracker.add_llm_cost(
            service="validation",
            amount=estimated_cost,
            tokens_used=llm_response.tokens_used,
            description=f"Clip validation: {clip_id}",
        )

        # Convert LLM response to criteria results
        llm_criteria = self._llm_to_criteria(llm_response)

        # Combine results
        all_results = deterministic_results + llm_criteria
        overall = (
            ValidationResult.PASS
            if llm_response.is_valid
            else ValidationResult.FAIL
        )

        return ClipValidation(
            clip_id=clip_id,
            start_time=segment.start_time,
            end_time=segment.end_time,
            overall_result=overall,
            criteria_results=all_results,
            transcript_segment=transcript_segment,
        )

    def _approve_clip(self, segment: HighlightSegment) -> None:
        """Mark a clip as approved.

        Adds to approved list and marks time range as used.

        Args:
            segment: The approved clip segment
        """
        self.approved_clips.append(segment)
        self.checker.mark_segment_used(segment.start_time, segment.end_time)
        self.used_segments.append((segment.start_time, segment.end_time))

    def _reject_clip(
        self,
        segment: HighlightSegment,
        validation: ClipValidation,
    ) -> None:
        """Add rejected clip to review queue.

        Args:
            segment: The rejected clip segment
            validation: Validation results showing why it failed
        """
        clip_id = self._get_clip_id(segment)

        # Track rejection count
        self.rejection_counts[clip_id] = self.rejection_counts.get(clip_id, 0) + 1

        # Add to review queue
        rejected = RejectedClip(
            clip_id=validation.clip_id,
            video_path=self.video_path,
            start_time=segment.start_time,
            end_time=segment.end_time,
            transcript_segment=validation.transcript_segment,
            rejection_reasons=validation.failure_reasons,
            validation_details={
                "criteria": [
                    {
                        "criterion": r.criterion,
                        "result": r.result.value,
                        "reason": r.reason,
                    }
                    for r in validation.criteria_results
                ]
            },
            replacement_attempts=self.rejection_counts[clip_id],
        )
        self.review_queue.add(rejected)

    def _find_replacement_candidates(
        self,
        count: int,
    ) -> list[HighlightSegment]:
        """Ask LLM to find replacement candidates.

        Args:
            count: Number of replacements needed

        Returns:
            List of new highlight segment candidates
        """
        if self.cost_tracker.ceiling_reached:
            return []

        # Get rejected clips for context
        rejected_clips = [
            ClipValidationResponse(
                clip_id=f"rejected_{i}",
                is_valid=False,
                issues=list(self.review_queue.list_all()[i].rejection_reasons) if i < len(self.review_queue.list_all()) else [],
            )
            for i in range(min(5, self.review_queue.count()))  # Limit context
        ]

        try:
            segments = self.llm.find_replacement_clips(
                rejected_clips=rejected_clips,
                transcript_text=self.transcript_text,
                used_segments=self.used_segments,
                target_count=count,
            )

            # Estimate cost for replacement search
            estimated_cost = 0.02  # ~$0.02 per replacement search
            self.cost_tracker.add_llm_cost(
                service="replacement_search",
                amount=estimated_cost,
                tokens_used=1000,
                description="Finding replacement clips",
            )

            return segments
        except Exception:
            return []

    def _get_segment_text(self, segment: HighlightSegment) -> str:
        """Extract transcript text for a segment.

        Uses the summary as fallback if no direct text available.

        Args:
            segment: The clip segment

        Returns:
            Transcript text for the segment
        """
        # For now, use summary as text (full transcript extraction would need timestamps)
        return segment.summary or f"Clip from {segment.start_time:.1f}s to {segment.end_time:.1f}s"

    def _get_clip_id(self, segment: HighlightSegment) -> str:
        """Generate a unique clip ID from segment times.

        Args:
            segment: The clip segment

        Returns:
            Unique clip identifier
        """
        return f"clip_{segment.start_time:.1f}_{segment.end_time:.1f}"

    def _llm_to_criteria(
        self,
        response: ClipValidationResponse,
    ) -> list[CriterionResult]:
        """Convert LLM validation response to CriterionResult list.

        Args:
            response: LLM validation response

        Returns:
            List of CriterionResult objects
        """
        results = []

        # Sentence boundaries (D002)
        results.append(CriterionResult(
            criterion="sentence_boundaries",
            result=ValidationResult.PASS if response.sentence_boundaries_ok else ValidationResult.FAIL,
            reason="Complete sentences" if response.sentence_boundaries_ok else "Incomplete sentence boundaries",
        ))

        # Topic complete (D003)
        results.append(CriterionResult(
            criterion="topic_complete",
            result=ValidationResult.PASS if response.topic_complete else ValidationResult.FAIL,
            reason="Topic complete" if response.topic_complete else "Topic incomplete or cut off",
        ))

        # Has hook (D004) - optional, so WARN not FAIL
        results.append(CriterionResult(
            criterion="has_hook",
            result=ValidationResult.PASS if response.has_hook else ValidationResult.WARN,
            reason="Has engaging hook" if response.has_hook else "No strong hook (optional)",
        ))

        # Standalone valid (D005)
        results.append(CriterionResult(
            criterion="standalone_valid",
            result=ValidationResult.PASS if response.standalone_valid else ValidationResult.FAIL,
            reason="Standalone valid" if response.standalone_valid else "Requires additional context",
        ))

        # Brand relevant (D006)
        results.append(CriterionResult(
            criterion="brand_relevant",
            result=ValidationResult.PASS if response.brand_relevant else ValidationResult.FAIL,
            reason="Brand relevant" if response.brand_relevant else "Not aligned with brand",
        ))

        # Transcript aligned (D007)
        results.append(CriterionResult(
            criterion="transcript_aligned",
            result=ValidationResult.PASS if response.transcript_aligned else ValidationResult.FAIL,
            reason="Timestamps aligned" if response.transcript_aligned else "Timestamp alignment issues",
        ))

        return results

    def _get_run_summary(
        self,
        rejected_segments: list[tuple[HighlightSegment, ClipValidation]],
    ) -> RunSummary:
        """Generate summary of the validation run.

        Args:
            rejected_segments: List of rejected clips with validations

        Returns:
            RunSummary with statistics
        """
        approved_count = len(self.approved_clips)
        rejected_count = len(rejected_segments)
        target_met = approved_count >= self.config.target_clips
        minimum_met = approved_count >= self.config.min_acceptable_clips

        # Determine termination reason
        if target_met:
            reason = "Target clip count reached"
        elif self.cost_tracker.ceiling_reached:
            reason = "Cost ceiling reached"
        elif rejected_count > 0 and approved_count < self.config.min_acceptable_clips:
            reason = "Transcript exhausted without meeting minimum"
        else:
            reason = "Transcript exhausted"

        return RunSummary(
            approved_count=approved_count,
            rejected_count=rejected_count,
            target_clips=self.config.target_clips,
            min_acceptable=self.config.min_acceptable_clips,
            target_met=target_met,
            minimum_met=minimum_met,
            iterations=self.iterations,
            total_cost_usd=self.cost_tracker.get_total(),
            total_cost_gbp=self.cost_tracker.get_total_gbp(),
            cost_ceiling_gbp=self.cost_tracker.ceiling_gbp,
            cost_ceiling_hit=self.cost_tracker.ceiling_reached,
            transcript_exhausted=not target_met and not self.cost_tracker.ceiling_reached,
            termination_reason=reason,
        )
