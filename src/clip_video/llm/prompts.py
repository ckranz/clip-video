"""Prompt templates for LLM highlight detection.

Provides structured prompts for analyzing transcripts and extracting
highlight-worthy segments with appropriate formatting for social media.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HighlightPromptBuilder:
    """Builder for highlight detection prompts.

    Creates structured prompts for LLM analysis of transcripts,
    customizable for different content types and target platforms.

    Attributes:
        target_platform: Target social media platform
        max_clip_duration: Maximum clip duration in seconds
        min_clip_duration: Minimum clip duration in seconds
        content_type: Type of content (conference, interview, etc.)
    """

    target_platform: str = "youtube_shorts"
    max_clip_duration: int = 60
    min_clip_duration: int = 15
    content_type: str = "conference_talk"

    def build_system_prompt(self) -> str:
        """Build the system prompt for highlight analysis.

        Returns:
            System prompt string
        """
        return f"""You are an expert video editor specializing in creating engaging social media content from {self.content_type} recordings.

Your task is to analyze video transcripts and identify the most compelling segments that would make great standalone clips for {self.target_platform}.

When selecting highlights, prioritize:
1. Strong opening hooks that grab attention in the first 3 seconds
2. Complete, self-contained thoughts or stories
3. Quotable insights or surprising facts
4. Emotional moments or demonstrations of passion
5. Actionable advice or clear takeaways
6. Technical explanations that are accessible and interesting

Avoid selecting segments that:
- Start or end mid-sentence
- Require context from earlier in the talk
- Contain inside jokes or references only attendees would understand
- Are purely transitional ("moving on to the next topic...")
- Have poor audio quality indicators (like "[inaudible]")

Target clip duration: {self.min_clip_duration}-{self.max_clip_duration} seconds.

Always provide:
- Exact start and end timestamps from the transcript
- A brief summary of the segment content
- A suggested hook/caption for social media
- Why this segment would perform well as a clip"""

    def build_analysis_prompt(
        self,
        transcript_text: str,
        session_description: str | None = None,
        target_clips: int = 3,
    ) -> str:
        """Build the analysis prompt for a specific transcript.

        Args:
            transcript_text: Transcript text with timestamps
            session_description: Optional session description
            target_clips: Number of clips to identify

        Returns:
            Analysis prompt string
        """
        description_section = ""
        if session_description:
            description_section = f"""
## Session Description
{session_description}
"""

        return f"""Analyze the following transcript and identify the top {target_clips} highlight-worthy segments.
{description_section}
## Transcript
{transcript_text}

## Instructions
Identify exactly {target_clips} segments that would make compelling social media clips.

For each segment, provide:
1. `start_time`: Start timestamp in seconds (e.g., 125.5)
2. `end_time`: End timestamp in seconds (e.g., 180.2)
3. `summary`: 1-2 sentence summary of the segment content
4. `hook_text`: Attention-grabbing caption for social media (max 100 characters)
5. `reason`: Why this segment would perform well as a clip
6. `topics`: List of key topics/themes in this segment
7. `quality_score`: Estimated engagement score from 0.0 to 1.0

## Output Format
Respond with a JSON object in this exact format:
```json
{{
  "session_summary": "Brief overall summary of the talk",
  "main_topics": ["topic1", "topic2", "topic3"],
  "segments": [
    {{
      "start_time": 125.5,
      "end_time": 180.2,
      "summary": "Speaker explains the key insight about...",
      "hook_text": "This changed how I think about...",
      "reason": "Strong emotional hook and clear takeaway",
      "topics": ["topic1", "topic2"],
      "quality_score": 0.9
    }}
  ]
}}
```

Return ONLY the JSON object, no additional text or markdown formatting outside the JSON."""

    def build_refinement_prompt(
        self,
        transcript_text: str,
        previous_segments: list[dict],
        feedback: str,
    ) -> str:
        """Build a refinement prompt based on feedback.

        Args:
            transcript_text: Original transcript
            previous_segments: Previously identified segments
            feedback: User feedback for refinement

        Returns:
            Refinement prompt string
        """
        segments_json = "\n".join(
            f"  - {seg['start_time']}-{seg['end_time']}: {seg['summary']}"
            for seg in previous_segments
        )

        return f"""Based on the feedback, please refine your highlight selections.

## Previous Selections
{segments_json}

## Feedback
{feedback}

## Original Transcript
{transcript_text}

Please provide updated segment selections in the same JSON format, incorporating the feedback."""

    def build_validation_system_prompt(self) -> str:
        """Build system prompt for clip validation.

        Returns:
            System prompt for validation
        """
        return """You are an expert video editor and quality assurance specialist.

Your task is to validate clip segments for quality before they are extracted from longer videos.

You evaluate clips against these quality criteria:
1. SENTENCE_BOUNDARIES: Does the clip start and end at natural sentence boundaries? No mid-sentence cuts.
2. TOPIC_COMPLETE: Is the topic/thought complete? Not cut off mid-explanation or missing crucial context.
3. HAS_HOOK: Does it have an engaging opening that hooks viewers in the first few seconds? (Optional but valued)
4. STANDALONE_VALID: Does it make sense on its own without requiring context from the rest of the video?
5. BRAND_RELEVANT: Does it align with the channel's brand, topics, and audience expectations?
6. TRANSCRIPT_ALIGNED: Do the timestamps align correctly with the transcript content?

Be strict but fair in your assessments. Minor issues can be noted but shouldn't fail a clip.
Major issues that would confuse viewers or harm engagement should cause a failure."""

    def build_validation_prompt(
        self,
        transcript_segment: str,
        start_time: float,
        end_time: float,
        clip_summary: str,
        brand_context: dict | None = None,
        full_transcript: str | None = None,
    ) -> str:
        """Build prompt for validating a specific clip.

        Args:
            transcript_segment: The transcript text for this clip
            start_time: Clip start time in seconds
            end_time: Clip end time in seconds
            clip_summary: Brief summary of clip content
            brand_context: Optional brand/channel context
            full_transcript: Optional full transcript for context

        Returns:
            Validation prompt string
        """
        brand_section = ""
        if brand_context:
            brand_section = f"""
## Brand Context
- Channel: {brand_context.get('channel_name', 'Unknown')}
- Topics: {', '.join(brand_context.get('topics', []))}
- Tone: {brand_context.get('tone', 'professional')}
- Audience: {brand_context.get('audience', 'general')}
"""

        context_section = ""
        if full_transcript:
            context_section = f"""
## Full Transcript (for context)
{full_transcript[:2000]}...
"""

        return f"""Validate the following clip segment for quality.

## Clip Information
- Start Time: {start_time:.1f}s
- End Time: {end_time:.1f}s
- Duration: {end_time - start_time:.1f}s
- Summary: {clip_summary}

## Transcript Segment
{transcript_segment}
{brand_section}{context_section}
## Instructions
Evaluate this clip against ALL quality criteria:
1. sentence_boundaries_ok: Does it start/end at natural sentence boundaries?
2. topic_complete: Is the topic/thought complete and not cut off?
3. has_hook: Does it have an engaging opening? (optional, nice-to-have)
4. standalone_valid: Does it make sense without surrounding context?
5. brand_relevant: Does it fit the brand/channel?
6. transcript_aligned: Do timestamps match the content?

## Output Format
Respond with ONLY a JSON object:
```json
{{
  "is_valid": true,
  "sentence_boundaries_ok": true,
  "topic_complete": true,
  "has_hook": false,
  "standalone_valid": true,
  "brand_relevant": true,
  "transcript_aligned": true,
  "issues": [],
  "suggestions": [],
  "confidence": 0.9
}}
```

Set is_valid to false if any REQUIRED criteria fail (sentence_boundaries_ok, topic_complete, standalone_valid, brand_relevant, transcript_aligned). has_hook is optional.

If invalid, provide specific issues and actionable suggestions for improvement."""

    def build_replacement_system_prompt(self) -> str:
        """Build system prompt for finding replacement clips.

        Returns:
            System prompt for replacement search
        """
        return f"""You are an expert video editor specializing in creating engaging social media content.

Your task is to find replacement clip segments from a transcript when previous selections were rejected.

When selecting replacements:
1. Avoid any time ranges that are already used or blacklisted
2. Look for segments that meet ALL quality criteria (complete sentences, standalone, etc.)
3. Prioritize high-engagement potential and strong hooks
4. Match the target duration range of {self.min_clip_duration}-{self.max_clip_duration} seconds
5. Consider the reasons previous clips were rejected to avoid similar issues

Focus on finding NEW segments, not variations of rejected ones."""

    def build_replacement_prompt(
        self,
        transcript_text: str,
        rejected_clips: list[dict],
        used_segments: list[tuple[float, float]],
        target_count: int = 1,
    ) -> str:
        """Build prompt for finding replacement clips.

        Args:
            transcript_text: Full transcript
            rejected_clips: List of rejected clip info with reasons
            used_segments: Time ranges to avoid (start, end tuples)
            target_count: Number of replacements needed

        Returns:
            Replacement search prompt
        """
        rejected_section = "## Rejected Clips (avoid similar issues)\n"
        for clip in rejected_clips:
            rejected_section += f"""
- Time: {clip.get('start_time', 0):.1f}s - {clip.get('end_time', 0):.1f}s
  Issues: {', '.join(clip.get('issues', ['Unknown']))}
"""

        blacklist_section = "## Time Ranges to Avoid (already used or rejected)\n"
        for start, end in used_segments:
            blacklist_section += f"- {start:.1f}s - {end:.1f}s\n"

        return f"""Find {target_count} replacement clip segment(s) from the transcript below.

{rejected_section}
{blacklist_section}
## Transcript
{transcript_text}

## Requirements
- Duration: {self.min_clip_duration}-{self.max_clip_duration} seconds each
- Must NOT overlap with any blacklisted time ranges
- Must have complete sentence boundaries
- Must be standalone and engaging
- Must have strong potential for social media engagement

## Output Format
Respond with ONLY a JSON object:
```json
{{
  "segments": [
    {{
      "start_time": 125.5,
      "end_time": 180.2,
      "summary": "Speaker explains...",
      "hook_text": "Caption for social media",
      "reason": "Why this segment is good",
      "topics": ["topic1"],
      "quality_score": 0.85
    }}
  ]
}}
```

Find exactly {target_count} new segment(s). Return an empty segments array if no suitable replacements exist."""


# Pre-built prompt builders for common use cases
CONFERENCE_PROMPT = HighlightPromptBuilder(
    target_platform="youtube_shorts",
    max_clip_duration=60,
    min_clip_duration=15,
    content_type="technical conference talk",
)

INTERVIEW_PROMPT = HighlightPromptBuilder(
    target_platform="linkedin",
    max_clip_duration=90,
    min_clip_duration=30,
    content_type="interview or podcast",
)

TUTORIAL_PROMPT = HighlightPromptBuilder(
    target_platform="tiktok",
    max_clip_duration=60,
    min_clip_duration=15,
    content_type="tutorial or educational content",
)
