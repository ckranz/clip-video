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
