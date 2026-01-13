"""Social copy generation for clip promotion.

Generates styled titles, descriptions, and captions for social media
based on brand-specific voice and style settings.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from clip_video.config import SocialCopyStyle


@dataclass
class SocialCopy:
    """Generated social media copy for a clip.

    Attributes:
        hook: Short attention-grabbing title/hook
        description: Longer description for the post
        hashtags: List of relevant hashtags
        topics: Key topics from the clip
        speaker_name: Name of the speaker (if known)
        talk_title: Title of the original talk
    """

    hook: str
    description: str
    hashtags: list[str]
    topics: list[str]
    speaker_name: str = ""
    talk_title: str = ""

    def to_markdown(self, video_filename: str = "") -> str:
        """Format as markdown for easy copy/paste.

        Args:
            video_filename: Optional filename for reference

        Returns:
            Markdown formatted string
        """
        lines = []

        if self.speaker_name or self.talk_title:
            lines.append(f"# {self.speaker_name}")
            if self.talk_title:
                lines.append(f"*{self.talk_title}*")
            lines.append("")

        lines.append("## Hook")
        lines.append(self.hook)
        lines.append("")

        lines.append("## Description")
        lines.append(self.description)
        lines.append("")

        if self.hashtags:
            lines.append("## Hashtags")
            lines.append(" ".join(f"#{tag}" for tag in self.hashtags))
            lines.append("")

        if self.topics:
            lines.append("## Topics")
            lines.append(", ".join(self.topics))
            lines.append("")

        if video_filename:
            lines.append("---")
            lines.append(f"*Video: {video_filename}*")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hook": self.hook,
            "description": self.description,
            "hashtags": self.hashtags,
            "topics": self.topics,
            "speaker_name": self.speaker_name,
            "talk_title": self.talk_title,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SocialCopy":
        """Create from dictionary."""
        return cls(
            hook=data.get("hook", ""),
            description=data.get("description", ""),
            hashtags=data.get("hashtags", []),
            topics=data.get("topics", []),
            speaker_name=data.get("speaker_name", ""),
            talk_title=data.get("talk_title", ""),
        )


def build_social_copy_prompt(
    style: "SocialCopyStyle",
    clip_summary: str,
    clip_transcript: str,
    speaker_name: str = "",
    talk_title: str = "",
    topics: list[str] | None = None,
) -> tuple[str, str]:
    """Build system and user prompts for social copy generation.

    Args:
        style: Social copy style settings
        clip_summary: Brief summary of the clip content
        clip_transcript: Full transcript text of the clip
        speaker_name: Name of the speaker
        talk_title: Title of the original talk
        topics: Key topics from the clip

    Returns:
        Tuple of (system_prompt, user_prompt)
    """
    # Build locale guidance
    locale_guidance = ""
    if style.locale == "british":
        locale_guidance = """
Use British English spelling and expressions:
- "colour" not "color", "behaviour" not "behavior"
- "whilst" is acceptable, "reckon" is fine
- Avoid Americanisms like "reach out", "touch base"
"""
    elif style.locale == "american":
        locale_guidance = """
Use American English spelling:
- "color" not "colour", "behavior" not "behaviour"
"""

    # Build tone guidance
    tone_guidance = {
        "informative": "Be direct and informative. Focus on what the viewer will learn.",
        "casual": "Keep it relaxed and conversational. Like sharing something interesting with a colleague.",
        "enthusiastic": "Show genuine excitement about the content without being over-the-top.",
        "professional": "Maintain a polished, professional tone suitable for business audiences.",
    }.get(style.tone, "Be clear and engaging.")

    # Build avoidance list
    avoid_section = ""
    if style.avoid_phrases:
        avoid_section = f"""
AVOID these phrases and patterns (they sound generic or AI-generated):
{chr(10).join(f'- "{phrase}"' for phrase in style.avoid_phrases)}
"""

    # Build preferred phrases
    preferred_section = ""
    if style.preferred_phrases:
        preferred_section = f"""
Consider using these brand-appropriate phrases where natural:
{chr(10).join(f'- "{phrase}"' for phrase in style.preferred_phrases)}
"""

    # Custom voice description
    voice_section = ""
    if style.voice_description:
        voice_section = f"""
Brand voice guidance:
{style.voice_description}
"""

    # Custom prompt additions
    custom_section = ""
    if style.custom_prompt:
        custom_section = f"""
Additional guidance:
{style.custom_prompt}
"""

    system_prompt = f"""You are writing social media copy to promote a video clip from a conference talk or presentation.

Your job is to write a short, engaging hook and description that will make people want to watch the clip.

{tone_guidance}

{locale_guidance}
{avoid_section}
{preferred_section}
{voice_section}
{custom_section}

Key principles:
1. Lead with the value or insight - what will viewers learn?
2. Be specific, not vague - mention concrete details from the content
3. Keep it genuine - don't oversell or use hype language
4. The hook should work as a standalone scroll-stopper
5. The description provides a bit more context

Output format:
Return a JSON object with these fields:
- "hook": Short attention-grabbing line (max {style.max_hook_length} chars)
- "description": Slightly longer description (max {style.max_description_length} chars)
- "hashtags": List of 3-5 relevant hashtags (without # symbol)

Return ONLY the JSON object, no additional text."""

    # Build user prompt with context
    context_parts = []

    if speaker_name:
        context_parts.append(f"Speaker: {speaker_name}")
    if talk_title:
        context_parts.append(f"Talk: {talk_title}")
    if topics:
        context_parts.append(f"Topics: {', '.join(topics)}")

    context_section = "\n".join(context_parts) if context_parts else ""

    user_prompt = f"""Write social media copy for this clip.

{context_section}

Summary: {clip_summary}

Transcript:
{clip_transcript}

Generate a hook and description that captures the key insight from this clip."""

    return system_prompt, user_prompt


def parse_social_copy_response(
    response_text: str,
    style: "SocialCopyStyle",
    speaker_name: str = "",
    talk_title: str = "",
    topics: list[str] | None = None,
) -> SocialCopy:
    """Parse LLM response into SocialCopy object.

    Args:
        response_text: Raw LLM response
        style: Social copy style settings
        speaker_name: Name of the speaker
        talk_title: Title of the talk
        topics: Key topics

    Returns:
        SocialCopy object
    """
    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                response_text = match.group(1)
        elif "```" in response_text:
            match = re.search(r"```\s*(.*?)\s*```", response_text, re.DOTALL)
            if match:
                response_text = match.group(1)

        data = json.loads(response_text.strip())
    except json.JSONDecodeError:
        # Fallback to extracting key parts
        data = {
            "hook": response_text[:style.max_hook_length],
            "description": "",
            "hashtags": [],
        }

    # Build hashtags list
    hashtags = data.get("hashtags", [])
    if style.include_hashtags and style.default_hashtags:
        # Add default hashtags, avoiding duplicates
        for tag in style.default_hashtags:
            tag_clean = tag.lstrip("#").lower()
            if tag_clean not in [h.lower() for h in hashtags]:
                hashtags.append(tag_clean)

    return SocialCopy(
        hook=data.get("hook", "")[:style.max_hook_length],
        description=data.get("description", "")[:style.max_description_length],
        hashtags=hashtags,
        topics=topics or [],
        speaker_name=speaker_name,
        talk_title=talk_title,
    )


def save_social_copy(
    social_copy: SocialCopy,
    output_path: Path,
    video_filename: str = "",
) -> Path:
    """Save social copy to markdown file.

    Args:
        social_copy: Generated social copy
        output_path: Path to save the markdown file
        video_filename: Optional video filename for reference

    Returns:
        Path to saved file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content = social_copy.to_markdown(video_filename)
    output_path.write_text(content, encoding="utf-8")

    return output_path
