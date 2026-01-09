"""Transcript storage, indexing, and import/export functionality.

This package provides:
- Enhanced transcript storage with JSON format
- Inverted index for fast word/phrase lookup across all transcripts
- Import/export for VTT and SRT subtitle formats
"""

from clip_video.transcript.import_export import (
    export_srt,
    export_vtt,
    import_srt,
    import_vtt,
)
from clip_video.transcript.index import (
    TranscriptIndex,
    WordOccurrence,
    PhraseMatch,
)
from clip_video.transcript.storage import (
    TranscriptStore,
)

__all__ = [
    # Storage
    "TranscriptStore",
    # Index
    "TranscriptIndex",
    "WordOccurrence",
    "PhraseMatch",
    # Import/Export
    "import_vtt",
    "import_srt",
    "export_vtt",
    "export_srt",
]
