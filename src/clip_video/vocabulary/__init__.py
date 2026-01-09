"""Vocabulary module for transcript correction.

Provides phonetic matching and fuzzy correction of technical terms
that are commonly mis-transcribed by speech-to-text systems.
"""

from clip_video.vocabulary.terms import VocabularyTerms
from clip_video.vocabulary.correction import TranscriptCorrector, CorrectionLog
from clip_video.vocabulary.phonetic import soundex, metaphone, levenshtein_distance

__all__ = [
    "VocabularyTerms",
    "TranscriptCorrector",
    "CorrectionLog",
    "soundex",
    "metaphone",
    "levenshtein_distance",
]
