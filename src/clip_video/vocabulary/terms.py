"""Vocabulary terms management.

Handles loading, managing, and querying domain-specific vocabulary
for transcript correction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class VocabularyTerms:
    """Manages vocabulary terms for transcript correction.

    Vocabulary is stored as a mapping of canonical terms to their
    known mis-transcriptions (variants).

    Example:
        vocabulary = VocabularyTerms()
        vocabulary.add_term("kubernetes", ["cooper netties", "kuber nettis"])
        canonical = vocabulary.get_canonical("cooper netties")  # Returns "kubernetes"
    """

    def __init__(self, vocabulary: dict[str, list[str]] | None = None):
        """Initialize vocabulary terms.

        Args:
            vocabulary: Initial vocabulary mapping canonical -> variants
        """
        # Canonical term -> list of variant spellings
        self._vocabulary: dict[str, list[str]] = {}

        # Reverse lookup: variant (lowercase) -> canonical
        self._variant_lookup: dict[str, str] = {}

        if vocabulary:
            for canonical, variants in vocabulary.items():
                self.add_term(canonical, variants)

    def add_term(self, canonical: str, variants: list[str]) -> None:
        """Add a term with its known variants.

        Args:
            canonical: The correct spelling
            variants: List of known mis-transcriptions
        """
        canonical_lower = canonical.lower()
        self._vocabulary[canonical_lower] = [v.lower() for v in variants]

        for variant in variants:
            self._variant_lookup[variant.lower()] = canonical_lower

    def remove_term(self, canonical: str) -> bool:
        """Remove a term and its variants.

        Args:
            canonical: The canonical term to remove

        Returns:
            True if term was found and removed
        """
        canonical_lower = canonical.lower()

        if canonical_lower not in self._vocabulary:
            return False

        # Remove variant lookups
        for variant in self._vocabulary[canonical_lower]:
            self._variant_lookup.pop(variant.lower(), None)

        del self._vocabulary[canonical_lower]
        return True

    def add_variant(self, canonical: str, variant: str) -> bool:
        """Add a variant to an existing term.

        Args:
            canonical: The canonical term
            variant: New variant to add

        Returns:
            True if variant was added, False if term doesn't exist
        """
        canonical_lower = canonical.lower()
        variant_lower = variant.lower()

        if canonical_lower not in self._vocabulary:
            return False

        if variant_lower not in self._vocabulary[canonical_lower]:
            self._vocabulary[canonical_lower].append(variant_lower)
            self._variant_lookup[variant_lower] = canonical_lower

        return True

    def remove_variant(self, canonical: str, variant: str) -> bool:
        """Remove a variant from a term.

        Args:
            canonical: The canonical term
            variant: Variant to remove

        Returns:
            True if variant was found and removed
        """
        canonical_lower = canonical.lower()
        variant_lower = variant.lower()

        if canonical_lower not in self._vocabulary:
            return False

        if variant_lower in self._vocabulary[canonical_lower]:
            self._vocabulary[canonical_lower].remove(variant_lower)
            self._variant_lookup.pop(variant_lower, None)
            return True

        return False

    def get_canonical(self, word: str) -> str | None:
        """Get canonical spelling for a word if it's a known variant.

        Args:
            word: Word to look up

        Returns:
            Canonical spelling if word is a variant, None otherwise
        """
        return self._variant_lookup.get(word.lower())

    def get_variants(self, canonical: str) -> list[str]:
        """Get all variants for a canonical term.

        Args:
            canonical: The canonical term

        Returns:
            List of variants (empty if term not found)
        """
        return self._vocabulary.get(canonical.lower(), []).copy()

    def get_all_terms(self) -> list[str]:
        """Get all canonical terms.

        Returns:
            List of all canonical terms
        """
        return list(self._vocabulary.keys())

    def get_all_variants(self) -> list[str]:
        """Get all known variants.

        Returns:
            List of all variant spellings
        """
        return list(self._variant_lookup.keys())

    def is_canonical(self, word: str) -> bool:
        """Check if a word is a canonical term.

        Args:
            word: Word to check

        Returns:
            True if word is a canonical term
        """
        return word.lower() in self._vocabulary

    def is_variant(self, word: str) -> bool:
        """Check if a word is a known variant.

        Args:
            word: Word to check

        Returns:
            True if word is a known variant
        """
        return word.lower() in self._variant_lookup

    def to_dict(self) -> dict[str, list[str]]:
        """Export vocabulary as dictionary.

        Returns:
            Dictionary mapping canonical terms to variants
        """
        return {k: v.copy() for k, v in self._vocabulary.items()}

    def generate_whisper_prompt(self) -> str:
        """Generate a prompt string for Whisper API conditioning.

        Creates a comma-separated list of canonical terms that can be
        passed to the Whisper API to improve transcription accuracy
        for domain-specific vocabulary.

        Returns:
            Prompt string for Whisper API
        """
        # Get all canonical terms
        terms = sorted(self._vocabulary.keys())

        if not terms:
            return ""

        # Create a natural-sounding prompt with the terms
        # Whisper's prompt conditioning works best with context
        term_list = ", ".join(terms)

        prompt = (
            f"This is a technical discussion that may include terms like: {term_list}. "
            "Please transcribe technical terms accurately."
        )

        return prompt

    def __len__(self) -> int:
        """Return number of canonical terms."""
        return len(self._vocabulary)

    def __contains__(self, word: str) -> bool:
        """Check if word is a canonical term or variant."""
        word_lower = word.lower()
        return word_lower in self._vocabulary or word_lower in self._variant_lookup

    @classmethod
    def from_dict(cls, vocabulary: dict[str, list[str]]) -> "VocabularyTerms":
        """Create VocabularyTerms from dictionary.

        Args:
            vocabulary: Dictionary mapping canonical -> variants

        Returns:
            VocabularyTerms instance
        """
        return cls(vocabulary)

    @classmethod
    def from_json_file(cls, path: Path | str) -> "VocabularyTerms":
        """Load vocabulary from a JSON file.

        Args:
            path: Path to JSON file

        Returns:
            VocabularyTerms instance

        Raises:
            FileNotFoundError: If file doesn't exist
            json.JSONDecodeError: If file is not valid JSON
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both flat format and nested format
        if "vocabulary" in data:
            vocabulary = data["vocabulary"]
        else:
            vocabulary = data

        return cls(vocabulary)

    def to_json_file(self, path: Path | str) -> None:
        """Save vocabulary to a JSON file.

        Args:
            path: Path to save to
        """
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._vocabulary, f, indent=2, ensure_ascii=False)

    @classmethod
    def load_cncf_starter(cls) -> "VocabularyTerms":
        """Load the CNCF starter vocabulary.

        Returns:
            VocabularyTerms with CNCF terminology
        """
        # Get path relative to this module
        module_dir = Path(__file__).parent
        starter_path = module_dir / "cncf_starter.json"

        if starter_path.exists():
            return cls.from_json_file(starter_path)

        # Fallback to hardcoded if file not found
        return cls(get_cncf_vocabulary())


def get_cncf_vocabulary() -> dict[str, list[str]]:
    """Get the CNCF starter vocabulary.

    Returns:
        Dictionary of CNCF terms and common mis-transcriptions
    """
    return {
        "kubernetes": ["cooper netties", "kuber nettis", "kuber netties", "cooper nettis"],
        "argocd": ["argo cd", "argo seedy", "argo c d", "argo cede"],
        "istio": ["is theo", "east io", "isto", "is tio"],
        "etcd": ["et cetera d", "e t c d", "et cd", "etcetera d"],
        "prometheus": ["pro metheus"],
        "grafana": ["graffana", "graph ana"],
        "helm": ["hellm"],
        "containerd": ["container d", "container dee"],
        "envoy": ["en voy"],
        "jaeger": ["jager", "yaeger", "yager"],
        "fluentd": ["fluent d", "fluent dee"],
        "falco": ["fallco"],
        "linkerd": ["linker d", "linker dee"],
        "cncf": ["c n c f", "cnc f", "see ncf"],
    }
