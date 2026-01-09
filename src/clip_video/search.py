"""Search functionality for finding words and phrases across brand video libraries.

Provides search commands that find all occurrences of a word/phrase across
all videos in a brand, ranking results by match quality and extracting
candidate clips for review.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from clip_video.transcript.index import (
    TranscriptIndex,
    TranscriptIndexManager,
    PhraseMatch,
    WordOccurrence,
)
from clip_video.storage import atomic_write_json, read_json


@dataclass
class SearchResult:
    """A single search result with ranking information.

    Attributes:
        phrase: The matched phrase text
        start: Start timestamp in seconds
        end: End timestamp in seconds
        project_name: Source project name
        video_id: Source video identifier
        confidence: Average confidence of matched words
        context_before: Text context before the match
        context_after: Text context after the match
        rank_score: Computed ranking score (higher = better match)
        clip_extracted: Whether a preview clip has been extracted
        clip_path: Path to extracted clip if available
    """

    phrase: str
    start: float
    end: float
    project_name: str
    video_id: str
    confidence: float = 1.0
    context_before: str = ""
    context_after: str = ""
    rank_score: float = 0.0
    clip_extracted: bool = False
    clip_path: Path | None = None

    @property
    def duration(self) -> float:
        """Duration of the match in seconds."""
        return self.end - self.start

    @property
    def source_key(self) -> str:
        """Unique key for the source video."""
        return f"{self.project_name}:{self.video_id}"

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "phrase": self.phrase,
            "start": self.start,
            "end": self.end,
            "project_name": self.project_name,
            "video_id": self.video_id,
            "confidence": self.confidence,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "rank_score": self.rank_score,
            "clip_extracted": self.clip_extracted,
            "clip_path": str(self.clip_path) if self.clip_path else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Create from dictionary."""
        return cls(
            phrase=data["phrase"],
            start=data["start"],
            end=data["end"],
            project_name=data["project_name"],
            video_id=data["video_id"],
            confidence=data.get("confidence", 1.0),
            context_before=data.get("context_before", ""),
            context_after=data.get("context_after", ""),
            rank_score=data.get("rank_score", 0.0),
            clip_extracted=data.get("clip_extracted", False),
            clip_path=Path(data["clip_path"]) if data.get("clip_path") else None,
        )

    @classmethod
    def from_phrase_match(cls, match: PhraseMatch) -> "SearchResult":
        """Create from a PhraseMatch object.

        Args:
            match: PhraseMatch from transcript index

        Returns:
            SearchResult object
        """
        # Calculate average confidence
        if match.words:
            avg_confidence = sum(w.confidence for w in match.words) / len(match.words)
        else:
            avg_confidence = 1.0

        return cls(
            phrase=match.phrase,
            start=match.start,
            end=match.end,
            project_name=match.project_name,
            video_id=match.video_id,
            confidence=avg_confidence,
        )


@dataclass
class SearchResults:
    """Collection of search results for a query.

    Attributes:
        query: Original search query
        brand_name: Brand being searched
        results: List of search results
        total_count: Total number of matches found
        created_at: Timestamp when search was performed
    """

    query: str
    brand_name: str
    results: list[SearchResult] = field(default_factory=list)
    total_count: int = 0
    created_at: str = ""

    @property
    def unique_videos(self) -> set[str]:
        """Get set of unique source video keys."""
        return {r.source_key for r in self.results}

    @property
    def by_video(self) -> dict[str, list[SearchResult]]:
        """Group results by source video."""
        grouped: dict[str, list[SearchResult]] = {}
        for result in self.results:
            key = result.source_key
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(result)
        return grouped

    def top_results(self, n: int = 10) -> list[SearchResult]:
        """Get top N results by rank score.

        Args:
            n: Number of results to return

        Returns:
            List of top results
        """
        return sorted(self.results, key=lambda r: r.rank_score, reverse=True)[:n]

    def filter_by_video(self, video_id: str) -> list[SearchResult]:
        """Get results for a specific video.

        Args:
            video_id: Video identifier

        Returns:
            List of results from that video
        """
        return [r for r in self.results if r.video_id == video_id]

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "query": self.query,
            "brand_name": self.brand_name,
            "results": [r.to_dict() for r in self.results],
            "total_count": self.total_count,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResults":
        """Create from dictionary."""
        return cls(
            query=data["query"],
            brand_name=data["brand_name"],
            results=[SearchResult.from_dict(r) for r in data.get("results", [])],
            total_count=data.get("total_count", 0),
            created_at=data.get("created_at", ""),
        )

    def save(self, path: Path) -> None:
        """Save search results to JSON file.

        Args:
            path: Path to save the file
        """
        atomic_write_json(path, self.to_dict())

    @classmethod
    def load(cls, path: Path) -> "SearchResults":
        """Load search results from JSON file.

        Args:
            path: Path to the file

        Returns:
            SearchResults object
        """
        data = read_json(path)
        return cls.from_dict(data)


class BrandSearcher:
    """Search engine for finding words/phrases across a brand's video library.

    Provides ranked search results with quality scoring and candidate
    clip extraction for review.

    Example usage:
        searcher = BrandSearcher("my_brand")
        results = searcher.search("kubernetes")
        for result in results.top_results(5):
            print(f"{result.phrase} at {result.start}s in {result.video_id}")
    """

    def __init__(
        self,
        brand_name: str,
        brands_root: Path | None = None,
    ):
        """Initialize the searcher.

        Args:
            brand_name: Name of the brand to search
            brands_root: Root directory for brands (default: ./brands)
        """
        self.brand_name = brand_name
        self.brands_root = brands_root or Path.cwd() / "brands"
        self.brand_path = self.brands_root / brand_name

        # Load or create index
        self.index_manager = TranscriptIndexManager(self.brands_root)
        self._index: TranscriptIndex | None = None

    @property
    def index(self) -> TranscriptIndex:
        """Get the transcript index, loading if needed."""
        if self._index is None:
            self._index = self.index_manager.get(self.brand_name)
        return self._index

    def search(
        self,
        query: str,
        max_results: int = 100,
        max_gap: float = 2.0,
        min_confidence: float = 0.0,
    ) -> SearchResults:
        """Search for a word or phrase across all videos.

        Args:
            query: Word or phrase to search for
            max_results: Maximum number of results to return
            max_gap: Maximum time gap between words in phrase (seconds)
            min_confidence: Minimum confidence threshold for results

        Returns:
            SearchResults object with ranked matches
        """
        from datetime import datetime

        # Perform search using transcript index
        matches = self.index.search_phrase(query, max_gap=max_gap)

        # Convert to SearchResult objects
        results = []
        for match in matches:
            result = SearchResult.from_phrase_match(match)

            # Filter by confidence
            if result.confidence < min_confidence:
                continue

            # Calculate rank score
            result.rank_score = self._calculate_rank_score(result, query)
            results.append(result)

        # Sort by rank score
        results.sort(key=lambda r: r.rank_score, reverse=True)

        # Limit results
        results = results[:max_results]

        return SearchResults(
            query=query,
            brand_name=self.brand_name,
            results=results,
            total_count=len(matches),
            created_at=datetime.now().isoformat(),
        )

    def search_multiple(
        self,
        queries: list[str],
        max_results_per_query: int = 50,
    ) -> dict[str, SearchResults]:
        """Search for multiple queries at once.

        Args:
            queries: List of words/phrases to search for
            max_results_per_query: Maximum results per query

        Returns:
            Dict mapping query to SearchResults
        """
        return {
            query: self.search(query, max_results=max_results_per_query)
            for query in queries
        }

    def _calculate_rank_score(self, result: SearchResult, query: str) -> float:
        """Calculate ranking score for a search result.

        Higher scores indicate better matches.

        Factors:
        - Confidence: Higher confidence = higher score
        - Duration: Shorter duration (more concise) = higher score
        - Exact match: Exact case match = bonus

        Args:
            result: SearchResult to score
            query: Original query

        Returns:
            Rank score (0.0 to 1.0+)
        """
        score = 0.0

        # Base score from confidence (0-1)
        score += result.confidence

        # Duration factor - prefer shorter, more concise matches
        # Ideal duration is 0.3-1.5 seconds per word
        words_in_query = len(query.split())
        ideal_min = words_in_query * 0.3
        ideal_max = words_in_query * 1.5

        if ideal_min <= result.duration <= ideal_max:
            score += 0.3  # Ideal duration bonus
        elif result.duration < ideal_min * 2 and result.duration > ideal_max / 2:
            score += 0.15  # Acceptable duration
        # Long or very short durations get no bonus

        # Exact phrase match bonus
        if result.phrase.lower() == query.lower():
            score += 0.2

        return score

    def get_search_results_path(self, query: str) -> Path:
        """Get the path for storing search results.

        Args:
            query: Search query

        Returns:
            Path to search results file
        """
        # Sanitize query for filename
        safe_query = "".join(c if c.isalnum() or c in " -_" else "_" for c in query)
        safe_query = safe_query.replace(" ", "_").lower()[:50]

        return self.brand_path / "search_results" / f"{safe_query}.json"

    def save_results(self, results: SearchResults) -> Path:
        """Save search results to file.

        Args:
            results: SearchResults to save

        Returns:
            Path to saved file
        """
        path = self.get_search_results_path(results.query)
        path.parent.mkdir(parents=True, exist_ok=True)
        results.save(path)
        return path

    def load_results(self, query: str) -> SearchResults | None:
        """Load previous search results.

        Args:
            query: Original search query

        Returns:
            SearchResults if found, None otherwise
        """
        path = self.get_search_results_path(query)
        if not path.exists():
            return None
        return SearchResults.load(path)

    def get_candidate_clips_path(self, query: str) -> Path:
        """Get the directory for storing candidate clips.

        Args:
            query: Search query

        Returns:
            Path to candidate clips directory
        """
        safe_query = "".join(c if c.isalnum() or c in " -_" else "_" for c in query)
        safe_query = safe_query.replace(" ", "_").lower()[:50]

        return self.brand_path / "search_results" / safe_query / "clips"


class MultiQuerySearcher:
    """Search for multiple words/phrases from an extraction list.

    Used to search for all targets from a lyrics file at once,
    tracking coverage and missing matches.
    """

    def __init__(
        self,
        brand_searcher: BrandSearcher,
    ):
        """Initialize the multi-query searcher.

        Args:
            brand_searcher: BrandSearcher instance to use
        """
        self.searcher = brand_searcher

    def search_extraction_list(
        self,
        extraction_list: "ExtractionList",
        max_results_per_target: int = 20,
    ) -> dict[str, SearchResults]:
        """Search for all targets in an extraction list.

        Args:
            extraction_list: ExtractionList from lyrics parser
            max_results_per_target: Max results per target

        Returns:
            Dict mapping target text to SearchResults
        """
        from clip_video.lyrics.phrases import ExtractionList

        results = {}

        for target in extraction_list.all_targets:
            # Search for main target
            target_results = self.searcher.search(
                target.text,
                max_results=max_results_per_target,
            )
            results[target.text] = target_results

            # Also search alternatives if any
            for alt in target.alternatives:
                if alt not in results:
                    alt_results = self.searcher.search(
                        alt,
                        max_results=max_results_per_target,
                    )
                    results[alt] = alt_results

        return results

    def get_coverage_report(
        self,
        extraction_list: "ExtractionList",
        results: dict[str, SearchResults],
        min_matches: int = 1,
    ) -> dict:
        """Generate a coverage report showing which targets have matches.

        Args:
            extraction_list: Original extraction list
            results: Search results dict
            min_matches: Minimum matches to consider "covered"

        Returns:
            Dict with coverage statistics
        """
        from clip_video.lyrics.phrases import ExtractionList

        covered = []
        missing = []

        for target in extraction_list.all_targets:
            # Check main target and alternatives
            found = False
            best_count = 0

            if target.text in results:
                best_count = len(results[target.text].results)
                if best_count >= min_matches:
                    found = True

            for alt in target.alternatives:
                if alt in results:
                    alt_count = len(results[alt].results)
                    if alt_count > best_count:
                        best_count = alt_count
                    if alt_count >= min_matches:
                        found = True

            if found:
                covered.append({
                    "text": target.text,
                    "source_line": target.source_line,
                    "match_count": best_count,
                })
            else:
                missing.append({
                    "text": target.text,
                    "source_line": target.source_line,
                    "is_phrase": target.is_phrase,
                })

        total = len(extraction_list.all_targets)
        covered_count = len(covered)

        return {
            "total_targets": total,
            "covered_count": covered_count,
            "missing_count": len(missing),
            "coverage_percent": (covered_count / total * 100) if total > 0 else 0,
            "covered": covered,
            "missing": missing,
        }
