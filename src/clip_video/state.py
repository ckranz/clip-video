"""Processing state machine for idempotent operations.

Tracks which videos have been transcribed, which clips extracted,
and which highlights generated. Enables resuming after interruption
and ensures operations are idempotent.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class OperationStatus(str, Enum):
    """Status of a processing operation."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class OperationType(str, Enum):
    """Types of operations tracked by the state machine."""

    TRANSCRIPTION = "transcription"
    CLIP_EXTRACTION = "clip_extraction"
    HIGHLIGHT_GENERATION = "highlight_generation"
    CLIP_RENDERING = "clip_rendering"
    CAPTION_BURN = "caption_burn"


@dataclass
class OperationRecord:
    """Record of a single operation on an item."""

    item_id: str  # Unique identifier (e.g., file path, clip id)
    operation: OperationType
    status: OperationStatus = OperationStatus.PENDING
    started_at: datetime | None = None
    completed_at: datetime | None = None
    output_path: str | None = None
    error_message: str | None = None
    attempts: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "item_id": self.item_id,
            "operation": self.operation.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "output_path": self.output_path,
            "error_message": self.error_message,
            "attempts": self.attempts,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "OperationRecord":
        """Create from dictionary."""
        return cls(
            item_id=data["item_id"],
            operation=OperationType(data["operation"]),
            status=OperationStatus(data.get("status", "pending")),
            started_at=(
                datetime.fromisoformat(data["started_at"])
                if data.get("started_at")
                else None
            ),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
            output_path=data.get("output_path"),
            error_message=data.get("error_message"),
            attempts=data.get("attempts", 0),
            metadata=data.get("metadata", {}),
        )

    def start(self) -> None:
        """Mark operation as in progress."""
        self.status = OperationStatus.IN_PROGRESS
        self.started_at = datetime.now()
        self.attempts += 1
        self.error_message = None

    def complete(self, output_path: str | None = None) -> None:
        """Mark operation as completed."""
        self.status = OperationStatus.COMPLETED
        self.completed_at = datetime.now()
        self.output_path = output_path
        self.error_message = None

    def fail(self, error_message: str) -> None:
        """Mark operation as failed."""
        self.status = OperationStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message

    def skip(self, reason: str = "Already processed") -> None:
        """Mark operation as skipped."""
        self.status = OperationStatus.SKIPPED
        self.error_message = reason

    def needs_processing(self) -> bool:
        """Check if this operation needs to be performed."""
        return self.status in (OperationStatus.PENDING, OperationStatus.FAILED)

    def is_complete(self) -> bool:
        """Check if operation is complete."""
        return self.status == OperationStatus.COMPLETED


class ProcessingState:
    """State machine for tracking processing progress.

    Tracks the state of all processing operations for a brand/project,
    enabling resume after interruption and ensuring idempotent operations.

    The state file uses atomic writes to prevent corruption on crash.

    Example:
        state = ProcessingState.load_or_create(state_path, "my_brand")

        # Check if video needs transcription
        if state.needs_processing("video1.mp4", OperationType.TRANSCRIPTION):
            state.start_operation("video1.mp4", OperationType.TRANSCRIPTION)
            state.save()

            try:
                transcript = transcribe(video)
                state.complete_operation(
                    "video1.mp4",
                    OperationType.TRANSCRIPTION,
                    output_path=str(transcript_path)
                )
            except Exception as e:
                state.fail_operation("video1.mp4", OperationType.TRANSCRIPTION, str(e))

            state.save()
    """

    def __init__(
        self,
        name: str,
        state_file: Path | None = None,
    ):
        """Initialize processing state.

        Args:
            name: Name of the brand/project
            state_file: Path to state file (for saving)
        """
        self.name = name
        self.state_file = state_file
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Operations indexed by (item_id, operation_type)
        self._operations: dict[tuple[str, OperationType], OperationRecord] = {}

        # Metadata about the processing session
        self.metadata: dict[str, Any] = {}

    def _key(self, item_id: str, operation: OperationType) -> tuple[str, OperationType]:
        """Create a key for the operations dict."""
        return (item_id, operation)

    def get_operation(
        self,
        item_id: str,
        operation: OperationType,
    ) -> OperationRecord | None:
        """Get an operation record.

        Args:
            item_id: Item identifier
            operation: Type of operation

        Returns:
            OperationRecord if found, None otherwise
        """
        return self._operations.get(self._key(item_id, operation))

    def get_or_create_operation(
        self,
        item_id: str,
        operation: OperationType,
    ) -> OperationRecord:
        """Get or create an operation record.

        Args:
            item_id: Item identifier
            operation: Type of operation

        Returns:
            Existing or new OperationRecord
        """
        key = self._key(item_id, operation)
        if key not in self._operations:
            self._operations[key] = OperationRecord(
                item_id=item_id,
                operation=operation,
            )
        return self._operations[key]

    def needs_processing(
        self,
        item_id: str,
        operation: OperationType,
    ) -> bool:
        """Check if an item needs processing for an operation.

        Args:
            item_id: Item identifier
            operation: Type of operation

        Returns:
            True if processing is needed
        """
        record = self.get_operation(item_id, operation)
        if record is None:
            return True
        return record.needs_processing()

    def is_complete(
        self,
        item_id: str,
        operation: OperationType,
    ) -> bool:
        """Check if an operation is complete for an item.

        Args:
            item_id: Item identifier
            operation: Type of operation

        Returns:
            True if operation is complete
        """
        record = self.get_operation(item_id, operation)
        if record is None:
            return False
        return record.is_complete()

    def start_operation(
        self,
        item_id: str,
        operation: OperationType,
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord:
        """Start an operation on an item.

        Args:
            item_id: Item identifier
            operation: Type of operation
            metadata: Optional metadata to store

        Returns:
            The operation record
        """
        record = self.get_or_create_operation(item_id, operation)
        record.start()
        if metadata:
            record.metadata.update(metadata)
        self.updated_at = datetime.now()
        return record

    def complete_operation(
        self,
        item_id: str,
        operation: OperationType,
        output_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> OperationRecord:
        """Mark an operation as complete.

        Args:
            item_id: Item identifier
            operation: Type of operation
            output_path: Path to output file
            metadata: Optional metadata to store

        Returns:
            The operation record
        """
        record = self.get_or_create_operation(item_id, operation)
        record.complete(output_path)
        if metadata:
            record.metadata.update(metadata)
        self.updated_at = datetime.now()
        return record

    def fail_operation(
        self,
        item_id: str,
        operation: OperationType,
        error_message: str,
    ) -> OperationRecord:
        """Mark an operation as failed.

        Args:
            item_id: Item identifier
            operation: Type of operation
            error_message: Error description

        Returns:
            The operation record
        """
        record = self.get_or_create_operation(item_id, operation)
        record.fail(error_message)
        self.updated_at = datetime.now()
        return record

    def skip_operation(
        self,
        item_id: str,
        operation: OperationType,
        reason: str = "Already processed",
    ) -> OperationRecord:
        """Mark an operation as skipped.

        Args:
            item_id: Item identifier
            operation: Type of operation
            reason: Skip reason

        Returns:
            The operation record
        """
        record = self.get_or_create_operation(item_id, operation)
        record.skip(reason)
        self.updated_at = datetime.now()
        return record

    def get_items_by_status(
        self,
        operation: OperationType,
        status: OperationStatus,
    ) -> list[str]:
        """Get all items with a specific status for an operation.

        Args:
            operation: Type of operation
            status: Status to filter by

        Returns:
            List of item IDs
        """
        return [
            record.item_id
            for key, record in self._operations.items()
            if key[1] == operation and record.status == status
        ]

    def get_pending_items(self, operation: OperationType) -> list[str]:
        """Get items that need processing for an operation.

        Args:
            operation: Type of operation

        Returns:
            List of item IDs that need processing
        """
        pending = self.get_items_by_status(operation, OperationStatus.PENDING)
        failed = self.get_items_by_status(operation, OperationStatus.FAILED)
        return pending + failed

    def get_completed_items(self, operation: OperationType) -> list[str]:
        """Get items that are complete for an operation.

        Args:
            operation: Type of operation

        Returns:
            List of item IDs that are complete
        """
        return self.get_items_by_status(operation, OperationStatus.COMPLETED)

    def get_failed_items(self, operation: OperationType) -> list[str]:
        """Get items that failed for an operation.

        Args:
            operation: Type of operation

        Returns:
            List of item IDs that failed
        """
        return self.get_items_by_status(operation, OperationStatus.FAILED)

    def add_items(
        self,
        item_ids: list[str],
        operation: OperationType,
    ) -> None:
        """Add multiple items for tracking.

        Args:
            item_ids: List of item identifiers
            operation: Type of operation
        """
        for item_id in item_ids:
            self.get_or_create_operation(item_id, operation)
        self.updated_at = datetime.now()

    def get_summary(self, operation: OperationType | None = None) -> dict[str, Any]:
        """Get summary statistics.

        Args:
            operation: Filter by operation type (None for all)

        Returns:
            Dictionary with progress statistics
        """
        records = [
            r for r in self._operations.values()
            if operation is None or r.operation == operation
        ]

        total = len(records)
        completed = sum(1 for r in records if r.status == OperationStatus.COMPLETED)
        failed = sum(1 for r in records if r.status == OperationStatus.FAILED)
        pending = sum(1 for r in records if r.status == OperationStatus.PENDING)
        in_progress = sum(1 for r in records if r.status == OperationStatus.IN_PROGRESS)
        skipped = sum(1 for r in records if r.status == OperationStatus.SKIPPED)

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "skipped": skipped,
            "completion_rate": (completed / total * 100) if total > 0 else 0.0,
        }

    def reset_failed(self, operation: OperationType | None = None) -> int:
        """Reset failed operations to pending status.

        Args:
            operation: Filter by operation type (None for all)

        Returns:
            Number of operations reset
        """
        count = 0
        for record in self._operations.values():
            if operation is not None and record.operation != operation:
                continue
            if record.status == OperationStatus.FAILED:
                record.status = OperationStatus.PENDING
                record.error_message = None
                count += 1

        if count > 0:
            self.updated_at = datetime.now()

        return count

    def reset_in_progress(self) -> int:
        """Reset in-progress operations to pending (for resume after crash).

        Returns:
            Number of operations reset
        """
        count = 0
        for record in self._operations.values():
            if record.status == OperationStatus.IN_PROGRESS:
                record.status = OperationStatus.PENDING
                count += 1

        if count > 0:
            self.updated_at = datetime.now()

        return count

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "operations": [
                record.to_dict() for record in self._operations.values()
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], state_file: Path | None = None) -> "ProcessingState":
        """Create from dictionary.

        Args:
            data: Dictionary data
            state_file: Path to state file

        Returns:
            ProcessingState instance
        """
        state = cls(
            name=data["name"],
            state_file=state_file,
        )
        state.created_at = datetime.fromisoformat(data["created_at"])
        state.updated_at = datetime.fromisoformat(data["updated_at"])
        state.metadata = data.get("metadata", {})

        for op_data in data.get("operations", []):
            record = OperationRecord.from_dict(op_data)
            state._operations[state._key(record.item_id, record.operation)] = record

        return state

    def save(self, path: Path | None = None) -> None:
        """Save state to file atomically.

        Args:
            path: Path to save to (uses self.state_file if None)
        """
        save_path = path or self.state_file
        if save_path is None:
            raise ValueError("No save path specified")

        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Atomic write using temp file + rename
        fd = None
        temp_path = None
        try:
            fd, temp_path = tempfile.mkstemp(
                suffix=".tmp",
                prefix=save_path.stem + "_",
                dir=save_path.parent,
            )

            # Write JSON data
            data = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
            os.write(fd, data.encode("utf-8"))
            os.fsync(fd)
            os.close(fd)
            fd = None

            # Atomic rename (Windows requires remove first)
            if os.name == "nt" and save_path.exists():
                save_path.unlink()
            os.rename(temp_path, save_path)
            temp_path = None

        except Exception as e:
            raise RuntimeError(f"Failed to save state to {save_path}: {e}") from e
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            if temp_path is not None:
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass

    @classmethod
    def load(cls, path: Path) -> "ProcessingState":
        """Load state from file.

        Args:
            path: Path to load from

        Returns:
            ProcessingState instance

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is invalid
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid state file {path}: {e}") from e

        return cls.from_dict(data, state_file=path)

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        name: str,
        reset_in_progress: bool = True,
    ) -> "ProcessingState":
        """Load state from file or create new if not exists.

        Args:
            path: Path to state file
            name: Name for new state
            reset_in_progress: Reset in-progress ops on load (for crash recovery)

        Returns:
            ProcessingState instance
        """
        path = Path(path)

        if path.exists():
            state = cls.load(path)
            state.state_file = path

            # Reset any in-progress operations (crash recovery)
            if reset_in_progress:
                reset_count = state.reset_in_progress()
                if reset_count > 0:
                    state.save()

            return state

        return cls(name=name, state_file=path)


class IdempotentProcessor:
    """Helper for processing items idempotently with state tracking.

    Example:
        state = ProcessingState.load_or_create(state_path, "brand")
        processor = IdempotentProcessor(state, OperationType.TRANSCRIPTION)

        for video in processor.filter_pending(videos, key_fn=str):
            with processor.process(str(video)) as op:
                result = transcribe(video)
                op.complete(output_path=str(result.path))
    """

    def __init__(
        self,
        state: ProcessingState,
        operation: OperationType,
        auto_save: bool = True,
    ):
        """Initialize processor.

        Args:
            state: Processing state to use
            operation: Operation type being performed
            auto_save: Whether to auto-save state after each operation
        """
        self.state = state
        self.operation = operation
        self.auto_save = auto_save

    def filter_pending(
        self,
        items: list[Any],
        key_fn: Any = str,
    ) -> list[Any]:
        """Filter items to only those needing processing.

        Args:
            items: List of items to filter
            key_fn: Function to get item ID from item

        Returns:
            List of items that need processing
        """
        return [
            item for item in items
            if self.state.needs_processing(key_fn(item), self.operation)
        ]

    def register_items(
        self,
        items: list[Any],
        key_fn: Any = str,
    ) -> None:
        """Register items for tracking.

        Args:
            items: List of items to register
            key_fn: Function to get item ID from item
        """
        self.state.add_items([key_fn(item) for item in items], self.operation)
        if self.auto_save:
            self.state.save()

    def process(self, item_id: str) -> "_ProcessContext":
        """Context manager for processing an item.

        Args:
            item_id: Item identifier

        Returns:
            Context manager for the processing operation
        """
        return _ProcessContext(self, item_id)


class _ProcessContext:
    """Context manager for idempotent processing."""

    def __init__(self, processor: IdempotentProcessor, item_id: str):
        self.processor = processor
        self.item_id = item_id
        self._record: OperationRecord | None = None

    def __enter__(self) -> "_ProcessContext":
        self._record = self.processor.state.start_operation(
            self.item_id,
            self.processor.operation,
        )
        if self.processor.auto_save:
            self.processor.state.save()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        if exc_type is not None and self._record is not None:
            # Exception occurred - mark as failed
            self._record.fail(str(exc_val))
            if self.processor.auto_save:
                self.processor.state.save()
        return False  # Don't suppress exceptions

    def complete(
        self,
        output_path: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Mark operation as complete.

        Args:
            output_path: Path to output file
            metadata: Optional metadata to store
        """
        self.processor.state.complete_operation(
            self.item_id,
            self.processor.operation,
            output_path=output_path,
            metadata=metadata,
        )
        if self.processor.auto_save:
            self.processor.state.save()

    def skip(self, reason: str = "Already processed") -> None:
        """Mark operation as skipped.

        Args:
            reason: Skip reason
        """
        self.processor.state.skip_operation(
            self.item_id,
            self.processor.operation,
            reason=reason,
        )
        if self.processor.auto_save:
            self.processor.state.save()
