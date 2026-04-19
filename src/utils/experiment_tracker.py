"""Local experiment tracking utilities for the stress detection project."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from uuid import uuid4


class ExperimentTrackerError(Exception):
    """Raised when experiment tracking operations fail."""


@dataclass
class ExperimentRecord:
    """Structured representation of a tracked experiment."""

    experiment_id: str
    experiment_name: str
    started_at: str
    status: str = "running"
    ended_at: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    model_versions: list[str] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the record to a dictionary."""

        return {
            "experiment_id": self.experiment_id,
            "experiment_name": self.experiment_name,
            "started_at": self.started_at,
            "status": self.status,
            "ended_at": self.ended_at,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "model_versions": self.model_versions,
            "artifacts": self.artifacts,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentRecord":
        """Build an experiment record from a dictionary."""

        required_keys = {"experiment_id", "experiment_name", "started_at", "status"}
        missing_keys = sorted(required_keys - set(payload))
        if missing_keys:
            raise ExperimentTrackerError(
                f"Experiment payload is missing required keys: {missing_keys}"
            )

        return cls(
            experiment_id=str(payload["experiment_id"]),
            experiment_name=str(payload["experiment_name"]),
            started_at=str(payload["started_at"]),
            status=str(payload["status"]),
            ended_at=payload.get("ended_at"),
            parameters=dict(payload.get("parameters", {})),
            metrics={key: float(value) for key, value in dict(payload.get("metrics", {})).items()},
            model_versions=[str(value) for value in payload.get("model_versions", [])],
            artifacts={key: str(value) for key, value in dict(payload.get("artifacts", {})).items()},
            notes=str(payload.get("notes", "")),
        )


class ExperimentTracker:
    """Track experiment parameters, metrics, versions, and results on local disk."""

    def __init__(self, base_directory: Union[str, Path] = "artifacts/experiments") -> None:
        """Initialize the tracker storage directory."""

        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)
        self._active_record: Optional[ExperimentRecord] = None

    def start_experiment(
        self,
        experiment_name: str,
        *,
        parameters: Optional[Dict[str, Any]] = None,
        notes: str = "",
    ) -> ExperimentRecord:
        """Start a new experiment run."""

        normalized_name = experiment_name.strip()
        if not normalized_name:
            raise ExperimentTrackerError("experiment_name cannot be empty.")

        experiment_id = uuid4().hex
        record = ExperimentRecord(
            experiment_id=experiment_id,
            experiment_name=normalized_name,
            started_at=datetime.now(timezone.utc).isoformat(),
            parameters=dict(parameters or {}),
            notes=notes,
        )
        self._active_record = record
        self._save_record(record)
        return record

    def log_parameters(self, parameters: Dict[str, Any]) -> None:
        """Log or update experiment parameters."""

        record = self._require_active_record()
        record.parameters.update(parameters)
        self._save_record(record)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Log or update experiment metrics."""

        record = self._require_active_record()
        record.metrics.update({key: float(value) for key, value in metrics.items()})
        self._save_record(record)

    def track_model_version(self, version: str) -> None:
        """Record a model version for the active experiment."""

        record = self._require_active_record()
        normalized_version = version.strip()
        if not normalized_version:
            raise ExperimentTrackerError("version cannot be empty.")
        if normalized_version not in record.model_versions:
            record.model_versions.append(normalized_version)
            self._save_record(record)

    def log_artifact(self, name: str, path: Union[str, Path]) -> None:
        """Attach an artifact path to the active experiment."""

        record = self._require_active_record()
        normalized_name = name.strip()
        if not normalized_name:
            raise ExperimentTrackerError("artifact name cannot be empty.")
        record.artifacts[normalized_name] = str(Path(path))
        self._save_record(record)

    def end_experiment(
        self,
        *,
        status: str = "completed",
        final_metrics: Optional[Dict[str, float]] = None,
    ) -> ExperimentRecord:
        """Finalize the active experiment."""

        record = self._require_active_record()
        if final_metrics:
            record.metrics.update({key: float(value) for key, value in final_metrics.items()})
        record.status = status
        record.ended_at = datetime.now(timezone.utc).isoformat()
        self._save_record(record)
        self._active_record = None
        return record

    def get_experiment(self, experiment_id: str) -> ExperimentRecord:
        """Load a tracked experiment by ID."""

        record_path = self._get_record_path(experiment_id)
        if not record_path.exists():
            raise ExperimentTrackerError(f"Experiment record not found: {experiment_id}")
        return self._load_record(record_path)

    def list_experiments(self) -> list[ExperimentRecord]:
        """List all tracked experiments."""

        records = []
        for path in sorted(self.base_directory.glob("*.json")):
            records.append(self._load_record(path))
        return records

    def _require_active_record(self) -> ExperimentRecord:
        """Return the active record or raise if none is active."""

        if self._active_record is None:
            raise ExperimentTrackerError("No active experiment. Call start_experiment first.")
        return self._active_record

    def _save_record(self, record: ExperimentRecord) -> None:
        """Persist a record to disk."""

        record_path = self._get_record_path(record.experiment_id)
        try:
            record_path.write_text(
                json.dumps(record.to_dict(), indent=4, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ExperimentTrackerError(f"Failed to save experiment record: {record.experiment_id}") from exc

    def _load_record(self, record_path: Path) -> ExperimentRecord:
        """Load a record from disk."""

        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
        except OSError as exc:
            raise ExperimentTrackerError(f"Failed to read experiment record: {record_path}") from exc
        except json.JSONDecodeError as exc:
            raise ExperimentTrackerError(f"Invalid experiment record JSON: {record_path}") from exc

        return ExperimentRecord.from_dict(payload)

    def _get_record_path(self, experiment_id: str) -> Path:
        """Return the path for an experiment record."""

        return self.base_directory / f"{experiment_id}.json"


__all__ = [
    "ExperimentRecord",
    "ExperimentTracker",
    "ExperimentTrackerError",
]
