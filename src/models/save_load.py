"""Model persistence utilities with metadata, versioning, and checksums."""

from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib


class ModelPersistenceError(Exception):
    """Raised when model persistence operations fail."""


@dataclass
class ModelMetadata:
    """Structured metadata saved alongside persisted model artifacts."""

    model_name: str
    version: str
    serialization_format: str
    created_at: str
    checksum: str
    artifact_path: str
    metadata_path: str
    experiment_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to a serializable dictionary."""

        return {
            "model_name": self.model_name,
            "version": self.version,
            "serialization_format": self.serialization_format,
            "created_at": self.created_at,
            "checksum": self.checksum,
            "artifact_path": self.artifact_path,
            "metadata_path": self.metadata_path,
            "experiment_metadata": self.experiment_metadata,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ModelMetadata":
        """Build metadata from a dictionary payload."""

        required_keys = {
            "model_name",
            "version",
            "serialization_format",
            "created_at",
            "checksum",
            "artifact_path",
            "metadata_path",
        }
        missing_keys = sorted(required_keys - set(payload.keys()))
        if missing_keys:
            raise ModelPersistenceError(
                f"Metadata payload is missing required keys: {missing_keys}"
            )
        return cls(
            model_name=str(payload["model_name"]),
            version=str(payload["version"]),
            serialization_format=str(payload["serialization_format"]),
            created_at=str(payload["created_at"]),
            checksum=str(payload["checksum"]),
            artifact_path=str(payload["artifact_path"]),
            metadata_path=str(payload["metadata_path"]),
            experiment_metadata=dict(payload.get("experiment_metadata", {})),
        )


class ModelPersistenceManager:
    """Persist models using pickle or joblib with metadata and checksums."""

    def __init__(self, base_directory: Union[str, Path] = "artifacts/models") -> None:
        """Initialize the persistence manager."""

        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(parents=True, exist_ok=True)

    def save_model(
        self,
        model: Any,
        *,
        model_name: str,
        version: str,
        serialization_format: str = "pickle",
        experiment_metadata: Optional[Dict[str, Any]] = None,
    ) -> ModelMetadata:
        """Persist a model artifact and companion JSON metadata."""

        normalized_format = serialization_format.strip().lower()
        if normalized_format not in {"pickle", "joblib"}:
            raise ModelPersistenceError(
                "serialization_format must be either 'pickle' or 'joblib'."
            )

        model_directory = self._get_version_directory(model_name=model_name, version=version)
        model_directory.mkdir(parents=True, exist_ok=True)

        artifact_extension = ".pkl" if normalized_format == "pickle" else ".joblib"
        artifact_path = model_directory / f"{model_name}{artifact_extension}"
        metadata_path = model_directory / f"{model_name}.metadata.json"

        self._serialize_model(model=model, path=artifact_path, serialization_format=normalized_format)
        checksum = self.compute_checksum(artifact_path)

        metadata = ModelMetadata(
            model_name=model_name,
            version=version,
            serialization_format=normalized_format,
            created_at=datetime.now(timezone.utc).isoformat(),
            checksum=checksum,
            artifact_path=str(artifact_path),
            metadata_path=str(metadata_path),
            experiment_metadata=dict(experiment_metadata or {}),
        )
        self._write_metadata(metadata, metadata_path)
        return metadata

    def load_model(
        self,
        *,
        model_name: str,
        version: str,
        verify_checksum: bool = True,
    ) -> tuple[Any, ModelMetadata]:
        """Load a persisted model and its metadata."""

        model_directory = self._get_version_directory(model_name=model_name, version=version)
        metadata_path = model_directory / f"{model_name}.metadata.json"
        if not metadata_path.exists():
            raise ModelPersistenceError(f"Metadata file not found: {metadata_path}")

        metadata = self.load_metadata(metadata_path)
        artifact_path = Path(metadata.artifact_path)
        if not artifact_path.exists():
            raise ModelPersistenceError(f"Model artifact not found: {artifact_path}")

        if verify_checksum:
            current_checksum = self.compute_checksum(artifact_path)
            if current_checksum != metadata.checksum:
                raise ModelPersistenceError(
                    "Model checksum verification failed. "
                    f"Expected {metadata.checksum}, found {current_checksum}."
                )

        model = self._deserialize_model(artifact_path, metadata.serialization_format)
        return model, metadata

    def load_metadata(self, metadata_path: Union[str, Path]) -> ModelMetadata:
        """Load model metadata from JSON."""

        path = Path(metadata_path)
        if not path.exists():
            raise ModelPersistenceError(f"Metadata file not found: {path}")

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ModelPersistenceError(f"Invalid JSON metadata file: {path}") from exc
        except OSError as exc:
            raise ModelPersistenceError(f"Unable to read metadata file: {path}") from exc

        return ModelMetadata.from_dict(payload)

    def list_versions(self, model_name: str) -> list[str]:
        """List all available persisted versions for a model."""

        model_root = self.base_directory / model_name
        if not model_root.exists():
            return []
        return sorted(
            [path.name for path in model_root.iterdir() if path.is_dir()]
        )

    def compute_checksum(self, artifact_path: Union[str, Path]) -> str:
        """Compute SHA-256 checksum for a saved artifact."""

        path = Path(artifact_path)
        if not path.exists():
            raise ModelPersistenceError(f"Cannot compute checksum. File not found: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8192), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _serialize_model(self, model: Any, path: Path, serialization_format: str) -> None:
        """Serialize a model using the requested backend."""

        try:
            if serialization_format == "pickle":
                with path.open("wb") as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
            else:
                joblib.dump(model, path)
        except (pickle.PickleError, OSError, TypeError) as exc:
            raise ModelPersistenceError(
                f"Failed to serialize model to {path} using {serialization_format}."
            ) from exc

    def _deserialize_model(self, path: Path, serialization_format: str) -> Any:
        """Deserialize a model using the requested backend."""

        try:
            if serialization_format == "pickle":
                with path.open("rb") as handle:
                    return pickle.load(handle)
            return joblib.load(path)
        except (pickle.PickleError, OSError, TypeError, EOFError) as exc:
            raise ModelPersistenceError(
                f"Failed to deserialize model from {path} using {serialization_format}."
            ) from exc

    def _write_metadata(self, metadata: ModelMetadata, path: Path) -> None:
        """Write model metadata to JSON."""

        try:
            path.write_text(
                json.dumps(metadata.to_dict(), indent=4, sort_keys=True),
                encoding="utf-8",
            )
        except OSError as exc:
            raise ModelPersistenceError(f"Failed to write metadata to {path}.") from exc

    def _get_version_directory(self, *, model_name: str, version: str) -> Path:
        """Compute the artifact directory for a model version."""

        normalized_name = model_name.strip()
        normalized_version = version.strip()
        if not normalized_name:
            raise ModelPersistenceError("model_name cannot be empty.")
        if not normalized_version:
            raise ModelPersistenceError("version cannot be empty.")
        return self.base_directory / normalized_name / normalized_version


__all__ = [
    "ModelMetadata",
    "ModelPersistenceError",
    "ModelPersistenceManager",
]
