"""Fuzzy membership function implementations with parameter tuning support."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np


ArrayLike = np.ndarray | list[float] | tuple[float, ...]


class MembershipFunctionError(Exception):
    """Raised when membership function configuration or tuning fails."""


@dataclass
class TuningBounds:
    """Parameter tuning bounds for membership functions."""

    lower: Dict[str, float]
    upper: Dict[str, float]

    def validate(self) -> None:
        """Validate lower and upper tuning bounds."""

        if set(self.lower) != set(self.upper):
            raise MembershipFunctionError("Tuning bound keys must match exactly.")
        for key in self.lower:
            if self.lower[key] > self.upper[key]:
                raise MembershipFunctionError(
                    f"Invalid tuning bounds for '{key}': lower bound exceeds upper bound."
                )


class BaseMembershipFunction(ABC):
    """Abstract base class for fuzzy membership functions."""

    def __init__(self, name: str, **parameters: float) -> None:
        """Initialize the membership function with named parameters."""

        self.name = name
        self.parameters: Dict[str, float] = {key: float(value) for key, value in parameters.items()}
        self.validate_parameters()

    @abstractmethod
    def compute(self, x: ArrayLike) -> np.ndarray:
        """Compute the degree of membership for the supplied values."""

    @abstractmethod
    def validate_parameters(self) -> None:
        """Validate the current parameter set."""

    def tune(
        self,
        parameter_updates: Optional[Dict[str, float]] = None,
        *,
        bounds: Optional[TuningBounds] = None,
        clip_to_bounds: bool = True,
    ) -> "BaseMembershipFunction":
        """Update tunable parameters while preserving valid configuration."""

        updates = parameter_updates or {}
        new_parameters = dict(self.parameters)

        for key, value in updates.items():
            if key not in new_parameters:
                raise MembershipFunctionError(
                    f"Unknown parameter '{key}' for membership function '{self.name}'."
                )
            new_parameters[key] = float(value)

        if bounds is not None:
            bounds.validate()
            for key, value in new_parameters.items():
                if key not in bounds.lower:
                    continue
                if clip_to_bounds:
                    new_parameters[key] = float(
                        np.clip(value, bounds.lower[key], bounds.upper[key])
                    )
                elif not bounds.lower[key] <= value <= bounds.upper[key]:
                    raise MembershipFunctionError(
                        f"Parameter '{key}'={value} violates tuning bounds "
                        f"[{bounds.lower[key]}, {bounds.upper[key]}]."
                    )

        self.parameters = new_parameters
        self.validate_parameters()
        return self

    def get_parameters(self) -> Dict[str, float]:
        """Return a defensive copy of the current parameters."""

        return dict(self.parameters)

    @staticmethod
    def _to_numpy(values: ArrayLike) -> np.ndarray:
        """Convert inputs to a one-dimensional NumPy float array."""

        array = np.asarray(values, dtype=float)
        return array

    @staticmethod
    def _clip_membership(values: np.ndarray) -> np.ndarray:
        """Keep membership values in the valid fuzzy range [0, 1]."""

        return np.clip(values, 0.0, 1.0)


class TriangularMembershipFunction(BaseMembershipFunction):
    """Triangular fuzzy membership function."""

    def __init__(self, a: float, b: float, c: float, name: str = "triangular") -> None:
        super().__init__(name=name, a=a, b=b, c=c)

    def compute(self, x: ArrayLike) -> np.ndarray:
        """Compute triangular membership values."""

        values = self._to_numpy(x)
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]

        rising = np.divide(
            values - a,
            b - a,
            out=np.zeros_like(values, dtype=float),
            where=(b - a) != 0,
        )
        falling = np.divide(
            c - values,
            c - b,
            out=np.zeros_like(values, dtype=float),
            where=(c - b) != 0,
        )
        membership = np.maximum(np.minimum(rising, falling), 0.0)
        membership = np.where(values == b, 1.0, membership)
        return self._clip_membership(membership)

    def validate_parameters(self) -> None:
        """Validate triangular parameter ordering."""

        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]
        if not a <= b <= c:
            raise MembershipFunctionError(
                "Triangular parameters must satisfy a <= b <= c."
            )
        if a == c:
            raise MembershipFunctionError("Triangular parameters must span a non-zero support.")


class TrapezoidalMembershipFunction(BaseMembershipFunction):
    """Trapezoidal fuzzy membership function."""

    def __init__(self, a: float, b: float, c: float, d: float, name: str = "trapezoidal") -> None:
        super().__init__(name=name, a=a, b=b, c=c, d=d)

    def compute(self, x: ArrayLike) -> np.ndarray:
        """Compute trapezoidal membership values."""

        values = self._to_numpy(x)
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]
        d = self.parameters["d"]

        left = np.divide(
            values - a,
            b - a,
            out=np.zeros_like(values, dtype=float),
            where=(b - a) != 0,
        )
        right = np.divide(
            d - values,
            d - c,
            out=np.zeros_like(values, dtype=float),
            where=(d - c) != 0,
        )
        plateau = np.ones_like(values, dtype=float)
        membership = np.minimum(np.minimum(left, plateau), right)
        membership = np.where((values >= b) & (values <= c), 1.0, membership)
        return self._clip_membership(np.maximum(membership, 0.0))

    def validate_parameters(self) -> None:
        """Validate trapezoidal parameter ordering."""

        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]
        d = self.parameters["d"]
        if not a <= b <= c <= d:
            raise MembershipFunctionError(
                "Trapezoidal parameters must satisfy a <= b <= c <= d."
            )
        if a == d:
            raise MembershipFunctionError("Trapezoidal parameters must span a non-zero support.")


class GaussianMembershipFunction(BaseMembershipFunction):
    """Gaussian fuzzy membership function."""

    def __init__(self, mean: float, sigma: float, name: str = "gaussian") -> None:
        super().__init__(name=name, mean=mean, sigma=sigma)

    def compute(self, x: ArrayLike) -> np.ndarray:
        """Compute Gaussian membership values."""

        values = self._to_numpy(x)
        mean = self.parameters["mean"]
        sigma = self.parameters["sigma"]
        membership = np.exp(-0.5 * np.square((values - mean) / sigma))
        return self._clip_membership(membership)

    def validate_parameters(self) -> None:
        """Validate Gaussian parameters."""

        sigma = self.parameters["sigma"]
        if sigma <= 0.0:
            raise MembershipFunctionError("Gaussian sigma must be strictly positive.")


class GeneralizedBellMembershipFunction(BaseMembershipFunction):
    """Generalized bell-shaped fuzzy membership function."""

    def __init__(self, a: float, b: float, c: float, name: str = "generalized_bell") -> None:
        super().__init__(name=name, a=a, b=b, c=c)

    def compute(self, x: ArrayLike) -> np.ndarray:
        """Compute generalized bell membership values."""

        values = self._to_numpy(x)
        a = self.parameters["a"]
        b = self.parameters["b"]
        c = self.parameters["c"]
        membership = 1.0 / (1.0 + np.power(np.abs((values - c) / a), 2.0 * b))
        return self._clip_membership(membership)

    def validate_parameters(self) -> None:
        """Validate generalized bell parameters."""

        if self.parameters["a"] == 0.0:
            raise MembershipFunctionError("Generalized bell parameter 'a' must be non-zero.")
        if self.parameters["b"] <= 0.0:
            raise MembershipFunctionError("Generalized bell parameter 'b' must be strictly positive.")


class SigmoidMembershipFunction(BaseMembershipFunction):
    """Sigmoid fuzzy membership function."""

    def __init__(self, slope: float, center: float, name: str = "sigmoid") -> None:
        super().__init__(name=name, slope=slope, center=center)

    def compute(self, x: ArrayLike) -> np.ndarray:
        """Compute sigmoid membership values."""

        values = self._to_numpy(x)
        slope = self.parameters["slope"]
        center = self.parameters["center"]
        exponent = np.clip(-slope * (values - center), -500.0, 500.0)
        membership = 1.0 / (1.0 + np.exp(exponent))
        return self._clip_membership(membership)

    def validate_parameters(self) -> None:
        """Validate sigmoid parameters."""

        slope = self.parameters["slope"]
        if slope == 0.0:
            raise MembershipFunctionError("Sigmoid slope must be non-zero.")


def build_membership_function(function_type: str, **parameters: float) -> BaseMembershipFunction:
    """Factory function for membership function creation."""

    normalized = function_type.strip().lower()
    registry = {
        "triangular": TriangularMembershipFunction,
        "trapezoidal": TrapezoidalMembershipFunction,
        "gaussian": GaussianMembershipFunction,
        "generalized_bell": GeneralizedBellMembershipFunction,
        "gbell": GeneralizedBellMembershipFunction,
        "sigmoid": SigmoidMembershipFunction,
    }

    if normalized not in registry:
        raise MembershipFunctionError(
            f"Unsupported membership function '{function_type}'. "
            f"Expected one of {sorted(registry.keys())}."
        )

    return registry[normalized](**parameters)


__all__ = [
    "BaseMembershipFunction",
    "GaussianMembershipFunction",
    "GeneralizedBellMembershipFunction",
    "MembershipFunctionError",
    "SigmoidMembershipFunction",
    "TriangularMembershipFunction",
    "TrapezoidalMembershipFunction",
    "TuningBounds",
    "build_membership_function",
]
