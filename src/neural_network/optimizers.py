"""Reusable optimizer implementations for NumPy-based neural networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, Optional

import numpy as np


ParameterDict = Dict[str, np.ndarray]
GradientDict = Dict[str, np.ndarray]


class Optimizer(ABC):
    """Abstract base class for parameter optimizers."""

    def __init__(self, learning_rate: float = 0.001) -> None:
        """Initialize the optimizer."""

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        self.learning_rate = float(learning_rate)

    @abstractmethod
    def step(self, parameters: ParameterDict, gradients: GradientDict) -> ParameterDict:
        """Update parameters using their gradients."""

    def set_learning_rate(self, learning_rate: float) -> None:
        """Update the optimizer learning rate."""

        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        self.learning_rate = float(learning_rate)

    @staticmethod
    def _validate_keys(parameters: ParameterDict, gradients: GradientDict) -> None:
        """Ensure parameter and gradient keys match."""

        parameter_keys = set(parameters.keys())
        gradient_keys = set(gradients.keys())
        if parameter_keys != gradient_keys:
            missing_in_gradients = sorted(parameter_keys - gradient_keys)
            missing_in_parameters = sorted(gradient_keys - parameter_keys)
            raise ValueError(
                "Parameter and gradient keys must match exactly. "
                f"Missing in gradients: {missing_in_gradients}; missing in parameters: {missing_in_parameters}."
            )

    @staticmethod
    def _copy_updated_parameters(parameters: ParameterDict) -> ParameterDict:
        """Return a detached copy of updated parameters."""

        return {name: np.array(value, copy=True) for name, value in parameters.items()}


class SGD(Optimizer):
    """Stochastic gradient descent optimizer."""

    def step(self, parameters: ParameterDict, gradients: GradientDict) -> ParameterDict:
        """Apply one SGD update step."""

        self._validate_keys(parameters, gradients)
        for name in parameters:
            parameters[name] -= self.learning_rate * gradients[name]
        return self._copy_updated_parameters(parameters)


class Momentum(Optimizer):
    """Momentum-based gradient descent optimizer."""

    def __init__(self, learning_rate: float = 0.001, momentum: float = 0.9) -> None:
        """Initialize momentum optimizer state."""

        super().__init__(learning_rate=learning_rate)
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must satisfy 0.0 <= momentum < 1.0.")
        self.momentum = float(momentum)
        self.velocity: dict[str, np.ndarray] = {}

    def step(self, parameters: ParameterDict, gradients: GradientDict) -> ParameterDict:
        """Apply one momentum update step."""

        self._validate_keys(parameters, gradients)
        for name in parameters:
            if name not in self.velocity:
                self.velocity[name] = np.zeros_like(parameters[name])
            self.velocity[name] = (
                self.momentum * self.velocity[name]
                - self.learning_rate * gradients[name]
            )
            parameters[name] += self.velocity[name]
        return self._copy_updated_parameters(parameters)


class RMSprop(Optimizer):
    """RMSprop adaptive learning-rate optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize RMSprop optimizer state."""

        super().__init__(learning_rate=learning_rate)
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta must satisfy 0.0 <= beta < 1.0.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive.")
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.squared_gradients: dict[str, np.ndarray] = {}

    def step(self, parameters: ParameterDict, gradients: GradientDict) -> ParameterDict:
        """Apply one RMSprop update step."""

        self._validate_keys(parameters, gradients)
        for name in parameters:
            if name not in self.squared_gradients:
                self.squared_gradients[name] = np.zeros_like(parameters[name])
            self.squared_gradients[name] = (
                self.beta * self.squared_gradients[name]
                + (1.0 - self.beta) * np.square(gradients[name])
            )
            parameters[name] -= (
                self.learning_rate
                * gradients[name]
                / (np.sqrt(self.squared_gradients[name]) + self.epsilon)
            )
        return self._copy_updated_parameters(parameters)


class Adam(Optimizer):
    """Adam optimizer with first and second moment estimation."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        """Initialize Adam optimizer state."""

        super().__init__(learning_rate=learning_rate)
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("beta1 must satisfy 0.0 <= beta1 < 1.0.")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must satisfy 0.0 <= beta2 < 1.0.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive.")
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.first_moment: dict[str, np.ndarray] = {}
        self.second_moment: dict[str, np.ndarray] = {}
        self.time_step = 0

    def step(self, parameters: ParameterDict, gradients: GradientDict) -> ParameterDict:
        """Apply one Adam update step."""

        self._validate_keys(parameters, gradients)
        self.time_step += 1

        for name in parameters:
            if name not in self.first_moment:
                self.first_moment[name] = np.zeros_like(parameters[name])
                self.second_moment[name] = np.zeros_like(parameters[name])

            self.first_moment[name] = (
                self.beta1 * self.first_moment[name]
                + (1.0 - self.beta1) * gradients[name]
            )
            self.second_moment[name] = (
                self.beta2 * self.second_moment[name]
                + (1.0 - self.beta2) * np.square(gradients[name])
            )

            first_unbiased = self.first_moment[name] / (1.0 - self.beta1 ** self.time_step)
            second_unbiased = self.second_moment[name] / (1.0 - self.beta2 ** self.time_step)

            parameters[name] -= (
                self.learning_rate
                * first_unbiased
                / (np.sqrt(second_unbiased) + self.epsilon)
            )

        return self._copy_updated_parameters(parameters)


class Adagrad(Optimizer):
    """Adagrad optimizer with accumulated squared gradients."""

    def __init__(self, learning_rate: float = 0.01, epsilon: float = 1e-8) -> None:
        """Initialize Adagrad optimizer state."""

        super().__init__(learning_rate=learning_rate)
        if epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive.")
        self.epsilon = float(epsilon)
        self.accumulator: dict[str, np.ndarray] = {}

    def step(self, parameters: ParameterDict, gradients: GradientDict) -> ParameterDict:
        """Apply one Adagrad update step."""

        self._validate_keys(parameters, gradients)
        for name in parameters:
            if name not in self.accumulator:
                self.accumulator[name] = np.zeros_like(parameters[name])
            self.accumulator[name] += np.square(gradients[name])
            parameters[name] -= (
                self.learning_rate
                * gradients[name]
                / (np.sqrt(self.accumulator[name]) + self.epsilon)
            )
        return self._copy_updated_parameters(parameters)


def build_optimizer(name: str, **kwargs: float) -> Optimizer:
    """Create an optimizer instance from a string identifier."""

    normalized_name = name.strip().lower()
    optimizer_registry = {
        "sgd": SGD,
        "momentum": Momentum,
        "rmsprop": RMSprop,
        "adam": Adam,
        "adagrad": Adagrad,
    }

    if normalized_name not in optimizer_registry:
        raise ValueError(
            f"Unsupported optimizer '{name}'. Expected one of {sorted(optimizer_registry.keys())}."
        )

    return optimizer_registry[normalized_name](**kwargs)


__all__ = [
    "Adagrad",
    "Adam",
    "GradientDict",
    "Momentum",
    "Optimizer",
    "ParameterDict",
    "RMSprop",
    "SGD",
    "build_optimizer",
]
