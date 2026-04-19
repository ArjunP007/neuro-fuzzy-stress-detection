"""Vectorized activation functions and derivatives for NumPy neural networks."""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


Array = np.ndarray


def relu(x: Array) -> Array:
    """Apply the ReLU activation function."""

    values = np.asarray(x, dtype=float)
    return np.maximum(0.0, values)


def relu_derivative(x: Array) -> Array:
    """Compute the derivative of ReLU."""

    values = np.asarray(x, dtype=float)
    return (values > 0.0).astype(float)


def leaky_relu(x: Array, alpha: float = 0.01) -> Array:
    """Apply the Leaky ReLU activation function."""

    values = np.asarray(x, dtype=float)
    return np.where(values > 0.0, values, alpha * values)


def leaky_relu_derivative(x: Array, alpha: float = 0.01) -> Array:
    """Compute the derivative of Leaky ReLU."""

    values = np.asarray(x, dtype=float)
    return np.where(values > 0.0, 1.0, alpha)


def sigmoid(x: Array) -> Array:
    """Apply the sigmoid activation function."""

    values = np.asarray(x, dtype=float)
    clipped = np.clip(values, -500.0, 500.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def sigmoid_derivative(x: Array) -> Array:
    """Compute the derivative of the sigmoid activation."""

    activated = sigmoid(x)
    return activated * (1.0 - activated)


def tanh(x: Array) -> Array:
    """Apply the hyperbolic tangent activation function."""

    values = np.asarray(x, dtype=float)
    return np.tanh(values)


def tanh_derivative(x: Array) -> Array:
    """Compute the derivative of tanh."""

    activated = tanh(x)
    return 1.0 - np.square(activated)


def softmax(x: Array) -> Array:
    """Apply the softmax activation function."""

    values = np.asarray(x, dtype=float)
    if values.ndim == 1:
        values = values.reshape(1, -1)

    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def softmax_derivative(x: Array) -> Array:
    """Compute the element-wise softmax derivative approximation."""

    probabilities = softmax(x)
    return probabilities * (1.0 - probabilities)


def elu(x: Array, alpha: float = 1.0) -> Array:
    """Apply the ELU activation function."""

    values = np.asarray(x, dtype=float)
    return np.where(values > 0.0, values, alpha * (np.exp(values) - 1.0))


def elu_derivative(x: Array, alpha: float = 1.0) -> Array:
    """Compute the derivative of ELU."""

    values = np.asarray(x, dtype=float)
    return np.where(values > 0.0, 1.0, alpha * np.exp(values))


def selu(
    x: Array,
    alpha: float = 1.6732632423543772,
    scale: float = 1.0507009873554805,
) -> Array:
    """Apply the SELU activation function."""

    values = np.asarray(x, dtype=float)
    return scale * np.where(values > 0.0, values, alpha * (np.exp(values) - 1.0))


def selu_derivative(
    x: Array,
    alpha: float = 1.6732632423543772,
    scale: float = 1.0507009873554805,
) -> Array:
    """Compute the derivative of SELU."""

    values = np.asarray(x, dtype=float)
    return scale * np.where(values > 0.0, 1.0, alpha * np.exp(values))


ACTIVATIONS: Dict[str, Callable[..., Array]] = {
    "relu": relu,
    "leaky_relu": leaky_relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "softmax": softmax,
    "elu": elu,
    "selu": selu,
}


DERIVATIVES: Dict[str, Callable[..., Array]] = {
    "relu": relu_derivative,
    "leaky_relu": leaky_relu_derivative,
    "sigmoid": sigmoid_derivative,
    "tanh": tanh_derivative,
    "softmax": softmax_derivative,
    "elu": elu_derivative,
    "selu": selu_derivative,
}


__all__ = [
    "ACTIVATIONS",
    "DERIVATIVES",
    "elu",
    "elu_derivative",
    "leaky_relu",
    "leaky_relu_derivative",
    "relu",
    "relu_derivative",
    "selu",
    "selu_derivative",
    "sigmoid",
    "sigmoid_derivative",
    "softmax",
    "softmax_derivative",
    "tanh",
    "tanh_derivative",
]
