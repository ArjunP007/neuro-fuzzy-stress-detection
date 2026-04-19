"""Reusable neural network layer implementations built with NumPy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class LayerGradients:
    """Container for layer gradient outputs."""

    input_gradient: np.ndarray
    weight_gradient: Optional[np.ndarray] = None
    bias_gradient: Optional[np.ndarray] = None
    gamma_gradient: Optional[np.ndarray] = None
    beta_gradient: Optional[np.ndarray] = None


class DenseLayer:
    """Fully connected layer with reusable forward and backward passes."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        weight_init: str = "he",
        random_state: int = 42,
    ) -> None:
        """Initialize dense layer parameters."""

        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be strictly positive.")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.weight_init = weight_init
        self.rng = np.random.default_rng(random_state)

        self.weights = self._initialize_weights()
        self.biases = np.zeros((1, self.output_dim), dtype=float)
        self._cached_inputs: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Compute the forward pass."""

        inputs = self._validate_inputs(inputs)
        self._cached_inputs = inputs
        return inputs @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray) -> LayerGradients:
        """Compute gradients for the backward pass."""

        if self._cached_inputs is None:
            raise RuntimeError("DenseLayer forward must be called before backward.")

        grad_output = self._validate_grad_output(grad_output)
        batch_size = grad_output.shape[0]

        weight_gradient = (self._cached_inputs.T @ grad_output) / batch_size
        bias_gradient = np.sum(grad_output, axis=0, keepdims=True) / batch_size
        input_gradient = grad_output @ self.weights.T

        return LayerGradients(
            input_gradient=input_gradient,
            weight_gradient=weight_gradient,
            bias_gradient=bias_gradient,
        )

    def _initialize_weights(self) -> np.ndarray:
        """Initialize weights using the requested strategy."""

        if self.weight_init == "he":
            scale = np.sqrt(2.0 / self.input_dim)
            return self.rng.normal(0.0, scale, size=(self.input_dim, self.output_dim))
        if self.weight_init == "xavier":
            scale = np.sqrt(1.0 / self.input_dim)
            return self.rng.normal(0.0, scale, size=(self.input_dim, self.output_dim))
        if self.weight_init == "xavier_uniform":
            limit = np.sqrt(6.0 / (self.input_dim + self.output_dim))
            return self.rng.uniform(-limit, limit, size=(self.input_dim, self.output_dim))
        if self.weight_init == "normal":
            return self.rng.normal(0.0, 0.01, size=(self.input_dim, self.output_dim))
        raise ValueError(f"Unsupported weight initialization strategy: {self.weight_init}")

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Validate dense layer input tensor."""

        array = np.asarray(inputs, dtype=float)
        if array.ndim != 2:
            raise ValueError("DenseLayer inputs must be a 2D array.")
        if array.shape[1] != self.input_dim:
            raise ValueError(
                f"DenseLayer expected input dimension {self.input_dim}, received {array.shape[1]}."
            )
        return array

    def _validate_grad_output(self, grad_output: np.ndarray) -> np.ndarray:
        """Validate gradient input for the dense layer."""

        array = np.asarray(grad_output, dtype=float)
        if array.ndim != 2:
            raise ValueError("DenseLayer grad_output must be a 2D array.")
        if array.shape[1] != self.output_dim:
            raise ValueError(
                f"DenseLayer expected gradient dimension {self.output_dim}, received {array.shape[1]}."
            )
        return array


class DropoutLayer:
    """Dropout regularization layer with train and inference behavior."""

    def __init__(self, dropout_rate: float = 0.5, random_state: int = 42) -> None:
        """Initialize dropout configuration."""

        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0.0 <= dropout_rate < 1.0.")

        self.dropout_rate = float(dropout_rate)
        self.keep_probability = 1.0 - self.dropout_rate
        self.rng = np.random.default_rng(random_state)
        self._mask: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Apply dropout mask during training and identity mapping during inference."""

        inputs = np.asarray(inputs, dtype=float)
        if inputs.ndim != 2:
            raise ValueError("DropoutLayer inputs must be a 2D array.")

        if not training or self.dropout_rate == 0.0:
            self._mask = None
            return inputs

        self._mask = self.rng.binomial(1, self.keep_probability, size=inputs.shape).astype(float)
        return (inputs * self._mask) / self.keep_probability

    def backward(self, grad_output: np.ndarray) -> LayerGradients:
        """Backpropagate through the dropout mask."""

        grad_output = np.asarray(grad_output, dtype=float)
        if grad_output.ndim != 2:
            raise ValueError("DropoutLayer grad_output must be a 2D array.")

        if self._mask is None or self.dropout_rate == 0.0:
            return LayerGradients(input_gradient=grad_output)

        input_gradient = (grad_output * self._mask) / self.keep_probability
        return LayerGradients(input_gradient=input_gradient)


class BatchNormalizationLayer:
    """Batch normalization layer supporting training and inference passes."""

    def __init__(
        self,
        feature_dim: int,
        *,
        momentum: float = 0.9,
        epsilon: float = 1e-5,
    ) -> None:
        """Initialize normalization parameters and running statistics."""

        if feature_dim <= 0:
            raise ValueError("feature_dim must be strictly positive.")
        if not 0.0 < momentum < 1.0:
            raise ValueError("momentum must satisfy 0.0 < momentum < 1.0.")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be strictly positive.")

        self.feature_dim = int(feature_dim)
        self.momentum = float(momentum)
        self.epsilon = float(epsilon)

        self.gamma = np.ones((1, self.feature_dim), dtype=float)
        self.beta = np.zeros((1, self.feature_dim), dtype=float)
        self.running_mean = np.zeros((1, self.feature_dim), dtype=float)
        self.running_var = np.ones((1, self.feature_dim), dtype=float)

        self._cached_inputs: Optional[np.ndarray] = None
        self._cached_normalized: Optional[np.ndarray] = None
        self._cached_batch_mean: Optional[np.ndarray] = None
        self._cached_batch_var: Optional[np.ndarray] = None

    def forward(self, inputs: np.ndarray, training: bool = True) -> np.ndarray:
        """Normalize activations using batch or running statistics."""

        inputs = self._validate_inputs(inputs)

        if training:
            batch_mean = np.mean(inputs, axis=0, keepdims=True)
            batch_var = np.var(inputs, axis=0, keepdims=True)
            normalized = (inputs - batch_mean) / np.sqrt(batch_var + self.epsilon)

            self.running_mean = self.momentum * self.running_mean + (1.0 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1.0 - self.momentum) * batch_var

            self._cached_inputs = inputs
            self._cached_normalized = normalized
            self._cached_batch_mean = batch_mean
            self._cached_batch_var = batch_var
        else:
            normalized = (inputs - self.running_mean) / np.sqrt(self.running_var + self.epsilon)

        return self.gamma * normalized + self.beta

    def backward(self, grad_output: np.ndarray) -> LayerGradients:
        """Compute gradients for batch normalization."""

        if (
            self._cached_inputs is None
            or self._cached_normalized is None
            or self._cached_batch_mean is None
            or self._cached_batch_var is None
        ):
            raise RuntimeError("BatchNormalizationLayer forward(training=True) must be called before backward.")

        grad_output = self._validate_inputs(grad_output)
        batch_size = grad_output.shape[0]

        normalized = self._cached_normalized
        inputs = self._cached_inputs
        batch_mean = self._cached_batch_mean
        batch_var = self._cached_batch_var

        gamma_gradient = np.sum(grad_output * normalized, axis=0, keepdims=True)
        beta_gradient = np.sum(grad_output, axis=0, keepdims=True)

        grad_normalized = grad_output * self.gamma
        inv_std = 1.0 / np.sqrt(batch_var + self.epsilon)

        grad_var = np.sum(
            grad_normalized * (inputs - batch_mean) * -0.5 * np.power(batch_var + self.epsilon, -1.5),
            axis=0,
            keepdims=True,
        )
        grad_mean = (
            np.sum(-grad_normalized * inv_std, axis=0, keepdims=True)
            + grad_var * np.mean(-2.0 * (inputs - batch_mean), axis=0, keepdims=True)
        )

        input_gradient = (
            grad_normalized * inv_std
            + grad_var * 2.0 * (inputs - batch_mean) / batch_size
            + grad_mean / batch_size
        )

        return LayerGradients(
            input_gradient=input_gradient,
            gamma_gradient=gamma_gradient / batch_size,
            beta_gradient=beta_gradient / batch_size,
        )

    def _validate_inputs(self, inputs: np.ndarray) -> np.ndarray:
        """Validate batch normalization input tensor."""

        array = np.asarray(inputs, dtype=float)
        if array.ndim != 2:
            raise ValueError("BatchNormalizationLayer inputs must be a 2D array.")
        if array.shape[1] != self.feature_dim:
            raise ValueError(
                f"BatchNormalizationLayer expected feature dimension {self.feature_dim}, received {array.shape[1]}."
            )
        return array


__all__ = [
    "BatchNormalizationLayer",
    "DenseLayer",
    "DropoutLayer",
    "LayerGradients",
]
