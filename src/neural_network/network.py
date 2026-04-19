"""NumPy-based neural network core for stress level detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, Optional, Sequence

import numpy as np


LOGGER = logging.getLogger(__name__)


@dataclass
class LayerParameters:
    """Container for layer weights and biases."""

    weights: np.ndarray
    biases: np.ndarray


@dataclass
class LayerCache:
    """Intermediate values required for backpropagation."""

    inputs: np.ndarray
    pre_activation: np.ndarray
    activation: np.ndarray
    dropout_mask: Optional[np.ndarray] = None


@dataclass
class TrainingHistory:
    """Track losses and metrics during optimization."""

    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


class NeuralNetwork:
    """Fully connected neural network implemented from scratch with NumPy."""

    def __init__(
        self,
        input_size: int,
        hidden_layers: Sequence[int],
        output_size: int,
        *,
        activation: str = "relu",
        output_activation: str = "softmax",
        weight_init: str = "he",
        optimizer: str = "adam",
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 100,
        momentum_beta: float = 0.9,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-8,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
        dropout_rate: float = 0.0,
        early_stopping: bool = True,
        patience: int = 15,
        min_delta: float = 1e-4,
        validation_split: float = 0.0,
        lr_scheduler: str = "none",
        lr_decay: float = 0.5,
        lr_decay_epochs: int = 20,
        min_learning_rate: float = 1e-6,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> None:
        """Initialize architecture, optimizer state, and training configuration."""

        self.input_size = int(input_size)
        self.hidden_layers = [int(units) for units in hidden_layers]
        self.output_size = int(output_size)
        self.activation_name = activation
        self.output_activation_name = output_activation
        self.weight_init = weight_init
        self.optimizer_name = optimizer
        self.learning_rate = float(learning_rate)
        self.initial_learning_rate = float(learning_rate)
        self.batch_size = int(batch_size)
        self.epochs = int(epochs)
        self.momentum_beta = float(momentum_beta)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_epsilon = float(adam_epsilon)
        self.l1_lambda = float(l1_lambda)
        self.l2_lambda = float(l2_lambda)
        self.dropout_rate = float(dropout_rate)
        self.early_stopping = bool(early_stopping)
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.validation_split = float(validation_split)
        self.lr_scheduler = lr_scheduler
        self.lr_decay = float(lr_decay)
        self.lr_decay_epochs = int(lr_decay_epochs)
        self.min_learning_rate = float(min_learning_rate)
        self.shuffle = bool(shuffle)
        self.random_state = int(random_state)

        self._validate_configuration()

        self.rng = np.random.default_rng(self.random_state)
        self.layer_sizes = [self.input_size] + self.hidden_layers + [self.output_size]
        self.parameters: list[LayerParameters] = self._initialize_parameters()
        self.velocity_w: list[np.ndarray] = [np.zeros_like(layer.weights) for layer in self.parameters]
        self.velocity_b: list[np.ndarray] = [np.zeros_like(layer.biases) for layer in self.parameters]
        self.adam_m_w: list[np.ndarray] = [np.zeros_like(layer.weights) for layer in self.parameters]
        self.adam_v_w: list[np.ndarray] = [np.zeros_like(layer.weights) for layer in self.parameters]
        self.adam_m_b: list[np.ndarray] = [np.zeros_like(layer.biases) for layer in self.parameters]
        self.adam_v_b: list[np.ndarray] = [np.zeros_like(layer.biases) for layer in self.parameters]
        self.training_history = TrainingHistory()
        self.best_parameters: Optional[list[LayerParameters]] = None
        self.best_val_loss = np.inf
        self._adam_step = 0
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Train the network using mini-batch gradient-based optimization."""

        X_train, y_train = self._prepare_inputs(X, y)
        X_train, y_train, X_validation, y_validation = self._prepare_validation_data(
            X_train,
            y_train,
            X_val,
            y_val,
        )

        best_epoch = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            batch_losses: list[float] = []

            for batch_X, batch_y in self._iterate_minibatches(X_train, y_train):
                probabilities, caches = self.forward(batch_X, training=True)
                loss = self.compute_loss(batch_y, probabilities)
                gradients = self.backward(batch_y, probabilities, caches)
                self._update_parameters(gradients)
                batch_losses.append(loss)

            epoch_train_loss = float(np.mean(batch_losses)) if batch_losses else np.nan
            self.training_history.train_loss.append(epoch_train_loss)
            self.training_history.learning_rates.append(self.learning_rate)

            epoch_val_loss = np.nan
            if X_validation is not None and y_validation is not None:
                val_probabilities, _ = self.forward(X_validation, training=False)
                epoch_val_loss = self.compute_loss(y_validation, val_probabilities)
                self.training_history.val_loss.append(epoch_val_loss)

                if epoch_val_loss + self.min_delta < self.best_val_loss:
                    self.best_val_loss = epoch_val_loss
                    self.best_parameters = self._copy_parameters()
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.early_stopping and patience_counter >= self.patience:
                    if verbose:
                        LOGGER.info("Early stopping triggered at epoch %d", epoch + 1)
                    break
            else:
                self.training_history.val_loss.append(np.nan)

            self._update_learning_rate(epoch + 1, epoch_val_loss)

            if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
                LOGGER.info(
                    "Epoch %d/%d | train_loss=%.6f | val_loss=%s | lr=%.6f",
                    epoch + 1,
                    self.epochs,
                    epoch_train_loss,
                    f"{epoch_val_loss:.6f}" if not np.isnan(epoch_val_loss) else "N/A",
                    self.learning_rate,
                )

        if self.best_parameters is not None:
            self.parameters = self._copy_parameters(self.best_parameters)
            if verbose:
                LOGGER.info("Restored best model from epoch %d", best_epoch + 1)

        self._is_fitted = True
        return self.training_history

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
    ) -> TrainingHistory:
        """Compatibility alias for fit."""

        return self.fit(X, y, X_val=X_val, y_val=y_val, verbose=verbose)

    def forward(self, X: np.ndarray, *, training: bool = False) -> tuple[np.ndarray, list[LayerCache]]:
        """Perform forward propagation through all layers."""

        activations = self._to_2d_float_array(X)
        caches: list[LayerCache] = []

        for layer_index, layer in enumerate(self.parameters[:-1]):
            z_values = activations @ layer.weights + layer.biases
            activated = self._apply_activation(z_values, self.activation_name)
            dropout_mask = None

            if training and self.dropout_rate > 0.0:
                dropout_mask = self.rng.binomial(1, 1.0 - self.dropout_rate, size=activated.shape)
                activated = (activated * dropout_mask) / (1.0 - self.dropout_rate)

            caches.append(
                LayerCache(
                    inputs=activations,
                    pre_activation=z_values,
                    activation=activated,
                    dropout_mask=dropout_mask,
                )
            )
            activations = activated

        output_layer = self.parameters[-1]
        output_z = activations @ output_layer.weights + output_layer.biases
        output_activation = self._apply_output_activation(output_z)
        caches.append(
            LayerCache(
                inputs=activations,
                pre_activation=output_z,
                activation=output_activation,
            )
        )
        return output_activation, caches

    def backward(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        caches: list[LayerCache],
    ) -> list[Dict[str, np.ndarray]]:
        """Perform backpropagation and compute parameter gradients."""

        labels = self._prepare_targets(y_true)
        one_hot_targets = self._one_hot_encode(labels)
        batch_size = labels.shape[0]

        gradients: list[Dict[str, np.ndarray]] = [dict() for _ in self.parameters]
        delta = (y_pred - one_hot_targets) / batch_size

        output_cache = caches[-1]
        gradients[-1]["weights"] = output_cache.inputs.T @ delta
        gradients[-1]["biases"] = np.sum(delta, axis=0, keepdims=True)

        if self.l1_lambda > 0.0:
            gradients[-1]["weights"] += self.l1_lambda * np.sign(self.parameters[-1].weights) / batch_size
        if self.l2_lambda > 0.0:
            gradients[-1]["weights"] += self.l2_lambda * self.parameters[-1].weights / batch_size

        for layer_index in range(len(self.parameters) - 2, -1, -1):
            next_weights = self.parameters[layer_index + 1].weights
            cache = caches[layer_index]
            delta = delta @ next_weights.T

            if cache.dropout_mask is not None:
                delta = (delta * cache.dropout_mask) / (1.0 - self.dropout_rate)

            delta *= self._activation_derivative(cache.pre_activation, self.activation_name)

            gradients[layer_index]["weights"] = cache.inputs.T @ delta
            gradients[layer_index]["biases"] = np.sum(delta, axis=0, keepdims=True)

            if self.l1_lambda > 0.0:
                gradients[layer_index]["weights"] += (
                    self.l1_lambda * np.sign(self.parameters[layer_index].weights) / batch_size
                )
            if self.l2_lambda > 0.0:
                gradients[layer_index]["weights"] += (
                    self.l2_lambda * self.parameters[layer_index].weights / batch_size
                )

        return gradients

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""

        probabilities, _ = self.forward(X, training=False)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""

        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate the model on a dataset."""

        X_eval, y_eval = self._prepare_inputs(X, y)
        probabilities = self.predict_proba(X_eval)
        loss = self.compute_loss(y_eval, probabilities)
        predictions = np.argmax(probabilities, axis=1)
        accuracy = float(np.mean(predictions == y_eval))
        return {"loss": loss, "accuracy": accuracy}

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute categorical cross-entropy plus optional regularization."""

        labels = self._prepare_targets(y_true)
        one_hot_targets = self._one_hot_encode(labels)
        clipped_predictions = np.clip(y_pred, 1e-12, 1.0)
        cross_entropy = -np.sum(one_hot_targets * np.log(clipped_predictions)) / labels.shape[0]

        l1_penalty = 0.0
        l2_penalty = 0.0
        if self.l1_lambda > 0.0:
            l1_penalty = self.l1_lambda * sum(np.sum(np.abs(layer.weights)) for layer in self.parameters) / labels.shape[0]
        if self.l2_lambda > 0.0:
            l2_penalty = 0.5 * self.l2_lambda * sum(np.sum(np.square(layer.weights)) for layer in self.parameters) / labels.shape[0]

        return float(cross_entropy + l1_penalty + l2_penalty)

    def _initialize_parameters(self) -> list[LayerParameters]:
        """Initialize all network weights according to the selected strategy."""

        parameters: list[LayerParameters] = []

        for input_units, output_units in zip(self.layer_sizes[:-1], self.layer_sizes[1:]):
            weights = self._initialize_weights(input_units, output_units)
            biases = np.zeros((1, output_units), dtype=float)
            parameters.append(LayerParameters(weights=weights, biases=biases))

        return parameters

    def _initialize_weights(self, input_units: int, output_units: int) -> np.ndarray:
        """Initialize one layer of weights."""

        if self.weight_init == "he":
            scale = np.sqrt(2.0 / input_units)
            return self.rng.normal(0.0, scale, size=(input_units, output_units))
        if self.weight_init == "xavier":
            scale = np.sqrt(1.0 / input_units)
            return self.rng.normal(0.0, scale, size=(input_units, output_units))
        if self.weight_init == "xavier_uniform":
            limit = np.sqrt(6.0 / (input_units + output_units))
            return self.rng.uniform(-limit, limit, size=(input_units, output_units))
        if self.weight_init == "uniform":
            return self.rng.uniform(-0.05, 0.05, size=(input_units, output_units))
        if self.weight_init == "normal":
            return self.rng.normal(0.0, 0.01, size=(input_units, output_units))
        raise ValueError(f"Unsupported weight initialization strategy: {self.weight_init}")

    def _iterate_minibatches(self, X: np.ndarray, y: np.ndarray) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Yield mini-batches for one training epoch."""

        sample_count = X.shape[0]
        indices = np.arange(sample_count)

        if self.shuffle:
            self.rng.shuffle(indices)

        for start in range(0, sample_count, self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield X[batch_indices], y[batch_indices]

    def _update_parameters(self, gradients: list[Dict[str, np.ndarray]]) -> None:
        """Apply optimizer updates to model parameters."""

        if self.optimizer_name == "gd":
            self._apply_gradient_descent(gradients)
            return
        if self.optimizer_name == "momentum":
            self._apply_momentum(gradients)
            return
        if self.optimizer_name == "adam":
            self._apply_adam(gradients)
            return
        raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

    def _apply_gradient_descent(self, gradients: list[Dict[str, np.ndarray]]) -> None:
        """Update parameters with vanilla gradient descent."""

        for layer, grads in zip(self.parameters, gradients):
            layer.weights -= self.learning_rate * grads["weights"]
            layer.biases -= self.learning_rate * grads["biases"]

    def _apply_momentum(self, gradients: list[Dict[str, np.ndarray]]) -> None:
        """Update parameters with momentum-based optimization."""

        for index, (layer, grads) in enumerate(zip(self.parameters, gradients)):
            self.velocity_w[index] = (
                self.momentum_beta * self.velocity_w[index]
                - self.learning_rate * grads["weights"]
            )
            self.velocity_b[index] = (
                self.momentum_beta * self.velocity_b[index]
                - self.learning_rate * grads["biases"]
            )
            layer.weights += self.velocity_w[index]
            layer.biases += self.velocity_b[index]

    def _apply_adam(self, gradients: list[Dict[str, np.ndarray]]) -> None:
        """Update parameters with the Adam optimizer."""

        self._adam_step += 1

        for index, (layer, grads) in enumerate(zip(self.parameters, gradients)):
            self.adam_m_w[index] = self.adam_beta1 * self.adam_m_w[index] + (1.0 - self.adam_beta1) * grads["weights"]
            self.adam_v_w[index] = self.adam_beta2 * self.adam_v_w[index] + (1.0 - self.adam_beta2) * np.square(grads["weights"])
            self.adam_m_b[index] = self.adam_beta1 * self.adam_m_b[index] + (1.0 - self.adam_beta1) * grads["biases"]
            self.adam_v_b[index] = self.adam_beta2 * self.adam_v_b[index] + (1.0 - self.adam_beta2) * np.square(grads["biases"])

            m_hat_w = self.adam_m_w[index] / (1.0 - self.adam_beta1 ** self._adam_step)
            v_hat_w = self.adam_v_w[index] / (1.0 - self.adam_beta2 ** self._adam_step)
            m_hat_b = self.adam_m_b[index] / (1.0 - self.adam_beta1 ** self._adam_step)
            v_hat_b = self.adam_v_b[index] / (1.0 - self.adam_beta2 ** self._adam_step)

            layer.weights -= self.learning_rate * m_hat_w / (np.sqrt(v_hat_w) + self.adam_epsilon)
            layer.biases -= self.learning_rate * m_hat_b / (np.sqrt(v_hat_b) + self.adam_epsilon)

    def _update_learning_rate(self, epoch: int, val_loss: float) -> None:
        """Adjust learning rate according to the configured scheduler."""

        if self.lr_scheduler == "none":
            return

        if self.lr_scheduler == "step" and epoch % self.lr_decay_epochs == 0:
            self.learning_rate = max(self.learning_rate * self.lr_decay, self.min_learning_rate)
            return

        if self.lr_scheduler == "plateau":
            if not np.isnan(val_loss) and val_loss > self.best_val_loss - self.min_delta:
                self.learning_rate = max(self.learning_rate * self.lr_decay, self.min_learning_rate)
            return

        raise ValueError(f"Unsupported learning rate scheduler: {self.lr_scheduler}")

    def _prepare_validation_data(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare validation data from explicit inputs or internal splitting."""

        if X_val is not None and y_val is not None:
            X_validation, y_validation = self._prepare_inputs(X_val, y_val)
            return X_train, y_train, X_validation, y_validation

        if self.validation_split <= 0.0:
            return X_train, y_train, None, None

        sample_count = X_train.shape[0]
        split_index = int(sample_count * (1.0 - self.validation_split))
        if split_index <= 0 or split_index >= sample_count:
            raise ValueError("validation_split results in an invalid training/validation partition.")

        X_validation = X_train[split_index:]
        y_validation = y_train[split_index:]
        X_train_subset = X_train[:split_index]
        y_train_subset = y_train[:split_index]
        return X_train_subset, y_train_subset, X_validation, y_validation

    def _prepare_inputs(self, X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Validate and normalize input arrays."""

        features = self._to_2d_float_array(X)
        labels = self._prepare_targets(y)

        if features.shape[0] != labels.shape[0]:
            raise ValueError("X and y must contain the same number of samples.")
        if features.shape[1] != self.input_size:
            raise ValueError(f"Expected input dimension {self.input_size}, received {features.shape[1]}.")

        return features, labels

    def _prepare_targets(self, y: np.ndarray) -> np.ndarray:
        """Convert targets to a one-dimensional integer array."""

        targets = np.asarray(y)
        if targets.ndim > 1:
            targets = targets.reshape(-1)
        if targets.size == 0:
            raise ValueError("Target array cannot be empty.")

        targets = targets.astype(int)
        if np.any(targets < 0) or np.any(targets >= self.output_size):
            raise ValueError("Target labels must be integer encoded in the range [0, output_size).")
        return targets

    @staticmethod
    def _to_2d_float_array(X: np.ndarray) -> np.ndarray:
        """Convert input to a two-dimensional float array."""

        array = np.asarray(X, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError("Input features must be a two-dimensional array.")
        return array

    def _one_hot_encode(self, y: np.ndarray) -> np.ndarray:
        """One-hot encode integer labels."""

        encoded = np.zeros((y.shape[0], self.output_size), dtype=float)
        encoded[np.arange(y.shape[0]), y] = 1.0
        return encoded

    def _apply_activation(self, values: np.ndarray, activation_name: str) -> np.ndarray:
        """Apply a hidden-layer activation function."""

        if activation_name == "relu":
            return np.maximum(0.0, values)
        if activation_name == "tanh":
            return np.tanh(values)
        if activation_name == "sigmoid":
            clipped = np.clip(values, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        if activation_name == "leaky_relu":
            return np.where(values > 0.0, values, 0.01 * values)
        raise ValueError(f"Unsupported activation function: {activation_name}")

    def _activation_derivative(self, values: np.ndarray, activation_name: str) -> np.ndarray:
        """Compute activation derivatives for backpropagation."""

        if activation_name == "relu":
            return (values > 0.0).astype(float)
        if activation_name == "tanh":
            return 1.0 - np.square(np.tanh(values))
        if activation_name == "sigmoid":
            sigmoid_values = self._apply_activation(values, "sigmoid")
            return sigmoid_values * (1.0 - sigmoid_values)
        if activation_name == "leaky_relu":
            return np.where(values > 0.0, 1.0, 0.01)
        raise ValueError(f"Unsupported activation derivative for: {activation_name}")

    def _apply_output_activation(self, values: np.ndarray) -> np.ndarray:
        """Apply the configured output activation."""

        if self.output_activation_name == "softmax":
            shifted = values - np.max(values, axis=1, keepdims=True)
            exp_values = np.exp(shifted)
            return exp_values / np.sum(exp_values, axis=1, keepdims=True)
        if self.output_activation_name == "sigmoid":
            clipped = np.clip(values, -500.0, 500.0)
            return 1.0 / (1.0 + np.exp(-clipped))
        raise ValueError(f"Unsupported output activation: {self.output_activation_name}")

    def _copy_parameters(
        self,
        source: Optional[list[LayerParameters]] = None,
    ) -> list[LayerParameters]:
        """Create a deep copy of model parameters."""

        layers = source if source is not None else self.parameters
        return [
            LayerParameters(weights=np.copy(layer.weights), biases=np.copy(layer.biases))
            for layer in layers
        ]

    def _validate_configuration(self) -> None:
        """Validate initialization settings."""

        if self.input_size <= 0:
            raise ValueError("input_size must be strictly positive.")
        if self.output_size <= 1:
            raise ValueError("output_size must be greater than 1.")
        if not self.hidden_layers:
            raise ValueError("hidden_layers must contain at least one layer.")
        if any(units <= 0 for units in self.hidden_layers):
            raise ValueError("All hidden layer sizes must be strictly positive.")
        if self.optimizer_name not in {"gd", "momentum", "adam"}:
            raise ValueError("optimizer must be one of {'gd', 'momentum', 'adam'}.")
        if self.weight_init not in {"he", "xavier", "xavier_uniform", "uniform", "normal"}:
            raise ValueError("Unsupported weight initialization strategy.")
        if self.activation_name not in {"relu", "tanh", "sigmoid", "leaky_relu"}:
            raise ValueError("Unsupported hidden activation function.")
        if self.output_activation_name != "softmax":
            raise ValueError("Only softmax output activation is currently supported for classification.")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be strictly positive.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be strictly positive.")
        if self.epochs <= 0:
            raise ValueError("epochs must be strictly positive.")
        if not 0.0 <= self.dropout_rate < 1.0:
            raise ValueError("dropout_rate must satisfy 0.0 <= dropout_rate < 1.0.")
        if self.patience <= 0:
            raise ValueError("patience must be strictly positive.")
        if not 0.0 <= self.validation_split < 1.0:
            raise ValueError("validation_split must satisfy 0.0 <= validation_split < 1.0.")
        if self.lr_scheduler not in {"none", "step", "plateau"}:
            raise ValueError("lr_scheduler must be one of {'none', 'step', 'plateau'}.")
        if not 0.0 < self.lr_decay <= 1.0:
            raise ValueError("lr_decay must satisfy 0.0 < lr_decay <= 1.0.")
        if self.lr_decay_epochs <= 0:
            raise ValueError("lr_decay_epochs must be strictly positive.")
        if self.min_learning_rate <= 0.0:
            raise ValueError("min_learning_rate must be strictly positive.")


__all__ = [
    "LayerCache",
    "LayerParameters",
    "NeuralNetwork",
    "TrainingHistory",
]
