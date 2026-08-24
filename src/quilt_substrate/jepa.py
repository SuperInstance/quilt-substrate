"""
jepa.py — Non-linear JEPA implementations (Open Q9).

The substrate's `predict()` is linear by default (uses the simple sum of inputs).
A non-linear JEPA can learn complex patterns from the witness log.

Two implementations:
1. `LinearJEPA` — the default, weighted sum of inputs
2. `MLPJEPA` — a small multi-layer perceptron, trained via gradient descent
3. `KnnJEPA` — k-nearest neighbors lookup in the witness log

All are callable: jepa(inputs: Dict[str, float]) -> float
"""
from __future__ import annotations
from typing import Dict, List, Tuple, Any, Optional
import math
import random


class LinearJEPA:
    """The default linear JEPA: weighted sum of inputs.

    weights are uniform 1/N where N is the number of inputs.
    """

    def __init__(self, weights: Optional[List[float]] = None):
        self.weights = weights or []

    def __call__(self, inputs: Dict[str, Any]) -> Any:
        if not inputs:
            return None
        values = []
        for k, v in inputs.items():
            if isinstance(v, (int, float)):
                values.append(float(v))
        if not values:
            return list(inputs.values())[0]
        n = len(values)
        if not self.weights or len(self.weights) != n:
            self.weights = [1.0 / n] * n
        return sum(w * v for w, v in zip(self.weights, values))


class MLPJEPA:
    """A small multi-layer perceptron JEPA.

    Architecture: input → hidden → output (1 neuron).
    Trained via simple gradient descent on the witness log.
    """

    def __init__(self, input_dim: int = 1, hidden_dim: int = 4, lr: float = 0.01):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        # Random initialization
        random.seed(42)
        self.w1 = [[random.uniform(-1, 1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.w2 = [random.uniform(-1, 1) for _ in range(hidden_dim)]
        self.b2 = 0.0

    def _relu(self, x: float) -> float:
        return max(0.0, x)

    def __call__(self, inputs: Dict[str, Any]) -> float:
        """Forward pass: predict the value from inputs."""
        if not inputs:
            return 0.0
        # Flatten inputs to a fixed-size vector (pad or truncate)
        x = [0.0] * self.input_dim
        for i, (k, v) in enumerate(inputs.items()):
            if i >= self.input_dim:
                break
            if isinstance(v, (int, float)):
                x[i] = float(v)
        # Hidden layer
        h = [self._relu(sum(x[i] * self.w1[i][j] for i in range(self.input_dim)) + self.b1[j])
             for j in range(self.hidden_dim)]
        # Output
        out = sum(h[j] * self.w2[j] for j in range(self.hidden_dim)) + self.b2
        return out

    def train_step(self, inputs: Dict[str, Any], target: float) -> float:
        """One step of training. Returns the loss."""
        # Forward
        x = [0.0] * self.input_dim
        for i, (k, v) in enumerate(inputs.items()):
            if i >= self.input_dim:
                break
            if isinstance(v, (int, float)):
                x[i] = float(v)
        h_pre = [sum(x[i] * self.w1[i][j] for i in range(self.input_dim)) + self.b1[j]
                 for j in range(self.hidden_dim)]
        h = [self._relu(v) for v in h_pre]
        out = sum(h[j] * self.w2[j] for j in range(self.hidden_dim)) + self.b2
        # Loss
        err = out - target
        loss = err * err
        # Backward (simplified)
        # Output layer
        for j in range(self.hidden_dim):
            self.w2[j] -= self.lr * err * h[j]
        self.b2 -= self.lr * err
        # Hidden layer (only for active ReLU)
        for j in range(self.hidden_dim):
            if h_pre[j] > 0:  # ReLU derivative
                grad = err * self.w2[j]
                for i in range(self.input_dim):
                    self.w1[i][j] -= self.lr * grad * x[i]
                self.b1[j] -= self.lr * grad
        return loss


class KnnJEPA:
    """A k-nearest-neighbors JEPA.

    Stores (inputs, target) pairs from the witness log. At predict time,
    finds the k most similar stored examples and averages their targets.
    """

    def __init__(self, k: int = 3):
        self.k = k
        self.examples: List[Tuple[Dict[str, Any], float]] = []

    def add(self, inputs: Dict[str, Any], target: float) -> None:
        """Add an example to the JEPA's memory."""
        self.examples.append((dict(inputs), float(target)))

    def __call__(self, inputs: Dict[str, Any]) -> Optional[float]:
        """Predict by finding the k most similar examples and averaging."""
        if not self.examples:
            return None
        # Compute distance to each example
        def distance(ex_inputs):
            # Euclidean distance over shared keys
            keys = set(inputs.keys()) & set(ex_inputs.keys())
            if not keys:
                return float('inf')
            total = 0.0
            for k in keys:
                v1 = inputs.get(k, 0)
                v2 = ex_inputs.get(k, 0)
                if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                    total += (float(v1) - float(v2)) ** 2
            return math.sqrt(total)
        # Sort by distance
        sorted_ex = sorted(self.examples, key=lambda ex: distance(ex[0]))
        # Average the k nearest
        k_nearest = sorted_ex[:self.k]
        return sum(target for _, target in k_nearest) / len(k_nearest)


def auto_train_jepa(cell, epochs: int = 50, jepa_type: str = "mlp") -> Any:
    """Auto-train a JEPA from a cell's witness log.

    The witness log records (inputs_dict, target_value) pairs. The
    trained JEPA can then predict the cell's value from its inputs.

    Args:
        cell: A Cell with witness log entries
        epochs: Number of training epochs
        jepa_type: "linear", "mlp", or "knn"
    """
    # Extract training data from the witness log
    # This is a simplification: in practice, the witness log records
    # reads, not training data. For now, we synthesize training data
    # from the cell's value history.
    if jepa_type == "mlp":
        model = MLPJEPA(input_dim=4, hidden_dim=4, lr=0.01)
        # Synthesize training data: random inputs → cell.value
        for epoch in range(epochs):
            for _ in range(10):
                inputs = {f"x{i}": random.uniform(0, 1) for i in range(4)}
                # The "target" is some function of the inputs
                target = cell.value if isinstance(cell.value, (int, float)) else 0.0
                # We don't know the true function, so use a linear combination
                # that approximates the cell's value
                target = sum(inputs.values()) / len(inputs) * target
                model.train_step(inputs, target)
        return model
    elif jepa_type == "knn":
        model = KnnJEPA(k=3)
        for _ in range(100):
            inputs = {f"x{i}": random.uniform(0, 1) for i in range(4)}
            target = cell.value if isinstance(cell.value, (int, float)) else 0.0
            model.add(inputs, target)
        return model
    else:
        return LinearJEPA()
