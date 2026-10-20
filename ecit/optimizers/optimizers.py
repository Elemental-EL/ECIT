from abc import ABC, abstractmethod

from ..layers.dense import DenseLayer
from ..math.matrix import Matrix


class Optimizer(ABC):
    @abstractmethod
    def step(self, layer: DenseLayer, target: Matrix) -> tuple[bool, Matrix, Matrix | None]:
        pass

class HebbianRule(Optimizer):
    def step(self, layer: DenseLayer, target: Matrix) -> tuple[bool, Matrix, Matrix | None]:
        if layer.last_input is None or layer.last_output is None:
            raise ValueError("Cannot call optimizer.step() without calling layer.forward()")
        if not layer.last_input.shape()[0] == 1:
            raise ValueError()
        if not target.shape() == (1, layer.out_features):
            raise ValueError()
        x = layer.last_input
        delta_w = x.transpose().matrix_mult(target)
        layer.weights = layer.weights.add(delta_w)
        if layer.use_bias and layer.biases is not None:
            delta_b = target.copy()
            layer.biases = layer.biases.add(delta_b)
        else: delta_b = None
        changed = (not delta_w.is_zero()) or (delta_b is not None and not delta_b.is_zero())
        return (changed, delta_w, delta_b)

class PerceptronRule(Optimizer):
    
    def __init__(self, learning_rate: float = 1.0) -> None:
        if 0.0 < learning_rate <= 1.0:
            self.learning_rate = learning_rate
        else: raise ValueError("Learning rate must be between 0.0 and 1.0")
    
    def step(self, layer: DenseLayer, target: Matrix) -> tuple[bool, Matrix, Matrix | None]:
        if layer.last_input is None or layer.last_output is None:
            raise ValueError("Cannot call optimizer.step() without calling layer.forward()")
        if not layer.last_input.shape()[0] == 1:
            raise ValueError()
        if not target.shape() == (1, layer.out_features):
            raise ValueError()
        x = layer.last_input
        y = layer.last_output
        if not y.shape()[0] == 1:
            raise ValueError("Invalid output shape")
        error_mask = y.not_equal_mask(target)
        masked_target = target.hadamard_mult(error_mask)
        scaled_target = masked_target.scalar_mult(self.learning_rate)
        delta_w = x.transpose().matrix_mult(scaled_target)
        layer.weights = layer.weights.add(delta_w)
        if layer.use_bias and layer.biases is not None:
            delta_b = scaled_target.copy()
            layer.biases = layer.biases.add(delta_b)
        else: delta_b = None
        changed = (not delta_w.is_zero()) or (delta_b is not None and not delta_b.is_zero())
        return (changed, delta_w, delta_b)
        