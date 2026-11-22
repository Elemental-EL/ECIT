from abc import ABC, abstractmethod

from ecit.layers.dense import DenseLayer
from ecit.layers.network import MultiLayerNetwork
from ecit.math.matrix import Matrix


class Optimizer(ABC):
    @abstractmethod
    def step(self, layer: DenseLayer | MultiLayerNetwork, target: Matrix) -> tuple[bool, Matrix, Matrix | None]:
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


class AdalineRule(Optimizer):
    
    def __init__(self, learning_rate: float = 0.1, threshold: float = 0.01) -> None:
            if 0.0 < learning_rate < 1.0:
                self.learning_rate = learning_rate
            else: raise ValueError("Learning rate must be between 0.0 and 1.0")
            if 0.0 < threshold <= 1.0:
                self.threshold = threshold
            else: raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def step(self, layer: DenseLayer, target: Matrix) -> tuple[bool, Matrix, Matrix | None]:
            if layer.last_input is None or layer.last_net_input is None:
                raise ValueError("Cannot call optimizer.step() without calling layer.forward()")
            if not layer.last_input.shape()[0] == 1:
                raise ValueError()
            if not target.shape() == (1, layer.out_features):
                raise ValueError()
            x = layer.last_input
            y = layer.last_net_input
            if not y.shape()[0] == 1:
                raise ValueError("Invalid output shape")
            error_mask = target.subtract(y)
            scaled_target = error_mask.scalar_mult(self.learning_rate)
            delta_w = x.transpose().matrix_mult(scaled_target)
            layer.weights = layer.weights.add(delta_w)
            if layer.use_bias and layer.biases is not None:
                delta_b = scaled_target.copy()
                layer.biases = layer.biases.add(delta_b)
            else: delta_b = None
            changed = (not delta_w.is_zero(self.threshold)) or (delta_b is not None and not delta_b.is_zero(self.threshold))
            return (changed, delta_w, delta_b)


class BackpropagationRule(Optimizer):
    
    def __init__(self, learning_rate: float = 1.0) -> None:
            if 0.0 < learning_rate <= 1.0:
                self.learning_rate = learning_rate
            else: raise ValueError("Learning rate must be between 0.0 and 1.0")
    
    def step(self, network: MultiLayerNetwork, target: Matrix) -> tuple[bool, list[Matrix], list[Matrix] | None]:
        for layer in network.layers:
            if layer.last_input is None or layer.last_net_input is None or layer.last_output is None:
                raise ValueError("Cannot call optimizer.step() without calling layer.forward()")
            if not layer.last_output.shape()[0] == 1:
                raise ValueError()
        if not target.shape() == (1, network.out_features):
            raise ValueError()
        delta_ws = []
        delta_bs = []
        current_delta: Matrix | None = None
        changed = False
        for i, layer in reversed(list(enumerate(network.layers))):
            x_in = layer.last_input
            net_in = layer.last_net_input
            out = layer.last_output
            f_prime = net_in.apply(layer.activation.derivative) # type: ignore
            if i == len(network.layers) - 1:
                error = target.subtract(out) # type: ignore
                current_delta = error.hadamard_mult(f_prime)
            else:
                next_layer = network.layers[i + 1]
                d_hidden = current_delta.matrix_mult(next_layer.weights.transpose()) # type: ignore
                current_delta = d_hidden.hadamard_mult(f_prime)
            scaled_delta = current_delta.scalar_mult(self.learning_rate)
            delta_w = x_in.transpose().matrix_mult(scaled_delta) # type: ignore
            delta_b = scaled_delta.copy() if layer.use_bias and layer.biases is not None else None
            delta_ws.insert(0, delta_w)
            delta_bs.insert(0, delta_b)
        for i, layer in enumerate(network.layers):
            layer.weights = layer.weights.add(delta_ws[i])
            if layer.biases and delta_bs[i] is not None:
                layer.biases = layer.biases.add(delta_bs[i])
            if (not delta_ws[i].is_zero()) or (delta_bs[i] is not None and not delta_bs[i].is_zero()):
                changed = True
        return (changed, delta_ws, delta_bs)