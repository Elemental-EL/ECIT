from enum import Enum

from ecit.activations.activations import Activation
from ecit.math.matrix import Matrix


class InitModes(Enum):
    Zeroes = "zeroes"
    Random_Uniform = "random_uniform"

class DenseLayer:
    def __init__(self, in_features: int, out_features: int, activation: Activation, use_bias: bool = True, init_mode: str = InitModes.Zeroes.value, seed= None) -> None:
        if in_features < 1 or out_features < 1:
            raise ValueError("Input and Output sizes should be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.use_bias = use_bias
        self.last_input = None
        self.last_net_input = None
        self.last_output = None
        if init_mode == InitModes.Zeroes.value:
            self.weights = Matrix.zeros(in_features, out_features)
            if use_bias:
                self.biases = Matrix.zeros(1, out_features)
            else: self.biases = None
        elif init_mode == InitModes.Random_Uniform.value:
            self.weights = Matrix.random_uniform(in_features, out_features, -0.5, 0.5, seed=seed)
            if use_bias:
                self.biases = Matrix.random_uniform(1, out_features, -0.5, 0.5, seed=seed if seed is None else seed + 1)
            else: self.biases = None
        else: raise ValueError("Initialization mode not valid")
    
    def forward(self, X: Matrix) -> Matrix:
        if not X.shape()[1] == self.in_features:
            raise ValueError("Input size doesn't match layer's number of input features")
        net_input = X.matrix_mult(self.weights)
        if self.use_bias and self.biases is not None:
            net_input = net_input.add(self.biases)
        output = net_input.apply(self.activation)
        self.last_input = X.copy()
        self.last_net_input = net_input.copy()
        self.last_output = output.copy()
        return output

    def __call__(self, X: Matrix) -> Matrix:
        return self.forward(X=X)
    
    def get_decision_hyperplane(self, output_index: int = 0):
        if not 0 <= output_index < self.out_features:
            raise ValueError("Output Index is out of bounds")
        weight_col = self.weights.col(output_index)
        bias = self.biases.matrix[0][output_index] if self.use_bias and self.biases is not None else 0.0
        threshold = getattr(self.activation, "threshold", 0.0)
        return (weight_col, bias, threshold)