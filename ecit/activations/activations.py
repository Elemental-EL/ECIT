import math
from abc import ABC, abstractmethod


class Activation(ABC):
    
    @abstractmethod
    def forward(self, net_input: float) -> float:
        pass
    
    @abstractmethod
    def derivative(self, net_input: float) -> float:
        pass
    
    def __call__(self, net_input: float) -> float:
        return self.forward(net_input)

class HardLimit(Activation):
    
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)
    
    def forward(self, net_input: float) -> float:
        return 1.0 if net_input >= self.threshold else 0.0
    
    def derivative(self, net_input: float) -> float:
        raise NotImplementedError("HardLimit function is non-differentiable at the threshold and zero elsewhere")

class SymmetricalHardLimit(Activation):
    
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)
    
    def forward(self, net_input: float) -> float:
        return 1.0 if net_input >= self.threshold else -1.0
    
    def derivative(self, net_input: float) -> float:
        raise NotImplementedError("SymmetricalHardLimit function is non-differentiable at the threshold and zero elsewhere")

class Linear(Activation):
    
    def forward(self, net_input: float) -> float:
        return net_input
    
    def derivative(self, net_input: float) -> float:
        return 1.0
    
class SaturatingLinear(Activation):
    
    def forward(self, net_input: float) -> float:
        if net_input < 0.0:
            return 0.0
        elif 0.0 <= net_input <= 1.0:
            return net_input
        else: return 1.0
    
    def derivative(self, net_input: float) -> float:
        return 1.0 if 0.0 <= net_input <= 1.0 else 0.0

class SymmetricSaturatingLinear(Activation):
    
    def forward(self, net_input: float) -> float:
        if net_input < -1.0:
            return -1.0
        elif -1.0 <= net_input <= 1.0:
            return net_input
        else: return 1.0
    
    def derivative(self, net_input: float) -> float:
        return 1.0 if -1.0 <= net_input <= 1.0 else 0.0

class LogSigmoid(Activation):
    
    def forward(self, net_input: float) -> float:
        return 1.0 / (1.0 + math.exp(-net_input))
    
    def derivative(self, net_input: float) -> float:
        net_input = self.forward(net_input)
        return net_input * (1.0 - net_input)

class HyperbolicTangentSigmoid(Activation):
    
    def forward(self, net_input: float) -> float:
        return math.tanh(net_input)
    
    def derivative(self, net_input: float) -> float:
        net_input = self.forward(net_input)
        return 1.0 - (net_input ** 2)

class PositiveLinear(Activation):
    
    def __init__(self, threshold: float = 0.0) -> None:
        self.threshold = float(threshold)
    
    def forward(self, net_input: float) -> float:
        return 0.0 if net_input < self.threshold else net_input
    
    def derivative(self, net_input: float) -> float:
        return 1.0 if net_input >= self.threshold else 0.0

class PerceptronStep(Activation):
    
    def __init__(self, threshold: float = 0.0, tolerance: float = 1e-9) -> None:
        if threshold < 0.0:
            raise ValueError("Threshold must be non-negative")
        self.threshold = float(threshold)
        self.tolerance = float(tolerance)
    
    def forward(self, net_input: float) -> float:
        if net_input > self.threshold + self.tolerance:
            return 1.0
        elif -self.threshold - self.tolerance <= net_input <= self.threshold + self.tolerance:
            return 0.0
        else: return -1.0
    
    def derivative(self, net_input: float) -> float:
            raise NotImplementedError("PerceptronStep function is non-differentiable at the threshold and zero elsewhere")