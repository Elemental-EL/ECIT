from ecit.math.matrix import Matrix

from .dense import DenseLayer


class MultiLayerNetwork:
    def __init__(self, layers: list[DenseLayer]) -> None:
        self.layers = layers
        if len(layers) < 2:
            raise ValueError("A MultiLayerNetwork cannot have less than 2 layers")
        for i in range(len(layers)-1):
            if layers[i].out_features != layers[i+1].in_features:
                raise ValueError("Mismatch between two or more sequential layers output and input size")
        for layer in layers:
            if layer.weights.is_zero():
                raise ValueError("Layers cannot have initial weight of zero in a MultiLayerNetwork")
    
    @property
    def in_features(self) -> int:
        return self.layers[0].in_features
    
    @property
    def out_features(self) -> int:
        return self.layers[-1].out_features
    
    @property
    def last_net_input(self) -> Matrix | None:
        return self.layers[-1].last_net_input
    
    @property
    def last_output(self) -> Matrix | None:
        return self.layers[-1].last_output
    
    def forward(self, X: Matrix) -> Matrix:
        for layer in self.layers:
            X = layer(X=X)
        return X

    def __call__(self, X: Matrix) -> Matrix:
            return self.forward(X=X)
    
    def get_weights_snapshot(self) -> list[tuple[Matrix, Matrix | None]]:
        weights_snapshot = []
        for layer in self.layers:
            weights_snapshot.append((layer.weights.copy(), layer.biases.copy() if layer.use_bias and layer.biases is not None else None))
        return weights_snapshot
    
    def load_weights_snapshot(self, weights_snapshot: list[tuple[Matrix, Matrix | None]]) -> None:
        if len(weights_snapshot) != len(self.layers):
            raise ValueError("Invalid weights snapshot")
        for i, layer in enumerate(self.layers):
            (w_snap, b_snap) = weights_snapshot[i]
            if w_snap.shape() == layer.weights.shape():
                layer.weights = w_snap.copy()
                layer.biases = b_snap.copy() if b_snap is not None else None
            else: raise ValueError("Invalid weight snapshot")
