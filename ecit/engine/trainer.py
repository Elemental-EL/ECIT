from ..layers.dense import DenseLayer
from ..math.matrix import Matrix
from ..optimizers.optimizers import Optimizer
from .callback import Callback, StepRecord


class TrainResult:
    def __init__(self, history: list[StepRecord], converged: bool, epochs_run: int, final_accuracy: float):
        self.history = history
        self.converged = converged
        self.epochs_run = epochs_run
        self.final_accuracy = final_accuracy

class Trainer:
    def __init__(self, max_epochs: int, stop_on_convergence: bool = True):
        if max_epochs < 1:
            raise ValueError("Invalid epoch number")
        self.max_epochs = max_epochs
        self.stop_on_convergence = stop_on_convergence
    
    def fit(self, layer: DenseLayer, optimizer: Optimizer, X: Matrix, T: Matrix, callbacks: list[Callback] | None = None) -> TrainResult:
        if not X.shape()[0] == T.shape()[0]:
            raise ValueError("Number of Samples must match")
        if X.shape()[1] != layer.in_features or T.shape()[1] != layer.out_features:
            raise ValueError("Invalid input or output sizes")
        if callbacks is None: callbacks = []
        
        history: list[StepRecord] = []
        converged = False
        epochs_run = 0
        num_samples = X.shape()[0]
        for cb in callbacks:
            cb.on_train_begin(layer)
        
        for epoch in range(1, self.max_epochs + 1):
            epochs_run = epoch
            epoch_changed = False
            for cb in callbacks:
                cb.on_epoch_begin(epoch, layer)
            for step_index in range(num_samples):
                x_p = X.row(step_index)
                t_p = T.row(step_index)
                y_p = layer.forward(x_p)
                net_input_p = layer.last_net_input
                assert net_input_p is not None
                step_changed, delta_w, delta_b = optimizer.step(layer, t_p)
                epoch_changed = epoch_changed or step_changed
                record = StepRecord(epoch=epoch, step_index=step_index, X=x_p, target=t_p, net_input=net_input_p, output=y_p, delta_w=delta_w, delta_b=delta_b, weights_after=layer.weights, biases_after=layer.biases, changed=step_changed)
                history.append(record)
                for cb in callbacks:
                    cb.on_step_end(record, layer)
            for cb in callbacks:
                cb.on_epoch_end(epoch, epoch_changed, layer)
            if self.stop_on_convergence is True and not epoch_changed:
                converged = True
                break
        accuracy = self.evaluate(layer, X, T)
        for cb in callbacks:
            cb.on_train_end(history, converged, epochs_run, layer)
        return TrainResult(history=history, converged=converged, epochs_run=epochs_run, final_accuracy=accuracy)

    def evaluate(self, layer: DenseLayer, X: Matrix, T: Matrix) -> float:
        if not X.shape()[0] == T.shape()[0]:
            raise ValueError("Number of Samples must match")
        if X.shape()[1] != layer.in_features or T.shape()[1] != layer.out_features:
            raise ValueError("Invalid input or output sizes")
        
        correct = 0
        num_samples = X.shape()[0]
        for i in range(num_samples):
            pred = layer(X.row(i))
            if pred.not_equal_mask(T.row(i)).is_zero():
                correct += 1
        return float(correct) / float(num_samples)