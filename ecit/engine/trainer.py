import random

from ecit.data.converters import *
from ecit.layers.dense import DenseLayer
from ecit.layers.network import MultiLayerNetwork
from ecit.math.matrix import Matrix
from ecit.optimizers.optimizers import BackpropagationRule, Optimizer

from .callback import Callback, EpochRecord, StepRecord


class TrainResult:
    def __init__(self, history: list[StepRecord], converged: bool | None, epochs_run: int, final_accuracy: float):
        self.history = history
        self.converged = converged
        self.epochs_run = epochs_run
        self.final_accuracy = final_accuracy

class MLNTrainResult:
    def __init__(self, history: list[EpochRecord], stop_reason: str, epochs_run: int, best_epoch: int) -> None:
        self.history = history
        self.stop_reason = stop_reason
        self.epochs_run = epochs_run
        self.best_epoch = best_epoch
        

class Trainer:
    def __init__(self, max_epochs: int, stop_on_convergence: bool | None = None):
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
        converged = False if self.stop_on_convergence is not None else None
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
            if self.stop_on_convergence is not None and self.stop_on_convergence is True and not epoch_changed:
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

class MLNTrainer:
    def __init__(self, max_epochs: int, target_mse: float = 1e-5, max_val_fails: int = 6, shuffle_each_epoch: bool = False, seed: int | None = None) -> None:
        self.max_epochs = max_epochs
        self.target_mse = target_mse
        self.max_val_fails = max_val_fails
        self.shuffle_each_epoch = shuffle_each_epoch
        self.seed = seed
    
    def fit(self, network: MultiLayerNetwork, optimizer: BackpropagationRule, X_train: Matrix, T_train: Matrix, X_val: Matrix | None = None, T_val: Matrix | None = None, X_test: Matrix | None = None, T_test: Matrix | None = None, verbose: bool = True) -> MLNTrainResult:
        if not X_train.shape()[0] == T_train.shape()[0]:
            raise ValueError("Number of Training Samples must match")
        if (X_val is not None and T_val is None) or (X_val is None and T_val is not None):
            raise ValueError("Either One of Input or Output Validation Samples are missing")
        if (X_test is not None and T_test is None) or (X_test is None and T_test is not None):
            raise ValueError("Either One of Input or Output Test Samples are missing")
        if X_val is not None and T_val is not None and not X_val.shape()[0] == T_val.shape()[0]:
            raise ValueError("Number of Validation Samples must match")
        if X_test is not None and T_test is not None and not X_test.shape()[0] == T_test.shape()[0]:
            raise ValueError("Number of Test Samples must match")
        if X_train.shape()[1] != network.in_features or T_train.shape()[1] != network.out_features:
            raise ValueError("Invalid input or output sizes for Training Samples")
        if X_val is not None and T_val is not None and (X_val.shape()[1] != network.in_features or T_val.shape()[1] != network.out_features):
            raise ValueError("Invalid input or output sizes for Validation Samples")
        if X_test is not None and T_test is not None and (X_test.shape()[1] != network.in_features or T_test.shape()[1] != network.out_features):
            raise ValueError("Invalid input or output sizes for Test Samples")
        history: list[EpochRecord] = []
        best_val_mse = float("inf")
        best_epoch = 0
        best_snapshot = network.get_weights_snapshot()
        val_fail_count = 0
        stop_reason = "Maximum epoch reached"
        epochs_run = 0
        val_mse = val_error = test_mse = test_error = None
        rng = random.Random(self.seed)
        indices = list(range(X_train.row_size))
        for epoch in range(1, self.max_epochs + 1):
            if self.shuffle_each_epoch:
                rng.shuffle(indices)
            for index in indices:
                x_p = X_train.row(index)
                t_p = T_train.row(index)
                network.forward(x_p)
                optimizer.step(network, t_p)
            train_mse = self.compute_mse(network=network, X=X_train, T=T_train)
            train_error = self.compute_classification_metrics(network, X_train, T_train)[1]
            if X_val is not None and T_val is not None:
                val_mse = self.compute_mse(network=network, X=X_val, T=T_val)
                val_error = self.compute_classification_metrics(network, X_val, T_val)[1]
                if val_mse < best_val_mse - 1e-12:
                    best_val_mse = val_mse
                    best_epoch = epoch
                    val_fail_count = 0
                    best_snapshot = network.get_weights_snapshot()
                else:
                    val_fail_count += 1
            if X_test is not None and T_test is not None:
                test_mse = self.compute_mse(network=network, X=X_test, T=T_test)
                test_error = self.compute_classification_metrics(network, X_test, T_test)[1]
            history.append(EpochRecord(epoch, train_mse, train_error, val_mse, val_error, test_mse, test_error, val_fail_count))
            epochs_run = epoch
            if verbose and (epoch == 1 or epoch % 10 == 0):
                val_str = f" | Val MSE: {val_mse:.6f} (fails: {val_fail_count})" if val_mse is not None else ""
                print(f"Epoch {epoch:4d} | Train MSE: {train_mse:.6f} | Train %E: {train_error * 100:.2f}%{val_str}")
            if train_mse <= self.target_mse:
                stop_reason = f"Target MSE ({self.target_mse}) reached"
                break
            if X_val is not None and val_fail_count >= self.max_val_fails:
                stop_reason = f"Validation stop (no improvement for {self.max_val_fails} epochs; restored epoch {best_epoch})"
                break
        if X_val is not None and best_epoch > 0:
            network.load_weights_snapshot(best_snapshot)
        if verbose and not (epoch == 1 or epoch % 10 == 0):
            val_str = f" | Val MSE: {val_mse:.6f} (fails: {val_fail_count})" if val_mse is not None else ""
            print(f"Epoch {epoch:4d} | Train MSE: {train_mse:.6f} | Train %E: {train_error * 100:.2f}%{val_str}")
        if verbose:
            print(f"Stop Reason: {stop_reason}")
        return MLNTrainResult(history, stop_reason, epochs_run, best_epoch)
    
    def compute_mse(self, network: MultiLayerNetwork, X: Matrix, T: Matrix) -> float:
        Y_hat = network(X=X)
        error = T.subtract(Y_hat)
        sq_error = error.hadamard_mult(error)
        mse = sum(sum(row) for row in sq_error.matrix) / (X.row_size * T.col_size)
        return mse
    
    def compute_classification_metrics(self, network: MultiLayerNetwork, X: Matrix, T: Matrix) -> tuple[float, float, list[list[int]]]:
        Y_hat = network(X=X)
        correct = 0
        num_samples = X.shape()[0]
        if T.col_size == 1:
            confusion = [[0, 0], [0, 0]]
            threshold = 0.0 if any(row[0] < 0.0 for row in T.matrix) else 0.5
            for i in range(num_samples):
                y_val = Y_hat.matrix[i][0]
                t_val = T.matrix[i][0]
                pred_idx = 1 if y_val >= threshold else 0
                true_idx = 1 if t_val >= threshold else 0
                confusion[pred_idx][true_idx] += 1
                correct += 1 if pred_idx == true_idx else 0
        else:
            num_classes = T.col_size
            confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
            for i in range(num_samples):
                y_row = Y_hat.matrix[i]
                t_row = T.matrix[i]
                pred_idx = y_row.index(max(y_row))
                true_idx = t_row.index(max(t_row))
                confusion[pred_idx][true_idx] += 1
                correct += 1 if pred_idx == true_idx else 0
        accuracy = float(correct) / float(num_samples)
        percent_error = 1.0 - accuracy
        return (accuracy, percent_error, confusion)