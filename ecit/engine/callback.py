from ..math.matrix import Matrix


class StepRecord:
    def __init__(self, epoch: int, step_index: int, X: Matrix, target: Matrix, net_input: Matrix, output: Matrix, delta_w: Matrix, delta_b: Matrix | None, weights_after: Matrix, biases_after: Matrix | None, changed: bool) -> None:
        self.epoch = epoch
        self.step_index = step_index
        self.X = X.copy()
        self.target = target.copy()
        self.net_input = net_input.copy()
        self.output = output.copy()
        self.delta_w = delta_w.copy()
        self.delta_b = delta_b.copy() if delta_b is not None else None
        self.weights_after = weights_after.copy()
        self.biases_after = biases_after.copy() if biases_after is not None else None
        self.changed = changed


class Callback:
    def on_train_begin(self, layer):
        pass

    def on_epoch_begin(self, epoch: int, layer):
        pass

    def on_step_end(self, record: StepRecord, layer):
        pass

    def on_epoch_end(self, epoch: int, epoch_changed: bool, layer):
        pass

    def on_train_end(self, history: list[StepRecord], converged: bool, epochs_run: int, layer):
        pass


class TableLoggerCallback(Callback):
    def on_train_begin(self, layer):
        n = layer.in_features
        headers = [
            "Epoch", "Step",
            *[f"x{i+1}" for i in range(n)],
            "t", "y_NI", "y",
            *[f"dW{i+1}" for i in range(n)], "db",
            *[f"W{i+1}" for i in range(n)], "b"
        ]
        header_line = " | ".join(f"{h:>6}" for h in headers)
        self.line_width = len(header_line)
        print("\n")
        print("-" * self.line_width)
        print(header_line)
        print("-" * self.line_width)
        
    def on_step_end(self, record: StepRecord, layer):
        x_vals = record.X.matrix[0]
        t_val = record.target.item()
        yni_val = record.net_input.item()
        y_val = record.output.item()
        dw_vals = [row[0] for row in record.delta_w.matrix]
        db_val = record.delta_b.item() if record.delta_b is not None else 0.0
        w_vals = [row[0] for row in record.weights_after.matrix]
        b_val = record.biases_after.item() if record.biases_after is not None else 0.0
        row_parts = [
            f"{record.epoch:>6d}",
            f"{record.step_index + 1:>6d}"
        ]
        float_vals = [
            *x_vals, 
            t_val, 
            yni_val, 
            y_val, 
            *dw_vals, 
            db_val, 
            *w_vals, 
            b_val
        ]
        row_parts.extend(f"{val:>6.2f}" for val in float_vals)
        row_line = " | ".join(row_parts)
        print(row_line)
    
    def on_train_end(self, history: list[StepRecord], converged: bool, epochs_run: int, layer):
        print("-" * self.line_width)
        print(f"Epochs Run: {epochs_run} | Converged: {converged} | Total Steps: {len(history)}")

class StepSnapshotCallback(Callback):
    def __init__(self) -> None:
        self.records: list[StepRecord] = []
        self.threshold: float = 0.0
    
    def on_train_begin(self, layer):
        self.records.clear()
        self.threshold = float(getattr(layer.activation, "threshold", 0.0))
    
    def on_step_end(self, record: StepRecord, layer):
        self.records.append(record)
    
    def get_hyperplane_at(self, index: int, output_index: int = 0) -> tuple[Matrix, float, float]:
        if index < 0 or index >= len(self.records):
            raise ValueError("Step index out of bounds")
        record = self.records[index]
        w_col = record.weights_after.col(output_index)
        b_val = record.biases_after.matrix[0][output_index] if record.biases_after is not None else 0.0
        return (w_col, b_val, self.threshold)