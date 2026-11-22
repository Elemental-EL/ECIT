import random

from ecit.math.matrix import Matrix


def _slice_rows(matrix: Matrix, indices: list[int]) -> Matrix | None:
    if len(indices) == 0:
        return None
    return Matrix([list(matrix.matrix[index]) for index in indices])

def train_val_test_split(X: Matrix, T: Matrix, train_ratio: float = 0.70, val_ratio: float = 0.15, test_ratio: float = 0.15, shuffle: bool = True, seed: int | None = 100) -> tuple[Matrix, Matrix, Matrix | None, Matrix | None, Matrix | None, Matrix | None]:
    if not (X.row_size == T.row_size and X.row_size >= 1):
        raise ValueError("Invalid Input or Output")
    if not (train_ratio > 0.0 and val_ratio >= 0.0 and test_ratio >= 0.0 and abs(train_ratio + val_ratio + test_ratio - 1.0) <= 1e-9):
        raise ValueError("Invalid Split Ratio")
    
    rng = random.Random(seed)
    indices = list(range(X.row_size))
    if shuffle:
        rng.shuffle(indices)
    
    n_train = round(X.row_size * train_ratio)
    if test_ratio == 0.0:
        n_val = X.row_size - n_train if val_ratio > 0.0 else 0
    else:
        n_val = round(X.row_size * val_ratio) if val_ratio > 0.0 else 0
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train : n_train + n_val]
    test_indices = indices[n_train + n_val :]
    
    X_train = _slice_rows(X, train_indices)
    T_train = _slice_rows(T, train_indices)
    assert X_train and T_train is not None
    X_val = _slice_rows(X, val_indices)
    T_val = _slice_rows(T, val_indices)
    X_test = _slice_rows(X, test_indices)
    T_test = _slice_rows(T, test_indices)
    
    return (X_train, T_train, X_val, T_val, X_test, T_test)