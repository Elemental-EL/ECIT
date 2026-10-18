from ..math.matrix import Matrix


def _to_bipolar(n: float) -> float:
    return 1.0 if n >= 0.5 else -1.0

def _to_binary(n: float) -> float:
    return 1.0 if n > 0.0 else 0.0

def binary_to_bipolar(matrix: Matrix) -> Matrix:
    return matrix.apply(_to_bipolar)

def bipolar_to_binary(matrix: Matrix) -> Matrix:
    return matrix.apply(_to_binary)

def encode_grid(grid_2d, active_symbol: str = "#", mode: str = "bipolar") -> Matrix:
    if mode.lower() == "bipolar":
        active_value = 1.0
        inactive_value = -1.0
    elif mode.lower() == "binary":
        active_value = 1.0
        inactive_value = 0.0
    else:
        raise ValueError("Encoding mode can only be either bipolar or binary")
    if len(grid_2d) < 1:
        raise ValueError("Grid cannot have size of zero")
    col_size = len(grid_2d[0])
    if col_size < 1:
        raise ValueError("Row cannot have length of zero")
    for row in grid_2d:
        if len(row) != col_size:
            raise ValueError("Row cannot have varying length")
    list_2d = []
    for i in range(len(grid_2d)):
        temp_list = []
        for j in range(col_size):
            if grid_2d[i][j] == active_symbol:
                temp_list.append(active_value)
            else: temp_list.append(inactive_value)
        list_2d.append(temp_list)
    return Matrix(list_2d).reshape(1, len(list_2d) * col_size)

def decode_grid(vector: Matrix, rows: int, cols: int, active_symbol: str = "#", inactive_symbol: str = ".", threshold: float = 0.0) -> list[str]:
    reshaped = vector.reshape(rows, cols)
    decoded_grid = []
    for i in range(rows):
        temp_list = []
        for j in range(cols):
            if reshaped.matrix[i][j] > threshold:
                temp_list.append(active_symbol)
            else: temp_list.append(inactive_symbol)
        decoded_grid.append("".join(temp_list))
    return decoded_grid

def encode_targets(class_indices: list[int], num_classes: int, mode: str = "bipolar") -> Matrix:
    if mode.lower() == "bipolar":
        positive = 1.0
        negative = -1.0
    elif mode.lower() == "binary":
        positive = 1.0
        negative = 0.0
    else:
        raise ValueError("Encoding mode can only be either bipolar or binary")
    if len(class_indices) == 0:
        raise ValueError("Class indices must not be empty")
    if num_classes < 2:
        raise ValueError("This function can only be used for multi-category instances")
    
    target_rows = []
    for c in class_indices:
        if not isinstance(c, int) or isinstance(c, bool):
            raise TypeError("Class index must be an integer")
        if 0 <= c < num_classes:
            row = [negative for _ in range(num_classes)]
            row[c] = positive
            target_rows.append(row)
        else:
            raise ValueError("Class index out of bounds")
    return Matrix(target_rows)
