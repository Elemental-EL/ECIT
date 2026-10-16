from __future__ import annotations

import copy
import itertools
import random


class Matrix:
    def __init__(self, list_2d: list[list[float]]) -> None:
        if len(list_2d) < 1:
            raise ValueError("Matrix cannot have size of zero")
        col_size = len(list_2d[0])
        if col_size < 1:
            raise ValueError("Row cannot have length of zero")
        for row in list_2d:
            if len(row) != col_size:
                raise ValueError("Row cannot have varying length")
            
        self.matrix = [[float(item) for item in row] for row in list_2d]
        self.row_size = len(list_2d)
        self.col_size = col_size
        
    def __str__(self) -> str:
        return "\n".join([str(row) for row in self.matrix])
    
    @classmethod
    def fill(cls, rows, columns, scalar):
        return cls([[float(scalar) for _ in range(columns)] for _ in range(rows)])
    
    @classmethod
    def zeros(cls, rows, columns):
        return cls.fill(rows=rows, columns=columns, scalar=0.0)
    
    @classmethod
    def ones(cls, rows, columns):
        return cls.fill(rows=rows, columns=columns, scalar=1.0)
    
    @classmethod
    def random_uniform(cls, rows, columns, lower_bound, upper_bound, seed=None):
        if seed is not None:
            random.seed(seed)
        return cls([[float(random.uniform(lower_bound, upper_bound)) for _ in range(columns)] for _ in range(rows)])
        
    @classmethod
    def from_row(cls, row: list[float]):
        return cls([row])
        
    @classmethod
    def from_col(cls, column: list[float]):
        return cls([[item] for item in column])
    
    def shape(self) -> tuple[int, int]:
        return (self.row_size, self.col_size)
    
    def copy(self):
        return copy.deepcopy(self)
    
    def transpose(self):
        return self.__class__([list(row) for row in zip(*self.matrix)])
    
    def reshape(self, row_size, col_size):
        if self.shape()[0] * self.shape()[1] == row_size * col_size:
            flat_iterator = (item for row in self.matrix for item in row)
            return self.__class__([list(itertools.islice(flat_iterator, col_size)) for _ in range(row_size)])
        else:
            raise ValueError("The matrix cannot be reshaped to this row and column size")
    
    def row(self, n):
        return self.__class__.from_row(self.matrix[n])
    
    def col(self, n):
        return self.__class__.from_col([row[n] for row in self.matrix])
    
    def item(self):
        if self.shape()[0] == 1 and self.shape()[1] == 1:
            return self.matrix[0][0]
        else:
            raise ValueError("There are more than one item in this matrix to unpack")
    
    def add(self, secondary_matrix: Matrix):
        if self.shape() == secondary_matrix.shape():
            result = copy.deepcopy(self)
            for i in range(self.shape()[0]):
                for j in range(self.shape()[1]):
                    result.matrix[i][j] += secondary_matrix.matrix[i][j]
            return result
        elif self.shape()[1] == secondary_matrix.shape()[1] and secondary_matrix.shape()[0] == 1:
            result = copy.deepcopy(self)
            for i in range(self.shape()[0]):
                for j in range(self.shape()[1]):
                    result.matrix[i][j] += secondary_matrix.matrix[0][j]
            return result
        elif self.shape()[1] == secondary_matrix.shape()[1] and self.shape()[0] == 1:
            result = copy.deepcopy(secondary_matrix)
            for i in range(secondary_matrix.shape()[0]):
                for j in range(secondary_matrix.shape()[1]):
                    result.matrix[i][j] += self.matrix[0][j]
            return result
        else:
            raise ValueError("Shape mismatch")
        
    def subtract(self, secondary_matrix: Matrix):
        if self.shape() == secondary_matrix.shape():
            result = copy.deepcopy(self)
            for i in range(self.shape()[0]):
                for j in range(self.shape()[1]):
                    result.matrix[i][j] -= secondary_matrix.matrix[i][j]
            return result
        elif self.shape()[1] == secondary_matrix.shape()[1] and secondary_matrix.shape()[0] == 1:
            result = copy.deepcopy(self)
            for i in range(self.shape()[0]):
                for j in range(self.shape()[1]):
                    result.matrix[i][j] -= secondary_matrix.matrix[0][j]
            return result
        elif self.shape()[1] == secondary_matrix.shape()[1] and self.shape()[0] == 1:
            result = copy.deepcopy(secondary_matrix)
            for i in range(secondary_matrix.shape()[0]):
                for j in range(secondary_matrix.shape()[1]):
                    result.matrix[i][j] = self.matrix[0][j] - secondary_matrix.matrix[i][j]
            return result
        else:
            raise ValueError("Shape mismatch")
    
    def scalar_mult(self, scalar: float):
        result = copy.deepcopy(self)
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                result.matrix[i][j] *= float(scalar)
        return result
    
    def hadamard_mult(self, secondary_matrix: Matrix):
        if self.shape() == secondary_matrix.shape():
            result = copy.deepcopy(self)
            for i in range(self.shape()[0]):
                for j in range(self.shape()[1]):
                    result.matrix[i][j] *= secondary_matrix.matrix[i][j]
            return result
        else:
            raise ValueError("Shape mismatch")
    
    def matrix_mult(self, secondary_matrix: Matrix):
        if self.shape()[1] == secondary_matrix.shape()[0]:
            secondary_matrix_T = secondary_matrix.transpose()
            multiplied_matrix = [[float(sum(a * b for a, b in zip(A_row, secondary_matrix_T_row))) for secondary_matrix_T_row in secondary_matrix_T.matrix] for A_row in self.matrix]
            return self.__class__(multiplied_matrix)
        else:
            raise ValueError("Shape mismatch")
    
    def apply(self, f):
        result = copy.deepcopy(self)
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                result.matrix[i][j] = float(f(self.matrix[i][j]))
        return result
    
    def not_equal_mask(self, secondary_matrix: Matrix, epsilon=1e-9):
        if self.shape() == secondary_matrix.shape():
            result = copy.deepcopy(self)
            for i in range(self.shape()[0]):
                for j in range(self.shape()[1]):
                    result.matrix[i][j] = 1.0 if abs(self.matrix[i][j] - secondary_matrix.matrix[i][j]) > epsilon else 0.0
            return result
        else:
            raise ValueError("Shape mismatch")
    
    def is_zero(self, epsilon=1e-9):
        for i in range(self.shape()[0]):
            for j in range(self.shape()[1]):
                if abs(self.matrix[i][j]) > epsilon:
                    return False
        return True