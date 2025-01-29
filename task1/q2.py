#!/usr/bin/env python

from time import time

from numpy import hstack, vstack, zeros, log
from numpy.random import rand, seed
import matplotlib.pyplot as plt

seed(1108)

# a
def divide_matrix(matrix):
	mid = matrix.shape[0] // 2
	return matrix[:mid, :mid], matrix[:mid, mid:], matrix[mid:, :mid], matrix[mid:, mid:]

def combine_matrices(C11, C12, C21, C22):
	top = hstack((C11, C12))
	bottom = hstack((C21, C22))
	return vstack((top, bottom))

def matrix_multiply_rec(A, B):
	n = A.shape[0]
	if n == 1:
		return A * B
	A11, A12, A21, A22 = divide_matrix(A)
	B11, B12, B21, B22 = divide_matrix(B)
	M1 = matrix_multiply_rec(A11+A22, B11+B22)
	M2 = matrix_multiply_rec(A21+A22, B11)
	M3 = matrix_multiply_rec(A11, B12-B22)
	M4 = matrix_multiply_rec(A22, B21-B11)
	M5 = matrix_multiply_rec(A11+A12, B22)
	M6 = matrix_multiply_rec(A21-A11, B11+B12)
	M7 = matrix_multiply_rec(A12-A22, B21+B22)
	C11 = M1 + M4 - M5 + M7
	C12 = M3 + M5
	C21 = M2 + M4
	C22 = M1 - M2 + M3 + M6
	return combine_matrices(C11, C12, C21, C22)

def pad_to_power_of_two(matrix):
	n, m = matrix.shape
	next_power_of_two = 1 << (max(n, m) - 1).bit_length()
	padded_matrix = zeros((next_power_of_two, next_power_of_two), dtype=matrix.dtype)
	padded_matrix[:n, :m] = matrix
	return padded_matrix

def matrix_multiply(A, B):
	n = A.shape[0]
	A = pad_to_power_of_two(A)
	B = pad_to_power_of_two(B)
	return matrix_multiply_rec(A, B)[:n, :n]

# b
# T(n) = a T(n/b) + f(n)
# a = 7, b = 2, f(n) = O(n^2)

# c
# c_crit = log(a) / log(b) = 2.807
# T(n) = O(n^2.807)
c_crit = log(7) / log(2)

# c
def theoretical(n):
	return n**c_crit

def test(n, num_tests=1):
	result = 0
	for _ in range(num_tests):
		A = rand(n, n)
		B = rand(n, n)
		start = time()
		matrix_multiply(A, B)
		result += time() - start
	return result / num_tests / theoretical(n)

test_sizes = [2**i for i in range(4, 9)]
test_results = [test(n) for n in test_sizes]
plt.plot(test_sizes, test_results)
plt.xlabel('Matrix size')
plt.ylabel('Time / Theoretical time')
plt.show()
# They largely agree, because the curve is nearly flat.
