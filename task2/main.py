#!/usr/bin/env python

from numpy import array, zeros, zeros_like, dot, diag, diagonal, exp, sign, isclose, pi, linspace, sin, sqrt
from numpy.linalg import norm
from numpy.random import rand, seed
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lobpcg
from matplotlib import pyplot as plt

seed(1108)

# 1
# Expressing the kth state:
# Write k in binary form, and then the ith bit (starting from 0) is 0 if ith spin is up, 1 if ith spin is down.
def is_nn_hopping(n, state1, state2):
	if state1 == state2:
		return 0
	d = state1 ^ state2 # find sites where the spin differs
	if state1&d==0 or state2&d==0: # one of the states has spin up on those sites
		return 0
	return int(d%3==0 and d//3&d//3-1==0) + int(d==(1<<n-1)+1) # condition 1: adjacent; condition 2: periodic

def get_spin(n, state, i):
	return 1/2 - (state >> i%n & 1)

def xxx_element(n, state1, state2):
	result = 0
	if state1 == state2:
		for i in range(n):
			result += 1/4 - get_spin(n, state1, i) * get_spin(n, state1, i+1)
	result -= is_nn_hopping(n, state1, state2)/2
	return result

# Build full array.
# O((2^n)^2) space and time complexity.
def xxx_hamiltonian(n):
	return [[xxx_element(n, i, j) for j in range(1<<n)] for i in range(1<<n)]

# Build sparse array.
# O(2^n poly(n)) space and time complexity, but still too slow for n=30.
def xxx_hamiltonian_sparse(n):
	result = csc_matrix((1<<n, 1<<n))
	for state in range(1<<n):
		if state == 0:
			continue
		if state >> n-1 == 1 and state & 1 == 0: # hopping by periodic boundary is possible
			hop_state = state ^ (1<<n-1) ^ 1
			result[state, hop_state] -= 1/2
			result[hop_state, state] -= 1/2
		mark_state = state & -2 # remove the lowest bit if it is 1
		# In every iteration, set the lowest nonzero bit of mark_state to 0
		while mark_state > 0:
			new_mark_state = mark_state & mark_state - 1
			one_down_state = (mark_state ^ new_mark_state) >> 1 # the bit next to the removed bit
			if state & one_down_state == 0: # hopping is possible
				hop_state = state ^ (one_down_state*3)
				result[state, hop_state] -= 1/2
				result[hop_state, state] -= 1/2
			mark_state = new_mark_state
		for i in range(n):
			result[state, state] += 1/4 - get_spin(n, state, i) * get_spin(n, state, i+1)
	return result

# 2
def gram_schmidt(A):
	m, n = A.shape
	Q = zeros((m, n))
	R = zeros((n, n))
	for i in range(n):
		v = A[:, i]
		for j in range(i):
			R[j, i] = dot(Q[:, j], A[:, i])
			v = v - R[j, i] * Q[:, j]
		R[i, i] = norm(v)
		Q[:, i] = v / R[i, i]
	return Q, R
def qr_diagonalize(H, tol=1e-10, max_iter=1000):
	H_k = array(H)
	for _ in range(max_iter):
		Q, R = gram_schmidt(H_k)
		H_k = R @ Q
		if norm(H_k - diag(diagonal(H_k))) < tol:
			break
	return H_k

# 3
# Never used. For matrices with size of 2^30, it is impossible to store the matrix.
# Also notice that the Cholesky decomposition is impossible on omega - H
# unless omega is larger than every eigenvalue of H (for positive definiteness).
def solve_cholesky(A, b):
	L = cholesky_decomposition(A)
	return backward_substitution(L.T, forward_substitution(L, b))
def cholesky_decomposition(A):
	n = A.shape[0]
	L = zeros_like(A)
	for i in range(n):
		for j in range(i + 1):
			sum_k = dot(L[i, :j], L[j, :j])
			if i == j:
				L[i, j] = sqrt(A[i, i] - sum_k)
			else:
				L[i, j] = (A[i, j] - sum_k) / L[j, j]
	return L
def forward_substitution(L, b):
	n = L.shape[0]
	y = zeros(n)
	for i in range(n):
		y[i] = (b[i] - dot(L[i, :i], y[:i])) / L[i, i]
	return y
def backward_substitution(LT, y):
	n = LT.shape[0]
	x = zeros(n)
	for i in range(n - 1, -1, -1):
		x[i] = (y[i] - dot(LT[i, i + 1:], x[i + 1:])) / LT[i, i]
	return x

# It seems impossible to find m eigenvalues of H within polynomial space and time of n, m.
# For O(2^n poly(n,m)) space and time, it is possible (using scipy.sparse.linalg.lobpcg), and is shown below.
# It runs too slowly and uses up my RAM, so I commented them out.
# Notice that because of gapless excitation (and because I told lobpcg to find smallest eigenvalues),
# all the found eigenvalues of H are close to zero, so the found eigenvalues of G are just 1/omega.
"""
	n = 30
	m = 5
	energies = lobpcg(xxx_hamiltonian_sparse(n), rand(1<<n, m), largest=False, maxiter=20)[0]
	omega = linspace(0, 3, 100)
	for energy in energies:
		plt.plot(omega, 1/(omega - energy))
	plt.xlabel(r'$\omega$')
	plt.ylabel(r'Eigenvalue of $G$')
	plt.ylim(-3, 3)
	plt.show()
"""

# 4
def magnon(n, p):
	result = zeros(1<<n, dtype=complex)
	for i in range(n):
		result[1<<i] = exp(1j*p*i)
	return result

# Instead of usual matrix multiplication, utilize sparsity,
# and also utilize magnon[1<<0] is 1.
# Find energy in O(poly(n)) time and O(1) space.
def energy(n, p):
	result = 0
	for i in range(n):
		result += xxx_element(n, 1<<0, 1<<i) * exp(1j*p*i)
	return result.real

n = 30
energies = [energy(n, 2*pi*k/n) for k in range(n)]
plt.scatter(range(n), energies)

k = linspace(0, n, 100)
plt.plot(k, 2*sin(pi*k/n)**2) # theoretical

plt.xlabel(r'$Np/2\pi$')
plt.ylabel('Energy')
plt.show()
