#!/usr/bin/env python

from numpy import array, zeros, dot, diag, diagonal, exp, sign, isclose, pi, linspace, sin
from numpy.linalg import norm
from matplotlib import pyplot as plt

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

def xxx_hamiltonian(n):
	return [[xxx_element(n, i, j) for j in range(1<<n)] for i in range(1<<n)]
# T(n) = O((2^n)^2)
# Very memory inefficient. It is never used.

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
# Cannot understand the question. How to plot a matrix?

# 4
# Put here to show that I understand what a magnon state is.
# It is never used.
def magnon(n, p):
	result = zeros(1<<n, dtype=complex)
	for i in range(n):
		result[1<<i] = exp(1j*p*i)
	return result

# instead of usual matrix multiplication, utilize sparsity,
# and also utilize magnon[1<<0] is 1.
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
