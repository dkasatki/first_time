import math as m
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
from sympy import symbols, diff, erf, lambdify

# Global
n = 5
u = 7


def data(x):
    y = []
    i = 0
    while i < len(x):
        y.append(m.erf(x[i]))
        i += 1
    return y
    # print(y)


def draw(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, l):
    # fig = plt.figure()
    plt.subplot(2, 2, l)
    plt.plot(x1, y1, color = 'black', marker = 'p', markersize = 4, linestyle = ' ') # f(x)
    plt.plot(x2, y2, color = 'red', marker = '', markersize = 4, linestyle = '-', label='Newton Interpolation') # N(x)
    plt.plot(x3, y3, color = 'blue', marker = '', markersize = 4, linestyle = '-', label='Cheb Interpolation') # cheb(x)
    plt.plot(x4, y4, color = 'blue', marker = '*', markersize = 4, linestyle = '') # dots cheb
    plt.plot(x5, y5, color = 'green', marker = '', markersize = 4, linestyle = '--') # error with derivative
    plt.grid(True, color = '#F8007D', linewidth = 0.5, linestyle = '-')
    plt.xlim(0, 3.5)
    plt.legend()


def interpolation(x, y, x2, v):
    f = np.zeros((v, v))
    i = 0
    k = len(x) - 1
    while i < k:
        j = 0
        while j < k:
            if(i == 0):
                f[i][j] = (y[j + 1] - y[j]) / (x[j + 1] - x[j])
            else:
                if(j < k - i):
                    f[i][j] = (f[i - 1][j + 1] - f[i - 1][j]) / (x[j + i + 1] - x[j])
            j += 1
        i += 1
    # print(f)
    f1 = np.zeros((v))
    i = 0
    while i < k:
        f1[i] = f[i][0]
        i += 1
    # print('f=', f1)
    i = 0
    p = np.zeros((len(x2)))
    q = np.ones((len(x2), len(x)))
    while i < len(x2):
        j = 0
        while j < k:
            l = 0
            while l < k:
                if(l <= j):
                    q[i][j] *= (x2[i] - x[l])
                l += 1
            p[i] += q[i][j] * f1[j]
            j += 1
        i += 1
    j = 0
    # print('q1=', q[1])
    # print('p=', p)
    P = np.zeros((len(x2), 1))
    while j < len(x2):
        P[j] = y[0] + p[j]
        j += 1
    return P


def error(x, x2):
    i = 0
    omega = np.ones((len(x2)))
    while i < len(x2):
        j = 0
        while j < len(x):
            omega[i] *= (x2[i] - x[j])
            j += 1
        if(omega[i] < 0):
            omega[i] = omega[i] * (-1)
        i += 1
    # derivative
    q = symbols('q')
    M = diff(erf(q), q, n + 1)
    M1 = lambdify(q, M, 'numpy')
    M2 = M1(x2)
    i = 0
    while i < len(M2):
        if(M2[i] < 0):
            M2[i] = M2[i] * (-1)
        i += 1
    M3 = max(M2)
    R = np.zeros((len(x2)))
    i = 0
    while i < len(x2):
        R[i] = (omega[i] * M3) / (m.factorial(n + 1))
        i += 1
    # print(M3)
    return R

def cheb(p):
    x = np.zeros((p))
    i = 1
    while i < len(x):
        x[i] = (m.pi / 2) + (m.pi / 2) * (m.cos(m.pi * (2 * i - 1) / (2 * p)))
        # print(x[i])
        i += 1
    return x


def main():
    # n = 5
    l1 = 1
    x1 = list(np.arange(0, 3.1415 + 0.2, 3.1415 / n))
    # x1 = list(np.arange(0.1, m.pi, n))
    x2 = np.linspace(0, 3.1415 + 0.2, 1000)
    y1 = data(x1)
    #
    x3 = list(np.arange(0, 3.1415, 1000))
    y = data(x2)
    # print(x1, y1)
    # print(x2)
    # print('')
    P = interpolation(x1, y1, x2, n)
    R = error(x1, x2)
    # print(R)
    i = 0
    Er1 = np.zeros((len(x2)))
    while i < len(x2):
        Er1[i] = P[i] - y[i]
        if(Er1[i] < 0):
            Er1[i] = Er1[i] * (-1)
        i += 1
    # print(P)
    t1 = cheb(n + 1)
    yt1 = data(t1)
    Pt1 = interpolation(t1, yt1, x2, n)
    draw(x1, y1, x2, P, x2, Pt1, t1, yt1, 0, 0, l1)
    l11 = 3
    ErPt1 = np.zeros((len(x2)))
    i = 0
    while i < len(x2):
        ErPt1[i] = Pt1[i] - y[i]
        if(ErPt1[i] < 0):
            ErPt1[i] = ErPt1[i] * (-1)
        i += 1
    z1 = np.zeros(len(y1))
    draw(x1, z1, x2, Er1, x2, ErPt1, t1, z1, x2, R, l11)
    plt.ylim(-0.005, 0.06)

    # u = 7
    l2 = 2
    x12 = list(np.arange(0, 3.1415 + 0.2, 3.1415 / u))
    x22 = np.linspace(0, 3.1415 + 0.2, 1000)
    y12 = data(x12)
    y22 = data(x22)
    P2 = interpolation(x12, y12, x22, u)
    Er2 = interpolation(x12, y12, x12, u)
    R2 = error(x12, x22)
    i = 0
    Er12 = np.zeros((len(x22)))
    while i < len(x22):
        Er12[i] = P2[i] - y22[i]
        if(Er12[i] < 0):
            Er12[i] = Er12[i] * (-1)
        i += 1
    # print(P)
    t12 = cheb(u + 1)
    yt12 = data(t12)
    Pt12 = interpolation(t12, yt12, x22, u)
    draw(x12, y12, x22, P2, x22, Pt12, t12, yt12, 0, 0, l2)
    l22 = 4
    ErPt12 = np.zeros((len(x22)))
    i = 0
    while i < len(x22):
        ErPt12[i] = Pt12[i] - y22[i]
        if(ErPt12[i] < 0):
            ErPt12[i] = ErPt12[i] * (-1)
        i += 1
    z12 = np.zeros(len(y12))
    draw(x12, z12, x22, Er12, x22, ErPt12, t12, z12, 0, 0, l22)
    plt.ylim(-0.0002, 0.0026)
    plt.show()


main()
