
import numpy as np
import pandas as pd
import time
import random
from random import sample
import os
from sparsecca import pmd
from sparsecca import cca_ipls

def gen_data(n1, n2, s1, s2):
    global A, B, C, X, Y
    
    random.seed(1)
    
    X = pd.read_csv(os.getcwd()+'/datasets/dna.csv',
                          encoding = 'utf-8',sep=',')  ### import the breast cancer DNA data
    
    X = np.array(X)
    
    sn1 = sample(range(len(X[0,:])), n1)
    X = X[:, sn1]
    X = np.matrix(X)
    
    
    Y = pd.read_csv(os.getcwd()+'/datasets/rna.csv',
                          encoding = 'utf-8',sep=',') ### import the breast cancer RNA data
    
    Y = np.array(Y)
    sm2 = sample(range(len(Y[0,:])), n2)
    Y = Y[:, sm2]
    Y = np.matrix(Y)
    
    N = 89
    
    B = sum([X[i,:].T*X[i,:] for i in range(N)])/N 
    C = sum([Y[i,:].T*Y[i,:] for i in range(N)])/N
    A = sum([X[i,:].T*Y[i,:] for i in range(N)])/N
    


def print_weights(name, weights):
    first = weights[:, 0] / np.max(np.abs(weights[:, 0]))
    print(name + ': ' + ', '.join(['{:.3f}'.format(item) for item in first]))

### Implement the algorithm of Witten et al. [41]
def PMD(l):
    start = time.time()
    np.random.seed(1)
    U, V, D = pmd(X.T @ Y, K=l, penaltyu=0.3, penaltyv=0.5, standardize=False)
    #0.05
    x_weights = U[:, 0]
    z_weights = V[:, 0]
    corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Y.T))[0, 1]
    
    print("Corrcoef for comp 1: " + str(corrcoef))
    
    end = time.time()
    ptime = end-start
    return x_weights, z_weights, abs(corrcoef), ptime

### Implement the algorithm of Parkhomenko et al. [37]    
def IPLS_mai(l):
    start = time.time()

    X_weights, Z_weights = cca_ipls(X, Y, alpha_lambda_ratio=1.0, beta_lambda_ratio=1.0,
         alpha_lambda=0, beta_lambda=0, niter=100, n_pairs=1, 
         standardize=False, eps=1e-4, glm_impl='pyglmnet')

    x_weights = X_weights[:, 0]
    z_weights = Z_weights[:, 0]
    corrcoef = np.corrcoef(np.dot(x_weights, X.T), np.dot(z_weights, Y.T))[0, 1]
    print("Corrcoef for comp 1: " + str(corrcoef))
    end = time.time()
    mtime = end-start
    
    return x_weights, z_weights, corrcoef, mtime
    
### Implement the algorithm of Chu et al. [10]
def algo_chu(n1, n2, delta, mux, muy, epsilon): 
    start = time.time()
    U, G1, Q1h = np.linalg.svd(X, full_matrices=False)
    positive_indices = G1 > 1e-4
    G1 = G1[positive_indices]
    U = U[:, positive_indices]
    Q1h = Q1h[positive_indices, :]

    
    V, G2, Q2h = np.linalg.svd(Y, full_matrices=False)
    positive_indices = G2 > 1e-4
    G2 = G2[positive_indices]
    V = V[:, positive_indices]
    Q2h = Q2h[positive_indices, :]

    
    temp = np.matrix(Q1h)*np.matrix(Q2h).T
    P1, G, P2h = np.linalg.svd(temp, full_matrices=True)
    P2 = P2h.T
    
    Wx = np.matrix(np.array([0]*n1)).T
    Vx = np.matrix(np.array([0]*n1)).T
    error = 1
    k=1
    T = 1e+4
    while error > epsilon:
        Vx = Vx + U*(np.matrix(np.diag(G1)).I*P1[:,0]-U.T*Wx)
        temp = [0]*n1
        for i in range(n1):
            if abs(Vx[i, 0]) > mux:
                temp[i] = delta*np.sign(Vx[i,0])*(abs(Vx[i, 0])-mux)
            else:
                temp[i] = 0
        Wx =  np.matrix(np.array(temp)).T
        
        error = np.linalg.norm(U.T*Wx - np.matrix(np.diag(G1)).I*P1[:,0],
                               ord = 'fro')
        # print(error)
        k = k+1
        if k > T:
            break
                               

    Wy = np.matrix(np.array([0]*n2)).T
    Vy = np.matrix(np.array([0]*n2)).T
    error = 1
    k = 1
    while error > epsilon:
        Vy = Vy + V*(np.matrix(np.diag(G2)).I*P2[:,0]-V.T*Wy)
        temp = [0]*n2
        for i in range(n2):
            if abs(Vy[i, 0]) > muy:
                temp[i] = delta*np.sign(Vy[i,0])*(abs(Vy[i, 0])-muy)
            else:
                temp[i] = 0
        Wy =  np.matrix(np.array(temp)).T

        error = np.linalg.norm(V.T*Wy - np.matrix(np.diag(G2)).I*P2[:,0],
                               ord = 'fro')
        k = k+1
        if k > T:
            break
        
    end = time.time()
    runtime = end-start
        
    return Wx, Wy, runtime


