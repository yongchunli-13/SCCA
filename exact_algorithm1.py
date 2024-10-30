
import os
import math
import numpy as np
import math
import datetime
import pandas as pd
import scipy
import random
import pandas as pd
from sympy import * 

### when s1 >= r and s2 >= r, return the optimal solution and value ####
def algo(n1, m2, s1, s2):
    global A
    global B
    global C
    
    r = 89
    
    X = pd.read_csv(os.getcwd()+'/datasets/dna.csv',
                          encoding = 'utf-8',sep=',')
    
    X = np.array(X)
    X = np.matrix(X)
    u, sigma, v = np.linalg.svd(X, full_matrices=False)
    Q, R, p = scipy.linalg.qr(v, pivoting=True)
    S1 = p[0:r]
    X = X[:,S1]

    
    Y = pd.read_csv(os.getcwd()+'/datasets/rna.csv',
                          encoding = 'utf-8',sep=',')
    
    Y = np.array(Y)
    Y = np.matrix(Y)
    u, sigma, v = np.linalg.svd(Y, full_matrices=False)
    Q, R, p = scipy.linalg.qr(v, pivoting=True)
    S2 = p[0:r]
    Y = Y[:,S2]
    
    B = X.T*X
    C = Y.T*Y
    A = X.T*Y
    
    temp1 = np.matrix(scipy.linalg.sqrtm(B.I))
    temp2 = np.matrix(scipy.linalg.sqrtm(C.I))
    u, sigma, v = np.linalg.svd(temp1 * A * temp2) 
    print(max(sigma))
    
    return S1, S2, max(sigma)

