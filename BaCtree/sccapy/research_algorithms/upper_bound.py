from typing import List, Tuple
import time
import numpy as np
import math
import scipy.linalg as LA
from numpy import (matrix, ix_)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData

def upper_bound(data: ProblemData, S0: List[int], S1: List[int]) -> Tuple[float, float, List[float]]:
    start_time = time.time()
    n1, n2 = data.n1, data.n2
    A, B, C = data.A, data.B, data.C

    zeroS1, zeroS2 = [], []
    for i in S0:
        if i < n1:
            zeroS1.append(i)
        else:
            zeroS2.append(i-n1)
            
    oneS1, oneS2 = [], []
    for i in S1:
        if i < n1:
            oneS1.append(i)
        else:
            oneS2.append(i-n1)
            
    sn = list(range(n1))
    sm = list(range(n2))
    print(len(S0))
    sel1 = []
    sel1 = list(set(sn) - set(zeroS1)) 
    
    sel2 = []
    sel2 = list(set(sm) - set(zeroS2))
    
    temp1 = B[ix_(sel1, sel1)]
    [a,b] = np.linalg.eigh(temp1) 
    a = a.real # engivalues
    b = b.real # engivectors
    engi_val = [0]*len(a)
    for l in range(len(a)):
        if(a[l] > 1e-8):
            engi_val[l] = 1/math.sqrt(a[l])
        else:
            engi_val[l] = 0

    temp1 = b*np.diag(engi_val)*b.T 
    # temp1 = matrix(LA.sqrtm(temp1.I))
    temp2 = C[ix_(sel2, sel2)]
    [a,b] = np.linalg.eigh(temp2) 
    a = a.real # engivalues
    b = b.real # engivectors
    engi_val = [0]*len(a)
    for l in range(len(a)):
        if(a[l] > 1e-8):
            engi_val[l] = 1/math.sqrt(a[l])
        else:
            engi_val[l] = 0

    temp2 = b*np.diag(engi_val)*b.T 
    # temp2 = matrix(LA.sqrtm(temp2.I))
    _, sigma, _ = svd(temp1 * A[ix_(sel1, sel2)] * temp2)
    UB = max(sigma)
    end_time = time.time()
    bound_time = end_time - start_time
    

    return UB, bound_time, []