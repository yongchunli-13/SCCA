from typing import List, Tuple
import math
import time
import numpy as np
import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData

def varfix(data: ProblemData, LB: float) -> Tuple[List[int], List[int], List[float], float, float, float]:
    start_time = time.time()
    n1, n2 = data.n1, data.n2
    A, B, C = data.A, data.B, data.C

    scores=[]
    OptS1 = []
    sn = list(range(n1))
    sm = list(range(n2))
    # temp2 = matrix(LA.sqrtm(C.I))
    temp2 = C
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
    for i in range(n1):
        sn.remove(i)
        temp1 = B[ix_(sn, sn)]
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
        _, sigma, _ = svd(temp1 * A[ix_(sn, sm)] * temp2)
        sn.append(i)
        scores.append(max(sigma))
        # print('S1:', i, max(sigma), LB)
        if max(sigma) < LB:
            # print('S1:', i, max(sigma), LB)
            OptS1.append(i)
            
    OptS2 = []
    sn = list(range(n1))
    sm = list(range(n2))
    temp1 = B
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
    
    for i in range(n2):
        sm.remove(i)
        temp2 = C[ix_(sm, sm)]
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
        _, sigma, _ = svd(temp1 * A[ix_(sn, sm)] * temp2)
        sm.append(i)
        scores.append(max(sigma))
        # print('S1:', i, max(sigma), LB)
        if max(sigma) < LB:
            # print('S2:', i, max(sigma), LB)
            OptS2.append(i)
            
    S1 = []
    for i in OptS1:
        S1.append(i)
    for i in OptS2:
        S1.append(i+n1)
    print("the fixed variables being 1 at the root node are ", S1, len(OptS1), len(OptS2))  
    
    # temp1 = matrix(LA.sqrtm(B.I))
    # temp2 = matrix(LA.sqrtm(C.I))
    # u, sigma, v = svd(temp1 * A * temp2)
    # for i in range(n1):
    #     scores[i] = u[i,0]
    # for i in range(n2):
    #     scores[i+n1] = v[0,i]
    runtime = time.time() - start_time
    return [], S1, scores, runtime, len(OptS1), len(OptS2)