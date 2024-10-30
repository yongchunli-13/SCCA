from typing import List, Tuple
import time
import numpy as np
import math
import scipy.linalg as LA
from numpy import (matrix, ix_)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData

def fval(data: ProblemData, S1: List[int]) -> Tuple[float, float]:
    start_time = time.time()
    n1 = data.n1
    A, B, C = data.A, data.B, data.C

    oneS1, oneS2 = [], []
    for i in S1:
        if i < n1:
            oneS1.append(i)
        else:
            oneS2.append(i-n1)
    
    temp1 = B[ix_(oneS1, oneS1)]
    [a,b] = np.linalg.eigh(temp1) 
    a = a.real # engivalues
    b = b.real # engivectors
    engi_val = [0]*len(a)
    for i in range(len(a)):
        if(a[i] > 1e-8):
            engi_val[i] = 1/math.sqrt(a[i])
        else:
            engi_val[i] = 0

    temp1 = b*np.diag(engi_val)*b.T 
    
    # temp1 = matrix(LA.sqrtm(temp1.I))

    temp2 = C[ix_(oneS2, oneS2)]
    # temp2 = matrix(LA.sqrtm(temp2.I))
    [a,b] = np.linalg.eigh(temp2) 
    a = a.real # engivalues
    b = b.real # engivectors
    engi_val = [0]*len(a)
    for i in range(len(a)):
        if(a[i] > 1e-8):
            engi_val[i] = 1/math.sqrt(a[i])
        else:
            engi_val[i] = 0

    temp2 = b*np.diag(engi_val)*b.T 
    _, sigma, _ = svd(temp1 * A[ix_(oneS1, oneS2)] * temp2) 
    
    runtime = time.time() - start_time

    return max(sigma), runtime 