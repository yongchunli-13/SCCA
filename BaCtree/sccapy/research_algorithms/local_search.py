from typing import List, Tuple
import math
import time
import numpy as np
import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd
from sccapy.research_algorithms.objective_eval import fval

from sccapy.utilities.problem_data import ProblemData
from sccapy.research_algorithms.greedy import greedy

def localsearch(data: ProblemData, intS0: List[int], intS1: List[int]) -> Tuple[float, float, List[int]]:
    start_time = time.time()

    n1, n2 = data.n1, data.n2
    s1, s2 = data.s1, data.s2
    A, B, C = data.A, data.B, data.C
    
    zeroS1, zeroS2 = [], []
    for i in intS0:
        if i < n1:
            zeroS1.append(i)
        else:
            zeroS2.append(i-n1)
    
    oneS1, oneS2 = [], []
    for i in intS1:
        if i < n1:
            oneS1.append(i)
        else:
            oneS2.append(i-n1)

    S1, S2, bestf, _ = greedy(data, intS0, intS1)
    print('greedy:', bestf)
    
    
    sn = list(range(n1))
    sn = list(set(sn) - set(zeroS1))
    sm = list(range(n2))
    sm = list(set(sn) - set(zeroS2))
    
    optimal = False

    unsel = []
    unsel = list(set(sn) - set(S1)) 
    
    unsel2 = []
    unsel2 = list(set(sm) - set(S2))

    
    while(optimal == False):
        optimal = True
        
        # first update row selection
        temp2 = C[ix_(S2, S2)]
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
        hat_S1 = []
        hat_S1 = list(set(S1) - set(oneS1)) 
        for i in hat_S1:
            for j in unsel:
                S1.remove(i)
                S1.append(j)
                
                temp1 = B[ix_(S1, S1)]
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
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestf:
                    # print(i)
                    optimal = False                             
                    bestf = LB
                    unsel.append(i)
                    unsel.remove(j)
                    break
                
                S1.append(i)
                S1.remove(j)
                
    
        temp1 = B[ix_(S1, S1)]
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
        hat_S2 = []
        hat_S2 = list(set(S2) - set(oneS2)) 
        for i in S2:
            for j in unsel2:
                S2.remove(i)
                S2.append(j)

                temp2 = C[ix_(S2, S2)]
                # temp2 = matrix(LA.sqrtm(temp2.I))
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
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
                if len(sigma) ==0:
                    LB = 0
                else:      
                    LB = max(sigma)
                
                if LB > bestf:
                    # print(i)
                    optimal = False                             
                    bestf = LB
                    unsel2.append(i)
                    unsel2.remove(j)
                    break
                
                S2.append(i)
                S2.remove(j)
                    
    end = time.time()
    total_time = end - start_time
    
    S = []
    for i in S1:
        S.append(i)
    for i in S2:
        S.append(i+n1)
    # print(len(S1), len(S2))
    # print('the lower bound at the root node is ',  bestf, fval(data, S)) 
    
    return bestf, total_time, S