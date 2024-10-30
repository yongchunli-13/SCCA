from typing import List, Tuple
import math
import time
import numpy as np
import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd

from sccapy.utilities.problem_data import ProblemData

def greedy(data: ProblemData, S0: List[int], S1: List[int]) -> Tuple[List[int], List[int], float, float]:
    
    start = time.time()
    
    n1, n2 = data.n1, data.n2
    s1, s2 = data.s1, data.s2
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
    # print(oneS1, oneS2)
    s1 = s1-len(oneS1)
    s2 = s2-len(oneS2)
    s = min(s1, s2)
    sn = list(range(n1))
    sn = list(set(sn) - set(zeroS1))
    sm = list(range(n2))
    sm = list(set(sm) - set(zeroS2))
    bestLB = 0.0
    # initialize two subsets
    if len(oneS1) ==0 and len(oneS2) ==0:
        bestLB = 0.0
        S1, S2 =[],[]
        temp = zeros([n1, n2])
        
        for i in range(n1):
            for j in range(n2):
                if C[j,j]>0 and B[i,i]>0:
                    temp[i,j] = abs(1/math.sqrt(B[i,i]) * A[i,j] * 1/math.sqrt(C[j,j]))
        
        sindex = unravel_index(argmax(abs(temp), axis=None), temp.shape)
        
        S1.append(sindex[0])
        S2.append(sindex[1])
        
        # print(np.max(temp), sindex[0], sindex[1])
    else:
        S1 = []
        S1 = oneS1
        S2 = []
        S2 = oneS2 
 
        if len(S1) == 0:
            bestLB = 0.0
            temp2 = C[ix_(S2, S2)]
            [a,b] = np.linalg.eigh(temp2) 
            a = a.real # engivalues
            b = b.real # engivectors
            engi_val = [0]*len(a)
            for t in range(len(a)):
                if(a[t] > 1e-8):
                    engi_val[t] = 1/math.sqrt(a[t])
                else:
                    engi_val[t] = 0

            temp2 = b*np.diag(engi_val)*b.T 
            for l in sn:
                S1.append(l)  
                temp1 = B[ix_(S1, S1)]
                [a,b] = np.linalg.eigh(temp1) 
                a = a.real # engivalues
                b = b.real # engivectors
                engi_val = [0]*len(a)
                for t in range(len(a)):
                    if(a[t] > 1e-8):
                        engi_val[t] = 1/math.sqrt(a[t])
                    else:
                        engi_val[t] = 0

                temp1 = b*np.diag(engi_val)*b.T 
                
                # temp1 = matrix(LA.sqrtm(temp1.I))
                
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                S1.remove(l)
                
            S1.append(bestl)
            # print(oneS1, oneS2, S1)
            # S1.append(sn[0])
        if len(S2) == 0:
            bestLB = 0.0
            temp1 = B[ix_(S1, S1)]
            [a,b] = np.linalg.eigh(temp1) 
            a = a.real # engivalues
            b = b.real # engivectors
            engi_val = [0]*len(a)
            for t in range(len(a)):
                if(a[t] > 1e-8):
                    engi_val[t] = 1/math.sqrt(a[t])
                else:
                    engi_val[t] = 0

            temp1 = b*np.diag(engi_val)*b.T 
            for l in sm:
                S2.append(l)  
                temp2 = C[ix_(S2, S2)]
                [a,b] = np.linalg.eigh(temp2) 
                a = a.real # engivalues
                b = b.real # engivectors
                engi_val = [0]*len(a)
                for t in range(len(a)):
                    if(a[t] > 1e-8):
                        engi_val[t] = 1/math.sqrt(a[t])
                    else:
                        engi_val[t] = 0

                temp2 = b*np.diag(engi_val)*b.T 
                
                # temp1 = matrix(LA.sqrtm(temp1.I))
                
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                S2.remove(l)
                
            S2.append(bestl)
            
            # print(oneS1, S2)
            # S2.append(sm[0])
            # S2 = sm[0]
        sel1 = oneS1
        sel2 = oneS2
        temp1 = B[ix_(sel1, sel1)]
        [a,b] = np.linalg.eigh(temp1) 
        a = a.real # engivalues
        b = b.real # engivectors
        engi_val = [0]*len(a)
        for t in range(len(a)):
            if(a[t] > 1e-8):
                engi_val[t] = 1/math.sqrt(a[t])
            else:
                engi_val[t] = 0

        temp1 = b*np.diag(engi_val)*b.T 
        # temp1 = matrix(LA.sqrtm(temp1.I))
        temp2 = C[ix_(sel2, sel2)]
        [a,b] = np.linalg.eigh(temp2) 
        a = a.real # engivalues
        b = b.real # engivectors
        engi_val = [0]*len(a)
        for t in range(len(a)):
            if(a[t] > 1e-8):
                engi_val[t] = 1/math.sqrt(a[t])
            else:
                engi_val[t] = 0

        temp2 = b*np.diag(engi_val)*b.T 
        # temp2 = matrix(LA.sqrtm(temp2.I))
        _, sigma, _ = svd(temp1 * A[ix_(sel1, sel2)] * temp2)
        
        bestf = max(sigma)
    
    
    for i in range(s-1):
        # sn = list(range(n1))
        # sn = list(set(sn) - set(zeroS1))
        unsel = []
        unsel = list(set(sn) - set(S1)) 
        bestLB = 0.0
        bestl = 0
   
        temp2 = C[ix_(S2, S2)]
        
        [a,b] = np.linalg.eigh(temp2) 
        a = a.real # engivalues
        b = b.real # engivectors
        engi_val = [0]*len(a)
        for t in range(len(a)):
            if(a[t] > 1e-8):
                engi_val[t] = 1/math.sqrt(a[t])
            else:
                engi_val[t] = 0

        temp2 = b*np.diag(engi_val)*b.T 
        
        # temp2 = matrix(LA.sqrtm(temp2.I))
        
        for l in unsel:
            S1.append(l)  
            temp1 = B[ix_(S1, S1)]
            [a,b] = np.linalg.eigh(temp1) 
            a = a.real # engivalues
            b = b.real # engivectors
            engi_val = [0]*len(a)
            for t in range(len(a)):
                if(a[t] > 1e-8):
                    engi_val[t] = 1/math.sqrt(a[t])
                else:
                    engi_val[t] = 0

            temp1 = b*np.diag(engi_val)*b.T 
            
            # temp1 = matrix(LA.sqrtm(temp1.I))
            
            _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
            
            LB = max(sigma)
            
            if LB > bestLB:
                bestLB = LB
                bestl = l
            S1.remove(l)
            
        S1.append(bestl)
        temp1 = B[ix_(S1, S1)]
        [a,b] = np.linalg.eigh(temp1) 
        a = a.real # engivalues
        b = b.real # engivectors
        engi_val = [0]*len(a)
        for t in range(len(a)):
            if(a[t] > 1e-8):
                engi_val[t] = 1/math.sqrt(a[t])
            else:
                engi_val[t] = 0

        temp1 = b*np.diag(engi_val)*b.T 
        
        # temp1 = matrix(LA.sqrtm(temp1.I))
        
        # then update set S2    
        # sm = list(range(n2))
        # sm = list(set(sm) - set(zeroS2))
        unsel = []
        unsel = list(set(sm) - set(S2)) 
        bestLB = 0.0
        bestl = 0
        for l in unsel:
            S2.append(l)
            temp2 = C[ix_(S2, S2)]
            [a,b] = np.linalg.eigh(temp2) 
            a = a.real # engivalues
            b = b.real # engivectors
            engi_val = [0]*len(a)
            for t in range(len(a)):
                if(a[t] > 1e-8):
                    engi_val[t] = 1/math.sqrt(a[t])
                else:
                    engi_val[t] = 0

            temp2 = b*np.diag(engi_val)*b.T 
            
            # temp2 = matrix(LA.sqrtm(temp2.I))
            _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 

            LB = 0.0
            LB =max(sigma)

            if LB > bestLB:
                bestLB = LB
                bestl = l
            S2.remove(l)
            
        S2.append(bestl)
        
    if s1 < s2:
        temp1 = B[ix_(S1, S1)]
        [a,b] = np.linalg.eigh(temp1) 
        a = a.real # engivalues
        b = b.real # engivectors
        engi_val = [0]*len(a)
        for t in range(len(a)):
            if(a[t] > 1e-8):
                engi_val[t] = 1/math.sqrt(a[t])
            else:
                engi_val[t] = 0

        temp1 = b*np.diag(engi_val)*b.T 
        
        # temp1 = matrix(LA.sqrtm(temp1.I))
        
        for i in range(s, s2):
            # only update set S2    
            unsel = []
            unsel = list(set(sm) - set(S2)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S2.append(l)
                temp2 = C[ix_(S2, S2)]
                [a,b] = np.linalg.eigh(temp2) 
                a = a.real # engivalues
                b = b.real # engivectors
                engi_val = [0]*len(a)
                for t in range(len(a)):
                    if(a[t] > 1e-8):
                        engi_val[t] = 1/math.sqrt(a[t])
                    else:
                        engi_val[t] = 0

                temp2 = b*np.diag(engi_val)*b.T 
                
                # temp2 = matrix(LA.sqrtm(temp2.I))
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 

                LB = 0.0
                LB =max(sigma)

                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                S2.remove(l)
                
            S2.append(bestl)
    else:
        
        temp2 = C[ix_(S2, S2)]
        [a,b] = np.linalg.eigh(temp2) 
        a = a.real # engivalues
        b = b.real # engivectors
        engi_val = [0]*len(a)
        for t in range(len(a)):
            if(a[t] > 1e-8):
                engi_val[t] = 1/math.sqrt(a[t])
            else:
                engi_val[t] = 0

        temp2 = b*np.diag(engi_val)*b.T 
        # temp2 = matrix(LA.sqrtm(temp2.I))
        
        for i in range(s, s1):
            # only update set S1
            unsel = []
            unsel = list(set(sn) - set(S1)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S1.append(l)  
                temp1 = B[ix_(S1, S1)]
                [a,b] = np.linalg.eigh(temp1) 
                a = a.real # engivalues
                b = b.real # engivectors
                engi_val = [0]*len(a)
                for t in range(len(a)):
                    if(a[t] > 1e-8):
                        engi_val[t] = 1/math.sqrt(a[t])
                    else:
                        engi_val[t] = 0

                temp1 = b*np.diag(engi_val)*b.T 
                # temp1 = matrix(LA.sqrtm(temp1.I))
                _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                    
                S1.remove(l)
                
            S1.append(bestl)
            
    end = time.time()
    total_time = (end-start)
    bestf = bestLB
    
    return  S1, S2, bestf, total_time  