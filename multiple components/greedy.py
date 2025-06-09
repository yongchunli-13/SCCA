
import math
import time
import numpy as np
import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd

def greedy(n1, n2, s1, s2, A, B, C, k): 
    
    start = time.time()
    

    s = min(s1, s2)
    sn = list(range(n1))
    sm = list(range(n2))
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
   
            _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 
            
            # LB = max(sigma)
            
            sorted_sigma = sorted(sigma, reverse=True)
            LB = sum(sorted_sigma[:k])
            
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
            
            _, sigma, _ = svd(temp1 * A[ix_(S1, S2)] * temp2) 

            LB = 0.0
            # LB =max(sigma)
            sorted_sigma = sorted(sigma, reverse=True)
            LB = sum(sorted_sigma[:k])
            
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
                sorted_sigma = sorted(sigma, reverse=True)
                LB = sum(sorted_sigma[:k])
                # LB =max(sigma)

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
                
                # LB = max(sigma)
                
                sorted_sigma = sorted(sigma, reverse=True)
                LB = sum(sorted_sigma[:k])
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                    
                S1.remove(l)
                
            S1.append(bestl)
            
    end = time.time()
    total_time = (end-start)
    bestf = bestLB
    
    return  S1, S2, bestf, total_time  