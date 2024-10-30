
import numpy as np
import pandas as pd
import mosek
from gurobipy import *
from gurobipy import GRB
from numpy import matrix
from numpy import array
from scipy.linalg import svd
import sys
import datetime
import random
from random import sample
import math
import scipy

def gen_data(n1, n2, k1, k2):
    
    global A, al, br
    global B, mu
    global C, nu
    
    np.random.seed(1)
    random.seed(1)
    
    B = np.matrix(np.random.normal(0, 1, (n1, n1)))
    B = B*B.T + np.matrix(np.eye(n1, dtype=float))
    
    C = np.matrix(np.random.normal(0, 1, (n2, n2)))
    C = C*C.T + np.matrix( np.eye(n2, dtype=float)) 
    
    NS1 = sample(list(range(n1)), n1-s1)
    NS2 = sample(list(range(n2)), n2-s2)
    
    u = np.matrix(np.random.uniform(0, 1, (n1)))
    for i in NS1:
        u[0,i] = 0   
    u = u/math.sqrt((u*B*u.T)[0,0])
    
    v = np.matrix(np.random.uniform(0, 1, (n2)))
    for i in NS2:
        v[0,i] = 0   
    v = v/math.sqrt((v*C*v.T)[0,0])
    
    l = np.random.uniform(0,1)
    A = l*B*u.T*v*C
   
    cov = np.block([[B, A], [A.T, C]])
    mean = [0.0]*(n1+n2)
    N = 5000
    np.random.seed(1)
    temp = np.random.multivariate_normal(mean, cov, N)
    X = np.matrix(temp[:,0:n1])
    Y = np.matrix(temp[:, n1:(n1+n2)])
    
    B = sum([X[i,:].T*X[i,:] for i in range(N)])/N
    [a,b] = np.linalg.eigh(B)
    mu = min(a)
    mu = [0]*n1
    for i in range(n1):
        S1 = list(range(n1))
        S1.remove(i)
        # print(S1)
        [a,b] = np.linalg.eigh(B[np.ix_(S1, S1)])
        mu[i] = min(a)
    mu = min(mu)
    
    C = sum([Y[i,:].T*Y[i,:] for i in range(N)])/N
    [a,b] = np.linalg.eigh(C)
    nu = [0]*n2
    for i in range(n2):
        S2 = list(range(n2))
        S2.remove(i)
        [a,b] = np.linalg.eigh(C[np.ix_(S2, S2)])
        nu[i] = min(a)
    nu = min(nu)

    A = sum([X[i,:].T*Y[i,:] for i in range(N)])/N
    
    U, s, V = svd(A) # SVD decomposition
    s[s<1e-6]=0 
    l = len(s)
    sindex = np.argsort(-s)
    opti = sindex[0]
    print(s[opti])
    
    A = s[opti]*np.matrix(np.array(U[:,opti])).T*np.matrix(np.array(V[opti,:]))
        
    al = math.sqrt(s[opti])*U[:,opti]
    br = math.sqrt(s[opti])* V[opti,:].T
    

def greedy(n1, m2, s1, s2):
    
    s = min(s1, s2)
    sn = list(range(n1))
    sm = list(range(m2))
    bestf = 0.0
    
    start = datetime.datetime.now()
    
    # initialize two subsets
    S1 = []
    S2 = []  
    temp = np.zeros([n1, m2])
    
    for i in range(n1):
        for j in range(m2):
            temp[i,j] = abs(1/math.sqrt(B[i,i]) * A[i,j] * 1/math.sqrt(C[j,j]))
    
    sindex = np.unravel_index(np.argmax(abs(temp), axis=None), temp.shape)
    
    S1.append(sindex[0])
    S2.append(sindex[1])    
    
    for i in range(s-1):
        print(i)
        # first update set S1    
        unsel = []
        unsel = list(set(sn) - set(S1)) 
        bestLB = 0.0
        bestl = 0
   
        temp2 = C[np.ix_(S2, S2)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        
        for l in unsel:
            S1.append(l)  
            temp1 = B[np.ix_(S1, S1)]
            temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
            u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 
            
            LB = max(sigma)
            
            if LB > bestLB:
                bestLB = LB
                bestl = l
            S1.remove(l)
            
        S1.append(bestl)
        temp1 = B[np.ix_(S1, S1)]
        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        
        # then update set S2    
        unsel = []
        unsel = list(set(sm) - set(S2)) 
        bestLB = 0.0
        bestl = 0
        for l in unsel:
            S2.append(l)
            temp2 = C[np.ix_(S2, S2)]
            temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
            u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 

            LB = 0.0
            LB =max(sigma)

            if LB > bestLB:
                bestLB = LB
                bestl = l
            S2.remove(l)
            
        S2.append(bestl)
        
    if s1 < s2:
        temp1 = B[np.ix_(S1, S1)]
        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        
        for i in range(s, s2):
            # only update set S2    
            unsel = []
            unsel = list(set(sm) - set(S2)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S2.append(l)
                temp2 = C[np.ix_(S2, S2)]
                temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 

                LB = 0.0
                LB =max(sigma)

                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                S2.remove(l)
                
            S2.append(bestl)
    else:
        
        temp2 = C[np.ix_(S2, S2)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        
        for i in range(s, s1):
            # only update set S1
            unsel = []
            unsel = list(set(sn) - set(S1)) 
            bestLB = 0.0
            bestl = 0
            for l in unsel:
                S1.append(l)  
                temp1 = B[np.ix_(S1, S1)]
                temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestLB:
                    bestLB = LB
                    bestl = l
                    
                S1.remove(l)
                
            S1.append(bestl)
            
    end = datetime.datetime.now()
    time = (end-start).seconds
    bestf = bestLB
    
    return  S1, S2, bestf, time  
    

def localsearch(n1, m2, s1, s2):
    start = datetime.datetime.now()
    S1, S2, bestf, time = greedy(n1, m2, s1, s2)
    
    sn = list(range(n1))
    sm = list(range(m2))
    
    optimal = False

    unsel = []
    unsel = list(set(sn) - set(S1)) 
    
    unsel2 = []
    unsel2 = list(set(sm) - set(S2))
    
    while(optimal == False):
        optimal = True
        
        # first update row selection
        temp2 = C[np.ix_(S2, S2)]
        temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
        
        for i in S1:
            for j in unsel:
                S1.remove(i)
                S1.append(j)
                
                temp1 = B[np.ix_(S1, S1)]
                temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 
                
                LB = max(sigma)
                
                if LB > bestf:
                    print(i)
                    optimal = False                             
                    bestf = LB
                    unsel.append(i)
                    unsel.remove(j)
                    break
                
                S1.append(i)
                S1.remove(j)
                

        temp1 = B[np.ix_(S1, S1)]
        temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
        for i in S2:
            for j in unsel2:
                S2.remove(i)
                S2.append(j)

                temp2 = C[np.ix_(S2, S2)]
                temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
                u, sigma, v = np.linalg.svd(temp1 * A[np.ix_(S1, S2)] * temp2) 


                if LB > bestf:
                    print(i)
                    optimal = False                             
                    bestf = LB
                    unsel2.append(i)
                    unsel2.remove(j)
                    break
                
                S2.append(i)
                S2.remove(j)
                
    end = datetime.datetime.now()
    time = (end-start).seconds
    
    return S1, S2, bestf, time
    

### MIQP of SCCA ###
def scca_perp(n1, k1, S1):
    E = np.eye(n1, dtype=int)
    B1 =  B - mu*E
    
    ## Cholesky factorization of B1    
    [s,b] = np.linalg.eigh(B1) 
    
    sqrt_eigen = [0]*n1
    for i in range(n1):
        if s[i]>0:
            sqrt_eigen[i] = math.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.zeros([n1, n1]) 
    for i in range(n1):
        for j in range(n1):
            V[i, j] = sqrt_eigen[i]*b[j,i]

    L = np.matrix(V)
    
    start = datetime.datetime.now()

    m = Model("scca_perp")
    
    #### Creat variables ####
    # zvar = m.addVars(n, vtype=GRB.BINARY, name="z")
    xvar = m.addVars(n1, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="x")
    y = m.addVars(n1, vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name="y")
    zvar = m.addVars(n1, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")
    uvar = m.addVars(n1, vtype=GRB.CONTINUOUS, lb=0.0, name="u")
    
    # m.addConstr(zvar[finx] == 0.0)
    #### Set objective ####
    m.setObjective(sum(al[i]*xvar[i] for i in range(n1)), GRB.MAXIMIZE)
    
    for i in range(n1):
        m.addConstr(sum(L[i, j]*xvar[j] for j in range(n1)) == y[i])
        
        m.addConstr(xvar[i]*xvar[i]  <= zvar[i]*uvar[i])
        
    
    m.addConstr(sum(y[i]*y[i] + mu*uvar[i] for i in range(n1)) <= 1)
    
    m.addConstr(sum(zvar[i] for i in range(n1)) <= k1)
    for i in range(n1): 
        if i in S1:
            zvar[i].start = 1
        else:
            zvar[i].start = 0
            
    m.params.OutputFlag = 1
    m.optimize()
    
    xsol = [0]*n1
    for i in range(n1):
        xsol[i] = xvar[i].x

    end = datetime.datetime.now()
    time = (end-start).seconds
    # print(xsol)
    return sum(al[i]*xsol[i] for i in range(n1)), time


### MIQP of SCCA ###
def scca_perp2(m2, k2, S2):
    E = np.eye(m2, dtype=int)
    C1 =  C - nu*E
    
    ## Cholesky factorization of B1    
    [s,b] = np.linalg.eigh(C1) 
    
    sqrt_eigen = [0]*m2
    for i in range(m2):
        if s[i]>0:
            sqrt_eigen[i] = math.sqrt(s[i])
        else:
            sqrt_eigen[i] = 0
               
    V = np.zeros([m2, m2]) 
    for i in range(m2):
        for j in range(m2):
            V[i, j] = sqrt_eigen[i]*b[j,i]

    L = np.matrix(V)
    
    start = datetime.datetime.now()

    m = Model("scca_perp2")
    
    #### Creat variables ####
    # zvar = m.addVars(n, vtype=GRB.BINARY, name="z")
    xvar = m.addVars(m2, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    y = m.addVars(m2, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="y")
    zvar = m.addVars(m2, vtype=GRB.CONTINUOUS, lb=0.0, ub=1.0, name="z")
    uvar = m.addVars(m2, vtype=GRB.CONTINUOUS, lb = 0.0, name="u")
    
    #### Set objective ####
    m.setObjective(sum(br[i]*xvar[i] for i in range(m2)), GRB.MAXIMIZE)
    
    for i in range(m2):
        m.addConstr(sum(L[i, j]*xvar[j] for j in range(m2)) == y[i])
        
        m.addConstr(xvar[i]*xvar[i]  <= zvar[i]*uvar[i])
        
    
    m.addConstr(sum(y[i]*y[i] + nu*uvar[i] for i in range(m2)) <= 1)
    
    
    m.addConstr(sum(zvar[i] for i in range(m2)) <= k2)
    
    for i in range(m2): 
        if i in S2:
            zvar[i].start = 1
        else:
            zvar[i].start = 0
            
    m.params.OutputFlag = 1
    m.optimize()
    
    xsol = [0]* m2
    for i in range(m2):
        xsol[i] = xvar[i].x

    end = datetime.datetime.now()
    time = (end-start).seconds
    # print(xsol)
    return sum(br[i]*xsol[i] for i in range(m2)), time

df_UB = pd.DataFrame(columns=('n1', 'm2', 's1', 's2', 
                              'opt_val', 'opt_time'))

loc = 0
for n1 in range(50, 251, 50):
    m2=n1
    for s1 in range(5, 11, 5): # set the values of s
        s2 =s1
        
        print("This is case", loc+1)
        gen_data(n1, m2, s1, s2)
        S1, S2, lval, ltime = localsearch(n1, m2, s1, s2)
        z, t = scca_perp(n1,  s1, S1)
        z2, t2 = scca_perp2(m2, s2, S2)
       
        
        df_UB.loc[loc] = np.array([n1, m2, s1, s2, 
                                   z*z2, t+t2])
        loc = loc+1 


