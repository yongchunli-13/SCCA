
import os
import numpy as np
import pandas as pd
import mosek
from mosek.fusion import *
from numpy import matrix
from numpy import array
from scipy.linalg import svd
import sys
import datetime
import random
from random import sample
import math
import scipy


def gen_data(n1, m2, k1, k2):
    
    global A
    global B, mu
    global C, nu
    
    np.random.seed(1)
    random.seed(1)
    
    B = np.matrix(np.random.normal(0, 1, (n1, n1)))
    B = B*B.T + 1*np.matrix(np.eye(n1, dtype=float))

    C = np.matrix(np.random.normal(0, 1, (m2, m2)))
    C = C*C.T + 1*np.matrix( np.eye(m2, dtype=float)) 
    
    NS1 = sample(list(range(n1)), 0)
    NS2 = sample(list(range(m2)), 0)
    
    u = np.matrix(np.random.uniform(0, 1, (n1)))
    for i in NS1:
        u[0,i] = 0   
    u = u/math.sqrt((u*B*u.T)[0,0])
    
    v = np.matrix(np.random.uniform(0, 1, (m2)))
    for i in NS2:
        v[0,i] = 0   
    v = v/math.sqrt((v*C*v.T)[0,0])
    
    l = np.random.uniform(0.9,1)

    A = l*B*u.T*v*C
    
    cov = np.block([[B, A], [A.T, C]])
    mean = [0.0]*(n1+m2)
    N = 5000
    np.random.seed(1)
    temp = np.random.multivariate_normal(mean, cov, N)
    X = np.matrix(temp[:,0:n1])
    Y = np.matrix(temp[:, n1:(n1+m2)])
    
    B = sum([X[i,:].T*X[i,:] for i in range(N)])/N
    mu = [0]*n1
    for i in range(n1):
        S1 = list(range(n1))
        S1.remove(i)
        # print(S1)
        [a,b] = np.linalg.eigh(B[np.ix_(S1, S1)])
        mu[i] = min(a)
    mu = min(mu)
    
    C = sum([Y[i,:].T*Y[i,:] for i in range(N)])/N
    nu = [0]*m2
    for i in range(m2):
        S2 = list(range(m2))
        S2.remove(i)
        # print(S2)
        [a,b] = np.linalg.eigh(C[np.ix_(S2, S2)])
        nu[i] = min(a)
    nu = min(nu)
    
    temp1 = B
    temp1 = np.matrix(scipy.linalg.sqrtm(temp1.I))
    
    temp2 = C
    temp2 = np.matrix(scipy.linalg.sqrtm(temp2.I))
    
    u, sigma, v = np.linalg.svd(temp1 * A * temp2)
    LB = max(sigma)
    print(LB)

    

### MISDP of SCCA ###
def scca_sdp(n1, m2, k1, k2):
    start = datetime.datetime.now()


    with Model("SCCA") as M:
        z1 = M.variable('z1', n1, Domain.inRange(0.0,1.0))
        z2 = M.variable('z2', m2, Domain.inRange(0.0,1.0))
        
        X1 = M.variable('X1', [n1, n1])
        X2 = M.variable('X2', [m2, m2])
        X3 = M.variable('X3', [n1, m2])
        
        #### constraints ##### 
        M.constraint(Expr.sum(z1), Domain.equalsTo(k1))
        M.constraint(Expr.sum(z2), Domain.equalsTo(k2))
        # M.constraint(Expr.sum(z1), Domain.lessThan(k1))
        # M.constraint(Expr.sum(z2), Domain.lessThan(k2))
        
       
        M.constraint(Expr.vstack(Expr.hstack(X1, X3), Expr.hstack(Expr.transpose(X3), X2)), Domain.inPSDCone(n1+m2))
   
        t1 = M.constraint(Expr.sum(Expr.mulElm(B, X1)), Domain.lessThan(1.0))
        t2 = M.constraint(Expr.sum(Expr.mulElm(C, X2)), Domain.lessThan(1.0))
        
        l1 = [0]*n1
        for i in range(n1):
            l1[i] = M.constraint(Expr.sub(X1.index(i,i), Expr.mul(1/mu, z1.index(i))), Domain.lessThan(0.0))
            # for j in range(n1):
            #     if j <i:
            #         M.constraint(Expr.sub(X1.index(i,j), Expr.mul(1/(2*mu), z1.index(i))), Domain.lessThan(0.0))
            #         M.constraint(Expr.add(X1.index(i,j), Expr.mul(-1/(2*mu), z1.index(i))), Domain.greaterThan(0.0))
                    
            # l1[i] = M.constraint(X1.index(i,i), Domain.lessThan(z1[i]/mu))
            # l1[i] = M.constraint(Expr.sub(X1.index(i,i), z1[i]/mu), Domain.lessThan(0.0))
          
        l2 = [0]*m2
        for i in range(m2):
            # l2[i] = M.constraint(X2.index(i,i), Domain.lessThan(z2[i]/nu))
            l2[i] = M.constraint(Expr.sub(X2.index(i,i), Expr.mul(1/nu, z2.index(i))), Domain.lessThan(0.0))
            # for j in range(m2):
            #     if j <i:
            #         M.constraint(Expr.sub(X2.index(i,j), Expr.mul(1/(2*nu), z2.index(i))), Domain.lessThan(0.0))
            #         M.constraint(Expr.add(X2.index(i,j), Expr.mul(-1/(2*nu), z2.index(i))), Domain.greaterThan(0.0))
                    
        
        M.objective("obj", ObjectiveSense.Maximize, Expr.sum(Expr.mulElm(A, X3))) 
        M.solve()
        end = datetime.datetime.now()
        time = (end-start).seconds
        
        print("running time is",time)
        print (M.getPrimalSolutionStatus())
        print ("SDP relaxation value is", M.primalObjValue())
        
        objval = M.primalObjValue()
        
        z1sol = M.getVariable('z1').level()
        z2sol = M.getVariable('z2').level()
        # print("solution z1 is", sum(z1sol), z1sol)
        # print("solution z2 is", sum(z2sol), z2sol)
        # theta  = t1.dual() + t2.dual()

        lam = [0]*(n1+m2)
        
        for i in range(n1):
            lam[i] = 1/mu * l1[i].dual()
            
        for i in range(m2):
            lam[i+n1] = 1/nu * l2[i].dual()
            
        lam = np.array(lam)[:,0]
    
    return objval, time, z1, z2, lam

