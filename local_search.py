from typing import List, Tuple
import math
import time
import numpy as np
import scipy.linalg as LA
from numpy import (unravel_index, argmax, matrix, ix_, zeros)
from numpy.linalg import svd
import greedy
import random
from random import sample
import pandas as pd
import os

greedy = greedy.greedy

def gen_data(n1, m2, s1, s2, dataname):
    # global A
    # global B
    # global C
    
    # random.seed(1)
    
    # X = pd.read_csv(os.getcwd()+'/datasets/dna.csv',
    #                       encoding = 'utf-8',sep=',')
    
    # X = np.array(X)
    
    # sn1 = sample(range(len(X[0,:])), n1)
    # X = X[:, sn1]
    # X = np.matrix(X)
    
    
    # Y = pd.read_csv(os.getcwd()+'/datasets/rna.csv',
    #                      encoding = 'utf-8',sep=',')
    
    # Y = np.array(Y)
    # sm2 = sample(range(len(Y[0,:])), m2)
    # Y = Y[:, sm2]
    # Y = np.matrix(Y)
    
    # N = 89
    
    # B = sum([X[i,:].T*X[i,:] for i in range(N)])/N 
    
 
    # C = sum([Y[i,:].T*Y[i,:] for i in range(N)])/N

    # A = sum([X[i,:].T*Y[i,:] for i in range(N)])/N
    # return (A, B, C)

    global A
    global B
    global C

    data = pd.read_table(os.getcwd()+'/datasets/'+dataname+'_txt',
      encoding = 'utf-8',sep=',')
    temp = data.drop(['Unnamed: 0'], axis=1)
    cov = np.matrix(np.array(temp))
    B = np.matrix(cov[0:n1, 0:n1])
    C = np.matrix(cov[n1:(n1+n2), n1:(n1+n2)])
    A = np.matrix(cov[0:n1, n1:(n1+n2)])

    return (A, B, C)

def localsearch(n1, n2, s1, s2, A, B, C): 
    start = time.time()
    S1, S2, gbestf, gtime = greedy(n1, n2, s1, s2, A, B, C)
    print('greedy:', gbestf, gtime)
    bestf = gbestf
    
    sn = list(range(n1))
    sm = list(range(n2))
    
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

        for i in S1:
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
    total_time = end - start
    
    return gbestf, gtime, bestf, total_time

loc = 0
df_LB = pd.DataFrame(columns=('n', 's', 'greedy_val', 'gtime', 'ls_val', 'lstime'))

list_n = [26, 30, 34, 57, 64, 77, 128, 385]
list_name = [ 'pol', 'wdbc', 'dermatology',
              'spambase', 'digits',  'buzz', 'gas', 'slice']

for i in range(len(list_n)):
    for s in range(5, 11, 5):
        n = list_n[i]
        data_name = list_name[i]
        n1 = int(n/2)
        n2 = n-n1
        s1 = s
        s2 = s
        
        # n1, m2, s1, s2 = 100, 100, 10, 10
        A, B, C = gen_data(n1, n2, s1, s2, data_name)
        gbestf, gtime, bestf, total_time = localsearch(n1, n2, s1, s2, A, B, C)
        print('case', loc+1, 'is:', gbestf, gtime, bestf, total_time)

        df_LB.loc[loc] = np.array([n, s, gbestf, gtime, bestf, total_time])
        loc = loc+1  


