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
import scipy

greedy = greedy.greedy

def gen_data(n1, n2, s1, s2,k):
    
    global X, Y, A, B, C
    
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
    
    # if n1 == 60:
    #     l = np.random.uniform(0.6,1)
    
    # if n1 == 80:
    #     l = np.random.uniform(0.5,1)
        
    # if n1 == 120:
    #     l = np.random.uniform(0.7,1)
    # if n1 <60 or n1==100:
    #     l = np.random.uniform(0,1)
    
    l = np.random.uniform(0,1)
    A = l*B*u.T*v*C
   
    cov = np.block([[B, A], [A.T, C]])
    mean = [0.0]*(n1+n2)
    N = 5000
    np.random.seed(1)
    temp = np.random.multivariate_normal(mean, cov, N)
    MX = np.matrix(temp[:,0:n1])
    MY = np.matrix(temp[:, n1:(n1+n2)])
    
    B = sum([MX[i,:].T*MX[i,:] for i in range(N)])/N
    
    C = sum([MY[i,:].T*MY[i,:] for i in range(N)])/N
    
    A = sum([MX[i,:].T*MY[i,:] for i in range(N)])/N
    
    U, s, V = svd(A) # SVD decomposition
    s[s<1e-6]=0 
    l = len(s)
    sindex = np.argsort(-s)
    opti = sindex[0]
    print(s[opti])
    
    A = s[opti]*np.matrix(np.array(U[:,opti]))*np.matrix(np.array(V[opti,:]))
        
    al = math.sqrt(s[opti])*U[:,opti]
    br = math.sqrt(s[opti])* V[opti,:].T
        
    # global A
    # global B
    # global C

    # data = pd.read_table(os.getcwd()+'/datasets/'+dataname+'_txt',
    #   encoding = 'utf-8',sep=',')
    # temp = data.drop(['Unnamed: 0'], axis=1)
    # cov = np.matrix(np.array(temp))
    # B = np.matrix(cov[0:n1, 0:n1])
    # C = np.matrix(cov[n1:(n1+n2), n1:(n1+n2)])
    # A = np.matrix(cov[0:n1, n1:(n1+n2)])
    
    [a,b] = np.linalg.eigh(B) 
    a = a.real # engivalues
    b = b.real # engivectors
    engi_val = [0]*len(a)
    for l in range(len(a)):
        if(a[l] > 1e-8):
            engi_val[l] = 1/math.sqrt(a[l])
        else:
            engi_val[l] = 0

    temp = b*np.diag(engi_val)*b.T 
    
    [a,b] = np.linalg.eigh(C) 
    a = a.real # engivalues
    b = b.real # engivectors
    engi_val = [0]*len(a)
    for l in range(len(a)):
        if(a[l] > 1e-8):
            engi_val[l] = 1/math.sqrt(a[l])
        else:
            engi_val[l] = 0

    temp1 = b*np.diag(engi_val)*b.T 
    
    K =  temp*A*temp1
    u, a, v = np.linalg.svd(K)
    sorted_sigma = sorted(a, reverse=True)
    LB = sum(sorted_sigma[:k])
    print(LB)
    return A, B, C, LB




def localsearch(n1, n2, s1, s2, A, B, C, k): 
    start = time.time()
    S1, S2, gbestf, gtime = greedy(n1, n2, s1, s2, A, B, C,k)
    # print('greedy:', gbestf, gtime)
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
                
                # LB = max(sigma)
                sorted_sigma = sorted(sigma, reverse=True)
                LB = sum(sorted_sigma[:k])
                
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
                    sorted_sigma = sorted(sigma, reverse=True)
                    LB = sum(sorted_sigma[:k])
                
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
df_LB = pd.DataFrame(columns=('n1', 'n2', 's1', 's2', 'k', 'greedy_val', 'gtime', 'ls_val', 'lstime', 'UB'))
ratio = []
list_n = [34, 57, 64, 77, 128, 385]
list_name = [ 'dermatology',
              'spambase', 'digits',  'buzz', 'gas', 'slice']


for n1 in range(50, 250, 50):
    n2=n1
    for s1 in range(5, 11, 5): # set the values of s
        s2 = s1
#         if s1==5:
#             k=2
#         else:
#             k=2
        # n1, m2, s1, s2 = 100, 100, 10, 10
        # A, B, C, UB = gen_data(n1, n2, s1, s2,k, data_name)
        A, B, C, UB = gen_data(n1, n2, s1, s2,k)
        gbestf, gtime, bestf, total_time = localsearch(n1, n2, s1, s2, A, B, C,k)
        print('case', loc+1, 'is:', gbestf, gtime, bestf, total_time, UB)
        ratio.append(bestf/UB)
        df_LB.loc[loc] = np.array([n1, n2, s1, s2,k, gbestf, gtime, bestf, total_time, UB])
        loc = loc+1  
  
# ratio = np.array(ratio)  
# data = ratio.reshape(8, 8)
# import seaborn as sns
# import matplotlib.pyplot as plt

# blue_cmap = sns.color_palette("Blues", as_cmap=True)

# plt.figure(figsize=(10, 8))
# ax = sns.heatmap(data, cmap=blue_cmap, annot=True, fmt=".3f", cbar=True)

# ax.set_xticklabels(np.arange(10, 81, 10))
# ax.set_yticklabels(np.arange(10, 81, 10))
# # Add titles and labels
# # plt.title("The ratio of SCCA and CCA correlations on spambase data")
# plt.xlabel("$s_1$", fontsize=20)
# plt.ylabel("$s_2$", fontsize=20)
# ax.invert_yaxis() 

# plt.savefig("synthetic.png", dpi=300) 
# plt.show()
