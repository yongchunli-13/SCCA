
from typing import Tuple
import random
from random import sample
import math
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import numpy as np
from scipy.linalg import svd

def gen_data(n1, n2, s1, s2, dataname) -> Tuple[np.matrix, np.matrix, np.matrix]:
    
    global A
    global B
    global C

    data = pd.read_table(os.path.dirname(os.getcwd())+'/datasets/'+dataname+'_txt',
      encoding = 'utf-8',sep=',')
    temp = data.drop(['Unnamed: 0'], axis=1)
    cov = np.matrix(np.array(temp))
    B = np.matrix(cov[0:n1, 0:n1])
    C = np.matrix(cov[n1:(n1+n2), n1:(n1+n2)])
    A = np.matrix(cov[0:n1, n1:(n1+n2)])

    return (A, B, C)