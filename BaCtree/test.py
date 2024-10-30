import numpy as np
import datetime
from sccapy.main import Problem
import pandas as pd
from sccapy.utilities.generate_data import gen_data
from sccapy.utilities.problem_data import ProblemData

from sccapy.research_algorithms.local_search import localsearch
from sccapy.research_algorithms.objective_eval import fval
from sccapy.research_algorithms.variable_fix import varfix
from sccapy.research_algorithms.upper_bound import upper_bound

loc = 0
df_fw = pd.DataFrame(columns=('n', 's', 'lb', 'ub', 'S1', 'S2', 'opttime'))

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

        prob: ProblemData = ProblemData(n1, n2, s1, s2)
        A, B, C = gen_data(n1, n2, s1, s2, data_name)
        setattr(prob, "A", np.matrix(np.copy(A)))
        setattr(prob, "B", np.matrix(np.copy(B)))
        setattr(prob, "C", np.matrix(np.copy(C)))
        
        start = datetime.datetime.now()
                
        lb, _ , _ = localsearch(prob, [], [])
        _, fixed_vars, var_scores, var_fixing_time, fix1, fix2 = varfix(prob, lb)
        ub, _, scores = upper_bound(prob, [], fixed_vars)
        
        
        # # print("OUTSIDE CALL GAP: ", ub - lb) # DEBUG
        
        prob1: Problem = Problem(n1, n2, s1, s2, A=A, B=B, C=C)
        
        prob1.solve(lower_bound_func=localsearch, upper_bound_func=upper_bound,
                    var_fix=varfix,  objective_func=fval)

        
        end = datetime.datetime.now()
        time = (end - start).seconds 
        df_fw.loc[loc] = np.array([n, s, lb, ub, fix1, fix2, time])
        loc = loc+1  
