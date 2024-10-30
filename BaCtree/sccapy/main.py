from typing import Callable, List, Tuple
# from sccapy.utilities.problem_data import ProblemData

from sccapy.utilities.problem_data import ProblemData
from sccapy.utilities.bounding_func import (LowerBounder, Bounder)
from sccapy.utilities.objective_func import Objective

from sccapy.research_algorithms.local_search import localsearch
from sccapy.research_algorithms.variable_fix import varfix
from sccapy.research_algorithms.upper_bound import upper_bound
from sccapy.research_algorithms.objective_eval import fval

from sccapy.tree.tree import Tree

class Problem:

    def __init__(self, n1: int, n2: int, s1: int, s2: int,
                
                 **kwargs) -> None:
        
        # Problem data has checks for the integers 
        self.data = ProblemData(n1, n2, s1, s2)

        for key, value in kwargs.items():
            # do a validity check here
            setattr(self.data, key, value)

        # keep track of whether solve has been attempted or not.
        self.var_fix_time = 0

    def solve(self, eps: float=1e-3, timeout: float=10,
                lower_bound_func: Callable[[ProblemData, List[int], List[int]], Tuple[float, float]] = None,
                upper_bound_func: Callable[[ProblemData, List[int], List[int]],
                                            Tuple[float, float]] = None,
                objective_func: Callable[[ProblemData, List[int]],
                                            Tuple[float, float]] = None,
                var_fix: Callable[[ProblemData, float],
                                            Tuple[List[int], List[int], List[float], float]] = None):
                                            
        lb, _, _ = lower_bound_func(self.data,[],[])
        _, fixed_vars, var_scores, var_fixing_time, _, _ = var_fix(self.data, lb)
        self.var_fix_time += var_fixing_time

        phi_ub: Bounder = Bounder(self.data, upper_bound_func)
        phi_lb: LowerBounder = LowerBounder(self.data, lower_bound_func)
        obj: Objective = Objective(self.data, objective_func)
        
        solution_tree: Tree = Tree(self.data.n1, self.data.n2, self.data.s1, self.data.s2,
                                   phi_ub=phi_ub, phi_lb=phi_lb, obj=obj)
        
        is_solved = solution_tree.solve(eps=eps, fixed_vars=fixed_vars, var_scores=var_scores, branch_strategy='dfs')


        # print(solution_tree._status)
        print(solution_tree.optsol, fval(self.data, solution_tree.optsol))
        
        if (is_solved):
            print ("SUCCESSFULLY SOLVED")
            print(solution_tree.LB)
        else:
            print("Not successfully solved")

    
    def generate_statistics():
        pass

if __name__ == '__main__':
    print("SUCCESS")