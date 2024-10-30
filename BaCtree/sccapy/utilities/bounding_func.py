from typing import Any, List, Tuple

from sccapy.utilities.problem_data import ProblemData

class Bounder:
    '''
    Template for bounding
    '''

    def __init__(self, data: ProblemData, proposed_func):
        self.data = data
        self._test_func(proposed_func)
        self.bounding_func = proposed_func
    
    def __call__(self, S0: List[int]=[], S1: List[int]=[]) -> tuple[float, float]:
        return self.bounding_func(self.data, S0, S1)
    
    def _test_func(self, proposed_func):
        pass

class LowerBounder:
    '''
    Template for lower bounding: proposed func should only take one argument: ProblemData
    '''

    def __init__(self, data: ProblemData, proposed_func):
        self.data = data
        self._test_func(proposed_func)
        self.bounding_func = proposed_func
    
    def __call__(self, S0: List[int]=[], S1: List[int]=[]) -> tuple[float, float]:
        return self.bounding_func(self.data, S0, S1)
    
    # def __call__(self) -> Tuple[float, float]:
    #     return self.bounding_func(self.data)
    
    def _test_func(self, proposed_func):
        pass    
