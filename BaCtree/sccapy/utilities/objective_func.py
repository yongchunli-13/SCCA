from typing import List, Tuple

from sccapy.utilities.problem_data import ProblemData

class Objective:
    """
    Template for evaluating the problem's objective.
    """

    def __init__(self, data: ProblemData, proposed_func) -> None:
        self.data = data
        # self._test_func(proposed_func)
        self.f0 = proposed_func

    def __call__(self, S1: List[int]) -> Tuple[float, float]:
        """
        Requires fully selected S1
        """
        return self.f0(self.data, S1)
    
    def _test_func(proposed_func):
        '''
        Try unit tests here
        '''
        pass