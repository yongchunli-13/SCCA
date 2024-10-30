import math
from typing import List
from functools import cached_property

from numpy import (setdiff1d)

class Node:
    num_instances: int = 0

    n1: int
    n2: int
    s1: int
    s2: int

    def __init__(self, fixed_in: List[int], fixed_out: List[int], s1_prime: int,
                 s2_prime: int, l1_prime: int, l2_prime: int) -> None:
        Node.num_instances += 1
        self.num_node = Node.num_instances # make sure this does what you think it does. TODO:unittest

        self.fixed_in: List[int] = fixed_in
        self.fixed_out: List[int] = fixed_out
        
        self.s1_prime : int = s1_prime
        self.s2_prime : int = s2_prime
        self.l1_prime : int = l1_prime
        self.l2_prime : int = l2_prime
        self._ub: float = math.inf # also is obj_val if this is a terminal leaf node

    def __eq__(self, other):
        return self._ub == other._ub
    
    def __lt__(self, other):
        return self._ub < other._ub
    
    @property
    def ub(self):
        return self._ub
    
    @ub.setter
    def ub(self, value):
        self._ub = value

    def lb(self):
        return self._lb
    
        
    # from here on I could use all cached properties...not of critical importance
    
    @property
    def s1_prime_full(self) -> bool:
        return self.s1_prime == Node.s1
    
    @property
    def s2_prime_full(self) -> bool:
        return self.s2_prime == Node.s2
    
    @property
    def l1_prime_full(self) -> bool:
        return self.l1_prime == Node.n1 - Node.s1
    
    @property
    def l2_prime_full(self) -> bool:
        return self.l2_prime == Node.n2 - Node.s2
    
    @property
    def is_x_internal_node(self) -> bool:
        return not (self.s1_prime_full or self.l1_prime_full)
    
    @property
    def is_x_terminal_leaf(self) -> bool:
        return self.s1_prime_full or self.l1_prime_full
    
    @property
    def is_y_internal_node(self) -> bool:
        return not (self.s2_prime_full or self.l2_prime_full)
    
    @property
    def is_y_terminal_leaf(self) -> bool:
        return self.s2_prime_full or self.l2_prime_full
    
    @property
    def is_terminal_leaf(self) -> bool:
        if self.is_x_terminal_leaf and self.is_y_terminal_leaf:
            _ = self.feasible_solution # go ahead and calculate solution
            return True
        else:
            return False
    
    # create a property to determine the feasible solution
    # only calculate once.
    @cached_property
    def feasible_solution(self) -> list[int]:
        """
        What are the possibilities here:
        1) s1_prime and s2_prime are full -> S1 is as it should be
        2) l1_prime and l2_prime are full -> S1 needs all remaining values
        3) s1_prime is full and l2_prime is full -> S1 needs remaining values selected from [n1, n1+n2-1]
        4) l1_prime is full and s2_prime is full -> S1 needs remaining values selected from [0, n1-1]
        """
        if self.s1_prime_full and self.s2_prime_full:
            return self.fixed_in
        elif self.l1_prime_full and self.l2_prime_full:
            return list(setdiff1d(list(range(Node.n1 + Node.n2)), self.fixed_out))
        elif self.s1_prime and self.l2_prime_full:
            selected_s2 = [i for i in self.fixed_in if i >= Node.n1]
            selected_l2 = [i for i in self.fixed_out if i >= Node.n1]
            remaining_indices = list(setdiff1d(list(range(Node.n1, Node.n1+Node.n2)), selected_s2))
            # remaining_indices currently includes selected_l2 values. Remove these:
            remaining_indices = [x for x in remaining_indices if x not in selected_l2]
            return self.fixed_in + remaining_indices
        elif self.l1_prime_full and self.s2_prime_full:
            selected_s1 = [i for i in self.fixed_in if i < Node.n1]
            selected_l1 = [i for i in self.fixed_out if i < Node.n1]
            remaining_indices = list(setdiff1d(list(range(self.n1)), selected_s1))
            remaining_indices = [x for x in remaining_indices if x not in selected_l1]
            return self.fixed_in + remaining_indices
        else:
            raise Exception("unexpected behavior calculating a node's feasible solution")