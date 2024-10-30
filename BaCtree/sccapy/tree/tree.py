from typing import Tuple, List, Optional
from numbers import Number
import time
import math
from copy import deepcopy
from enum import Enum

from numpy import argmax, argmin

from sccapy.tree.node import Node
from sccapy.utilities.bounding_func import (Bounder, LowerBounder)
from sccapy.utilities.objective_func import Objective

class BranchStrategy(Enum):
    DFS = 1
    SHRINK = 2

class Tree:

    def __init__(self, n1: int, n2: int, s1: int, s2: int,
                 phi_ub: Bounder, phi_lb: LowerBounder, obj: Objective) -> None:
        
        self.n1: int = n1
        self.n2: int = n2
        self.s1: int = s1
        self.s2: int = s2
        Node.n1 = n1
        Node.n2 = n2
        Node.s1 = s1
        Node.s2 = s2
        
        self.phi_ub: Bounder = phi_ub
        self.phi_lb: LowerBounder = phi_lb
        self.f0: Objective = obj
        
        self.LB: float = -math.inf
        self.UB: float = math.inf
        self.nodes: List[Node] = []
        self.feasible_leaves: List[Node] = [] # only will be >= initial_LB
        self.branch_strategy: BranchStrategy = BranchStrategy.DFS
        
        ### Research Specific Objects ###
        self.known_fixed_in: Optional[List[int]] = None
        self.variable_scores: Optional[List[float]] = None
        
        ### Framework Metrics ###
        self._status: Optional[str] = None
        self._value = None
        self.num_iter: int = 0
        self.LB_update_iterations: List[int] = []
        self.solve_time: float = 0 # total enumeration time
        self.ub_bound_time: float = 0 # total runtime of ub function
        self.lb_bound_time: float = 0 # total runtime of lb function
        self.obj_time: float = 0 # total runtime of obj function
        self.initial_gap: float = math.inf
        self.initial_LB: float = -math.inf
        self.initial_UB: float = math.inf
        self.optsol: Optional[List[int]] = None

    @property
    def gap(self):
        return self.UB - self.LB 
    
    def solve(self, eps: Number=1e-3, timeout: Number=60, fixed_vars: List[int]=None,
              var_scores: List[float]=None, branch_strategy:str="dfs") -> bool:
        """Enumerate a branch and bound tree to solve the SCCA problem to global optimality
        using the bounding and objective functions passed into the tree upon its construction.

        Populates the :code:'status' and :code:'value' attributes on the
        tree object as a side-effect.

        Arguments
        ----------
        eps: positive float, optional
            The desired optimality tolerance.
            The default tolerance is 1e-3.
        timeout: float, optional
            The number of minutes solve will run before terminating.
            The default timeout is after 60 minutes.
        fixed_vars: List[int], optional
            Variable elements known to be fixed in (i.e. x[i] = 1 forall i in fixed_vars).
        var_scores: List[float], optional
            Scores used to determine variable branching priority.
        branch_strategy: str, optional
            The method for choosing subproblems in the tree.
            Defaults to depth first search (dfs). Any other input
            will result in an upper bound shrinking strategy.
        
        Returns
        -------
        bool: Whether or not the problem was solved to global optimality.
        
        Raises
        ------
        AssertionError
            Raised if epsilon or timeout are not Numbers.
            Raised if var_scores doesn't contain enough scores.
        ValueError
            Raised if a fixed variable index is negative or >= n1 + n2
        """
        
        ######### SETUP START #########
        
        start_time = time.time()
        loop_time = time.time() - start_time

        if branch_strategy != "dfs":
            self.branch_strategy: BranchStrategy = BranchStrategy.SHRINK

        assert isinstance(eps, Number), "eps must be a Number"
        assert eps > 0, "eps must be positive."
        assert isinstance(timeout, Number), "timeout must be a Number"
        # keep track of these for metric purposes
        self.eps = eps
        self.timeout = timeout

        # check if there are variable scores
        if var_scores != None:
            assert len(var_scores) == self.n1 + self.n2, "there must be variable scores for all variables"
            self.variable_scores = deepcopy(var_scores)

        # check if there are fixed variables
        # create root node, gap, LB accordingly
        S0, S1 = [], []
        if fixed_vars != None:
            for i in range(len(fixed_vars)):
                assert isinstance(fixed_vars[i], int), "the fixed_vars list must contain integers."
            S1 = deepcopy(fixed_vars)
            num_s1_fixed, num_s2_fixed = self._fix_vars(fixed_vars)
        
        self._create_root_node(S0, S1, num_s1_fixed, num_s2_fixed)

        ######### SETUP END #########

        
        ######### MAIN #########
        
        while (self.UB-self.LB > eps and timeout > (loop_time / 60)):
            self.num_iter += 1
            node: Node = self._choose_subproblem()

            # check pruning again: possible LB has been updated since 
            # node was added to L_k
            if node.ub <= self.LB + eps:
                ub_node = max(self.nodes)
                self.UB = ub_node.ub
                continue
            
            # split problem handles updating LB (if possible)
            # and handles adding the new subproblems to nodes (L_k)
            self._split_problem(node)

            # Moved the following to evaluate node function
            # ub_node = max(self.nodes)
            # self.UB = ub_node.ub

            loop_time = time.time() - start_time

            if (self.gap > eps and len(self.nodes) == 0):
                raise Exception("Node list is empty but GAP is unsatisfactory.")
            
            # if self.num_iter%20 ==0:
            print(f"Iteration ={self.num_iter} | current LB = {self.LB} | current UB = {self.UB}", end = "\r") 
            
            # print(f"Iteration {self.num_iter} | current LB = {self.LB}  | Number of Open Subproblems = {len(self.nodes)}"
            #     + f" | Total Running Time = {loop_time} seconds ", end = "\r") 
        
        if (timeout < loop_time / 60):
            self._status = "solve timed out."
            return False
        
        if self.UB-self.LB < 1e-3:
            self._status = "global optimal found."
            return True
        
        self._status = "global optimal found."
        # print(self.f0(S1=node.feasible_solution))
        self._value = self.LB
        self.solve_time = time.time() - start_time
        return True
    
    def _fix_vars(self, proposed_fixed_vars):
            s1_count, s2_count = 0, 0
            for i in proposed_fixed_vars:
                if 0 <= i < self.n1:
                    s1_count += 1
                elif self.n1 <= i <= self.n1 + self.n2 - 1:
                    s2_count += 1
                else:
                    raise ValueError("fixed variable indices should be nonnegative and less than n1 + n2.")
            return (s1_count, s2_count)


    # def _fix_vars(self, proposed_fixed_vars):
    # DELETE THIS. WOULD ONLY MAKE SENSE IF SHRINKING SUBPROBLEMS, WHICH YOU ARE NOT.
    #         s1_count, s2_count = 0, 0
    #         for i in proposed_fixed_vars:
    #             if 0 <= i < self.n1:
    #                 s1_count += 1
    #             elif self.n1 <= i <= self.n1 + self.n2 - 1:
    #                 s2_count += 1
    #             else:
    #                 raise ValueError("fixed variable indices should be nonnegative and less than n1 + n2.")
            
    #         self.n1 -= s1_count
    #         self.s1 -= s1_count
    #         Node.n1 -= s1_count
    #         Node.s1 -= s1_count

    #         self.n2 -= s2_count
    #         self.s2 -= s2_count
    #         Node.n2 -= s2_count
    #         Node.s2 -= s2_count

    def _create_root_node(self, S0, S1, s1_prime, s2_prime):
        root_node: Node = Node(fixed_in=S1, fixed_out=S0, s1_prime=s1_prime, s2_prime=s2_prime,
                               l1_prime=0, l2_prime=0)
        
        root_node.ub, root_ub_time, _ = self.phi_ub(root_node.fixed_out, root_node.fixed_in)
        self.ub_bound_time += root_ub_time
        root_lb, root_lb_time, rootsol = self.phi_lb([], [])
        self.lb_bound_time += root_lb_time
        
        # self.variable_scores = scores
        self.UB = root_node.ub
        self.LB = root_lb
        self.optsol = rootsol
        self.initial_gap = self.UB - self.LB
        self.nodes.append(root_node)
    
    def _choose_subproblem(self) -> Node:
        if self.branch_strategy == BranchStrategy.SHRINK:
            return self.nodes.pop(argmax(self.nodes))
        
        else:
            return self.nodes.pop()

    def _split_problem(self, node: Node):
        """
        For visualization purposes, assume "left" subproblem corresponds to discarding
        a variable while a "right" subproblem corresponds to selecting a variable.

        Don't need to handle if node is leaf since all leaves are passed into
        feasible solutiosn only, not L_k.

        when adding to L_k see if pruning should take place
        also see if we need to update LB.
        updating UB happens in while loop.

        also need to handle the leaf node conditions.

        just handle the conditions. Let node class handle computations

        For a chosen variable there are six possible conditions to consider
        WLOG, for x variable
        1. internal node -> create two subproblems
        2. s1_prime = s1 -> create left subproblem
        3. l1_prime = n1_prime - s_i -> create right subproblem

        Note that the following conditions are handled prior to the if statements:
        1. WLOG node is_x_terminal_leaf -> only y indices are left in var_scores_prime
        2. node is a terminal leaf node -> doesn't get added to L_k to begin with.
        
        """
        # TODO (later): add functionality for random branching

        var_scores_prime = self._create_var_scores_prime(node)
        chosen_var = argmin(var_scores_prime)

        # print("BRANCHING VAR: ", chosen_var) # DEBUG
        # print("VAR SCORES PRIME", var_scores_prime)

        if (chosen_var < self.n1 and node.is_x_internal_node):
            # print("x internal 2 new")  # DEBUG
            self._create_left_subproblem(node, chosen_var, x_branch=True)
            self._create_right_subproblem(node, chosen_var, x_branch=True)
        elif (chosen_var < self.n1 and node.s1_prime_full):
            # print("x internal 1 new")  # DEBUG
            self._create_left_subproblem(node, chosen_var, x_branch=True)
        elif (chosen_var < self.n1 and node.l1_prime_full):
            # print("x internal 1 new")  # DEBUG
            self._create_right_subproblem(node, chosen_var, x_branch=True)
        elif (chosen_var >= self.n1 and node.is_y_internal_node):
            # print("y internal 2 new")  # DEBUG
            self._create_left_subproblem(node, chosen_var, x_branch=False)
            self._create_right_subproblem(node, chosen_var, x_branch=False)
        elif (chosen_var >= self.n1 and node.s2_prime_full):
            # print("y internal 1 new")  # DEBUG
            self._create_left_subproblem(node, chosen_var, x_branch=False)
        elif (chosen_var >= self.n1 and node.l2_prime_full):
            # print("y internal 1 new")  # DEBUG
            self._create_right_subproblem(node, chosen_var, x_branch=False)
        else:
            raise Exception("Branching code ran into an unexpected case") # TODO: add information here, i.e. print something
    
    def _create_var_scores_prime(self, node: Node) -> List[float]:
        """
        use math.inf since we branch on least score 

        ensure that the var scores computed by the variable fixing algorithm
        cannot achieve math.inf

        TODO: TEST THIS.
        """
        to_return = []
        
        if node.is_x_terminal_leaf:
            to_return = [math.inf if i < self.n1 else self.variable_scores[i]
                    for i in range(self.n1 + self.n2)]
        elif node.is_y_terminal_leaf:
            to_return = [math.inf if i >= self.n1 else self.variable_scores[i]
                    for i in range(self.n1 + self.n2)]
        else:
            to_return = deepcopy(self.variable_scores)
            
        for index in node.fixed_out:
            to_return[index] = math.inf
        
        for index in node.fixed_in:
            to_return[index] = math.inf

        return to_return

    def _create_right_subproblem(self, node: Node, branch_idx: int, x_branch: bool) -> None:
        """
        fixes in:
        - adds the new index to fixed_in
        - increments s1_prime if x_branch == True else increments s2_prime
        - creates corresponding node
        """
        S1 = deepcopy(node.fixed_in) + [branch_idx]
        new_subproblem: Node
        if x_branch:
            new_subproblem = Node(S1, node.fixed_out, node.s1_prime + 1, node.s2_prime, node.l1_prime,
                                    node.l2_prime)
        else:
            new_subproblem: Node = Node(S1, node.fixed_out, node.s1_prime, node.s2_prime + 1, node.l1_prime,
                                    node.l2_prime)
        
        self._evaluate_node(new_subproblem)
    

    def _create_left_subproblem(self, node: Node, branch_idx: int, x_branch: bool) -> None:
        """
        fixes out:
        - adds the new index to fixed_out
        - increments l1_prime if x_branch == True else increments l2_prime
        - creates corresponding node
        """
        S0 = deepcopy(node.fixed_out) + [branch_idx]
        new_subproblem: Node
        if x_branch:
            new_subproblem = Node(node.fixed_in, S0, node.s1_prime, node.s2_prime, node.l1_prime+1,
                                    node.l2_prime)
        else:
            new_subproblem: Node = Node(node.fixed_in, S0, node.s1_prime, node.s2_prime, node.l1_prime,
                                    node.l2_prime+1)
        
        self._evaluate_node(new_subproblem)

    def _evaluate_node(self, node: Node) -> None:
        # note that if you want generalilzation of bounding/obj functions then they need
        # to return bound val, bounding time respectively.

        if node.is_terminal_leaf:
            node.ub, obj_find_time = self.f0(S1=node.feasible_solution)
            self.obj_time += obj_find_time
            ub_node = max(self.nodes + [node])
            self.UB = ub_node.ub
            if node.ub > self.LB:
                self.LB = node.ub
                self.LB_update_iterations.append(self.num_iter)
                print("LB UPDATED") # DEBUG
            self.feasible_leaves.append(node)
        else:

            node.ub, bound_time, _ = self.phi_ub(S0=node.fixed_out, S1=node.fixed_in)
            if self.num_iter%1000 ==0 and self.num_iter >= 100:
                node.lb, lbbound_time, currensol = self.phi_lb(S0=node.fixed_out, S1=node.fixed_in) # YL
                if self.LB < node.lb:
                    self.LB = max(self.LB, node.lb)
                    self.optsol = currensol
            # if self.num_iter%10 ==0 and self.num_iter >= 100:
            #     self.variable_scores = scores
            ub_node = max(self.nodes + [node])
            
            # self.variable_scores = scores
            self.ub_bound_time += bound_time
            self.UB = ub_node.ub
            if node.ub > self.LB + self.eps: # PRUNING
                self.nodes.append(node)
               
    
                
                
                