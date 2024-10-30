class ProblemData():
    '''
    Use this wrapper class to store whatever data is needed for
    a specific problem instance.
    
    Example
    -------
    If using all defaults, then A, B, C need to be passed in
    from the synthetic data generator
    '''
    
    def __init__(self, n1: int, n2: int, s1: int, s2: int) -> None:
        assert n1 > s1, 'n1 must be greater than s1'
        assert n2 > s2, 'n2 must be greater than s2'
        
        self.n1: int = n1
        self.n2: int = n2
        self.s1: int = s1
        self.s2: int = s2