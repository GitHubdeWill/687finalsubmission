# Function class for cma to optimize
from mdp.policies.f_policy import FBSoftmax
from hcope_helper import *

class PDIS_CMA:
    """The function that CMA-ES is trying to optimize"""
    
    def __init__(self, Dc, pi_b, Ds_size, delta, c):
        self.Dc = Dc
        self.pi_b = pi_b
        self.Ds_size = Ds_size
        self.delta = delta
        self.c = c

    def __call__(self, theta):
        # Construct pi_c from theta
        nnumStates = self.pi_b.numStates
        nnumActions = self.pi_b.numActions
        niOrder = self.pi_b.iOrder
        ndOrder = self.pi_b.dOrder
        pi_c = FBSoftmax(nnumStates, nnumActions, niOrder, ndOrder)
        pi_c.parameters = theta
        x = -pdis(self.Dc, pi_c, self.pi_b)

        # Estimate if pass safety test or not
        passed = est_safety_test(self.Dc, pi_c, self.pi_b, self.Ds_size, self.delta, self.c)
        if not passed:
            return min(x, 0)+100000
        return x
