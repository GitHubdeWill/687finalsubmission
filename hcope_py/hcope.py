import numpy as np
from mdp.history import History
from mdp.policies.f_policy import FBSoftmax
from hcope_functions import PDIS_CMA
from scipy.stats import t as tdist
from hcope_helper import *

import cma.test
import cma

class HCOPE(object):
    """
        The class HCOPE will take:

        data D with list of
            history H that is generated through running pi_b
            H will be a T * 3 * EpisodeLength

        D: data of multiple History
        pi_b: safety policy
        Ds_count: the number of behaviour data
        c: the threshold for safety test
        delta = 0.05: t test side probability
        gamma=1.0: discount of policy
        cma_sigma=0.5: sigma for CMA-ES
        init_theta: init theta for CMA-ES, default to pi_b
    """    
    def __init__(self, D, pi_b, Ds_count, c, delta = 0.01, gamma=1.0, cma_sigma=3, init_theta=None):
        # Initialize parameters
        self.D = D
        self.Ds_count = Ds_count  # number of safety history
        self.gamma = gamma
        self.pi_b = pi_b
        self.delta = delta
        self.c = c

        self.episodeCount = len(D)

        # cma param, making place holder for candidate theta
        if init_theta is None:
            init_theta = pi_b.parameters.copy()
        self.es = cma.CMAEvolutionStrategy(init_theta, cma_sigma)

        # cma function to be optimized
        self.cma_func = PDIS_CMA(self.Dc, pi_b, Ds_count, delta, c)

    @property
    def Dc(self):
        return self.D[self.Ds_count:]

    @property
    def Ds(self):
        return self.D[:self.Ds_count]


    def test_pdis(self, pi_e):
        """Estimate J(pi_e) on safety data"""
        return pdis(self.Ds, pi_e, self.pi_b)

    def get_policies(self, number=10):
        """Select one theta_c use CMA-ES"""
        if (self.es.stop()):
            print("Converged")
        (proposed_theta_cs, test_results) = self.es.ask_and_eval(self.cma_func, number=number)
        self.es.tell(proposed_theta_cs, test_results)
        self.es.logger.add()
        
        return proposed_theta_cs, test_results

    # def plot_es(self):
    #     self.es.logger.plot()

    def sample_theta_c(self, number=10):
        """
            sample multiple theta_c and return them if passed safety test

            return [theta, function_value, pdis_on_safety_data]
        """
        # Do candidate selection
        solution_found = False
        theta_c = None
        ret = []
        while (not solution_found):
            print("getting new set of thetas")
            proposed_theta_cs, test_results = self.get_policies()

            idx = np.argmin(test_results)
            print(test_results)

            # Safety test
            nnumStates = self.pi_b.numStates
            nnumActions = self.pi_b.numActions
            niOrder = self.pi_b.iOrder
            ndOrder = self.pi_b.dOrder
            pi_c = FBSoftmax(nnumStates, nnumActions, niOrder, ndOrder)
            
            for theta_c, func_ret in zip(proposed_theta_cs, test_results):
                pi_c.parameters = theta_c
                passed = safety_test(self.Ds, pi_c, self.pi_b, len(self.Ds), self.delta, self.c)
                if passed:
                    ret.append((theta_c, func_ret, self.test_pdis(pi_c)))
                solution_found = solution_found or passed
        print("sampled",len(ret),"results")
        return ret


