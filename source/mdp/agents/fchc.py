import numpy as np
from .bbo_agent import BBOAgent

from typing import Callable


class FCHC(BBOAgent):
    """
    First-choice hill-climbing (FCHC) for policy search is a black box optimization (BBO)
    algorithm. This implementation is a variant of Russell et al., 2003. It has 
    remarkably fewer hyperparameters than CEM, which makes it easier to apply. 
    
    Parameters
    ----------
    sigma (float): exploration parameter 
    theta (np.ndarray): initial mean policy parameter vector
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates a provided policy.
        input: policy (np.ndarray: a parameterized policy), numEpisodes
        output: the estimated return of the policy 
    """

    def __init__(self, theta:np.ndarray, sigma:float, evaluationFunction:Callable, numEpisodes:int=10):
        self._origin_theta = theta

        self._name = "fchc"
        self._theta = theta
        self._sigma = sigma
        self._evaluationFunction = evaluationFunction
        self._numEpisodes = numEpisodes

        # Initial Evaluation
        self.j_hat = self._evaluationFunction(theta, numEpisodes)


    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._theta

    def train(self)->np.ndarray:
        # print(self._theta)
        theta_prime = np.random.multivariate_normal(self._theta, self._sigma * np.eye(len(self._theta)))

        new_j_hat = self._evaluationFunction(theta_prime, self._numEpisodes)

        if new_j_hat > self.j_hat:
            self._theta = theta_prime
            self.j_hat = new_j_hat

        # print(self.j_hat)

        # print(self._theta)
        
        return self._theta

    def reset(self)->None:
        self._theta = self._origin_theta
