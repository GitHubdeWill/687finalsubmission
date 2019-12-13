import numpy as np
from .bbo_agent import BBOAgent
from pprint import pprint

from typing import Callable

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(name)s; %(levelname)s:%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

import inspect
import re
def slog(x):
    """log var status with name"""
    frame = inspect.currentframe().f_back
    s = inspect.getframeinfo(frame).code_context[0]
    r = re.search(r"\((.*)\)", s).group(1)
    print("{} = {}".format(r,x))

class CEM(BBOAgent):
    """
    The cross-entropy method (CEM) for policy search is a black box optimization (BBO)
    algorithm. This implementation is based on Stulp and Sigaud (2012). Intuitively,
    CEM starts with a multivariate Gaussian dsitribution over policy parameter vectors.
    This distribution has mean thet and covariance matrix Sigma. It then samples some
    fixed number, K, of policy parameter vectors from this distribution. It evaluates
    these K sampled policies by running each one for N episodes and averaging the
    resulting returns. It then picks the K_e best performing policy parameter
    vectors and fits a multivariate Gaussian to these parameter vectors. The mean and
    covariance matrix for this fit are stored in theta and Sigma and this process
    is repeated.

    Parameters
    ----------
    sigma (float): exploration parameter
    theta (numpy.ndarray): initial mean policy parameter vector
    popSize (int): the population size
    numElite (int): the number of elite policies
    numEpisodes (int): the number of episodes to sample per policy
    evaluationFunction (function): evaluates the provided parameterized policy.
        input: theta_p (numpy.ndarray, a parameterized policy), numEpisodes
        output: the estimated return of the policy
    epsilon (float): small numerical stability parameter
    """

    def __init__(self, theta:np.ndarray, sigma:float, popSize:int, numElite:int, numEpisodes:int, evaluationFunction:Callable, epsilon:float=0.0001):
        self._name = "cem"

        self._origin_theta = theta

        # Parameters passed by the constructor
        self._theta = theta
        self._popSize = popSize
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._evaluationFunction = evaluationFunction
        self._epsilon = epsilon

        # model param
        self._Sigma = sigma * np.eye(len(theta))
        self._original_Sigma = self._Sigma


    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        # parameter will be hstack(cov, theta)
        # return np.hstack(self._Sigma, self._theta)
        return self._theta

    def train(self)->np.ndarray:

        # new_theta = self._theta
        # new_cov = self._Sigma

        thetas = []
        Jhats = []
        for k in range(self._popSize):
            theta_k = np.random.multivariate_normal(self._theta, self._Sigma)
            J_hat_k = self._evaluationFunction(theta_k, self._numEpisodes)
            thetas.append(theta_k)
            Jhats.append(J_hat_k)

        # logger.debug("Mean return: "+str(np.mean(Jhats)))
        # Sorted by J
        sorted_theta_J_hat_pairs = sorted(zip(thetas, Jhats), key=lambda x: x[1], reverse=True)

        # pprint(sorted_theta_J_hat_pairs)

        sorted_thetas = []
        sorted_Jhats = []

        for x, y in sorted_theta_J_hat_pairs:
            sorted_thetas.append(x)
            sorted_Jhats.append(y)

        sorted_thetas = np.array(sorted_thetas)
        sorted_Jhats = np.array(sorted_Jhats)

        # pprint(sorted_Jhats)

        # logger.debug("Mean return for Elite: "+str(np.mean(sorted_Jhats[:self._numElite])))

        # slog(sorted_thetas.shape)
        # slog(sorted_Jhats.shape)

        elite_thetas = sorted_thetas[:self._numElite, :]

        new_theta = np.mean(elite_thetas, axis=0)

        # pprint(new_theta)

        theta_deltas = elite_thetas - new_theta
        cov_sum_term = np.zeros_like(self._Sigma)
        for theta_delta in theta_deltas:
            cov_sum_term += np.outer(theta_delta, theta_delta)

        new_cov = (1/(self._epsilon + self._numElite)) * (self._epsilon * np.eye(len(self._theta)) + cov_sum_term)

        self._theta = new_theta
        self._Sigma = new_cov

        return new_theta


    def reset(self)->None:
        self._theta = self._origin_theta
        self._Sigma = self._original_Sigma
