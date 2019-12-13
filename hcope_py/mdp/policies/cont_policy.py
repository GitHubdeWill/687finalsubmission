import numpy as np
from .skeleton import Policy
from typing import Union

np.random.seed(1213971)

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


class PolySoftmax(Policy):
    """
    A Poly Softmax Policy (bs)
    with Linear Function Approximation

    Parameters

    """

    def __init__(self, epsilon = 0.001):
        '''m = 16, |A| = 2'''
        # Basis: 1,a,b,c,d,ab,bc,cd,ac,bd,ad, abc, abd, acd, bcd, abcd
        self.numStates = 4
        self.degree = 2
        self._theta = np.zeros(((self.degree+1)**self.numStates, 2))
        self.epsilon = epsilon
        self._sigma = 1


    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length m x|A|, 16 * 2
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of siactionactionze m x|A|, 16 * 2.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state, action=None)->Union[float, np.ndarray]:
        if action == None:
            return self.getActionProbabilities(state)
        else:
            return self.getActionProbabilities(state)[action]

    def getExpansion (self, state):
        a,b,c,d = state
        features = []
        for i in range(self.degree+1):
            for j in range(self.degree+1):
                for k in range(self.degree+1):
                    for l in range(self.degree+1):
                        features.append(a**i*b**j*c**k*d**l)

        return features

    def sampleAction(self, state)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        probs = self.getActionProbabilities(state)

        # for i in range(len(probs)):
        #     if probs[i] < 1 and probs[i] > 0:
        #         pass
        #     else:
        #         if probs
        actions = [0,1]
        sampled_action = np.random.choice(actions, p=probs)
        return sampled_action

    def getActionProbabilities(self, state)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        x = self._sigma * self._theta.T.dot(self.getExpansion(state))
        z = x - np.max(x)
        numerator = np.exp(z)
        denominator = np.sum(numerator)
        softmax = numerator/denominator
        return softmax
    
        # den = np.sum(self.epsilon+np.exp())
        # nums = self.epsilon+np.exp(self._sigma * self._theta.T.dot(self.getExpansion(state)))
        # # print(nums / den)
        # return (nums) / (den)