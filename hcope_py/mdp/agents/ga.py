import numpy as np
from .bbo_agent import BBOAgent

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

class GA(BBOAgent):
    """
    A canonical Genetic Algorithm (GA) for policy search is a black box 
    optimization (BBO) algorithm. 
    
    Parameters
    ----------
    populationSize (int): the number of individuals per generation
    numEpisodes (int): the number of episodes to sample per policy         
    evaluationFunction (function): evaluates a parameterized policy
        input: a parameterized policy theta, numEpisodes
        output: the estimated return of the policy            
    numElite (int): the number of top individuals from the current generation
                    to be copied (unmodified) to the next generation
    initPopulationFunction: initializes the first generation of individuals. 
    """

    def __init__(self, populationSize:int, evaluationFunction:Callable, initPopulationFunction:Callable, numElite:int=1, numEpisodes:int=10, truncationIndex:int=10):
        self._name = "GA"
        self._populationSize = populationSize
        self._evaluationFunction = evaluationFunction
        self._truncationIndex = truncationIndex
        self._numElite = numElite
        self._numEpisodes = numEpisodes
        self._initPopulationFunction = initPopulationFunction

        self._learning_param = 2.5

        # Initial agent properties
        self._initial_population = populationSize
        self._population = self._initPopulationFunction(self._initial_population)

        self._population_0 = np.copy(self._population[0])
        self._best_jhat = self._evaluationFunction(self._population_0, self._numEpisodes)


    @property
    def name(self)->str:
        return self._name
    
    @property
    def parameters(self)->np.ndarray:
        return self._population_0

    def _mutate(self, parent:np.ndarray)->np.ndarray:
        """
        Perform a mutation operation to create a child for the next generation.
        The parent must remain unmodified. 
        
        output:
            child -- a mutated copy of the parent
        """
        return parent + self._learning_param * np.random.normal(0,1,parent.shape[0])

    # Assuming generation control happens outside of the train function
    def train(self)->np.ndarray:
        thetas = []
        Jhats = []

        self._truncationIndex = min(self._populationSize, self._truncationIndex)

        for k in range(self._populationSize):
            theta_k = self._population[k]
            J_hat_k = self._evaluationFunction(theta_k, self._numEpisodes)
            thetas.append(theta_k)
            Jhats.append(J_hat_k)

        # logger.debug("Mean return: "+str(np.mean(Jhats)))
        # Sorted by J
        sorted_theta_J_hat_pairs = sorted(zip(thetas, Jhats), key=lambda x: x[1], reverse=True)


        sorted_thetas = []
        sorted_Jhats = []

        for x, y in sorted_theta_J_hat_pairs:
            sorted_thetas.append(x)
            sorted_Jhats.append(y)

        # sorted_thetas = np.array(sorted_thetas)
        # sorted_Jhats = np.array(sorted_Jhats)

        parents = sorted_thetas[:self._truncationIndex]

        # Parent in next gen
        next_gen = sorted_thetas[:self._numElite]
        
        # Adding Children
        for i in range(self._populationSize - self._numElite):
            parent = parents[np.random.randint(len(parents))]
            next_gen.append(self._mutate(parent))

        # Update best theta
        for k in next_gen:
            J_hat_k = self._evaluationFunction(k, self._numEpisodes)
            if J_hat_k > self._best_jhat:
                self._population_0 = np.copy(k)
                self._best_jhat = J_hat_k
        
        
        self._population = next_gen
        print(self._best_jhat)
        return self._population_0


    def reset(self)->None:
        self._population = self._initPopulationFunction(self._populationSize)

        self._population_0 = np.copy(self._population[0])
        self._best_jhat = self._evaluationFunction(self._population_0, self._numEpisodes)