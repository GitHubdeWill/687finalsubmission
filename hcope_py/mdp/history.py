import numpy as np

class History(object):
    """class that represents a trajectory of an episode"""
    def __init__(self, states, actions, rewards):
        """
            The constructor will take all states, actions and rewards for each episode (we probably don't care gamma etc.)
            We assume finite action space.
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.episodeLength = len(actions)

    def __str__(self):
        return "History: [return: "+str(np.sum(self.rewards))+"\nstates: ["+str(self.states)+"];\nactions: ["+str(self.actions)+"];\nrewards:["+str(self.rewards)+"]\n"