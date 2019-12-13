import numpy as np

from mdp.agents import skeleton as base_agent
from mdp.environments import cartpole as cp
from mdp.environments import gridworld as gw
from mdp.environments import skeleton as base_env
from mdp.history import History
from mdp.policies import cont_policy as cont
from mdp.policies import skeleton as base_policy
from mdp.policies import tabular_softmax as tabp
from mdp.policies.f_policy import FBSoftmax
from mdp.policies.fb import FourierBasis

class PolicyEvaluation:
    def __init__(self, p, env):
        self.Pol = p
        self.Env = env
        self.results = []

    def run_policy(self, policy_param, num_episodes):

            world = self.Env()
            policy = self.Pol
            policy.parameters=policy_param

            J_hats = []
            steps = []
            states = []
            actions = []
            rewards = []

            for n in range(num_episodes):
                ret = 0
                step = 0

                states.append([])
                actions.append([])
                rewards.append([])

                while not world.isEnd:
                    states[n].append(world.state)

                    step += 1
                    action = policy.sampleAction(world.state)
                    world.step(action)
                    discounted_reward = world.reward #* world.gamma**step
                    ret += discounted_reward

                    actions[n].append(action)
                    rewards[n].append(discounted_reward)

                J_hats.append(ret)
                self.results.append(ret)
                steps.append(step)
                world.reset()
            
            D = []

            for i in range(num_episodes):
                history = History(states[i], actions[i], rewards[i])
                D.append(history)

            return np.mean(J_hats), D, np.max(J_hats)
