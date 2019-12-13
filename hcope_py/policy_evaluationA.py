import numpy as np

from mdp.policies.f_policy import FBSoftmax
from mdp.policies.fb import FourierBasis

from mdp.history import History

from mdp.agents import skeleton as base_agent
from mdp.environments import skeleton as base_env
from mdp.environments import gridworld as gw
from mdp.environments import cartpole as cp
from mdp.policies import skeleton as base_policy
from mdp.policies import tabular_softmax as tabp
from mdp.policies import cont_policy as cont

from mdp.policies.fourier_policy import TabularSoftmaxContinuous

class tabp_carp_evaluation:
    def __init__(self, p:TabularSoftmaxContinuous, env:cp.Cartpole):
        self.Pol = p
        self.Env = env
        self.results = []

    def evaluate_policy(self, policy_param, num_episodes):

            world = self.Env()
            policy = self.Pol()

            # logger.info("Setting policy Params...")
            policy.parameters=policy_param

            J_hats = []
            steps = []

            # logger.info("evaluating for "+str(num_episodes)+" episodes")

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
                    # logger.info("Taking "+str(action)+" ends up in state "+str(world.state))
                    discounted_reward = world.reward #* world.gamma**step
                    # logger.info("reward: "+str(discounted_reward))
                    ret += discounted_reward

                    actions[n].append(action)
                    rewards[n].append(discounted_reward)

                # logger.info("Episode "+str(n)+" Terminated w/ return "+str(ret))

                J_hats.append(ret)
                self.results.append(ret)
                steps.append(step)
                world.reset()
            # logger.debug("Avg Step: "+ str(np.mean(steps)))
            
            D = []

            for i in range(num_episodes):
                history = History(states[i], actions[i], rewards[i])
                D.append(history)

            return np.mean(J_hats), D

policy = TabularSoftmaxContinuous(4, 2)
numEpisode = 10

environment = cp.Cartpole

t = tabp_carp_evaluation(policy, environment)

mean_J_hat, D = t.evaluate_policy(policy, numEpisode)

print(D)