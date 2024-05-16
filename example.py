import numpy as np
import gym
from CommonsGame.constants import *
import time
numAgents = 1

env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=10, mapSketch=smallMapV2)#, mapSketch=smallMap)
env.reset()
for t in range(10000):
    nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    nObservations, nRewards, nDone, nInfo = env.step(nActions)
    if t%100 == 0:
        print(t)
        env.render()
