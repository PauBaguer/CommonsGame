import logging
import time

from agent import Agent
import gym
import numpy as np
from utils import plot_learning_curve, save_frames_as_gif_big_map, save_observations_as_gif
from CommonsGame.constants import *
import threading


def handle_agent_choose_action(agent, observation):
    if type(observation) == type(None):
        action = -1
    else:
        action = agent.choose_action(observation)
    return action


def handle_agent_learn(agent, observation, observation_, action, reward, done):
    if type(observation_) == type(None):
        return
    if type(observation) == type(None):
        observation = np.copy(agent.state_memory[-1])

    agent.store_transition(observation, action, reward, observation_, done)
    agent.learn()

def do_plots_and_gifs(episode, frames, obs, scores, eps_history):
    # PLOT LEARNING CURVE
    x = [j + 1 for j in range(episode)]
    path = "./Results/multi-agent/learning-curves/"
    filename = f'learning-curve-episode-{episode}.png'
    plot_learning_curve(x, scores, eps_history, path + filename)

    path = "./Results/multi-agent/gifs/"
    filename = f"gif-episode-{episode}.gif"
    path_obs = "./Results/multi-agent/gifs/"
    filename_obs = f"gif-agent0-episode-{episode}.gif"

    t2 = threading.Thread(target=save_observations_as_gif, name="Saving gif", args=(obs, path_obs, filename_obs))
    t2.daemon = True

    t = threading.Thread(target=save_frames_as_gif_big_map, name="Saving gif", args=(frames, t2, path, filename))
    t.daemon = True
    t.start()

    print("Started gif saving threads")

def run_episode(episode, env, agents, numAgents, save_episodes_as_gifs, scores, eps_history):
    frames = []
    obs = []
    score = 0
    done = [False]
    observations, _ = env.reset()
    while not done[0]:
        # Save map render to do gif
        if episode in save_episodes_as_gifs:
            frames.append(env.render(mode="rgb_array"))

        # Each agent chooses its action.
        actions = []
        for ag in range(numAgents):
            action = handle_agent_choose_action(agents[ag], observations[ag])
            actions.append(action)

        # Actions are played, rewards are received.
        observations_, rewards, done, info = env.step(actions)
        score += rewards[0]

        # Save observation of agent 0 to do gif
        if episode in save_episodes_as_gifs:
            obs.append(observations[0])

        # Learn from current action-reward combo
        for ag in range(numAgents):
            handle_agent_learn(agents[ag], observations[ag], observations_[ag], actions[ag], rewards[ag], done[ag])

        # Current observations will be next old observations
        observations = observations_

    # Save scores
    scores.append(score)
    eps_history.append(agents[0].epsilon)

    avg_score = np.mean(scores[-100:])

    print('Episode {} Average Score: {:.2f} Epsilon {:.2f}'.format(episode, avg_score, agents[0].epsilon))
    if episode in save_episodes_as_gifs:
        do_plots_and_gifs(episode, frames, obs, scores, eps_history)

def main():
    # Hyperparameters
    n_episodes = 10050
    save_episodes_as_gifs = [10, 500, 5000, 10000]
    numAgents = 10
    visualRadius = 5

    input_dims = [visualRadius*2+1, visualRadius*2+1, 3]
    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=visualRadius, mapSketch=bigMap)#, mapSketch=smallMap)
    gym.logger.setLevel(logging.CRITICAL)
    agents = [Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=8,
                  eps_end=0.01, input_dims=input_dims, lr=0.003)
              for _ in range(numAgents)]
    scores, eps_history = [], []

    for episode in range(1, n_episodes + 1):
        run_episode(episode, env, agents, numAgents, save_episodes_as_gifs, scores, eps_history)




if __name__ == "__main__":
    main()