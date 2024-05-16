import logging
from agent import Agent
import gym
import numpy as np
from utils import plot_learning_curve, save_frames_as_gif_big_map, save_observations_as_gif
from CommonsGame.constants import *
import threading


def main():
    # Hyperparameters
    n_episodes = 10050
    save_episodes_as_gifs = [10, 500, 5000, 10000]
    numAgents = 1
    visualRadius = 5


    input_dims = [visualRadius*2+1, visualRadius*2+1, 3]
    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=visualRadius, mapSketch=bigMap)#, mapSketch=smallMap)
    gym.logger.setLevel(logging.CRITICAL)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=8,
                  eps_end=0.01, input_dims=input_dims, lr=0.003)
    scores, eps_history = [], []

    for episode in range(1,n_episodes+1):
        frames = []
        obs = []
        score = 0
        done = [False]
        observation, _ = env.reset()

        while not done[0]:
            if episode in save_episodes_as_gifs:

                frames.append(env.render(mode="rgb_array"))

            action = agent.choose_action(observation[0])
            observation_, reward, done, info = env.step([action])
            score += reward[0]
            agent.store_transition(observation[0], action, reward[0], observation_[0], done[0])

            if episode in save_episodes_as_gifs:
                obs.append(observation[0])

            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('Episode {} Average Score: {:.2f} Epsilon {:.2f}'.format(episode, avg_score, agent.epsilon))
        if episode in save_episodes_as_gifs:
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




if __name__ == "__main__":
    main()