import logging
from agent import Agent
import gym
import numpy as np
from utils import plot_learning_curve, save_frames_as_gif
from CommonsGame.constants import *
import threading


def main():

    n_episodes = 10050
    save_episodes_as_gifs = [10, 500, 5000, 10000]
    numAgents = 1
    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=10, mapSketch=smallMapV2)#, mapSketch=smallMap)
    gym.logger.setLevel(logging.CRITICAL)
    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=8,
                  eps_end=0.01, input_dims=[21, 21, 3], lr=0.003)
    scores, eps_history = [], []

    for episode in range(1,n_episodes+1):
        frames = []
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


            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('Episode {} Average Score: {:.2f} Epsilon {:.2f}'.format(episode, avg_score, agent.epsilon))
        if episode in save_episodes_as_gifs:
            path = "./Results/single-agent/gifs/"
            filename = f"gif-episode-{episode}.gif"
            t = threading.Thread(target=save_frames_as_gif, name="Saving gif", args=(frames, path, filename))
            t.daemon = True
            t.start()
            print("Started gif saving thread")

            # PLOT LEARNING CURVE
            x = [j + 1 for j in range(episode)]
            path = "./Results/single-agent/learning-curves/"
            filename = f'learning-curve-episode-{episode}.png'
            t2 = threading.Thread(target=plot_learning_curve, name="Saving gif", args=(x, scores, eps_history, path + filename))
            t2.daemon = True
            t2.start()




if __name__ == "__main__":
    main()