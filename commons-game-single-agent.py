from agent import Agent
import gym
import numpy as np
from utils import plot_learning_curve
def main():

    numAgents = 1
    env = gym.make('CommonsGame:CommonsGame-v0', numAgents=numAgents, visualRadius=10)#, mapSketch=smallMap)

    agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=8,
                  eps_end=0.01, input_dims=[21, 21, 3], lr=0.003)
    scores, eps_history = [], []
    n_games = 5000

    for i in range(n_games):
        score = 0
        done = [False]
        observation = env.reset()

        while not done[0]:
            action = agent.choose_action(observation[0])
            observation_, reward, done, info = env.step([action])
            score += reward[0]
            agent.store_transition(observation[0], action, reward[0], observation_[0], done[0])


            agent.learn()
            observation = observation_
        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])

        print('Episode {} Average Score: {:.2f} Epsilon {:.2f}'.format(i, avg_score, agent.epsilon))

    x = [i + 1 for i in range(n_games)]
    filename = 'lunar_lander_test.png'
    plot_learning_curve(x, scores, eps_history, filename)

if __name__ == "__main__":
    main()