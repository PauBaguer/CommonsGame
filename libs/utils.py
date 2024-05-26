import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from libs.socialmetrics import SocialMetrics
import pandas as pd
import time

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Training Steps", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t-100):(t+1)])

    ax2.scatter(x, running_avg, color="C1")
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    ax2.set_ylabel('Score', color="C1")
    ax2.yaxis.set_label_position('right')
    ax2.tick_params(axis='y', colors="C1")

    fig.savefig(filename)
    print("Plotted learning curve!")

def plot_social_metrics(x, social_metrics_history : [SocialMetrics], filename):
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(6,10))
    utilitarian_eff = [m.utilitarian_eff for m in social_metrics_history]
    equality = [m.equality for m in social_metrics_history]
    sustainability = [m.sustainability for m in social_metrics_history]
    peace = [m.peace for m in social_metrics_history]


    window_size = 100
    for i, m in enumerate([utilitarian_eff, sustainability, equality, peace]):
        df = pd.DataFrame(m, columns=['value'])
        df['rolling_avg'] = df['value'].rolling(window=window_size, min_periods=25).mean()

        ax[i].plot(df.index, df['value'], alpha=0.5)
        ax[i].plot(df.index, df['rolling_avg'])


    ax[0].set_ylabel('Efficiency (U)', fontsize=14)
    ax[1].set_ylabel('Sustainability (S)', fontsize=14)
    ax[2].set_ylabel('Equality (E)', fontsize=14)
    ax[3].set_ylabel('Peacefulness (P)', fontsize=14)

    ax[3].set_xlabel('Episode', fontsize=14)

    ax[0].yaxis.grid(linestyle='--')
    ax[1].yaxis.grid(linestyle='--')
    ax[2].yaxis.grid(linestyle='--')
    ax[3].yaxis.grid(linestyle='--')


    fig.tight_layout()
    fig.savefig(filename)
    print("plotted social metrics!")


"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filename>
"""
def save_frames_as_gif(frames, t2, path='./', filename='gym_animation.gif'):
    fig = plt.figure(8, figsize=(frames[0].shape[1] / 32, frames[0].shape[0] / 32), dpi=512)
    fig.suptitle('tick: 0', fontsize=3, fontweight='bold', fontfamily='monospace')
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        fig.suptitle(f'tick: {i}', fontsize=3, fontweight='bold')

    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("Saved gif!")
    t2.start()


def save_frames_as_gif_big_map(frames, t2, path='./', filename='gym_animation.gif'):
    fig = plt.figure(8, figsize=(frames[0].shape[1] / 64, frames[0].shape[0] / 64), dpi=512)
    fig.suptitle('tick: 0', fontsize=3, fontweight='bold', fontfamily='monospace')
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        fig.suptitle(f'tick: {i}', fontsize=3, fontweight='bold')

    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("Saved gif!")
    t2.start()
def save_observations_as_gif(frames, path='./', filename='gym_animation.gif'):
    fig = plt.figure(3, figsize=(frames[0].shape[1] / 32, frames[0].shape[0] / 32), dpi=512)
    fig.suptitle('0', fontsize=3, fontweight='bold', fontfamily='monospace')
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        if type(frames[i]) == type(None):
            return
        patch.set_data(frames[i])
        fig.suptitle(f'{i}', fontsize=3, fontweight='bold')

    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("Saved gif!")
