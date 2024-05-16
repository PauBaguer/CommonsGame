import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
import time

def plot_learning_curve(x, scores, epsilons, filename):
    fig = plt.figure(1)
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




"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filename>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    time.sleep(10)

    fig = plt.figure(2, figsize=(frames[0].shape[1] / 32, frames[0].shape[0] / 32), dpi=512)
    fig.suptitle('tick: 0', fontsize=3, fontweight='bold', fontfamily='monospace')
    print(frames[0].shape)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])
        fig.suptitle(f'tick: {i}', fontsize=3, fontweight='bold')

    anim = animation.FuncAnimation(fig, animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)
    print("Saved gif!")