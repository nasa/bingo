"""
A simple example of an animated plot
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def animate_data(list_of_best_indv_values):

    fig, ax = plt.subplots()

    num_generations = len(list_of_best_indv_values)
    x = np.arange(0, len(list_of_best_indv_values[0]))
    y = list_of_best_indv_values
    zero = [0]*len(x)
    polygon = ax.fill_between(x, zero, y[0], color='b', alpha=0.3)
    points, = ax.plot(x, y[0], 'bs')
    points.set_label('Generation :' + str(0))
    legend = ax.legend(loc='upper right', shadow=True)


    def animate(i):
        ax.collections.clear()
        polygon = ax.fill_between(x, zero, y[i], color='b', alpha=0.3)
        points.set_ydata(y[i])  # update the data
        points.set_label('Generation :' + str(i))
        legend = ax.legend(loc='upper right')
        return points, polygon, legend


    # Init only required for blitting to give a clean slate.
    def init():
        points.set_ydata(np.ma.array(x, mask=True))
        return points, polygon, points

    plt.xlabel('Chromosome Value Index', fontsize=15)
    plt.ylabel('Value Magnitude', fontsize=15)
    plt.title("Values of Best Individual in Island", fontsize=15)
    plt.ylim(-0.01, 2.05)
    ax.tick_params(axis='y', labelsize=15)
    ax.tick_params(axis='x', labelsize=15)

    ani = animation.FuncAnimation(fig, animate, num_generations, init_func=init,
                                interval=250, blit=True)
    plt.show()