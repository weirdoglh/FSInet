# manipulation
import numpy as np

# plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

__all__ = [
    'visSpk',
    'visCurve',
    'animF',
    'multiPlot',
]

# plot spikes
def visSpk(ts, es, path=None):
    """Visualization of spiking activities in gdf form

    Args:
        ts (array): time points
        es (array): spike events
        path (string, optional): save path of image. Defaults to None.
    """
    plt.figure(figsize=(np.max(ts)/500, np.max(es)/50))
    plt.scatter(ts, es, s=1, marker='.')
    plt.xlabel('Time [ms]')
    plt.ylabel('Neuron')
    plt.tight_layout()

    if path is not None:
        plt.savefig(path + '.png', dpi=500)
        plt.close()
    else:
        plt.show()

def visCurve(xs, ys, x_label, y_label, title, path, labels=None, x_lim=None, y_lim=None):
    """Visualization of curves with detailed setting

    Args:
        xs (arrays): x of curves
        ys (arrays): y of curves
        x_label (string): x unit
        y_label (string): y unit
        title (string): figure title
        path (string): save path
        labels (list, optional): lable of curves. Defaults to None.
        x_lim (list, optional): x limits. Defaults to None.
        y_lim (list, optional): y limits. Defaults to None.
    """
    fig, ax = plt.subplots(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.title.set_text(title)
    if labels is not None:
        for x, y, label in zip(xs, ys, labels):
            ax.plot(x, y, label=label)
        ax.legend()
    else:
        for x, y in zip(xs, ys):
            ax.plot(x, y)
    if x_lim:
        ax.set_xlim(x_lim)
    if y_lim:
        ax.set_ylim(y_lim)

    if path is None:
        plt.show()
    else:
        fig.savefig(path + '.png')
        plt.close()

def animF(data, path=None):
    """Create animation from data frames, each frame is a 2d matrix

    Args:
        data (array): array of 2d matrices
        path (string, optional): save path of video. Defaults to None.
    """
    # generate anim
    fig, ax = plt.subplots(1)
    ims = []
    for state in data:
        ims.append([ax.imshow((state))])
    ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True,)

    # save anim
    if path is not None:
        ani.save(path + ".mp4", writer='ffmpeg')
    plt.close()

def multiPlot(multiEvents, xlim, labels, path=None):
    """
    Visualization of multi measurements in NEST simulator for multiple neurons per type.
    Each neuron type gets a unique color, individual neurons are plotted with transparency,
    and the average trace per type is plotted in bold.

    Args:
        multiEvents (list of lists): [[events for neurons of type1], [events for type2], ...]
        xlim (list): [xmin, xmax]
        labels (list): labels for neuron types
        path (string, optional): save path of figure. Defaults to None.

    Returns:
        tuple: tsArr, vmArr, gexArr, ginArr
    """
    smoothKern = np.ones(50) / 50

    # Prepare arrays
    tsArr, vmArr, gexArr, ginArr = [], [], [], []
    for type_events in multiEvents:
        ts_type, vm_type, gex_type, gin_type = [], [], [], []
        for events in type_events:
            ts_type.append(events['times'])
            vm_type.append(events['V_m'])
            gex_type.append(np.convolve(events['g_ex'], smoothKern, 'same'))
            gin_type.append(np.convolve(events['g_in'], smoothKern, 'same'))
        tsArr.append(ts_type)
        vmArr.append(vm_type)
        gexArr.append(gex_type)
        ginArr.append(gin_type)

    # Colors for each neuron type
    colors = plt.cm.tab10(np.linspace(0, 1, len(labels)))

    def plot_traces(ax, dataArr, ylabel, ratio=False):
        for i, ts_type in enumerate(tsArr):
            color = colors[i]
            # Plot individual neurons with transparency
            for idx, ts in enumerate(ts_type):
                if ratio:
                    grat = np.divide(gexArr[i][idx], ginArr[i][idx], out=np.zeros_like(gexArr[i][idx]), where=ginArr[i][idx] != 0)
                    ax.plot(ts, grat, color=color, alpha=0.3, linewidth=0.8)
                else:
                    ax.plot(ts, dataArr[i][idx], color=color, alpha=0.3, linewidth=0.8)
                # Label only the first neuron of each type
                if idx == 0:
                    ax.plot([], [], color=color, label=labels[i])
            # Plot average trace in bold
            avg_trace = np.mean(np.vstack(dataArr[i]), axis=0)
            ax.plot(tsArr[i][0], avg_trace, color=color, linewidth=2)
        ax.set_xlim(xlim)
        ax.set_xlabel('Time [ms]')
        ax.set_ylabel(ylabel)
        ax.legend()

    fig, axs = plt.subplots(2, 2, figsize=(20, 10))

    # V_m
    axs[0, 0].set_title('Membrane Potential')
    plot_traces(axs[0, 0], vmArr, 'V_m [mV]')

    # g_ex
    axs[0, 1].set_title('Excitatory Conductance')
    plot_traces(axs[0, 1], gexArr, 'g_ex [nS]')

    # g_in
    axs[1, 0].set_title('Inhibitory Conductance')
    plot_traces(axs[1, 0], ginArr, 'g_in [nS]')

    # g_ex/g_in ratio
    axs[1, 1].set_title('g_ex / g_in Ratio')
    plot_traces(axs[1, 1], vmArr, 'Ratio', ratio=True)

    fig.suptitle('Multimeter Recordings')
    plt.subplots_adjust(wspace=0.3, hspace=0.3)

    if path is None:
        plt.show()
    else:
        plt.savefig(path + '.png')
        plt.close()

    return tsArr, vmArr, gexArr, ginArr

