import matplotlib as mpl
mpl.use('tkagg')
from matplotlib import pyplot as plt

params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
}
mpl.rcParams.update(params)


def post_ax_update(ax, kwargs):
    if 'set_ylabel' in kwargs:
        ax.set_ylabel(kwargs['set_ylabel'])
    if 'set_xlabel' in kwargs:
        ax.set_xlabel(kwargs['set_xlabel'])

    ax.grid(linestyle='-', linewidth=0.5, alpha=0.3)
    if 'no_spines' in kwargs:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)
    return ax


def plot_hist(data, fig=None, ax=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    bins = kwargs['bins'] if 'bins' in kwargs else 30
    color = kwargs['color'] if 'color' in kwargs else '#2f528f'
    rwidth = kwargs['rwidth'] if 'rwidth' in kwargs else 0.9
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
    ax.hist(data,
            bins=bins,
            color=color,
            alpha=alpha,
            rwidth=rwidth)
    ax = post_ax_update(ax, kwargs)
    if 'save_fig' in kwargs:
        fig.savefig(kwargs['save_fig'], bbox_inches='tight',
                    dpi=300, transparent=True)
    return fig, ax


def plot_2d_samples(data, fig=None, ax=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    color = kwargs['color'] if 'color' in kwargs else '#2f528f'
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
    marker = kwargs['marker'] if 'marker' in kwargs else 'o'
    ax.scatter(x=data[0], y=data[1],
               c=color,
               alpha=alpha,
               marker=marker)
    ax = post_ax_update(ax, kwargs)
    if 'save_fig' in kwargs:
        fig.savefig(kwargs['save_fig'], bbox_inches='tight',
                    dpi=300, transparent=True)
    return fig, ax


def show():
    plt.show()
