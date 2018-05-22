import numpy as np
import matplotlib as mpl
# mpl.use('tkagg')
from matplotlib import pyplot as plt
from matplotlib.ticker import LinearLocator, FuncFormatter
# plt.rcParams["font.family"] = "Times New Roman"

params = {
    'axes.labelsize': 10,
    'font.size': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'font.family': 'Times New Roman',
    'figure.figsize': [4, 4],
    # 'lines.markeredgewidth': 0,
    'lines.linestyle': '--',
    'lines.linewidth': 0.75,
}
mpl.rcParams.update(params)


def nosigfig(x, pos):
    return int(round(x))

TickFormatter = FuncFormatter(nosigfig)


def post_ax_update(ax, kwargs):
    if 'set_ylabel' in kwargs:
        ax.set_ylabel(kwargs['set_ylabel'])
    if 'set_xlabel' in kwargs:
        ax.set_xlabel(kwargs['set_xlabel'])
    if 'title' in kwargs:
        ax.set_title(kwargs['title'])
    if 'legend' in kwargs:
        ax.legend(kwargs['legend'])

    ax.grid(linestyle='-', linewidth=0.5, alpha=0.3)
    if 'no_spines' in kwargs:
        for spine in plt.gca().spines.values():
            spine.set_visible(False)

    if 'numticks' in kwargs:
        ax.xaxis.set_major_locator(LinearLocator(numticks=kwargs['numticks']))
        ax.yaxis.set_major_locator(LinearLocator(numticks=kwargs['numticks']))
        ax.xaxis.set_major_formatter(TickFormatter)
        ax.yaxis.set_major_formatter(TickFormatter)

    if 'mode' in kwargs:
        ax = plot_poly(kwargs['mode'], ax)

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
            rwidth=rwidth,
            histtype='step', cumulative=-1, density=True,
            )

    ax.xaxis.set_major_locator(LinearLocator(numticks=5))
    ax.yaxis.set_major_locator(LinearLocator(numticks=5))
    # ax.set_xlim([0.7, 1.5]) # lin ez
    ax.set_xlim([2.8, 4.2])  # lin hard
    # ax.set_xlim([0.0, 0.05]) # quad
    # ax.set_xlim([0, 60])  # bilin
    ax.set_ylim([0.0, 1.0])

    ax = post_ax_update(ax, kwargs)
    if 'save_fig' in kwargs:
        fig.savefig(kwargs['save_fig'], bbox_inches='tight',
                    dpi=300)
    return fig, ax


def plot_2d_samples(data, fig=None, ax=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    color = kwargs['color'] if 'color' in kwargs else '#2f528f'
    alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
    marker = kwargs['marker'] if 'marker' in kwargs else 'o'
    zorder = kwargs['zorder'] if 'zorder' in kwargs else 1
    ax.scatter(x=data[0], y=data[1],
               c=color,
               alpha=alpha,
               marker=marker,
               zorder=zorder)
    ax = post_ax_update(ax, kwargs)
    if 'save_fig' in kwargs:
        fig.savefig(kwargs['save_fig'], bbox_inches='tight',
                    dpi=300)
    return fig, ax


def plot_lines(data, fig=None, ax=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots(1, 1)
    color = kwargs['color'] if 'color' in kwargs else '#2f528f'
    # alpha = kwargs['alpha'] if 'alpha' in kwargs else 1.0
    # marker = kwargs['marker'] if 'marker' in kwargs else 'o'
    zorder = kwargs['zorder'] if 'zorder' in kwargs else 1
    ax.semilogy(data[0], data[1], marker='s',
                c=color,
                # alpha=alpha,
                # marker=marker,
                zorder=zorder)
    ax = post_ax_update(ax, kwargs)

    if 'save_fig' in kwargs:
        fig.savefig(kwargs['save_fig'], bbox_inches='tight',
                    dpi=300)
    return fig, ax


def plot_poly(mode, ax, **kwargs):
    if mode == 'noisy_box_dist' or mode == 'box_dist':
        x = [9, 16, 16, -1, -1, 9, 9]
        y = [-1, -1, 16, 16, 9, 9, -1]
        ax.plot(x, y, color='0.01', **kwargs)
    else:
        raise NotImplementedError('i dont know the feasible set.')

    return ax


def show():
    plt.show()
