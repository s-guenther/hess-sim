#!/usr/bin/env python3
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import pchip_interpolate

from hessemssim import (SimSetup, Storage, InputData, FilterController,
                        DeadzoneController, FuzzyController, MPCController,
                        NeuralController, PassthroughFallbackController)


def generate_paper_plots():
    """Generates data for two time series for each of the 5 EMS and calls
    `plot_comparison()`with this data."""
    rng = np.random.default_rng(42)
    ts_list = []
    for _ in range(2):
        xi = np.linspace(0, 1, 10)
        yi = rng.random((10,)) - 0.75
        yi /= np.max(np.abs(yi))
        xx = np.linspace(0, 1, 100)
        yy = pchip_interpolate(xi, yi, xx)
        ts_list.append(yy)

    list_of_setup_lists = [sim_mult_ems_for_one_ts(ts) for ts in ts_list]
    subfiglbls = 'Filter Deadzone Fuzzy MPC Neural'.split()
    title = 'Random Signal'
    kwargs = dict(figsize=(8.9/2.54, 15/2.54))
    fig_axs_tuples = [plot_comparison(slist, subfiglbls, title, **kwargs)
                      for slist in list_of_setup_lists]
    return list_of_setup_lists, fig_axs_tuples


def sim_mult_ems_for_one_ts(ts, stor_para=None, con_para=None):
    if stor_para is None:
        stor_para = (
            # base
            dict(energy=0.8, power=0.3, efficiency=0.93, selfdischarge=1e-5),
            # peak
            dict(energy=0.2, power=0.7, efficiency=0.97, selfdischarge=1e-3)
        )
    if con_para is None:
        con_para = (
            # filter
            dict(fcut=0.7, finit=0, gain=0),
            # deadzone
            dict(threshold_pos=0.3, threshold_neg=-0.3,
                 out_max=0.7, out_min=-0.7, gain=0, base_max=0.3,
                 base_min=-0.3, peak_max=0.7, peak_min=-0.7),
            # fuzzy
            dict(cut=0.3),
            # mpc
            dict(cut=0.3),
            # neural
            dict(cut=0.3),
        )

    base = (Storage(**stor_para[0]))
    peak = (Storage(**stor_para[1]))

    controllers = (
        FilterController(**con_para[0]),
        DeadzoneController(**con_para[1]),
        FuzzyController(**con_para[2]),
        MPCController(**con_para[3]),
        NeuralController(**con_para[4]),
    )
    fb_con = PassthroughFallbackController()

    inputdata = InputData(np.linspace(1/len(ts), 1, len(ts)), ts)

    setups = [SimSetup(inputdata, base, peak, con, fb_con)
              for con in controllers]
    return setups


def plot_comparison(list_of_sim_setups, subfiglabels=None, title=None,
                    **kwargs):
    """Plot function that recieves a list of SimSetup objects. Generates a
    plot with a subfigure for each SimSetup, calls the .plot() method of the
    SimSetup object to plot the power time series into each axes.
    Optionally adds subfigure labels and title."""
    nplots = len(list_of_sim_setups)
    # Generate subfiglabels
    startlabels = [f'({letter})' for letter
                   in 'abcdefghijklmnopqrstuvwxyz'][:nplots]
    if subfiglabels is None:
        subfiglabels = ['' for _ in range(nplots)]

    if len(subfiglabels) != nplots:
        raise ValueError('Number of provided subfiglabels does not match '
                         'number of provided sim setups')

    labels = [f'{letter} {label}' for letter, label
              in zip(startlabels, subfiglabels)]

    # Generate figure and axes objects
    fig, axs = plt.subplots(nplots, 1, sharex='all', **kwargs,
                            gridspec_kw=dict(top=0.96, bottom=0.09, left=0.065,
                                             right=1, hspace=0.06))
    if title is not None:
        fig.suptitle(title, ha='left', va='top', x=0, y=1)

    for ax, setup, label in zip(axs, list_of_sim_setups, labels):
        setup.plot(axs=ax, make_legend=False)
        # Add
        ax.text(0.015, 1.0, label,
                transform=ax.transAxes, va='top', ha='left')
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False,
                       left=False, right=False,
                       labelbottom=False, labelleft=False)
        ax.grid(0)
        ax.set_xlabel('')
        ax.set_xlim([0, 1])
        ax.set_ylim([-1.05, 1.05])
        ax.set_ylabel('Power $p$')
        ax.spines['top'].set_visible(False)    # Remove the top spine
        ax.spines['right'].set_visible(False)  # Remove the right spine
        ax.spines['bottom'].set_visible(False) # Remove the bottom spine
        ax.spines['left'].set_visible(False)   # Remove the left spine
        ax.plot([0, 0], [-1.1, 1.1], 'k', linewidth=1)

    ax.set_xlabel('Time $t$')  # noqa
    ax.legend('Input Base Peak Mismatch'.split(),
              ncol=4, columnspacing=1.2, fontsize='small', loc='lower center',
              bbox_to_anchor=(0.5, -0.58),
              bbox_transform=ax.transAxes)
    # lines = ax.get_lines()  # noqa
    # fig.legend(lines, ['Input Power', 'Base Power', 'Peak Power', 'Mismatch'],
    #            loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=1)
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.1)
    return fig, ax


if __name__ == '__main__':
    SETUPS, FIGAXS = generate_paper_plots()
    dummybreakpoint = True
