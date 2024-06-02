#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.patheffects as PathEffects  # noqa
import scipy.io as sio
from scipy import fft
from hessemssim import (SimSetup, Storage, InputData, FilterController,
                        DeadzoneController, FuzzyController, MPCController,
                        PassthroughFallbackController)


ESTSS_PATH = Path(__file__).parent / 'estss64.pkl'


def get_estss_data(path=ESTSS_PATH, which='all'):
    data = pd.read_pickle(path)
    neg = data.iloc[:, :32]
    posneg = data.iloc[:, 32:]
    showcase = data.loc[:, [15, 1_000_009]].iloc[:333, :]
    tup = (data, neg, posneg, showcase)
    ddict = {'all': data, 'neg': neg, 'posneg': posneg, 'showcase': showcase}
    if which is None or which == 'all':
        return data
    elif which == 'neg':
        return neg
    elif which == 'posneg':
        return posneg
    elif which == 'showcase':
        return showcase
    elif which == 'as_tuple':
        return tup
    elif which == 'as_dict':
        return ddict
    else:
        options = 'neg posneg showcase as_tuple as_dict'.split()
        msg = (f'Unknown identifier for `which`, must be in {options}, '
               f'found {which}')
        raise ValueError(msg)


def mat_export(
        path='examples/estss64.mat',
        datasets=get_estss_data(which='as_tuple'),  # noqa
        varnames=('all', 'neg', 'posneg', 'showcase')
):
    data_dict = {k: v.to_numpy() for k, v in zip(varnames, datasets)}
    return sio.savemat(path, data_dict)


def mat_import(path='examples/estss64_hybrid_results.mat'):
    varnames = ('res_all', 'res_neg', 'res_posneg', 'res_showcase')
    header = [
        'hyb_pot', 'rel_pot', 'hyb_skew', 'crosspoint_within', 'overdim',
        'cut', 'ebase', 'pbase', 'epeak', 'ppeak', 'eboth', 'pboth', 'esingle',
        'psingle'
    ]
    indexes = {
        'neg': np.arange(32),
        'posneg': np.arange(32) + 1_000_000,
        'showcase': [15, 1_000_009]
    }
    indexes['all'] = np.concatenate([indexes['neg'], indexes['posneg']])

    data = sio.loadmat(path)
    data_dict = {k[4:]: v for k, v in data.items() if k in varnames}
    for key in data_dict.keys():
        ind = indexes[key]
        data_dict[key] = pd.DataFrame(
            data=data_dict[key], index=ind, columns=header
        )
    return data_dict


TS = get_estss_data(which='as_dict')
HYB = mat_import()


def generate_estss64_plots():
    ts_ids = TS['all'].columns
    sim_args = [_sim_args_general(ts_id) for ts_id in ts_ids]
    list_of_setup_lists = [sim_mult_ems_for_one_ts(*args) for args in sim_args]
    titles = [f'estss{ts_id}' for ts_id in ts_ids]
    kwargs = dict(figsize=(20, 11.25))
    subfiglbls = 'Filter Deadzone Fuzzy MPC Neural'.split()
    plt.rcParams['path.simplify'] = False
    plt.rcParams['path.simplify_threshold'] = 1.0
    fig_axs_tuples = [plot_comparison(slist, subfiglbls, title, **kwargs)
                      for slist, title
                      in zip(list_of_setup_lists, titles)]
    # for (fig, ax), title in zip(fig_axs_tuples, titles):
    #     fig.savefig(f'{title}.png', format='png', dpi=300)
    return list_of_setup_lists, fig_axs_tuples


def generate_paper_plots():
    """Generates data for two time series for each of the 5 EMS and calls
    `plot_comparison()`with this data."""

    sim_args = [_sim_args_for_showcase_15(), _sim_args_for_showcase_09()]
    list_of_setup_lists = [sim_mult_ems_for_one_ts(*args) for args in sim_args]
    # subfiglbls = ['Filter-based EMS',
    #               'Deadzone-based EMS',
    #               'Fuzzy-logic-based EMS',
    #               'Model-predictive-control-based EMS']
    subfiglbls = ['', '', '', '']
    kwargs = dict(figsize=(8.9/2.54, 11.5/2.54))
    fig_axs_tuples = [plot_comparison(slist, subfiglbls, **kwargs)
                      for slist
                      in list_of_setup_lists]
    return list_of_setup_lists, fig_axs_tuples


def _sim_args_for_showcase_15():
    ts, stor_para, con_para = _sim_args_general(15, 'showcase')

    con_para[0]['finit'] = -0.35
    con_para[0]['fcut'] = 0.8

    con_para[1]['gain'] = 0.2

    return ts, stor_para, con_para


def _sim_args_for_showcase_09():
    ts, stor_para, con_para = _sim_args_general(1_000_009, 'showcase')

    con_para[0]['fcut'] = 3.5
    con_para[0]['finit'] = -0.0
    con_para[0]['gain'] = 0

    con_para[1]['gain'] = 0.1

    return ts, stor_para, con_para


def _sim_args_general(ts_id, which='all', overdim=1.1):
    ts = TS[which][ts_id].to_numpy()
    res = HYB[which].loc[ts_id, :]

    stor_para = (
        # base
        dict(
            energy=res['ebase']*overdim,
            power=res['pbase']*overdim,
            efficiency=0.93,
            selfdischarge=0.01
        ),
        # peak
        dict(
            energy=res['epeak']*overdim,
            power=res['ppeak']*overdim,
            efficiency=0.97,
            selfdischarge=0.25
        ),
    )

    con_para = (
        # filter
        dict(
            fcut=_estimate_filter_cutoff(ts, res['ebase']/res['eboth']),
            finit=ts[0]*res['cut'],
            gain=1e-2
        ),
        # deadzone
        dict(
            threshold_pos=res['cut'],
            threshold_neg=-res['cut'],
            out_max=stor_para[1]['power'],
            out_min=-stor_para[1]['power'],
            gain=1e-2,
            base_max=stor_para[0]['power'],
            base_min=-stor_para[0]['power'],
            peak_max=stor_para[1]['power'],
            peak_min=-stor_para[1]['power'],
        ),
        # fuzzy
        dict(epeak_max=stor_para[1]['energy']),
        # mpc
        dict(
            pbase_max=stor_para[0]['power'],
            pbase_min=-stor_para[0]['power'],
            ebase_max=stor_para[0]['energy'],
            ppeak_max=stor_para[1]['power'],
            ppeak_min=-stor_para[1]['power'],
            epeak_max=stor_para[1]['energy'],
            ref=stor_para[1]['energy']/2,
            pred_horizon=50,
            w1=1e5,
            w2=0.1,
            input_data=InputData(np.linspace(1/len(ts), 1, len(ts)), ts)
        )
    )

    return ts, stor_para, con_para


def _estimate_filter_cutoff(ts, ratio, adjust=3):
    f_ts = fft.fft(ts)[1:int(len(ts)/2)]
    power_spec = np.abs(f_ts)  # removed square from official power spectrum
    cum_power = np.cumsum(power_spec)
    total = cum_power[-1]
    ind = np.searchsorted(cum_power, ratio*total)
    estimate = ind/len(cum_power)*adjust
    return estimate


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
                            gridspec_kw=dict(top=1, bottom=0.065,
                                             left=0.047, right=1,
                                             hspace=0.10))
    if title is not None:
        fig.suptitle(title, ha='left', va='top', x=0, y=1)

    for ax, setup, label in zip(axs, list_of_sim_setups, labels):
        setup.plot(axs=ax, make_legend=False)
        # Add
        txt = ax.text(0.01, 1, label,
                      transform=ax.transAxes, va='top', ha='left')
        txt.set_path_effects(
            [PathEffects.withStroke(linewidth=3, foreground='w')]
        )
        ax.tick_params(axis='both', which='both',
                       bottom=False, top=False,
                       left=False, right=False,
                       labelbottom=False, labelleft=False)
        ax.grid(0)
        ax.set_xlabel('')
        ax.set_xlim([0, 1])
        ax.set_ylim([1.05, -1.05])
        ax.set_ylabel('Power $p$', labelpad=-1)
        ax.spines['top'].set_visible(False)     # Remove the top spine
        ax.spines['right'].set_visible(False)   # Remove the right spine
        ax.spines['bottom'].set_visible(False)  # Remove the bottom spine
        ax.spines['left'].set_visible(False)    # Remove the left spine
        ax.plot([0, 0], [-1.1, 1.1], 'k', linewidth=1)

    ax.set_xlabel('Time $t$', labelpad=-10)  # noqa
    ax.legend('Input Base Peak Mismatch'.split(),
              ncol=4, columnspacing=1.2, fontsize='small', loc='lower center',
              bbox_to_anchor=(0.5, -0.33),
              bbox_transform=ax.transAxes)
    # lines = ax.get_lines()  # noqa
    # fig.legend(lines,
    #            ['Input Power', 'Base Power', 'Peak Power', 'Mismatch'],
    #            loc='lower center', bbox_to_anchor=(0.5, 0.1), ncol=1)
    # fig.tight_layout()
    # fig.subplots_adjust(bottom=0.1)
    return fig, ax


if __name__ == '__main__':
    SETUPS, FIGAXS = generate_paper_plots()
    dummybreakpoint = True
