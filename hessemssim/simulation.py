#!/usr/bin/env python3
"""This file incorporates all classes and logic to perform a hybrid storage
simulation with a specified controller (and fallback controller). Namely,
it contains all implementations of specific controllers, and implementations
of the storage. It also incorporates a simulation setup implementation,
which orchestrates the simulation of the components.

The simulation is carried out as a discrete time step simulation and all
component logic is written in this way. I.e. the model is not formulated as
an ODE solved by an ODE solver but the components implement the analytic
solution of the ODE for a step function. I.e. a timeseries is not
interpreted as a continuous signal but as a stair function consisting of
individual steps.

This form of implementation drastically decreases computing time
which is important as the model and its simulation is subject to an (
evolutionary black box) optimisation in the larger scope of the toolbox."""

# ###############################
# #####   IMPORTANT NOTES   #####
# ###############################
#
# The input and output vectors of the simulation components (sim() method)
# are defined as follows:
#
# inputvec = [time, power, dtime]
# controller outvec = [p_base, p_peak], [internals]
# fallback controller outvec: [p_base, p_peak], [internals]
# base storage outvec: [p_base, e_base], [internals]
# peak storage outvec: [p_peak, e_peak], [internals]
#
# The first output argument is always an exactly defined tuple, so the other
# simulation components can rely on the format and definition. The second
# output argument is a generic tuple of arbitrary content and size which the
# component uses by itself to implement dynamic behaviour (retrieve
# information about its state of last step)
#
# To informally describe the structure of the internals output tuple,
# each simulation component specifies the property self.state_names (
# e.g. see FilterController). The main intention of this property is
# documentation, but it can also be used for e.g. generic automated plotting
# functions
#
# The output vecs are the input vecs for the adequate other simulation
# components. A simulation component can choose which individual elements of
# these vectors are needed, or may choose to omit a vector in the
# calculation altogether, but nevertheless the method signature must
# incorporate this value (e.g. the FilterController.sim() method does not need
# basevec input as an argument, but still retains the order of positional
# arguments list by explicitly leaving out the basevec argument with "_")
#
# More specifically, the simulation components must implement the following
# function signature for the sim method:
# controller:
#   [pbase, ppeak], [internals] = sim(inputvec, basevec, peakvec, internalsvec)
# fallback controller:
#   [pbase, ppeak], [internals] = sim(inputvec, convec, basevec, peakvec, \
#                                     internalsvec)
# base storage
#   [power, energy], [internals] = sim(inputvec, fbconvec[0], internalsvec)
# peak storage
#   [power, energy], [internals] = sim(inputvec, fbconvec[1], internalsvec)
#
# The simulation setup gathers all outvecs (and internal outvecs) of all
# simulation components for postprocessing
#
# At last, the simulation components all implement the method get_init(). As
# the components (may) implement dynamic behaviour, they need an initial
# condition at the start of the simulation. This initial condition is
# constructed by this method. See FilterController.get_init() as an example
#
# A simulation component must be instantiatable without specifying
# parameters, i.e. The __init__ of
# each simulation component either have no arguments, or each argument must
# provide a default value.
#
# Each simulation component inherits the get_defaults() method from the base
# class without the need to reimplement it.
#
# The simulation is run within a loop which calculates scalar
# equations in each step: For maximum performance, use python built in types
# (int, float) for the .sim() method of derived SimComponent objects. Scalar
# arithmetics are much faster with these compared to numpy. numpy types
# provide advantage for vector arithmetics. Difference in speed is roughly a
# factor of 10

from abc import ABC
from copy import copy
from dataclasses import dataclass
import inspect
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from hessemssim import util
from hessemssim.misc import COLORS


class SimSetup:
    def __init__(self, inputdata, base, peak, controller, fallback,
                 sim_after_init=True):
        """Initialize a SimSetup Object. It takes the different objects and
        builds the entire simulation with them.

        Input Parameters:
            inputdata   all relevant input (not yet clear what is needed)
            base        a storage object for the base storage
            peak        a storage object for the peak storage
            controller  a controller object
            fallback    a fallback controller object
            sim_after_init    if True (default) the simulation is carried
                        out at the end of construction"""

        self.inputdata = inputdata
        self.controller = controller
        self.fallback = fallback
        self.base = base
        self.peak = peak

        # initialize output vars
        self._data = None
        self._results = None

        self.is_simulated = False

        if sim_after_init:
            self.simulate()

    @property
    def data(self):
        """will organise all available data into a single class"""
        if self._data is None:
            warn('No data available, run simulation first, returning None')
        return self._data

    @property
    def results(self):
        """Will calculate several output criteria with the given data"""
        if self._results is None:
            warn('No data available, run simulation first, returning None')
        return self._results

    def simulate(self):
        """Simulates the complete setup from start to end"""
        # ### Variable naming conventions within this method:
        # # The following variables store the complete simulation data over
        # # the whole timeframe
        # con           controller output array
        # con_state     controller state output array
        # fallb         fallback controller output array
        # fallb_state   fallback controller state output array
        # base          base storage output array
        # base_state    base storage state output array
        # peak          peak storage output array
        # peak_state    peak storage state output array
        # # The following variables ar used within the loop to store the
        # # results of one individual simulation step:
        # c     controller output vector, one step
        # cs    controller state output vector, one step
        # f     fallback controller output vector, one step
        # fs    fallback controller state output vector, one step
        # b     base storage output vector, one step
        # bs    base storage state output vector, one step
        # p     peak storage output vector, one step
        # ps    peak storage state output vector, one step

        # initialize components with first step
        c, cs = self.controller.get_init()
        f, fs = self.fallback.get_init()
        b, bs = self.base.get_init()
        p, ps = self.peak.get_init()

        # allocate outputs:
        nsteps = len(self.inputdata) + 1

        def init_zeros(col_shape, rows=nsteps):
            return np.zeros((rows, len(col_shape)))

        con, con_state = init_zeros(c), init_zeros(cs)
        fallb, fallb_state = init_zeros(f), init_zeros(fs)
        base, base_state = init_zeros(b), init_zeros(bs)
        peak, peak_state = init_zeros(p), init_zeros(ps)

        # write init to output arrays
        con[0, :], con_state[0, :] = c, cs
        fallb[0, :], fallb_state[0, :] = f, fs
        base[0, :], base_state[0, :] = b, bs
        peak[0, :], peak_state[0, :] = p, ps

        # loop through .sim() method and write to output arrays
        for ii in np.arange(nsteps):
            if ii == 0:
                # Skip the first step because the initial conditions are
                # already written to the output arrays
                continue
            # current step ii is calculated with the help of the previous
            # input data ii-1
            idata = self.inputdata(ii - 1)
            c, cs, f, fs, b, bs, p, ps = self._sim(idata, cs, fs, b, bs, p, ps)
            # write output (implicit conversion to numpy)
            con[ii, :], con_state[ii, :] = c, cs
            fallb[ii, :], fallb_state[ii, :] = f, fs
            base[ii, :], base_state[ii, :] = b, bs
            peak[ii, :], peak_state[ii, :] = p, ps

        # write output vars to pandas objects (saves to object)
        self._simout_to_pandas(self.inputdata.data, con, con_state, fallb,
                               fallb_state, base, base_state, peak, peak_state)
        # generate integral results (saves to object)
        self._calc_results()

        self.is_simulated = True

    def _sim(self, indata, csl, fsl, bl, bsl, pl, psl):
        """Simulate one step for each component"""
        # # Inputs:
        # indata   input vector [t, p, dt] for one step
        # csl      controller state input vector, last step
        # fsl      fallback controller state input vector, last step
        # bl       base storage input vector, last step
        # bsl      base storage input output vector, last step
        # pl       peak storage input vector, last step
        # psl      peak storage input output vector, last step
        # # Outputs:
        # c        controller output vector, updated
        # cs       controller state output vector, updated step
        # f        fallback controller output vector, updated step
        # fs       fallback controller state output vector, updated step
        # b        base storage output vector, updated step
        # bs       base storage state output vector, updated step
        # p        peak storage output vector, updated step
        # ps       peak storage state output vector, updated step
        c, cs = self.controller.sim(indata, bl, pl, csl)
        f, fs = self.fallback.sim(indata, c, bl, pl, fsl)
        b, bs = self.base.sim(indata, f[0], bsl)
        p, ps = self.peak.sim(indata, f[1], psl)
        return c, cs, f, fs, b, bs, p, ps

    def _simout_to_pandas(self, indata, con, con_state, fallb, fallb_state,
                          base, base_state, peak, peak_state):
        """Stores the simulation data in a pandas object."""
        # Make header - create a 3 leveled MultiHeader,
        # level0: port (i.e. simin, simout, state)
        # level1: component (i.e. input, controller, fallback, base, peak)
        # level2: varnames (i.e. time, p_in, p_base, p_peak, ...)
        level10 = ['input'] * 3
        level11 = ['controller'] * 2 + \
                  ['fallback'] * 2 + \
                  ['base'] * 2 + \
                  ['peak'] * 2
        level12 = ['controller'] * len(self.controller.state_names) + \
                  ['fallback'] * len(self.fallback.state_names) + \
                  ['base'] * len(self.base.state_names) + \
                  ['peak'] * len(self.peak.state_names)

        level20 = ['time', 'p_in', 'dtime']  # inputdata
        level21 = ['p_base', 'p_peak',  # controller
                   'p_base', 'p_peak',  # fallback
                   'p_base', 'e_base',  # base
                   'p_peak', 'e_peak']  # peak
        level22 = self.controller.state_names + self.fallback.state_names + \
            self.base.state_names + self.peak.state_names

        level2 = level20 + level21 + level22
        level1 = level10 + level11 + level12
        level0 = ['simin'] * len(level20) + \
                 ['simout'] * len(level21) + \
                 ['state'] * len(level22)

        header = \
            pd.MultiIndex.from_arrays([level0, level1, level2],
                                      names=['port', 'component', 'variable'])

        # concat simulation data
        simin = self._add_zero_timestep(indata)
        simout = np.hstack([con, fallb, base, peak])
        state = np.hstack([con_state, fallb_state, base_state, peak_state])

        # create and write pandas array
        self._data = pd.DataFrame(np.hstack([simin, simout, state]),
                                  columns=header)

    @staticmethod
    def _add_zero_timestep(indata):
        """Adds the zero timestep to the input data. This is done for the
        output to be able to plot the initial conditions at t=0 correctly."""
        t, p, dt = 0, 0, 0
        return np.vstack([(t, p, dt), indata])

    def _calc_results(self):
        """Will calculate several output criteria with the given data"""
        p_base = self.data['simout']['base']['p_base']
        e_base = self.data['simout']['base']['e_base']
        p_peak = self.data['simout']['peak']['p_peak']
        e_peak = self.data['simout']['peak']['e_peak']
        fb_stepins = self.data['state']['fallback']['active']
        error_base = self.data['state']['base']['error']
        error_peak = self.data['state']['peak']['error']

        p_basepeak = p_base + p_peak

        ecap_base = self.base.energy
        ecap_peak = self.peak.energy
        prated_base = self.base.power
        prated_peak = self.peak.power

        dt = self.data['simin']['input']['dtime']
        signal = self.data['simin']['input']['p_in']
        ecap_single = util.ideal_energy(signal, dt)
        prated_single = util.rated_power_from_signal(signal)
        e_single = np.cumsum(signal*dt) - min(np.cumsum(signal*dt))

        enorm_base = util.enorm(signal, ecap_base, dt)
        pnorm_base = util.pnorm(signal, prated_base)
        enorm_peak = util.enorm(signal, ecap_peak, dt)
        pnorm_peak = util.pnorm(signal, prated_peak)

        # Sim Results
        sim = SimulationResults(
            util.total_pnorm_mismatch(signal, p_basepeak, dt),
            util.max_pnorm_mismatch(signal, p_basepeak),
            util.nnorm_p_mismatch(signal, p_basepeak, dt),
            util.norm_stepins(fb_stepins * (fb_stepins > 0), dt),
            util.norm_stepins(fb_stepins * (fb_stepins < 0), dt),
            util.norm_stepins(error_base, dt),
            util.norm_stepins(error_peak, dt),
            util.total_norm_dim(enorm_base, enorm_peak,
                                pnorm_base, pnorm_peak))

        # Base Results
        base = StorageResults(
            util.cycles_from_power(p_base, dt, ecap_base),
            util.stress_discrete(p_base, np.max(np.abs(prated_base))),
            util.high_power(p_base, dt, prated_base),
            util.limit_soc(e_base, dt, ecap_base),
            enorm_base,
            pnorm_base,
            ecap_base,
            prated_base,
            util.specific_power(prated_base, ecap_base))

        # Peak Results
        peak = StorageResults(
            util.cycles_from_power(p_peak, dt, ecap_peak),
            util.stress_discrete(p_peak, np.max(np.abs(prated_peak))),
            util.high_power(p_peak, dt, prated_peak),
            util.limit_soc(e_peak, dt, ecap_peak),
            enorm_peak,
            pnorm_peak,
            ecap_peak,
            prated_peak,
            util.specific_power(prated_peak, ecap_peak))

        # Single Results
        single = StorageResults(
            util.cycles_from_power(signal, dt),
            util.stress_discrete(signal, np.max(np.abs(prated_single))),
            util.high_power(signal, dt),
            util.limit_soc(e_single, dt, ecap_single),
            util.enorm(signal, ecap_single, dt),
            util.pnorm(signal, prated_single),
            ecap_single,
            prated_single,
            util.specific_power(prated_single, ecap_single))

        # Write to output
        self._results = \
            Results(sim, base, peak, single)

    def plot(self, *args):
        """Alias for plot_timeseries"""
        return self.plot_timeseries(*args)

    def plot_timeseries(self, extended='auto'):
        """Plots timeseries data: power and energy of storages as function of
        time as well as some states and error codes which are useful for
        interpretation.
        Input:
            extended   True, False, or 'auto' (default). If True,
                       additional  extended state and controller information
                       will be plotted. If 'auto', extended information will
                       be plotted in case the simulation contains
                       errors/power mismatches
        Output:
            fig        The figure object
            ax         The axes objects within a list (length=2 if
                       extended=False or length=4 if extended=True)
            ."""
        # preparing the data for plotting
        if not self.is_simulated:
            warn('No data available, run simulation first. Skipping plot.')
            return
        if extended not in (True, False, 'auto'):
            raise ValueError("'extended', must True, False or 'auto'")

        time = self.data['simin']['input']['time']
        p_in = self.data['simin']['input']['p_in']
        e_base = self.data['simout']['base']['e_base']
        e_peak = self.data['simout']['peak']['e_peak']
        soc_base = self.data['state']['base']['soc']
        soc_peak = self.data['state']['peak']['soc']
        p_base = self.data['simout']['base']['p_base']
        p_peak = self.data['simout']['peak']['p_peak']

        turquoise = COLORS['turquoise']
        violet = COLORS['violet']
        ochre = COLORS['ochre']

        # create 2 or 4 axes, depending on input 'extended'
        if extended == 'auto':
            # choose True if there are some errors during simulation,
            # choose false if everything went fine
            extended = (self.results.sim.n_mismatch > 0) | \
                       (self.results.sim.successful_fb_stepins > 0) | \
                       (self.results.sim.failed_fb_stepins > 0)
        if extended:
            fig, ax = plt.subplots(4, 1, sharex='all',
                                   height_ratios=(6, 4, 3, 2),
                                   figsize=(10, 8))
        else:
            fig, ax = plt.subplots(2, 1, sharex='all',
                                   height_ratios=(3, 2),
                                   figsize=(10, 5))

        # power plots
        ax[0].step(time, p_in, color='k', linewidth=2)
        ax[0].stackplot(time, [p_base * (p_base > 0), p_peak * (p_peak > 0)],
                        step='pre', colors=(turquoise, violet), alpha=0.5)
        ax[0].fill_between(time, p_in, p_base + p_peak, step='pre',
                           hatch='/', facecolor='none')
        ax[0].stackplot(time, [p_base * (p_base < 0), p_peak * (p_peak < 0)],
                        step='pre', colors=(turquoise, violet), alpha=0.5)
        ax[0].step(time, np.zeros(time.shape), color='k', linewidth=0.5)
        ax[0].grid(1)
        ax[0].legend(['input', 'base', 'peak', 'difference'],
                     loc='center left', bbox_to_anchor=(1.1, 0.5))

        # energy plots
        axt = ax[1].twinx()
        axt.fill_between(time, soc_base, color=turquoise, alpha=0.2,
                         edgecolor=None)
        axt.fill_between(time, soc_peak, color=violet, alpha=0.2,
                         edgecolor=None)
        ax[1].plot(time, e_base + e_peak, color='k', linewidth=2)
        ax[1].plot(time, e_base, color=turquoise, linewidth=2)
        ax[1].plot(time, e_peak, color=violet, linewidth=2)
        ax[1].step(time, np.zeros(time.shape), color='k', linewidth=0.5)
        ax[1].grid(1)
        ax[1].legend(['total', 'base', 'peak'],
                     loc='upper left', bbox_to_anchor=(1.1, 1))
        axt.legend(['soc base', 'soc peak'],
                   loc='lower left', bbox_to_anchor=(1.1, 0))

        ax[0].set_ylabel('power')
        ax[1].set_ylabel('energy')
        axt.set_ylabel('soc')

        if extended:
            d_con_fb_base = self.data['state']['fallback']['base_diff']
            d_con_fb_peak = self.data['state']['fallback']['peak_diff']
            d_con_fb_total = d_con_fb_base + d_con_fb_peak
            d_fb_stor_base = self.data['simout']['fallback']['p_base'] - p_base
            d_fb_stor_peak = self.data['simout']['fallback']['p_peak'] - p_peak
            d_fb_stor_total = d_fb_stor_base + d_fb_stor_peak

            error_base = self.data['state']['base']['error']
            error_peak = self.data['state']['peak']['error']
            error_fb = self.data['state']['fallback']['active']

            # diff plots
            ax[2].step(time, d_con_fb_base, color=turquoise, linewidth=2)
            ax[2].step(time, d_con_fb_peak, color=violet, linewidth=2)
            ax[2].step(time, d_con_fb_total, color='k', linewidth=2)
            ax[2].fill_between(time, d_con_fb_base,
                               d_con_fb_base - d_fb_stor_base,
                               color=turquoise,
                               step='pre', hatch='-', facecolor='none')
            ax[2].fill_between(time, d_fb_stor_peak, color=violet,
                               step='pre', hatch='|', facecolor='none')
            ax[2].fill_between(time, d_fb_stor_total, color=ochre,
                               step='pre', facecolor=ochre, alpha=0.5,
                               edgecolor=ochre, linewidth=2)

            ax[2].legend(['diff con-fb base', 'diff con-fb peak',
                          'diff con-fb total',
                          'diff fb-stor base', 'diff fb-stor peak',
                          'diff fb_stor_total'],
                         loc='center left', bbox_to_anchor=(1.1, 0.5))

            # error code plots
            lfb, lb, lp = 0, -0.1, -0.2
            ax[3].fill_between(time, lfb, lfb + 0.08*(error_fb == 1),
                               color=COLORS['ochre1'], step='pre')
            ax[3].fill_between(time, lfb, lfb + 0.08*(error_fb == 2),
                               color=COLORS['ochre2'], step='pre')
            ax[3].fill_between(time, lfb, lfb + 0.08*(error_fb == 3),
                               color=COLORS['ochre3'], step='pre')
            ax[3].fill_between(time, lfb, lfb + 0.08*(error_fb == -1),
                               color=COLORS['2nd1'], step='pre')
            ax[3].fill_between(time, lfb, lfb + 0.08*(error_fb == -2),
                               color=COLORS['2nd2'], step='pre')
            ax[3].fill_between(time, lfb, lfb + 0.08*(error_fb == -3),
                               color=COLORS['2nd3'], step='pre')

            ax[3].fill_between(time, lb, lb + 0.08*(error_base == -1),
                               color=COLORS['base1'], step='pre')
            ax[3].fill_between(time, lb, lb + 0.08*(error_base == -2),
                               color=COLORS['base2'], step='pre')
            ax[3].fill_between(time, lb, lb + 0.08*(error_base == -3),
                               color=COLORS['base3'], step='pre')

            ax[3].fill_between(time, lp, lp + 0.08*(error_peak == -1),
                               color=COLORS['peak1'], step='pre')
            ax[3].fill_between(time, lp, lp + 0.08*(error_peak == -2),
                               color=COLORS['peak2'], step='pre')
            ax[3].fill_between(time, lp, lp + 0.08*(error_peak == -3),
                               color=COLORS['peak3'], step='pre')

            ax[3].annotate('Fallback Controller', (0, lfb + 1e-2),
                           color=COLORS['ochre5'])
            ax[3].annotate('Base Storage', (0, lb + 1e-2),
                           color=COLORS['base5'])
            ax[3].annotate('Peak Storage', (0, lp + 1e-2),
                           color=COLORS['peak5'])

            ax[3].legend(['success power', 'success soc', 'success both',
                          'fail power', 'fail soc', 'fail both',
                          'power bound', 'soc bound', 'bound both',
                          'power bound', 'soc bound', 'bound both'],
                         loc='upper left', bbox_to_anchor=(0.0, -0.3),
                         ncol=4)

            ax[2].grid(1)
            ax[2].set_ylabel('power error')
            ax[3].grid(1)
            ax[3].set_ylabel('error codes')
            ax[3].set_yticks([])

        ax[-1].set_xlabel('time')
        fig.tight_layout()
        return fig, ax

    def plot_results(self):

        res = self.results

        def to_res_array(sres):
            restup = (sres.cycles, sres.stress, sres.high_power,
                      sres.limit_soc, sres.enorm, sres.pnorm, sres.spec_power)
            return np.array(restup)

        base = to_res_array(res.base)
        peak = to_res_array(res.peak)
        single = to_res_array(res.single)
        sim = [res.sim.n_mismatch, res.sim.successful_fb_stepins,
               res.sim.failed_fb_stepins, res.sim.base_errors,
               res.sim.peak_errors]

        fig, ax = plt.subplots(1, 1, figsize=(10, 3))

        storlabels = ['cycles', 'stress', 'high power', 'limit soc',
                      'norm energy', 'norm power', 'specific power']
        simlabels = ['n_mismatch', 'successful fb stepins',
                     'failed fb stepins', 'base errors', 'peak errors']

        x1 = np.arange(len(storlabels))
        x2 = x1[-1] + 1 + np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        width = 0.35

        # line plots at reference y=1
        ax.plot([x1[0]-0.5, x1[-1]+0.5], [1, 1], color=COLORS['grey1'])
        ax.plot([x2[0]-0.3, x2[-1]+0.3], [1, 1], color=COLORS['grey1'])

        # plot bars for base and peak next to each other and sim separate to
        # the rest
        rects1 = ax.bar(x1 - width/2, base/single, width, label='base',
                        color=COLORS['base'])
        rects2 = ax.bar(x1 + width/2, peak/single, width, label='peak',
                        color=COLORS['peak'])
        rects3 = ax.bar(x2, sim, width, label='sim', color=COLORS['grey3'])

        # add numbers above bars
        ax.bar_label(rects1, labels=["%.2f" % n for n in base], padding=3)
        ax.bar_label(rects2, labels=["%.2f" % n for n in peak], padding=3)
        ax.bar_label(rects3, labels=["%.2f" % n for n in sim], padding=3)

        # add xticks for base/peak and add annotation for sim bars
        ax.set_xticks(np.concatenate([x1, [x2[2]]]),
                      storlabels + ['sim results'])
        ax.annotate('\n'.join(simlabels), (x2[0] - width/2, 1.5))

        ax.set_yticks([])
        ax.legend()
        ax.set_title('Base and Peak integral results normalized by Single | '
                     'Sim Results')
        fig.tight_layout()

        return fig, ax


@dataclass
class StorageResults:
    cycles: float
    stress: float
    high_power: float
    limit_soc: float
    enorm: float
    pnorm: float
    energy: float
    power: float
    spec_power: float


@dataclass
class SimulationResults:
    norm_mismatch: float
    max_mismatch: float
    n_mismatch: float
    successful_fb_stepins: float
    failed_fb_stepins: float
    base_errors: float
    peak_errors: float
    total_norm_dim: float


@dataclass
class Results:
    """Class to store all results"""
    sim: SimulationResults
    base: StorageResults
    peak: StorageResults
    single: StorageResults


# Base class for simulation components
class SimComponent(ABC):
    def __init__(self, *args, **kwargs):
        self.state_names = []

    def sim(self, *args):
        pass

    def get_init(self):
        pass

    @classmethod
    def get_defaults(cls):
        sign = inspect.signature(cls).parameters
        return {key: val.default for key, val in sign.items()}

    def get_current_para_set(self):
        keys = self.get_defaults().keys()
        return {key: getattr(self, key) for key in keys}

    def props_to_para_dict(self, subset=False, full=False):
        fullset = self.get_current_para_set()

        if subset:
            para = {k: fullset[k] for k in subset}  # noqa
        else:
            para = fullset

        if not full:
            for key, val in list(para.items()):
                if val is None:
                    para.pop(key)
        return para


class InputData:
    def __init__(self, time, value, *, strip_zero=True, downsample=False,
                 upsample=False):

        super().__init__()

        if downsample and upsample:
            raise ValueError("'downsample' and 'upsample' cannot both be set")
        if strip_zero:
            time, value = self._strip_zero(copy(time), copy(value))
        if downsample:
            time, value = self._downsample(time, value)
        if upsample:
            time, value = self._upsample(time, value)

        self._validate_input(time, value)

        self.time = np.array(time)
        self.val = np.array(value)
        self.dt = np.diff(np.concatenate([[0], self.time]))
        self.data = np.stack([self.time, self.val, self.dt]).transpose()

        # name decoding of the generic "internals" output vector of sim
        # (this simulation component does not have any internal states)
        self.state_names = []

    @staticmethod
    def _strip_zero(time, value):
        # Test if time starts with zero, if so - delete first value
        if time[0] == 0:
            warn('Stripping first value with timestamp = 0')
            return time[1:], value[1:]
        else:
            # Leave as is otherwise
            return time, value

    @staticmethod
    def _validate_input(time, val):
        """Test if input is monotonous and of same length"""
        is_strictly_monotonous = all(dt > 0 for dt in np.diff(time))
        if not is_strictly_monotonous:
            msg = 'Time Vector is not strictly monotonous'
            raise ValueError(msg)

        is_same_length = len(time) == len(val)
        if not is_same_length:
            msg = 'Time vector has not the same length as value vector'
            raise ValueError(msg)

    @staticmethod
    def _downsample(time, val):
        warn('Downsample option not implemented, doing nothing...')
        return time, val

    @staticmethod
    def _upsample(time, val):
        warn('Upsample option not implemented, doing nothing...')
        return time, val

    def __len__(self):
        return len(self.time)

    def __call__(self, ii):
        """Returns [time, val, dt] for index i"""
        return self.data[ii, :].tolist()
