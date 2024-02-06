#!/user/bin/env python3
"""This module gathers some loose functions that are used at various points
within the toolbox from various modules, functions or classes.

This module contains functions implementing logic, i.e. some kind of
mathematical formula to calculate something specific to the scientific
domain in contrast to the module 'misc' which contains common functions that
convert, parse or process data structures or similar."""

import numpy as np
from scipy import interpolate


# Integral results for results() of SimSetup

def total_pnorm_mismatch(reference, actual, dt=None, tol=1e-12):
    """Sums base and peak power errors and normalises by entire power"""
    if dt is None:
        dt = np.ones(reference.shape)

    ref = reference * dt
    diff = (reference - actual) * dt
    diff[(-tol < diff) * (diff < tol)] = 0
    mismatch = np.sum(np.abs(diff)) / np.sum(np.abs(ref))

    return mismatch


def max_pnorm_mismatch(reference, actual, tol=1e-12):
    """Calculates maximum power mismatch, normalised by maximum power"""
    maxpower = max(abs(reference))
    maxdiff = max(abs(reference - actual))
    maxdiff = 0 if maxdiff < tol else maxdiff
    return maxdiff / maxpower


def nnorm_p_mismatch(reference, actual, dt=None, tol=1e-12):
    """Determines the number of timesteps an error occurred, normalised to
    the length of the data """
    if dt is None:
        dt = np.ones(reference.shape)

    diff = (reference - actual) * dt
    diff[(-tol < diff) * (diff < tol)] = 0
    nonzero_length = np.sum((np.abs(diff) > 0) * dt)
    total_length = np.sum(dt)

    return nonzero_length / total_length


def cycles_from_power(power, dt=None, capacity=None):
    if dt is None:
        dt = np.ones(power.shape)
    if capacity is None:
        capacity = ideal_energy(power, dt)

    through = np.abs(np.sum(power * (power <= 0) * dt))

    return through / capacity


def cycles_from_energy(energy, capacity=None):
    """Determines the number of equivalent cycles the storage went through.
    The result is a good estimate as the information of the first step is
    missing. Will converge for large number of steps"""
    if capacity is None:
        capacity = np.max(energy)

    # reconstruct power from energy (time vector cancels out and is not needed)
    power = np.diff(energy)

    return cycles_from_power(power, capacity=capacity)


def stress(power, dt=None, rated_power=None):
    """Calculates the stress of the base storage, ie how strongly does the
    power fluctuate normalised to the rated power"""
    if dt is None:
        dt = np.ones(power.shape)
    if rated_power is None:
        rated_power = rated_power_from_signal(power)

    t = np.cumsum(dt)
    dp = np.gradient(power, t)

    return np.sum(np.abs(dp) * dt) / rated_power


def stress_discrete(power, rated_power=None):
    if rated_power is None:
        rated_power = rated_power_from_signal(power)
    return np.sum(np.abs(np.diff(power))) / rated_power


def high_power(power, dt=None, rated_power=None, cut=0.8):
    """Percentage of timesteps with a higher power than the cutoff power, which
    is a multiplication of the cutoff value and the maximal power of the
    storage system. Variable rated_power is a vector with the form
    [minimal power, maximal power]. It's possible to enter rated_power as a
    single value if the maximal and minimal power differ only by sign."""
    if dt is None:
        dt = np.ones(power.shape)
    if rated_power is None:
        rated_power = np.array([np.min(power), np.max(power)])

    if np.size(rated_power) == 1:
        rated_power = [-rated_power, rated_power]

    plow = rated_power[0]
    phigh = rated_power[1]
    total_length = np.sum(dt)
    hp_length = np.sum((power > cut * phigh) * dt) + np.sum(
        (power < cut * plow) * dt)

    return hp_length / total_length


def limit_soc(energy, dt=None, capacity=None, cutlow=0.1, cuthigh=0.9,
              upsample=100):
    """Percentage of timesteps where the state of charge is not between
    the cutoff points. This indicates how often the storage is almost full
    or empty, which is an indicator for stress.
    CAUTION: upsampling routine expects that signal starts with t=0 and
    dt(t=0) = 0"""
    if dt is None:
        dt = np.ones(energy.shape)
    if capacity is None:
        capacity = np.max(energy)

    # The SOC is a trapezodial function of time (assuming power is a step
    # function). Treating the SOC as a step function as well will only
    # approximate the limit_soc value. This should be fine and sufficient in
    # most cases, only in cases with a coarse time resolution, e.g. if a
    # storage gets charged from 0 to 1 in a single timestep, it will
    # calculate an inaccurate value. To circumvent this issue, the function
    # will linearily interpolate the energy signal and perform the
    # calculations on the interpolation.
    x = np.arange(len(dt))
    xx = np.linspace(0, x[-1], upsample * (len(x) - 1) + 1)
    dtint = interpolate.interp1d(x, dt, kind='next')
    energyint = interpolate.interp1d(x, energy, kind='linear')
    dt = dtint(xx) / 10
    energy = energyint(xx)

    total_length = np.sum(dt)
    soc = energy / capacity
    lim_length = np.sum((soc < cutlow) * dt + (soc > cuthigh) * dt)

    return lim_length / total_length


def enorm(signal, capacity, dt=None):
    """Calculates the factor by which the storage is oversized (or undersized)
    relative to the necessary storage size (for a single storage) regarding
    the energy capacity"""
    if dt is None:
        dt = np.ones(signal.shape)

    return capacity / ideal_energy(signal, dt)


def pnorm(signal, rated_power):
    """Calculates the factor by which the power bound is oversized (or
    undersized) relative to the maximum power of the input data"""
    if np.size(rated_power) == 1:
        rated_power = [-rated_power, rated_power]

    idealpower = np.max(np.abs(signal))
    pnorm_two_vec = rated_power / idealpower

    return np.max(np.abs(pnorm_two_vec))


def ideal_energy(signal, dt=None):
    """Calculates the energy content an ideal storage needs to fulfil the
    signal."""
    if dt is None:
        dt = np.ones(signal.shape)
    energy = np.cumsum(signal * dt)
    return np.max(energy) - np.min(energy)


def rated_power_from_signal(signal):
    """Determines the maximum absolute power of a signal."""
    return [np.min(signal), np.max(signal)]


def specific_power(rated_power, energy_capacity):
    """Returns the specific discharge power, argument rated power is a 2x1
    vector [discharge, charge]"""
    return -rated_power[0] / energy_capacity


def specific_power_from_signal(signal, dt=None, kind='symmetric'):
    if dt is None:
        dt = np.ones(signal.shape)

    energy = ideal_energy(signal, dt)
    rated_power = rated_power_from_signal(signal)

    if kind == 'symmetric':
        power = np.max(np.abs(rated_power))
    elif kind == 'discharge':
        power = -rated_power[0]
    elif kind == 'charge':
        power = rated_power[1]
    elif kind == 'both':
        power = rated_power
    else:
        raise ValueError(f"'kind' is '{kind}',must be 'symmetric', 'charge', "
                         f"'discharge', or 'both'")

    return power/energy


def norm_stepins(stepins, dt=None):
    """Count the number of nonzero fallback states (this equals timesteps
    where the fallback controller intervened the original controller) and
    normalize by number of time steps."""
    if dt is None:
        dt = np.ones(stepins.shape)

    total_length = np.sum(dt)
    stepins_length = np.sum((stepins != 0) * dt)

    return stepins_length / total_length


def total_norm_dim(enorm_base, enorm_peak, pnorm_base, pnorm_peak):
    """Returns the vector length ( of base and peak storage added) wherre a
    vector is defined as [energy, power] normalizsed by the single storage
    dimensions."""
    eb, ep, pb, pp = enorm_base, enorm_peak, pnorm_base, pnorm_peak
    return np.sqrt((eb + ep) ** 2 + (pb + pp) ** 2)
