"""This module contains all controllers as a Simulation Component. See
simulation.py for more information."""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from hessemssim.simulation import SimComponent


# Controller classes, derived from SimComponent

class ProportionalController(SimComponent):
    def __init__(self, cut=0.5):
        """Distributes the incoming power to the storages according to
        cut, where cut is the share that is passed to base storage.

        Input Parameters:
            cut     default 0.5, between 0 and 1, share that distributes to
                    the base storage, the difference is passed to peak"""

        # call superclass
        super().__init__()

        self.cut = cut
        self.state_names = []

    def sim(self, inputvec, _, __, ___):
        pin = inputvec[1]
        pbase = pin*self.cut
        ppeak = pin - pbase
        internals = []
        return [pbase, ppeak], internals

    def get_init(self):
        """Does not need any init, returning zero and [] to comply
        signature."""
        return [0, 0], []


class FilterController(SimComponent):
    def __init__(self, fcut=0.5, finit=0, k=1e-1, eref=0.5):
        """Initialize a FilterController Object. It implements the
        controller strategy 'filter control', i.e. a linear and
        discretized first order low pass filter with a proportional feedback
        loop that compensates deviations of the peak storage energy content
        from a set-point.


        Input Parameters:
            fcut    Filter cutoff frequency of the low pass filter
            finit   Initial integration value ("last step" of first
                    simulation step within the simulation)
            k       Feedback factor/gain of the peak energy feedback loop
            eref    Reference value for the peak energy feedback loop"""

        # call superclass
        super().__init__()
        # Parameters of the controller
        self.fcut = fcut
        self.finit = finit
        self.k = k
        self.eref = eref
        # name decoding of the generic "states" output vector of sim
        self.state_names = ['pfilt', 'pcomp']

    def sim(self, inputvec, _, peakvec, statevec):
        """Implements the behaviour of the FilterController, i.e. first order
        low pass filter function in discrete form with proportional feedback.
        See
        https://en.wikipedia.org/wiki/Low-pass_filter#Discrete
        -time_realization
        for more information."""
        # get relevant variables from input vectors
        dt = inputvec[2]
        pin = inputvec[1]
        last_pfilt = statevec[0]
        epeak = peakvec[1]

        # convert filter cutoff fc to alpha
        alpha = 2 * np.pi * dt * self.fcut / (2 * np.pi * dt * self.fcut + 1)
        # compute filter step
        pfilt = alpha * pin + (1 - alpha) * last_pfilt
        # compute pcomp (feedback of energy content of peak)
        pcomp = (epeak - self.eref) * self.k

        # write output
        pbase = pfilt + pcomp
        ppeak = pin - pbase
        states = [pfilt, pcomp]
        return [pbase, ppeak], states

    def get_init(self):
        """The initial output state of pbase and ppeak is not of importance
        and arbitrarily set to zero. This also holds true for the internal
        state 'pfilt' and 'pcomp'. What matters is the initial value of
        the filter. Consequently, it is set within the internals tuple. """
        pbase = 0
        ppeak = 0
        internals = [self.finit, 0]
        return [pbase, ppeak], internals


class RuleBasedController(SimComponent):
    pass


class FuzzyController(SimComponent):
    def __init__(self, pnormnegmem=(-1, -1, -0.5, -0.2),
                 pnormlowmem=(-0.4, -0.2, 0.2, 0.4),
                 pnormposmem=(0.2, 0.5, 1, 1),
                 fpeaklowmem=(0, 0, 0.2, 0.4),
                 fpeakmedmem=(0.2, 0.4, 0.6, 0.8),
                 fpeakhimem=(0.6, 0.8, 1, 1),
                 rpeaklowmem=(-0.5, -0.3, 0.2, 0.3),
                 rpeakmedmem=(0.3, 0.4, 0.6, 0.7),
                 rpeakhimem=(0.7, 0.8, 1, 1)):
        # default values are placeholders
        """Initialize a FuzzyController Object and creates a fuzzy model that
        can be used to determine power partition.

        Input Parameters:
            pnormnegmem      values for the trapezoid membership function for
                             negative normalised power
            pnormlowmem      values for the trapezoid membership function for
                             low normalised power
            pnormposmem      values for the trapezoid membership function for
                             positive normalised power
            fpeaklowmem      values for the trapezoid membership function for a
                             low peak storage loading degree
            fpeakmedmem      values for the trapezoid membership function for a
                             medium peak storage loading degree
            fpeakhimem       values for the trapezoid membership function for a
                             high peak storage loading degree
            rpeaklowmem      values for the trapezoid membership function for a
                             low peak storage power share
            rpeakmedmem      values for the trapezoid membership function for a
                             medium peak storage power share
            rpeakhimem       values for the trapezoid membership function for a
                             high peak storage power share"""

        # call superclass
        super().__init__()
        # Parameters of the controller
        self.pNormNegMem = pnormnegmem
        self.pNormLowMem = pnormlowmem
        self.pNormPosMem = pnormposmem
        self.fPeakLowMem = fpeaklowmem
        self.fPeakMedMem = fpeakmedmem
        self.fPeakHiMem = fpeakhimem
        self.rPeakLowMem = rpeaklowmem
        self.rPeakMedMem = rpeakmedmem
        self.rPeakHiMem = rpeakhimem
        # name decoding of the generic "internals" output vector of sim
        self.internals_names = []

        # New Antecedent/Consequent objects hold universe variables and
        # membership functions
        pnorm = ctrl.Antecedent(np.arange(-1, 1, 0.01), 'pnorm')
        fpeak = ctrl.Antecedent(np.arange(0, 1, 0.01), 'fpeak')
        rpeak = ctrl.Consequent(np.arange(0, 1, 0.01), 'rpeak')

        # building the membership functions
        pnorm['negative'] = fuzz.trapmf(pnorm.universe, pnormnegmem)
        pnorm['low'] = fuzz.trapmf(pnorm.universe, pnormlowmem)
        pnorm['positive'] = fuzz.trapmf(pnorm.universe, pnormposmem)

        fpeak['low'] = fuzz.trapmf(fpeak.universe, fpeaklowmem)
        fpeak['medium'] = fuzz.trapmf(fpeak.universe, fpeakmedmem)
        fpeak['high'] = fuzz.trapmf(fpeak.universe, fpeakhimem)

        rpeak['low'] = fuzz.trapmf(rpeak.universe, rpeaklowmem)
        rpeak['medium'] = fuzz.trapmf(rpeak.universe, rpeakmedmem)
        rpeak['high'] = fuzz.trapmf(rpeak.universe, rpeakhimem)

        # define the rules for the fuzzy relationship between input and output
        # variables
        rule1 = ctrl.Rule(pnorm['negative'] & fpeak['low'], rpeak['low'])
        rule2 = ctrl.Rule(pnorm['negative'] &
                          fpeak['medium'], rpeak['medium'])
        rule3 = ctrl.Rule(pnorm['negative'] & fpeak['high'], rpeak['high'])
        rule4 = ctrl.Rule(pnorm['low'] & fpeak['low'], rpeak['low'])
        rule5 = ctrl.Rule(pnorm['low'] & fpeak['medium'], rpeak['low'])
        rule6 = ctrl.Rule(pnorm['low'] & fpeak['high'], rpeak['low'])
        rule7 = ctrl.Rule(pnorm['positive'] & fpeak['low'], rpeak['high'])
        rule8 = ctrl.Rule(pnorm['positive'] &
                          fpeak['medium'], rpeak['medium'])
        rule9 = ctrl.Rule(pnorm['positive'] & fpeak['high'], rpeak['low'])

        # creating the fuzzy model
        strategy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4,
                                            rule5, rule6, rule7, rule8, rule9])
        self.strategy = ctrl.ControlSystemSimulation(strategy_ctrl)

    def sim(self, inputvec, _, peakvec, internalsvec):
        """Inserts the current power into the fuzzy model to determine the
        distribution to the base and peak storage"""
        # get relevant variables from input vectors
        pin = inputvec[2]
        epeak = peakvec[1]

        # inserts the starting values in the existing fuzzy model

        # define the inputs
        self.strategy.input['P_norm'] = pin
        self.strategy.input['F_peak'] = epeak

        # initiate the fuzzy model
        self.strategy.compute()

        # returning the output
        rpeakout = self.strategy.output['rPeak']
        ppeak = rpeakout * pin
        pbase = pin - ppeak
        # write output
        internals = []
        return [pbase, ppeak], internals

    def get_init(self):
        """The initial output state of pbase and ppeak is not of importance
        and arbitrarily set to zero. The initial value for the internal vector
        is empty, since there are no internal values yet."""
        pbase = 0
        ppeak = 0
        internals = []
        return [pbase, ppeak], internals


class MPCController(SimComponent):
    # def __init__(self):
    #     pass

    def sim(self, inputvec, basevec, peakvec, internalsvec):
        pass

    def get_init(self):
        pass


class NeuralController(SimComponent):
    pass


class SimpleDeadzoneController(SimComponent):
    def __init__(self, cut=(0.5, 0.5), power_max=(-1, 1)):
        """Implements a simple deadzone controller, where power excess to a
        threshold value defined by cut and power_max is passed to the peak
        storage and the rest is handled by the base storage."""
        # expand input if necessary
        super().__init__()
        if np.size(cut) == 1:
            cut = (cut, cut)
        if np.size(power_max) == 1:
            power_max = (-power_max, power_max)

        # A few plausibility checks
        if power_max[0] > 0 or power_max[1] < 0:
            msg = 'Discharge power must be negative, charge power positive'
            raise ValueError(msg)

        self.cut = cut
        self.power_max = power_max
        self.state_names = []

    def sim(self, inputvec, _, __, ___):
        """Performs one simulation step. Power within a threshold is passed to
        base and excess to this threshold to peak."""
        p_in = inputvec[1]
        maxbase = self.power_max[1]*self.cut[1]
        minbase = self.power_max[0]*self.cut[0]
        if p_in >= 0:
            p_base = min(p_in, maxbase)
        else:
            p_base = max(p_in, minbase)
        p_peak = p_in - p_base
        return [p_base, p_peak], []

    def get_init(self):
        """Does not have states. Returns arbitrarily zero for base and peak
        storage and returns empty list for states."""
        return [0, 0], []


# Fallback Controller, derived from SimComponent

class FallbackController(SimComponent):
    def __init__(self, power_base=(float('-inf'), float('inf')),
                 power_peak=(float('-inf'), float('inf')),
                 energy_base=float('inf'),
                 energy_peak=float('inf')):
        """Initialize a FallbackController Object. It ensures that the storages
        will not get charged when they are full or discharged when they are
        empty.
        Default values are inf, so the FallbackController will pass through
        everything without alteration.

        Input Parameters:
            power_base  maximal discharge/charge power of the base storage
                        1x2 vector
            power_peak  maximal discharge/charge power of the peak storage
                        1x2 vector
            energy_base energy capacity of the base storage
            energy_peak energy capacity of the peak storage"""
        if np.size(power_base) == 1:
            power_base = [-power_base, power_base]
        if np.size(power_peak) == 1:
            power_base = [-power_peak, power_peak]
        # call superclass
        super().__init__()
        # Parameters of the controller
        self.power_base = power_base
        self.power_peak = power_peak
        self.energy_base = energy_base
        self.energy_peak = energy_peak
        # name decoding of the generic "internals" output vector of sim
        self.state_names = ['base_diff', 'peak_diff', 'active']

    def sim(self, inputvec, convec, basevec, peakvec, internalsvec):
        """Checks if the peak and base storage can handle the power that the
        Controller intends. Will set a new power partition if necessary. """

        # get relevant variables from input vectors
        pin = inputvec[1]
        pbasecon = convec[0]
        ppeakcon = convec[1]
        ebase = basevec[1]
        epeak = peakvec[1]

        # changing the base or peak storage power if necessary
        if (ebase <= 0) and (pbasecon < 0) and (pin > 0):
            pbase = 0
            ppeak = min(pin, self.power_peak[1])
            wasactive = 1
        elif (ebase <= 0) and (pbasecon < 0) and (pin < 0):
            pbase = 0
            ppeak = max(pin, self.power_peak[0])
            wasactive = 1
        elif (ebase >= self.energy_base) and (pbasecon > 0) and (pin > 0):
            pbase = 0
            ppeak = min(pin, self.power_peak[1])
            wasactive = 1
        elif (ebase >= self.energy_base) and (pbasecon > 0) and (pin < 0):
            pbase = 0
            ppeak = max(pin, self.power_peak[0])
            wasactive = 1
        elif (epeak <= 0) and (ppeakcon < 0) and (pin > 0):
            ppeak = 0
            pbase = min(pin, self.power_base[1])
            wasactive = 1
        elif (epeak <= 0) and (ppeakcon < 0) and (pin < 0):
            ppeak = 0
            pbase = max(pin, self.power_base[0])
            wasactive = 1
        elif (epeak >= self.energy_peak) and (ppeakcon > 0) and (pin > 0):
            ppeak = 0
            pbase = min(pin, self.power_base[1])
            wasactive = 1
        elif (epeak >= self.energy_peak) and (ppeakcon > 0) and (pin < 0):
            ppeak = 0
            pbase = max(pin, self.power_base[0])
            wasactive = 1
        else:
            pbase = pbasecon
            ppeak = ppeakcon
            wasactive = 0

        basediff = pbase - convec[0]
        peakdiff = ppeak - convec[1]

        states = [basediff, peakdiff, wasactive]

        return [pbase, ppeak], states

    def get_init(self):
        """The initial output state of pbase and ppeak is not of importance
        and arbitrarily set to zero. The initial value for the internal vector
        is empty, since there are no internal values yet."""
        pbase = 0
        ppeak = 0
        states = [0, 0, 0]
        return [pbase, ppeak], states


class PassthroughFallbackController(SimComponent):
    def __init__(self, *args):
        """This fallback controller does not perform any checks, it just
        passes the powers of the preceding controller to the storages
        without intervening if any limits are hit."""
        super().__init__(*args)
        self.state_names = ['base_diff', 'peak_diff', 'active']

    def sim(self, _, convec, __, ___, ____):
        """Performs one simulation step. It just passes the input to the
        output without doing anything."""
        states = [0, 0, 0]

        return convec, states

    def get_init(self):
        """The initial output state of pbase and ppeak is not of importance
        and arbitrarily set to zero. The initial value for the internal vector
        is empty, since there are no internal values yet."""
        return [0, 0], [0, 0, 0]
