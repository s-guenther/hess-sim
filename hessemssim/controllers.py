"""This module contains all controllers as a Simulation Component. See
simulation.py for more information."""

import numpy as np

from hessems.deadzone import deadzone
from hessems.lowpass import lowpass
from hessems.fuzzy import fuzzy, build_controller_with_serialized_para

from hessemssim.simulation import SimComponent


# ##
# ## Simple additional controllers for testing purposes
# ##

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


# ##
# ## Controllers that originate in the HESS-EMS package
# ##

class FilterController(SimComponent):
    def __init__(self, fcut=None, gain=None, ref=None, finit=0):
        """Initialize a FilterController Object. It implements the
        controller strategy 'filter control', i.e. a linear and
        discretized first order low pass filter with a proportional feedback
        loop that compensates deviations of the peak storage energy content
        from a set-point.


        Input Parameters:
            fcut    Filter cutoff frequency of the low pass filter
            gain    Feedback factor/gain of the peak energy feedback loop
            ref    Reference value for the peak energy feedback loop
            finit   Initial integration value ("last step" of first
                    simulation step within the simulation)"""

        # call superclass
        super().__init__()
        # Parameters of the controller
        self.fcut = fcut
        self.gain = gain
        self.ref = ref
        self.finit = finit
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
        pin = inputvec[1]
        dt = inputvec[2]
        last_pfilt = statevec[0]
        epeak = peakvec[1]

        para = self.props_to_para_dict(subset=['gain', 'ref', 'fcut'])
        base, peak, filt, feedback = lowpass(pin, dt, last_pfilt, epeak, para)

        return [base, peak], [filt, feedback]

    def get_init(self):
        """The initial output state of pbase and ppeak is not of importance
        and arbitrarily set to zero. This also holds true for the internal
        state 'pfilt' and 'pcomp'. What matters is the initial value of
        the filter. Consequently, it is set within the internals tuple. """
        pbase = 0
        ppeak = 0
        internals = [self.finit, 0]
        return [pbase, ppeak], internals


class DeadzoneController(SimComponent):
    def __init__(self, slope_pos=None, slope_neg=None, out_max=None,
                 out_min=None, threshold_pos=None, threshold_neg=None,
                 gain=None, window_up=None, window_low=None, base_max=None,
                 base_min=None, peak_max=None, peak_min=None):
        # call superclass
        super().__init__()
        # write parameters
        self.slope_pos = slope_pos
        self.slope_neg = slope_neg
        self.out_max = out_max
        self.out_min = out_min
        self.threshold_pos = threshold_pos
        self.threshold_neg = threshold_neg
        self.gain = gain
        self.window_up = window_up
        self.window_low = window_low
        self.base_max = base_max
        self.base_min = base_min
        self.peak_max = peak_max
        self.peak_min = peak_min
        # name decoding of the generic "states" output vector of sim
        self.state_names = [
            'base_pre',
            'peak_pre',
            'feedback',
            'feedback_pre'
        ]

    def sim(self, inputvec, _, peakvec, __):
        pin = inputvec[1]
        epeak = peakvec[1]
        para = self.props_to_para_dict()
        base, peak, bpre, ppre, feedback, fpre = \
            deadzone(pin, epeak, para)
        return [base, peak], [bpre, ppre, feedback, fpre]

    def get_init(self):
        pbase = 0
        ppeak = 0
        internals = [0, 0, 0, 0]
        return [pbase, ppeak], internals


class FuzzyController(SimComponent):
    def __init__(
            self,
            # Membership function support points for input1 `pin`
            in1_a_u=None, in1_a_r=None,
            in1_b_l=None, in1_b_u=None, in1_b_r=None,
            in1_c_l=None, in1_c_u=None, in1_c_r=None,
            in1_d_l=None, in1_d_u=None, in1_d_r=None,
            in1_e_l=None, in1_e_u=None,
            # Membership function support points for input2 `epeak`
            in2_a_u=None, in2_a_r=None,
            in2_b_l=None, in2_b_u=None, in2_b_r=None,
            in2_c_l=None, in2_c_u=None, in2_c_r=None,
            in2_d_l=None, in2_d_u=None, in2_d_r=None,
            in2_e_l=None, in2_e_u=None,
            # Membership function support points for output `pbase`
            out_a_u=None, out_a_r=None,
            out_b_l=None, out_b_u=None, out_b_r=None,
            out_c_l=None, out_c_u=None, out_c_r=None,
            out_d_l=None, out_d_u=None, out_d_r=None,
            out_e_l=None, out_e_u=None
    ):
        # call superclass
        super().__init__()

        # Membership function support points for input1 `pin`
        self.in1_a_u = in1_a_u; self.in1_a_r = in1_a_r  # noqa
        self.in1_b_l = in1_b_l; self.in1_b_u = in1_b_u; self.in1_b_r = in1_b_r  # noqa
        self.in1_c_l = in1_c_l; self.in1_c_u = in1_c_u; self.in1_c_r = in1_c_r  # noqa
        self.in1_d_l = in1_d_l; self.in1_d_u = in1_d_u; self.in1_d_r = in1_d_r  # noqa
        self.in1_e_l = in1_e_l; self.in1_e_u = in1_e_u  # noqa

        # Membership function support points for input2 `epeak`
        self.in2_a_u = in2_a_u; self.in2_a_r = in2_a_r  # noqa
        self.in2_b_l = in2_b_l; self.in2_b_u = in2_b_u; self.in2_b_r = in2_b_r  # noqa
        self.in2_c_l = in2_c_l; self.in2_c_u = in2_c_u; self.in2_c_r = in2_c_r  # noqa
        self.in2_d_l = in2_d_l; self.in2_d_u = in2_d_u; self.in2_d_r = in2_d_r  # noqa
        self.in2_e_l = in2_e_l; self.in2_e_u = in2_e_u  # noqa

        # Membership function support points for output `pbase`
        self.out_a_u = out_a_u; self.out_a_r = out_a_r  # noqa
        self.out_b_l = out_b_l; self.out_b_u = out_b_u; self.out_b_r = out_b_r  # noqa
        self.out_c_l = out_c_l; self.out_c_u = out_c_u; self.out_c_r = out_c_r  # noqa
        self.out_d_l = out_d_l; self.out_d_u = out_d_u; self.out_d_r = out_d_r  # noqa
        self.out_e_l = out_e_l; self.out_e_u = out_e_u  # noqa

        self._paras_updated = True
        self._paralist = list(self.get_defaults().keys())
        self._controller = None
        self.build_controller()
        self.state_names = []

    def __setattr__(self, name, value):
        if name in self._paralist:
            self._paras_updated = True
        super().__setattr__(name, value)

    def build_controller(self):
        para = self.props_to_para_dict()
        self._controller = build_controller_with_serialized_para(para)
        self._paras_updated = False

    @property
    def controller(self):
        if self._paras_updated:
            self.build_controller()
        return self._controller

    def sim(self, inputvec, _, peakvec, __):
        pin = inputvec[1]
        epeak = peakvec[1]
        return fuzzy(pin, epeak, self.controller), []

    def get_init(self):
        pbase = 0
        ppeak = 0
        internals = []
        return [pbase, ppeak], internals


class MPCController(ProportionalController):
    pass


# ##
# ## Fallback Controller
# ##

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
