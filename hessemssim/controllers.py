"""This module contains all controllers as a Simulation Component. See
simulation.py for more information."""

import numpy as np

from hessems.deadzone import deadzone
from hessems.lowpass import lowpass
from hessems.fuzzy import fuzzy, build_controller_with_serialized_para
from hessems.mpc import mpc

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
        """
        Initialize a DeadzoneController object.
        
        It implements the deazone-based ems that distributes the power
        exclusively to one storage until a threshold value, and the excess to
        the other. 

        Various parameterizations possible that change the layout of the
        controller from base-prioritized, peak-prioritized and combined.
        Further, different feedback algorithms are representable.
        
        All parameters are optional and have default values if not set.
        Defaults are defined in and loaded from the hessems package.
        
        Parameters:
        -----------
        slope_pos : float
            Positive slope of the (saturated) deadzone function
        slope_neg : float
            Negative slope of the (saturated) deadzone function
        out_max : float
            Maximum output of the (saturated) deadzone function
        out_min : float
            Minimum output of the (saturated) deadzone function 
            input as negative value
        threshold_pos : float
            Positive threshold value of the (saturated) deadzone 
            function
        threshold_neg : float
            Positive threshold value of the (saturated) deadzone 
            function, input as negative value
        gain : float
            SOC feedback gain
        window_up : float
            Upper window value of the feedback logic
        window_low : float
            Lower window value of the feedback logic
        base_max : float
            Rated positive power of base storage (charge)
        base_min : float
            Rated negative power of base storage (discharge, input as
            negative value)
        peak_max : float
            Rated positive power of peak storage (charge)
        peak_min : float
            Rated negative power of peak storage (discharge, input as
            negative value)
        """
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
        """Simulates one step of the ems within the simulation framework by
        delegating the calculation to the hessems toolbox"""
        pin = inputvec[1]
        epeak = peakvec[1]
        para = self.props_to_para_dict()
        base, peak, bpre, ppre, feedback, fpre = \
            deadzone(pin, epeak, para)
        return [base, peak], [bpre, ppre, feedback, fpre]

    def get_init(self):
        """Initial state. Everything set to zero as the state does not matter
        as the deadzone-based ems calculation is stateless. But format is
        needed to comply base class signature."""
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
            out_e_l=None, out_e_u=None,
            # others
            epeak_max=1
    ):
        """
        Fuzzy-logic-based Energy Management Strategy (EMS)

        A Fuzzy-logic controller with two inputs (input power and peak energy
        storage) and one output (base power). The other output (peak power) is
        the difference from input to base. Each in and output has 5
        triangular/trapezodial membership functions (trapezodial at edges). The
        support points are defined via `para`.
        The ruleset is fixed, as well as the aggregation method (bounded sum),
        implication method (min), and defuzzification method (center of gravity).

        Default controller is used if `controller` is not provided,
        see `build_controller()` for more information.

        Parameters
        ----------
        *args, **kwargs:
            39 Parameters to define the exact positioning of the triangular
            and trapezodial membership functions. For details, see the fuzzy
            module of the hessems package, espeacially the function
            `build_controller_with_serialized_para()` as well as the global
            module variables

        Returns
        -------
        base
            Power dispatched to base storage.
        peak
            Power dispatched to peak storage.
        """
        # call superclass
        self._paralist = list(self.get_defaults().keys())
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

        # others
        self.epeak_max = epeak_max

        # other defs and executions
        self._paras_updated = True
        self._controller = None
        self.build_controller()
        self.state_names = []

    def __setattr__(self, name, value):
        """Special overload of the dunder-method that additionally checks if
        one of the membership parameters is changed. If this is the case, this
        change is saved in self._paras_updated. Then the fuzzy controller
        object can be rebuild before executing the next computation."""
        if name == '_paralist':
            super().__setattr__(name, value)
            return None
        if name in self._paralist:
            self._paras_updated = True
        super().__setattr__(name, value)

    def build_controller(self):
        """Generates the fuzzy controller object incorporates the logic and
        performs the actual calculation"""
        para = self.props_to_para_dict(exclude=['epeak_max'])
        self._controller = build_controller_with_serialized_para(para)
        self._paras_updated = False

    @property
    def controller(self):
        """Returns the fuzzy controller object"""
        if self._paras_updated:
            self.build_controller()
        return self._controller

    def sim(self, inputvec, _, peakvec, __):
        """Simulates one step by passing the input and internal fuzzy
        controller object to the fuzzy function of the hessems package."""
        pin = inputvec[1]
        epeak = peakvec[1]
        fpeak = epeak/self.epeak_max
        return fuzzy(pin, fpeak, self.controller), []

    def get_init(self):
        """Initial state. Everything set to zero as the state does not matter
        as the deadzone-based ems calculation is stateless. But format is
        needed to comply base class signature."""
        pbase = 0
        ppeak = 0
        internals = []
        return [pbase, ppeak], internals


class MPCController(SimComponent):
    def __init__(self,
            input_data=None, pred_horizon=50, pred_method='full',
            w1=None, w2=None, w3=None, ref=None,
            ebase_max=None, epeak_max=None,
            pbase_max=None, pbase_min=None, ppeak_max=None, ppeak_min=None,
            tau_base=None, tau_peak=None, eta_base=None, eta_peak=None
    ):
        """
        Model-predictive-control-based Energy Management strategy
    
        Model-predictive-control calculation that weights the difference of the
        peak storage energy content to a reference and the utilization of the base
        storage as well as a penalty for the difference between output and target
        trajectory in the objective.
    
        Prediction length is implicitely set by the length of the power_in and
        dtime vectors.
        Wrapper function for the MPCModel class, which instantiates the object,
        calls the build() and solve() method and returns the computed results (with
        the .results_as_tuple() method).
    
        The default values for the optional parameters are defined in and
        loaded from the hessems package. See STD_PARA and STD_PARA_DEFINITION
        within the mpc submodule for further information.
    
        Parameters
        ----------
        input_data : InputData object
            Input Data object of the simulation framework which must be the
            same as the one used for the simulation itself. It can be empty if
            the prediction method `pred_method` is 'naive'.
        pred_horizon : float, default: 50
            Prediction horizon in simulation steps.
        pred_method : string, default: 'full'
            Can be 'full' or 'naive'.
        w1 : float, optional
            Penalty weight for mismatch of output to reference trajectory
        w2 : float, optional
            Weight for difference of peak energy to reference energy
        w3 :  float, optional
            Weight for base storage usage (power base)
        ref :  float, optional
            Reference energy for peak storage
         Fixed parameters
        pbase_max : float
            Maximum base storage power (charge)
        pbase_min :  float
            Minimum base storage power (discharge, enter neg. value)
        ppeak_max :  float
            Maximum peak storage power (charge)
        ppeak_min :  float
            Minimum peak storage power (discharge, enter neg. value)
        ebase_max :  float
            Maximum energy content of base storage
        epeak_max : float
            Maximum energy content of peak storage
        tau_base : float
            Self-discharge rate of base storage
        tau_peak : float
            Self-discharge rate of peak storage
        eta_base : float
            Efficiency of base storage
        eta_peak : float
            Efficiency of peak storage
        """
        # call superclass
        super().__init__()
        # write args into object
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.ref = ref
        self.ebase_max = ebase_max
        self.epeak_max = epeak_max
        self.pbase_max = pbase_max
        self.pbase_min = pbase_min
        self.ppeak_max = ppeak_max
        self.ppeak_min = ppeak_min
        self.tau_base = tau_base
        self.tau_peak = tau_peak
        self.eta_base = eta_base
        self.eta_peak = eta_peak
        # write args into object that are not used by the mpc-based ems of
        # hessems but used by the controller object to correctly call the inner
        # ems 
        self.input_data = input_data
        self.pred_horizon = pred_horizon
        self.pred_method = pred_method
        # define states (none, but var needed in framework)
        self.state_names = []
        # Choose prediction strategy
        if pred_method == 'full':
            self._predict = self._predict_full
        elif pred_method == 'naive':
            self._predict = self._predict_naive
        else:
            msg = (f'Unknown Prediction method pred_method={pred_method}. '
                   f'Must be "full" or "naive".')
            raise ValueError(msg)


    def sim(self, inputvec, basevec, peakvec, _):
        """Simulates one step of the ems within the simulation framework by
        delegating the calculation to the hessems toolbox"""
        # construct prediction
        pin, dt = self._predict(inputvec)
        # get parameters
        para = self.props_to_para_dict(
            exclude=['input_data', 'pred_horizon', 'pred_method']
        )
        # get state variables
        eb = basevec[1]
        ep = peakvec[1]
        # pass to mpc-based ems
        base, peak, *_ = mpc(pin, dt, eb, ep, para)
        return [base, peak], []

    def get_init(self):
        """Initial state. Everything set to zero as the state does not matter
        as the deadzone-based ems calculation is stateless. The mpc controller
        needs the state of the storages, but there are no controller internal
        states. Format is needed to comply base class signature."""
        return [0, 0], []

    def _predict_naive(self, inputvec):
        """Get the input power vector for the ems computation based on a naive
        forcast (simply repeats the current value for p prediction steps)"""
        pin = np.ones(self.pred_horizon)*inputvec[1]
        dt = np.ones(self.pred_horizon)*inputvec[2]
        return pin, dt

    def _predict_full(self, inputvec):
        """Get the input power vector for the ems computation based on a full
        or perfect prediction by loading the adequate future data from the
        InputData object for p prediction steps"""
        # find time from inputvec in self.input_data, construct prediction
        tvec = self.input_data.time
        t = inputvec[0]
        ind = np.where(tvec == t)[0][0]
        pin = self.input_data.val[ind:ind+self.pred_horizon]
        dt= self.input_data.dt[ind:ind+self.pred_horizon]
        return pin, dt

    def _predict(self, inputvec):
        """Will be overwritten in __init__ with the adequate strategy"""
        return _predict_naive(inputvec)


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
