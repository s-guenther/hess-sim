"""This module contains all storages as a Simulation Component. See
simulation.py for more information."""

import numpy as np

from hessemssim.simulation import SimComponent


# Storage class, derived from SimComponent
class UnifiedStorage(SimComponent):
    def __init__(self, energy=1, power=(-1, 1),
                 efficiency=(0.95, 0.95), selfdischarge=1e-5, init=1):
        """Initialize a Storage Object. It determines the power the
        storage unit will receive considering power and energy limits.
        Accordingly, it will calculate the new loading degree.

        Input Parameters:
            energy         Energy capacity of the storage, default 1
            power          2-tuple with the maximum discharging and
                           charging power, default 1
            efficiency     2-tuple with the discharging efficiency and the
                           charging efficiency of the storage, default 0.95
            selfdischarge  self-discharge rate in x per unit time, default 1e-3
                           e.g. sd value = 1e-5, time unit: seconds,
                           storage looses 0.01 per mil of is current energy
                           per second
            init           initial energy as state of energy, i.e. 1 equals
                           100% charged"""

        # call superclass
        super().__init__()
        # Parameters of the storage
        self._efficiency = None
        self.efficiency = efficiency
        self._power = None
        self.power = power
        self.energy = energy
        self.selfdischarge = selfdischarge
        self.init = init
        # name decoding of the generic "state" output vector of sim
        self.state_names = ['energy', 'soc', 'power_diff', 'error']

    @property
    def power(self):
        return self._power

    @power.setter
    def power(self, value):
        if np.size(value) == 1:
            value = [-value, value]
        # A few plausibility checks
        if value[0] > 0 or value[1] < 0:
            msg = 'Discharge power must be negative, charge power positive'
            raise ValueError(msg)
        self._power = value

    @property
    def specific_power(self):
        """Returns the larger specific power (charge or discharge)"""
        return np.max([p/self.energy for p in self.power])

    @property
    def efficiency(self):
        return self._efficiency

    @efficiency.setter
    def efficiency(self, value):
        if np.size(value) == 1:
            value = [value, value]
        self._efficiency = value

    def sim(self, inputvec, fallbackval, internalsvec):
        """Checks if the peak and base storage can handle the power that the
        Controller intends. Will set a new power partition if necessary.
        internalsvec contains the following encoding:
        [energyold,    # last energy value/integrator value
         soc           # last soc value
         power_diff,   # last deviation reference/actual
         error]        # 0 - everything fine,
                       # -1 - power bound hit
                       # -2 - soc bound hit
                       # -3 - both"""

        # get relevant variables from input vectors
        dt = inputvec[2]
        energyold = internalsvec[0]

        p_in = fallbackval
        power_error = 0

        # set power to power boundaries if necessary
        if p_in > self.power[1]:
            power_error = 1
            p_in = self.power[1]
        elif p_in < self.power[0]:
            power_error = -1
            p_in = self.power[0]
        else:
            # No bounds hit
            pass

        #  calculate new energy of the storage
        sdpower = -self.selfdischarge*energyold
        if p_in > 0:
            energynew = energyold + sdpower*dt + p_in*self.efficiency[0]*dt
        else:
            energynew = energyold + sdpower*dt + p_in/self.efficiency[1]*dt

        # check whether energy boundaries are violated
        if energynew > self.energy:
            energynew = self.energy
            p_in = (energynew - energyold) / dt / self.efficiency[0] - sdpower
            soc_error = 1
        elif energynew < 0:
            energynew = 0
            p_in = (energynew - energyold) / dt * self.efficiency[1] + sdpower
            soc_error = -1
        else:
            soc_error = 0

        # write output
        energy = energynew
        soc = energy/self.energy
        internals = [energy, soc, power_error, soc_error]
        return [p_in, energy], internals

    def get_init(self):
        """The initial output state of pbase and ppeak is not of importance
        and arbitrarily set to zero. The initial value for the internal vector
        is empty, since there are no internal values yet."""
        power = 0
        # init energy is encoded as state of energy, i.e. 1 equals 100% charged
        # but in the calculation it must be handled as absolute energy
        internals = [self.init*self.energy, self.init, 0, 0]
        return [power, self.init*self.energy], internals
