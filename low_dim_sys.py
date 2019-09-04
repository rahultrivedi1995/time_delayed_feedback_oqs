"""Defines an interface with a low-dimensional system."""
import abc
import numpy as np

import qutip

from typing import Callable

class LowDimensionalSystem(metaclass=abc.ABCMeta):
    """Defines a low-dimensional system that are basic units in open-systems."""

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """Returns the dimensionality of the system."""
        raise NotImplementedError()

    @abc.abstractmethod
    def get_system_hamiltonian(self, time: float) -> qutip.qobj.Qobj:
        """Returns the hamiltonian corresponding to the system at given time.

        Args:
            time: The time at which to compute the hamiltonian.

        Returns:
            The hamiltonian at the specified time instant as a qutip object.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_coupling_operator(self, time: float) -> qutip.qobj.Qobj:
        """Returns the coupling operator corresponding to the given time.

        Args:
            time: The time-instant at which to compute the coupling operator.

        Returns:
            The coupling operator at the specified time.
        """
        raise NotImplementedError()


class TwoLevelSystem(LowDimensionalSystem):
    """Defines a two-level low dimensional system."""
    def __init__(self,
                 gamma: float,
                 delta: float) -> None:
        """Creates a new `TwoLevelSystem` object.

        Args:
            gamma: The decay rate of the two-level system.
            delta: The detuning between the two-level system and the reference
                frequency.
        """
        # Two-level system parameters.
        self._gamma = gamma
        self._delta = delta

        # Qutip operator for the two-level system.
        self._sigma = qutip.destroy(2)
        self._num_op = self._sigma.dag() * self._sigma

    @property
    def dim(self) -> int:
        """Returns the dimensionality of the system."""
        return 2

    def get_system_hamiltonian(self, time: float) -> qutip.qobj.Qobj:
        """Returns the system hamiltonian for the two-level system.

        Args:
            time: The time at which to get the system hamiltonian.

        Returns:
            The hamiltonian for the two-level system.
        """
        return self._delta * self._num_op

    def get_coupling_operator(self, time: float) -> qutip.qobj.Qobj:
        """Returns the coupling operator between the system and the waveguide.

        Args:
            time: The time at which to get the coupling operator.

        Returns:
            The coupling operator at the specified time.
        """
        return np.sqrt(self._gamma) * self._sigma


class DrivenTwoLevelSystem(LowDimensionalSystem):
    """Defines a two-level system with a laser drive."""
    def __init__(self,
                 gamma: float,
                 delta: float,
                 drive_amp: Callable[[float], float]) -> None:
        """Creates a new `DrivenTwoLevelSystem` object.

        Args:
            gamma: The decay rate of the emitter.
            delta: The detuning of the emitter from the reference frequency.
            drive_amp: The driving amplitude of the laser. Note that this is
                assumed to be the amplitude after removing an exponential
                oscillating at the laser frequency.
        """
        # Parameters for the driven two-level system.
        self._gamma = gamma
        self._delta = delta
        self._drive_amp = drive_amp

        # Commonly used operator for the two-level system.
        self._sigma = qutip.destroy(2)
        self._num_op = self._sigma.dag() * self._sigma
        self._drive_op = self._sigma.dag() + self._sigma

    @property
    def dim(self) -> int:
        """Returns the dimensionality of the system."""
        return 2


    def get_system_hamiltonian(self, time: float) -> qutip.qobj.Qobj:
        """Returns the system hamiltonian for the two-level system.

        Args:
            time: The time at which to get the system hamiltonian.

        Returns:
            The hamiltonian for the two-level system.
        """
        # Current laser amplitude.
        amp = self._drive_amp(time)
        return self._delta * self._num_op + amp * self._drive_op

    def get_coupling_operator(self, time: float) -> qutip.qobj.Qobj:
        """Returns the coupling operator between the sytem and waveguide.

        Args:
            time: The time at which to get the coupling operator.

        Returns:
            The coupling operator at the specified time.
        """
        return np.sqrt(self._gamma) * self._sigma

