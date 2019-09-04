"""Module for matrix product state representation of an open quantum system."""
import abc
import numpy as np

import tncontract
import qutip

import low_dim_sys
from typing import List, Tuple


class OpenQuantumSystemMps(metaclass=abc.ABCMeta):
    """Defines the matrix product state representation of an open system MPS.

    At the `kth` time-step, the MPS has `k` sites corresponding to the
    waveguide, one bin corresponding to the system.
    """
    def __init__(self,
                 sys_init_state: qutip.qobj.Qobj,
                 dim: int,
                 num_init_wg_sites: int,
                 thresh: float = 1e-6) -> None:
        """Initializes the chiral waveguide object.

        Args:
            sys_init_state: The initial state of the system.
            dim: The dimensionality of the hilbert space of each waveguide bin.
            num_int_wg_sites: The number of waveguide sites to have in the MPS
                initially.
            thresh: The threshold to use for SVD during the MPS update.
        """
        # Initialize the MPS state. The MPS state is initialized with one site
        # in the waveguide (in its vacuum state), and one site corresponding to
        # the system.
        self._state = tncontract.qutip_conv.qobjlist_to_mps(
                [sys_init_state] +
                [qutip.basis(dim, 0) for _ in range(num_init_wg_sites)])
        self._dim = dim
        self._thresh = thresh

    @property
    def dim(self) -> int:
        """Returns the dimensions of each site in the wavguide."""
        return self._dim

    @abc.abstractmethod
    def _get_propagator(
            self, tstep: int) -> Tuple[tncontract.Tensor, List[float]]:
        """Function to get the propagator at the current time-step.

        Args:
            tstep: The index of the current time step, starting from 0.

        Returns:
            The propagator as a `tncontract.Tensor`.
        """
        raise NotImplementedError()

    def _apply_propagator(self,
                          prop: tncontract.Tensor,
                          delays: List[float]) -> None:
        """Apply the propagator to the matrix product state.

        The propagator is applied on the system, together with time-bins as
        specified by the `delays`.

        Args:
            prop: The propagator, specified as a tensor object. The tensor
                should be defined in the following hilbert space:
                        H_s x H_b1 x H_b2 .... H_bk
                where `H_s` is the hilbert space of the system, `H_bi` are the
                hilbert spaces of the bins. Note that the bins `b1, b2 ... bk`
                are the same as those specified by the argument `delays`.
            delays: The delays of the waveguide bins which are being acted on by
                the propagator. Note that the order of the delays should be
                consistent with the order of the operators used while
                constructing the propagator object. Also, the delays are
                expressed in units of the discretization time `dt` (delay of
                1 is equal to a natural unit of `dt`).
        """
        # Sort the delays.
        sorted_delays = np.sort(delays)

        # Swap the bins corresponding to the delays to the front of the system
        # site. This is important since the gate defined by `prop` can be
        # efficiently applied on adjacent sites. Note that we swap the first
        # delay to final index of 1, second delay to the final index of 2 and
        # so on (index of 0 corresponds to the system site).
        final_indices = []
        for k, delay in enumerate(sorted_delays):
            if delay == k:
                # In this case, the bin is already in the right position, and
                # thus no swaps are required. Note that we append `k + 1` to
                # the final index since the site with index 0 is the system
                # site, and consequently the first waveguide bin starts from
                # index 1.
                final_indices.append(k + 1)
            else:
                # We want the bin at `delay` to be at index `k + 1` at the
                # end of all swaps. We start iterating from `delay` and go upto
                # `k + 1`, all the while performing a swap between the current
                # index,, and one larger than the current index. Following this
                # process, we swap the bin at `delay + 1` with the bin at
                # `k + 1`.
                for temp_ind in range(delay, k, -1):
                    # Note that the swap gate applied at index `i` swaps `i` and
                    # `i + 1`.
                    self._state.swap_gate(temp_ind,
                                          threshold=self._thresh)
                final_indices.append(k + 1)

        # Apply the propagator.
        self._state.apply_gate(prop, 0, threshold=self._thresh)

        # Swap the bins back into their original (delayed) positions.
        for k in range(len(delays) - 1, -1, -1):
            swapped_ind = final_indices[k]
            delay = sorted_delays[k]
            if delay != (swapped_ind - 1):
                # Swap back to original position. In this case, we iterate from
                # `swapped_index` to `delay`, and swap the current index and
                # one next to it.
                for temp_ind in range(swapped_ind, delay + 1):
                    self._state.swap_gate(temp_ind, threshold=self._thresh)

    def _expand_mps(self) -> None:
        """Expands the MPS by one bin to prepare for the next time step.

        The expansion has two components - one is to add a bin before the
        system site, and then swap the system site with the added bin. This
        increases the number of waveguide bins by 1, while at the same time
        maintaining the order of the system and waveguide bins.
        """
        # Concatenate a waveguide bin to the system state.
        bin_to_concat = tncontract.qutip_conv.qobjlist_to_mps(
                [qutip.basis(self._dim, 0)])
        self._state.data = np.concatenate(([bin_to_concat.data[0]],
                                           self._state.data), axis=0)

        # Swap the concatenated bin with the system bin to bring back the MPS
        # into the assumed form.
        self._state.swap_gate(0)

    def _compute_system_expectation(self, sys_op: tncontract.Tensor) -> float:
        """Computes the expectation value of a specified system operator.

        Args:
            sys_op: The system operator, specified as a QuTip object.

        Returns:
            The expectation value of the system operator with respect to the
            current system state.
        """
        # Copy the current state. This seems to be needed since computing the
        # expectation seems to change the system state.
        state_copy = self._state.copy()

        # Finally, compute the system operator.
        return state_copy.expval(sys_op, 0)


    def simulate(self,
                 num_tsteps: int,
                 ops_to_compute: List[qutip.qobj.Qobj],
                 num_log_steps: int = 20) -> List[np.ndarray]:
        """Performs a time step.

        Args:
            num_tsteps: The number of time steps to perform.
            ops_to_compute: A list of operators to compute during the performed
                time-steps.
            num_log_steps: Number of steps to log after.

        Returns:
            A list of arrays corresponding to the expectation value of the
            operators computed as a function of time.
        """
        print("Threshold = ", self._thresh)
        # Convert the operators to tensors.
        ops_to_compute_as_tensors = []
        for op in ops_to_compute:
            ops_to_compute_as_tensors.append(
                    tncontract.qutip_conv.qobj_to_tensor(op))

        # Perform the time-step.
        exp_vals = [[] for _ in ops_to_compute_as_tensors]
        for tstep in range(num_tsteps):
            if tstep % num_log_steps == 0 or tstep == num_tsteps - 1:
                print("On time step ", tstep)
            # Calculate the expectation value for the current time-step.
            for op, exp_val in zip(ops_to_compute_as_tensors, exp_vals):
                exp_val.append(self._compute_system_expectation(op).data)

            # Calculate the propagator and delays for the current time-step.
            prop, delays = self._get_propagator(tstep)

            # Applying the propagator.
            self._apply_propagator(prop, delays)

            # Expanding the MPS.
            self._expand_mps()

        return exp_vals

class ChiralWgMarkovianCoupling(OpenQuantumSystemMps):
    """Implements the matrix product state simulation for a Markovian system."""

    def __init__(self,
                 sys_init_state: qutip.qobj.Qobj,
                 dim: int,
                 sys: low_dim_sys.LowDimensionalSystem,
                 dt: float) -> None:
        """Creates a `ChiralWgMarkovianCoupling` object.

        Args:
            sys_init_state: The initial state of the system.
            dim: The dimensions of each waveguide bin.
            sys: The low dimensional system coupling to the waveguide.
            dt: The time-stepping value used for the matrix product state
                representation.
        """
        super(ChiralWgMarkovianCoupling, self).__init__(sys_init_state, dim, 1)
        self._sys = sys
        self._dt = dt

    def _get_propagator(self,
                        tstep: int) -> Tuple[tncontract.Tensor, List[int]]:
        """Computes the propagator for the current time-step.

        Note that the propagator is a tensor

        Args:
            tstep: The time-step at which to apply the propagator.

        Returns:
            The propagator corresponding to the current time-step.
        """
        # Operator for the waveguide bin.
        dB = qutip.destroy(self._dim)
        # Setup the system hamiltonian and the coupling operator.
        sys_hamil = qutip.tensor(
                self._sys.get_system_hamiltonian(tstep * self._dt),
                qutip.qeye(self._dim))
        coup_op = self._sys.get_coupling_operator(tstep * self._dt)
        # Get the interaction hamiltonian.
        int_hamil = (qutip.tensor(coup_op, dB.dag()) +
                     qutip.tensor(coup_op.dag(), dB)) / np.sqrt(self._dt)
        # Compute the propagator.
        prop = tncontract.qutip_conv.qobj_to_tensor(
                (-1.0j * (sys_hamil + int_hamil) * self._dt).expm())
        return prop, [0]
