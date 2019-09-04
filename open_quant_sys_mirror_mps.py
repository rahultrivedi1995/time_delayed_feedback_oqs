"""Module for simulating low-dimensional system in front of a mirror."""
import abc
import numpy as np

import tncontract
import tncontract.qutip_conv
import qutip

import low_dim_sys
import open_quant_sys_mps as oqs
from typing import List, Tuple


class WgIdealMirror(oqs.OpenQuantumSystemMps):
    """Defines an open quantum system in front of an ideal mirror."""
    def __init__(self,
                 sys_init_state: qutip.qobj.Qobj,
                 fb_delay: float,
                 ph_mirror: float,
                 ph_delay: float,
                 dim: int,
                 sys: low_dim_sys.LowDimensionalSystem,
                 dt: float,
                 thresh: float = 1e-6) -> None:
        """Creates an `OpenQuantumSystemIdealMirror` object.

        Args:
            sys_init_state: The initial state of the system.
            fb_delay: The delay in the feedback path (which is `2x` delay to
                the mirror).
            ph_mirror: The phase provided by the ideal mirror upon reflection.
                This is assumed to be frequency independent.
            ph_delay: The phase accumulated by the field at reference frequency
                on travelling from the low-dimensional system to the mirror and
                back.
            dim: The dimensionality of the waveguide bin.
            sys: Low dimensional system coupling to the waveguide.
            dt: The time-stepping value used for the matrix product state.
            thresh: The threshold to use for SVD during MPS update.
        """
        # Note that since this is a delayed feedback system, at `t = 0`, the
        # bins at `t = 0` and `t = fb_delay` are entangled. Consequently, our
        # initial MPS size has to have bins up till the last delay.
        super(WgIdealMirror, self).__init__(sys_init_state, dim, 
                                            int(fb_delay / dt) + 1, thresh)
        self._sys = sys
        self._dt = dt
        self._fb_delay = fb_delay
        self._ph_mirror = ph_mirror
        self._ph_delay = ph_delay

    def _get_propagator(self,
                        tstep: int) -> Tuple[tncontract.Tensor, List[int]]:
        """Computes the propagator for the current time-step.

        Note that the propagator here is a rank 6 tensor, that acts on the
        the system, the most recent time bin and the time bin delayed by
        twice the delay between the low dimensional system and the mirror.

        Args:
            tstep: The time-step at which to compute the propagator.

        Returns:
            The propagator for the current time-step, as well as the delays of
            the bins at which it acts. Note that for this problem, it acts at
            bins at a distance of 0 and `2 * self._dist_mirror` (in units of
            `self._dt`).
        """
        # Setup the annihilation operator for the waveguide bin.
        dB = qutip.destroy(self._dim)

        # Setup the hamiltonian for the system and the coupling operator.
        sys_hamil = qutip.tensor(
                self._sys.get_system_hamiltonian(tstep * self._dt),
                qutip.qeye(self._dim), qutip.qeye(self._dim))
        coup_op = self._sys.get_coupling_operator(tstep * self._dt)
        # Get the interaction hamiltonian. The interaction hamiltonian has two
        # components, one corresponding to the direct emission, and the
        # second corresponding to interaction between the reflected field and
        # the emitter.
        int_direct = qutip.tensor(coup_op, dB.dag(),
                                  qutip.qeye(self._dim)) / np.sqrt(2 * self._dt)
        int_fb = qutip.tensor(coup_op, qutip.qeye(self._dim),
                              dB.dag()) / np.sqrt(2 * self._dt)
        int_hamil = (
                np.exp(0.5j * self._ph_delay) * int_direct +
                np.exp(-0.5j * self._ph_delay) * int_direct.dag() +
                np.exp(-1.0j * (0.5 * self._ph_delay + self._ph_mirror)) * int_fb +
                np.exp(
                    1.0j * (0.5 * self._ph_delay + self._ph_mirror)) * int_fb.dag())
        # Compute the propagator.
        prop = tncontract.qutip_conv.qobj_to_tensor(
                (-1.0j *  (sys_hamil + int_hamil) * self._dt).expm())
        return prop, [0, int(self._fb_delay / self._dt)]


class WgPartialMirror(oqs.OpenQuantumSystemMps):
    """An open quantum system coupled to a waveguide with a partial mirror.

    A partial mirror between two waveguides can be described by a point
    interaction hamiltonian in between the two waveguides."""

    def __init__(self,
                 sys_init_state: qutip.qobj.Qobj,
                 fb_delay: float,
                 mirror_refl: float,
                 mirror_ph: float,
                 ph_delay: float,
                 dim: int,
                 sys: low_dim_sys.LowDimensionalSystem,
                 dt: float,
                 thresh: float = 1e-6) -> None:
        """Creates a new `WgPartialMirror` object.

        Args:
            sys_init_state: The initial state of the system.
            fb_delay: The delay within the feedback path.
            mirror_refl: The reflection coefficient of the mirror.
            mirror_ph: The phase imparted by the mirror.
            ph_delay: The phase due to the delay. This is dependent on the
                central frequency multiplied by the time delay in the return
                path.
            dim: The dimensionality of the waveguide bins. Note that this
                dimensionality is for the forward and backward propagating
                waves individually.
            sys: The low dimensional system under consideration.
            dt: The discretization used along time.
        """
        # Note that since this is a delayed feedback system, at `t = 0`, the
        # bins at `t = 0` and `t = fb_delay` are entangled. Consequently, our
        # initial MPS size has to have bins up till the last delay.
        super(WgPartialMirror, self).__init__(sys_init_state, dim**2,
                                              int(fb_delay / dt) + 1,
                                              thresh)
        self._fb_delay = fb_delay
        self._coup_const = self._get_coupling_constant(mirror_refl, mirror_ph)
        self._ph_delay = ph_delay
        self._sys = sys
        self._dt = dt
        self._dim_per_dir = dim

    @property
    def coup_const(self) -> complex:
        return self._coup_const

    def _get_coupling_constant(self, mirror_refl, mirror_ph):
        """Calculate the coupling constant between the waveguide modes.

        The presence of the mirror at `x = 0` is equivalent to the following
        hamiltonian:
                H = V [int a(w) dw] [int b^dagger(w) dw] + hc
        This function computes the coupling constant `V` in terms of the mirror
        reflection and phase using the following results:
                |V| = 2 / |r| * (1 + sqrt(1 - |r|^2))
                ph(V) = -pi / 2 - ph(r)

        Args:
            mirror_refl: The reflection coefficient of the mirror. Note that
                this is equivalent to `|r|` described above.
            mirror_ph: The phase of the reflection coefficient of the mirror.
        """
        coup_const_amp = 2 / mirror_refl * (1 + np.sqrt(1 - mirror_refl**2))
        coup_const_ph = -0.5 * np.pi - mirror_ph
        return coup_const_amp * np.exp(1.0j * coup_const_ph)

    def _construct_wg_ops(self) -> Tuple[qutip.qobj.Qobj, qutip.qobj.Qobj]:
        """This function constructs waveguide operators.

        In the MPS implementation of a waveguide with both forward and backward
        propagating modes, each MPS site spans the tensor product of the hilbert
        space of the forward and backward propagating modes. Consequently, the
        hilbert space is of size `dim**2`, and the annihilation operators have
        to be unpacked accordingly.

        Returns:
            The operators corresponding to the forward and backward propagating
            modes.
        """
        # Construct separate operators for the forward and backward propagating
        # modes time-bins.`dR` refers to the operator corresponding to a right
        # propagating (forward) time-bin, and `dL` refers the the operator
        # corresponding to a left propagating (backward) time-bin.
        dR = qutip.tensor(qutip.destroy(self._dim_per_dir),
                          qutip.qeye(self._dim_per_dir))
        dL = qutip.tensor(qutip.qeye(self._dim_per_dir),
                          qutip.destroy(self._dim_per_dir))

        # Flatten the operators. This amounts to changing `dims` of the
        # operators to `[[D**2], [D**2]]` from `[[D, D], [D, D]]` else
        # tncontract will interpret the operator to act on two sites instead
        # of 1.
        dR.dims = [[self._dim], [self._dim]]
        dL.dims = [[self._dim], [self._dim]]
        return dR, dL

    def _get_propagator(self,
                        tstep: int) -> Tuple[tncontract.Tensor, List[int]]:
        """Get the propagator for the current time-step.

        Args:
            tstep: The time-step to compute the propagator for.

        Returns:
            The propagator as well as a list of delays at which to apply the
            propagator.
        """
        # The annihilation operators for the waveguide bins for the forward and
        # backward waveguide modes.
        dR, dL = self._construct_wg_ops()

        # Since we are using MPS in the rotating frame corresponding to the
        # waveguide and mirror hamiltonian, the hamiltonian defining the
        # dynamical map corresponding to the propagator is given by
        #                   H = H_TLS + V
        # where `H_TLS` is the hamiltonian of the TLS (in the rotating frame),
        # and `V` is the interaction hamiltonian between the TLS and the
        # waveguide mode.

        # The hamiltonian due to coupling between the system and the waveguide
        # mode (`V`).
        # Calculating the transmission and reflection coefficient from the
        # mirror.
        coup_mag = np.abs(self._coup_const)
        tran = (1 - coup_mag**2 / 4) / (1 + coup_mag**2 / 4)
        refl = -1j * self._coup_const.conj() / (1 + coup_mag**2 / 4)
        coup_op = self._sys.get_coupling_operator(tstep * self._dt)
        # Construct the waveguide operator `O` such that the interaction
        # hamiltonian between the system and the waveguide is given by
        #           `O * s.dag()  + O.dag() * s`
        # where `s` is the system coupling operator. Note that for the given
        # problem, `O` will be constructed as a tensor product of operators
        # acting on the time bin `t` and `t - tau`, where `t` is the current
        # time and `tau` is the time of the round-trip path from the system to
        # the mirror and back.
        if tstep * self._dt < 0.5 * self._fb_delay:
            wg_op = (dR * np.exp(0.5j * self._ph_delay) +
                     dL * np.exp(-0.5j * self._ph_delay)
                    ) / np.sqrt(2 * self._dt)
            wg_op = qutip.tensor(wg_op, qutip.qeye(self._dim))
        else:
            wg_op_r = (qutip.tensor(tran * dR, qutip.qeye(self._dim)) +
                       qutip.tensor(qutip.qeye(self._dim), refl * dL))
            wg_op_l = qutip.tensor(dL, qutip.qeye(self._dim))
            wg_op = (wg_op_r * np.exp(0.5j * self._ph_delay) +
                     wg_op_l * np.exp(-0.5j * self._ph_delay)
                    ) / np.sqrt(2 * self._dt)
        # Construct the interaction hamiltonian.
        sys_wg_hamil = qutip.tensor(coup_op.dag(), wg_op)
        sys_wg_hamil += sys_wg_hamil.dag()

        # System hamiltonian (`H_TLS`).
        sys_hamil = qutip.tensor(
                self._sys.get_system_hamiltonian(tstep * self._dt),
                qutip.qeye(self._dim), qutip.qeye(self._dim))

        # Calculating the full propagator.
        prop = (-1.0j * self._dt *
                (sys_hamil + sys_wg_hamil)).expm()
        return (tncontract.qutip_conv.qobj_to_tensor(prop),
                np.array([0, int(self._fb_delay / self._dt)]))





