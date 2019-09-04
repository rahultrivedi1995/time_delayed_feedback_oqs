"""Script to study the impact of mirror non-ideality on feedback system."""
import h5py
import os
import sys
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import qutip

import fdtd_spontaneous_emission as fdtd
import low_dim_sys
import open_quant_sys_mirror_mps as oqs_mirror_mps


def emitter_decay_sims() -> None:
    """Runs simulations of emitter decay with non ideal MPS."""
    # Some Parameter values for the system being simulated.
    gamma = 1.0 # Decay rate of the emitter.
    delta = 0.0 # Detuning of emitter frequency from reference frequency.
    delay = 4.0 / gamma # Delay between emitter and mirror.
    ph_mirror = 0 # Reflection phases of the mirror.
    refls_mirror = [0.2, 0.4, 0.6, 0.8, 1.0] # reflection coefficients.
    ph_delay = np.pi # Phase corresponding to the delay between emitter to
                     # mirror and back at the reference frequency.

    # Numerical parameters.
    dims = 2 # Dimensionality of each waveguide bin.
    dt = 0.05 / gamma # Time-step. Also used for meshing waveguide in MPS.
    num_tsteps = 1000 # Number of time-steps to perform.
    thresh = 1e-2 # Threshold at which to truncate Schmidth decomposition at
                  # each step of MPS.

    # Open a h5 file for storing simulation results.
    data = h5py.File("results/impact_emitter_decay.h5", "w")
    param_grp = data.create_group("parameters")
    param_grp.create_dataset("gamma", data=gamma)
    param_grp.create_dataset("delta", data=delta)
    param_grp.create_dataset("delay", data=delay)
    param_grp.create_dataset("ph_mirror", data=ph_mirror)
    param_grp.create_dataset("refls_mirror", data=np.array(refls_mirror))
    param_grp.create_dataset("ph_delay", data=ph_delay)
    param_grp.create_dataset("dims", data=dims)
    param_grp.create_dataset("dt", data=dt)
    param_grp.create_dataset("num_tsteps", data=num_tsteps)
    param_grp.create_dataset("thresh", data=thresh)
    sim_res_grp = data.create_group("simulation_results")

    # MPS with non-ideal mirror MPS code but with reflectivity = 1.
    print("Running nonideal mirror MPS simulations.")
    results_nonideal = []
    for refl_mirror in refls_mirror:
        print("On refl_mirror = {}".format(refl_mirror))
        # Define the system.
        sys = low_dim_sys.TwoLevelSystem(gamma, delta)
        mps_ideal = oqs_mirror_mps.WgPartialMirror(qutip.basis(2, 1),
                                                   delay, refl_mirror,
                                                   ph_mirror, ph_delay,
                                                   dims, sys, dt,
                                                   thresh)
        result = mps_ideal.simulate(num_tsteps,
                                    [qutip.create(2) * qutip.destroy(2)], 50)
        results_nonideal.append(result[0])
    sim_res_grp.create_dataset("mps_nonideal", data=np.array(results_nonideal))
    data.close()


def emitter_drive_sims() -> None:
    """Run simulations of emitter near mirror with a coherent drive."""
    # Some Parameter values for the system being simulated.
    gamma = 1.0 # Decay rate of the emitter.
    delta = 0.0 # Detuning of emitter frequency from reference frequency.
    delay = 4.0 / gamma # Delay between emitter and mirror.
    ph_mirror = 0 # Phase of the mirror.
    refls_mirror = [0.2, 0.4, 0.6, 0.8, 1.0] # Reflection of the mirror.
    ph_delay = np.pi # Phase corresponding to the delay between emitter to
                     # mirror and back at the reference frequency.
    pulse_width = 2 / gamma
    pulse_amp = 0.1 * gamma

    # Numerical parameters.
    dims = 2 # Dimensionality of each waveguide bin.
    dt = 0.05 / gamma # Time-step. Also used for meshing waveguide in MPS.
    num_tsteps = 1000 # Number of time-steps to perform.
    thresh = 1e-2 # Threshold at which to truncate Schmidth decomposition at
                  # each step of MPS.

    # Open a h5 file for storing simulation results.
    data = h5py.File("results/impact_emitter_drive.h5", "w")
    param_grp = data.create_group("parameters")
    param_grp.create_dataset("gamma", data=gamma)
    param_grp.create_dataset("delta", data=delta)
    param_grp.create_dataset("delay", data=delay)
    param_grp.create_dataset("ph_mirror", data=ph_mirror)
    param_grp.create_dataset("refls_mirror", data=refls_mirror)
    param_grp.create_dataset("ph_delay", data=ph_delay)
    param_grp.create_dataset("dims", data=dims)
    param_grp.create_dataset("dt", data=dt)
    param_grp.create_dataset("num_tsteps", data=num_tsteps)
    param_grp.create_dataset("thresh", data=thresh)
    param_grp.create_dataset("pulse_width", data=pulse_width)
    param_grp.create_dataset("pulse_amp", data=pulse_amp)
    sim_res_grp = data.create_group("simulation_results")

    # Define the pulse.
    def square_pulse(t: float) -> float:
        if t < pulse_width:
            return pulse_amp
        else:
            return 0

    def exponential_pulse(t: float) -> float:
        return pulse_amp * np.exp(-t / pulse_width)

    # Run the nonideal MPS simulations.
    print("Running nonideal MPS simulations.")
    results_mps_nonideal = []
    for refl_mirror in refls_mirror:
        print("On refl_mirror = {}".format(refl_mirror))
        # Define the low dimensional system.
        sys = low_dim_sys.DrivenTwoLevelSystem(gamma, delta, exponential_pulse)
        mps_nonideal = oqs_mirror_mps.WgPartialMirror(qutip.basis(2, 0), delay,
                                                      refl_mirror, ph_mirror,
                                                      ph_delay,
                                                      dims, sys, dt, thresh)
        result = mps_nonideal.simulate(num_tsteps,
                                       [qutip.create(2) * qutip.destroy(2)],
                                       50)
        results_mps_nonideal.append(result[0])
    sim_res_grp.create_dataset("mps_nonideal",
                               data=np.array(results_mps_nonideal))


if __name__ == "__main__":
    print("Performing emitter decay simulations.")
    emitter_decay_sims()
    print("Performing emitter drive simulations.")
    emitter_drive_sims()

