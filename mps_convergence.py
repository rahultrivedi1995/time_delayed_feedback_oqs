"""Script to study the convergence of MPS with ideal mirror."""
import h5py
import os
import sys
import numpy as np
import qutip

import fdtd_spontaneous_emission as fdtd
import low_dim_sys
import open_quant_sys_mirror_mps as oqs_mirror_mps


def convergence_with_dt() -> None:
    """Runs simulations with different discretization dt."""
    # The parameters assumed here lead to a bound state. We only study
    # convergence of a system with these parameters.
    gamma = 1.0
    delta = 0.0
    delay = 4.0 / gamma
    ph_mirror = 0
    ph_delay = np.pi
    sim_time = 50

    # Numerical parameters.
    dims = 2 # Enough while considering a decay problem.
    dt_fdtd = 0.001 # Use a very small `dt` for FDTD simulations, since these
                    # are very fast.
    thresh = 0.01

    #dt_mps = [0.15, 0.1, 0.075,  0.05, 0.025]
    dt_mps = [1.0, 0.5, 0.15, 0.1, 0.05]

    # Open h5 file for storing simulations.
    data = h5py.File("results/mps_convergence_dt.h5")
    data.create_dataset("dt", data=np.array(dt_mps))

    # Run the FDTD simulations.
    result_fdtd = fdtd.tls_ideal_mirror(gamma, delay, ph_delay, ph_mirror,
                                        dt_fdtd, int(sim_time // dt_fdtd))
    data.create_dataset("fdtd", data=np.array(result_fdtd))

    # Run the MPS simulations.
    mps_grp = data.create_group("mps")
    results_mps = []
    for dt in dt_mps:
        print("On dt = {}".format(dt))
        sys = low_dim_sys.TwoLevelSystem(gamma, delta)
        mps_ideal = oqs_mirror_mps.WgPartialMirror(qutip.basis(2, 1),
                                                   delay, 1.0, ph_mirror,
                                                   ph_delay, dims, sys, dt,
                                                   thresh)
        result = mps_ideal.simulate(int(sim_time // dt),
                                    [qutip.create(2) * qutip.destroy(2)], 50)
        mps_grp.create_dataset("dt_{}".format(dt), data=np.array(result[0]))
    data.close()


def convergence_with_thresh() -> None:
    """Runs simulations with different discretization dt."""
    # The parameters assumed here lead to a bound state. We only study
    # convergence of a system with these parameters.
    gamma = 1.0
    delta = 0.0
    delay = 4.0 / gamma
    ph_mirror = 0
    ph_delay = np.pi
    sim_time = 50

    # Numerical parameters.
    dims = 2 # Enough while considering a decay problem.
    dt_fdtd = 0.001 # Use a very small `dt` for FDTD simulations, since these
                    # are very fast.
    dt_mps = 0.05
    thresh = [0.5, 0.25, 0.1, 0.05, 0.01]


    # Open h5 file for storing simulations.
    data = h5py.File("results/mps_convergence_thresh.h5")

    # Run the FDTD simulations.
    result_fdtd = fdtd.tls_ideal_mirror(gamma, delay, ph_delay, ph_mirror,
                                        dt_fdtd, int(sim_time // dt_fdtd))
    data.create_dataset("fdtd", data=np.array(result_fdtd))

    # Run the MPS simulations.
    mps_grp = data.create_group("mps")
    results_mps = []
    for tol in thresh:
        sys = low_dim_sys.TwoLevelSystem(gamma, delta)
        mps_ideal = oqs_mirror_mps.WgPartialMirror(qutip.basis(2, 1),
                                                   delay, 1.0, ph_mirror,
                                                   ph_delay, dims, sys, dt_mps,
                                                   tol)
        result = mps_ideal.simulate(int(sim_time // dt_mps),
                                    [qutip.create(2) * qutip.destroy(2)], 50)
        mps_grp.create_dataset("tol_{}".format(tol), data=np.array(result[0]))
    data.close()





if __name__ == "__main__":
    convergence_with_thresh()



