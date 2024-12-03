from misc.common_controller_params import Params, Sim
import numpy as np
import matplotlib.pyplot as plt

def plot_ref_traj(ref_traj: np.ndarray):
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(ref_traj[0, :], "--")
    axs[1].plot(ref_traj[1, :], "--")
    axs[0].set_ylabel(f"pos (m)")
    axs[1].set_ylabel("vel (ms-1)")
    axs[1].set_xlabel(f"time step k")
    axs[0].legend(["reference"])
    plt.show()
if __name__ == "__main__":
    ref_traj = Sim.leader_trajectory.get_seeded_leader_trajectory(113)
    plot_ref_traj(ref_traj)