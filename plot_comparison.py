import pickle
import numpy as np
from misc.common_controller_params import Params, Sim
import matplotlib.pyplot as plt
from misc.common_controller_params import Params, Sim
# np.random.seed(19)
nx_l = 2
def plot_comparison(seed):
    # load the data
    sim = Sim()
    leader_trajectory = sim.leader_trajectory
    # leader_x = leader_trajectory.get_leader_trajectory()

    seeds = map(int, np.random.SeedSequence(seed).generate_state(1))
    for episode, current_seed in zip(range(1), seeds):
        leader_x = leader_trajectory.get_seeded_leader_trajectory(current_seed)
        
    with open("evaluate_data.pkl", "rb") as f:
        rlmpc_data = pickle.load(f)
    
    with open("purempc_data.pkl", "rb") as f:
        purempc_data = pickle.load(f)
    
    rlmpc_X = rlmpc_data['states']
    rlmpc_U = rlmpc_data['actions']
    rlmpc_R = rlmpc_data['rewards']
    rlmpc_r_tracking = rlmpc_data['tracking_costs']
    rlmpc_r_fuel = rlmpc_data['fuel_costs']
    
    purempc_X = purempc_data['X']
    purempc_U = purempc_data['U']
    purempc_R = purempc_data['R']
    purempc_r_tracking = purempc_data['r_tracking']
    purempc_r_fuel = purempc_data['r_fuel']

    # plot the data
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(leader_x[0, :], "--", color = "black")
    axs[1].plot(leader_x[1, :], "--", color = "black")
    for i in range(1):
        axs[0].plot(rlmpc_X[:, nx_l * i], label="RLMPC")
        axs[0].plot(purempc_X[:, nx_l * i], label="PureMPC")
        axs[1].plot(rlmpc_X[:, nx_l * i + 1], label="RLMPC")
        axs[1].plot(purempc_X[:, nx_l * i + 1], label="PureMPC")
    axs[0].set_ylabel(f"pos (m)")
    axs[1].set_ylabel("vel (ms-1)")
    axs[1].set_xlabel(f"time step k")
    axs[0].legend(["reference"])
    axs[0].legend()
    axs[1].legend()

    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    # set legend    
    # axs.plot(rlmpc_U, label="RLMPC")
    # axs.plot(purempc_U, label="PureMPC")

    for i in range(1):
        axs[0].plot(rlmpc_U[:, nx_l * i], label="RLMPC")
        axs[0].plot(purempc_U[:, nx_l * i], label="PureMPC")
        axs[1].plot(rlmpc_U[:, nx_l * i + 1], label="RLMPC")
        axs[1].plot(purempc_U[:, nx_l * i + 1], label="PureMPC")
    
    axs[0].set_ylabel("throttle input")
    axs[1].set_ylabel("gear choice")
    axs[1].set_xlabel(f"time step k")
    axs[0].legend()
    axs[1].legend()

    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    axs.plot(rlmpc_r_tracking, label="RLMPC")
    axs.plot(purempc_r_tracking, label="PureMPC")
    axs.set_ylabel("tracking cost")
    axs.set_xlabel(f"time step k")
    axs.legend()

    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    axs.plot(rlmpc_r_fuel, label="RLMPC")
    axs.plot(purempc_r_fuel, label="PureMPC")
    axs.set_ylabel("fuel cost")
    axs.set_xlabel(f"time step k")   
    axs.legend()

    # plot aggregated tracking cost
    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    rlmpc_total_tracking_costs = [sum(rlmpc_r_tracking[: i + 1]) for i in range(len(rlmpc_r_tracking))]
    purempc_total_tracking_costs = [sum(purempc_r_tracking[: i + 1]) for i in range(len(purempc_r_tracking))]
    axs.plot(rlmpc_total_tracking_costs, label="RLMPC")
    axs.plot(purempc_total_tracking_costs, label="PureMPC")
    axs.set_ylabel("total tracking cost")
    axs.set_xlabel(f"time step k")
    axs.legend()



    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    rlmpc_total_fuel_costs = [sum(rlmpc_r_fuel[: i + 1]) for i in range(len(rlmpc_r_fuel))]
    purempc_total_fuel_costs = [sum(purempc_r_fuel[: i + 1]) for i in range(len(purempc_r_fuel))]
    axs.plot(rlmpc_total_fuel_costs, label="RLMPC")
    axs.plot(purempc_total_fuel_costs, label="PureMPC")
    axs.set_ylabel("total fuel cost")
    axs.set_xlabel(f"time step k")
    axs.legend()

    plt.show()
if __name__ == '__main__':
    plot_comparison(seed = Sim.seed) 