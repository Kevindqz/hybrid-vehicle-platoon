import pickle
from misc.common_controller_params import Params, Sim
import matplotlib.pyplot as plt

# load pickle file and return the data
def load_data(file):
    with open(file, 'rb') as f:
        train_data = pickle.load(f)
    return train_data

def plot_fleet_any_episode(n, X, U, R, leader_state):
    # fix in one plot
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(leader_state[0, :], "--")
    axs[1].plot(leader_state[1, :], "--")
    for i in range(n):
        axs[0].plot(X[:,2 * i])
        axs[1].plot(X[:,2 * i + 1])
    axs[0].set_ylabel(f"pos (m)")
    axs[1].set_ylabel("vel (ms-1)")
    axs[1].set_xlabel(f"time step k")
    axs[0].legend(["reference"])
   

    # plot control input and total cost (only 1 vehicle for now)
    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    axs.plot(U)
    axs.set_ylabel("control input")
    axs.set_xlabel(f"time step k")

    # _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    # axs.plot(R.squeeze())
    # axs.set_ylabel("total cost")
    # axs.set_xlabel(f"time step k")

    # plot tracking and fuel cost individually

    # _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    # axs.plot(r_tracking)
    # axs.set_ylabel("tracking cost")
    # axs.set_xlabel(f"time step k")

    # _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    # axs.plot(r_fuel)
    # axs.set_ylabel("fuel cost")
    # axs.set_xlabel(f"time step k")

    # # plot aggregated fuel consumption cost
    # _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    # total_fuel_costs = [sum(r_fuel[: i + 1]) for i in range(len(r_fuel))]
    # axs.plot(total_fuel_costs)
    # axs.set_ylabel("total fuel cost")
    # axs.set_xlabel(f"time step k")

    plt.show()

if __name__ == '__main__':
    file = 'train_data.pkl'
    data = load_data(file)

    sim = Sim()
    leader_trajectory = sim.leader_trajectory
    leader_x = leader_trajectory.get_leader_trajectory()
    episode_idx = 2999
    X = data["states"][episode_idx].squeeze()
    U = data["actions"][episode_idx].squeeze()
    R = data["rewards"][episode_idx]

    plot_fleet_any_episode(1, X, U, R, leader_x)