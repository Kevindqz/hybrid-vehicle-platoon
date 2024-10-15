import matplotlib.pyplot as plt

plt.rc("text", usetex=True)
plt.rc("font", size=14)
plt.style.use("bmh")

nx_l = 2


def plot_fleet(n, X, U, R, r_tracking, r_fuel, leader_state, violations=None):
    # fix in one plot
    _, axs = plt.subplots(2, 1, constrained_layout=True, sharex=True)
    axs[0].plot(leader_state[0, :], "--")
    axs[1].plot(leader_state[1, :], "--")
    for i in range(n):
        axs[0].plot(X[:, nx_l * i])
        axs[1].plot(X[:, nx_l * i + 1])
    axs[0].set_ylabel(f"pos (m)")
    axs[1].set_ylabel("vel (ms-1)")
    axs[1].set_xlabel(f"time step k")
    axs[0].legend(["reference"])
    # if violations is not None:
    #    axs[0].plot(violations)

    # plot acceleration (only 1 vehicle for now)
    # _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    # axs.plot(acc)
    # axs.set_ylabel("acceleration")
    # axs.set_xlabel(f"time step k")

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

    # plot aggregated tracking cost
    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    total_tracking_costs = [sum(r_tracking[: i + 1]) for i in range(len(r_tracking))]
    axs.plot(total_tracking_costs)
    axs.set_ylabel("total tracking cost")
    axs.set_xlabel(f"time step k")

    # plot aggregated fuel consumption cost
    _, axs = plt.subplots(1, 1, constrained_layout=True, sharex=True)
    total_fuel_costs = [sum(r_fuel[: i + 1]) for i in range(len(r_fuel))]
    axs.plot(total_fuel_costs)
    axs.set_ylabel("total fuel cost")
    axs.set_xlabel(f"time step k")

