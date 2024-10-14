import pickle

import numpy as np
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl import Agent
from mpcrl.wrappers.envs import MonitorEpisodes

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from models import Platoon
from mpcs.new_mpc import MpcMldCentNew, SolverTimeRecorder

# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)


class TrackingCentralizedAgentNew(Agent):
    def __init__(self, mpc: MpcMld, ep_len: int, N: int, leader_x: np.ndarray) -> None:
        self.ep_len = ep_len
        self.N = N
        self.leader_x = leader_x

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        self.bin_var_counts = np.zeros((ep_len, 1))
        super().__init__(mpc, mpc.fixed_parameters)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.fixed_parameters["leader_traj"] = self.leader_x[
            :, timestep : (timestep + self.N + 1)
        ]
        self.solve_times
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        self.fixed_parameters["leader_traj"] = self.leader_x[:, 0 : self.N + 1]
        return super().on_episode_start(env, episode, state)


def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    seed: int = 1,
    thread_limit: int | None = None,
    leader_index=0,
):
    n = 1  # num cars
    N = sim.N  # controller horizon
    ep_len = sim.ep_len  # length of episode (sim len)
    ts = Params.ts
    masses = sim.masses

    spacing_policy = sim.spacing_policy
    leader_trajectory = sim.leader_trajectory
    leader_x = leader_trajectory.get_leader_trajectory()
    # vehicles
    platoon = Platoon(n, vehicle_type=sim.vehicle_model_type, masses=masses)
    systems = platoon.get_vehicle_system_dicts(ts)

    # env
    env = MonitorEpisodes(
        TimeLimit(
            PlatoonEnv(
                n=n,
                platoon=platoon,
                leader_trajectory=leader_trajectory,
                spacing_policy=spacing_policy,
                start_from_platoon=sim.start_from_platoon,
                real_vehicle_as_reference=sim.real_vehicle_as_reference,
                ep_len=sim.ep_len,
                leader_index=leader_index,
                quadratic_cost=sim.quadratic_cost,
            ),
            max_episode_steps=ep_len,
        )
    )

    mpc = SolverTimeRecorder(MpcMldCentNew(N, systems[0]))

    # agent
    agent = TrackingCentralizedAgentNew(mpc, ep_len, N, leader_x)

    agent.evaluate(env=env, episodes=1, seed=seed, raises=True)

    if len(env.observations) > 0:
        X = env.observations[0].squeeze()
        U = env.actions[0].squeeze()
        R = env.rewards[0]
    else:
        X = np.squeeze(env.ep_observations)
        U = np.squeeze(env.ep_actions)
        R = np.squeeze(env.ep_rewards)

    print(f"Return = {sum(R.squeeze())}")
    print(f"Violations = {env.unwrapped.viol_counter}")
    print(f"Run_times_sum: {sum(mpc.solver_time)}")
    print(f"average_bin_vars: {sum(agent.bin_var_counts)/len(agent.bin_var_counts)}")

    if plot:
        plot_fleet(n, X, U, R, leader_x, violations=env.unwrapped.viol_counter[0])

    if save:
        with open(
            f"cent_{sim.id}_seed_{seed}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(agent.solve_times, file)
            pickle.dump(agent.node_counts, file)
            pickle.dump(env.unwrapped.viol_counter[0], file)
            pickle.dump(leader_x, file)


if __name__ == "__main__":
    simulate(Sim(), save=False, seed=3, leader_index=0)
