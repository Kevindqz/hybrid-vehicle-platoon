import pickle
from typing import Any, Optional

import numpy as np
from dmpcpwa.mpc.mpc_mld import MpcMld
from gymnasium import Env, spaces
from gymnasium.core import ObsType, ActType
from gymnasium.wrappers import TimeLimit
from mpcrl import Agent
from mpcrl.wrappers.envs import MonitorEpisodes
from numpy._typing import NDArray
import numpy.typing as npt

from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from models import Platoon
from mpcs.new_mpc import MpcMldCentNew, SolverTimeRecorder
from mpcs.new_fuel_mpc import FuelMpcCentNew

# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet
import matplotlib.pyplot as plt
import time
import logging




np.random.seed(2)

class TrackingCentralizedAgentNew(Agent):
    def __init__(self, mpc, ep_len: int, N: int, leader_x: np.ndarray) -> None:
        self.ep_len = ep_len
        self.N = N
        self.leader_x = leader_x
        self.mpc = mpc
        super().__init__(mpc, mpc.fixed_parameters)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.fixed_parameters["leader_traj"] = self.leader_x[
            :, timestep : (timestep + self.N + 1)
        ]
        # self.solve_times
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        self.fixed_parameters["leader_traj"] = self.leader_x[:, 0 : self.N + 1]
        return super().on_episode_start(env, episode, state)
    
    def evaluate(
        self,
        env: Env[ObsType, ActType],
        episodes: int,
        deterministic: bool = True,
        seed: int = None,
        raises: bool = True,
        env_reset_options: Optional[dict[str, Any]] = None,
    ) -> npt.NDArray[np.floating]:
        """Evaluates the agent in a given environment."""
        rng = np.random.default_rng(seed)
        self.reset(rng)
        returns = np.zeros(episodes)
        self.on_validation_start(env)
        seeds = map(int, np.random.SeedSequence(seed).generate_state(episodes))
        tracking_cost_list = []
        fuel_cost_list = []
        performance_list = []
        runtimes = []
        # runtimes = []
        for episode, current_seed in zip(range(episodes), seeds):
            total_tracking_cost = 0
            total_fuel_cost = 0
            performance = 0
            # total_solver_time = 0
            state, info = env.reset(seed = current_seed, options=env_reset_options)
            self.leader_x = info["leader_trajectory"]
            truncated, terminated, timestep = False, False, 0
            self.on_episode_start(env, episode, state)
            start_time = time.time()
            while not (truncated or terminated):
                action, sol = self.state_value(state, deterministic)
                if not sol.success:
                    self.on_mpc_failure(episode, timestep, sol.status, raises)

                state, r, truncated, terminated, info = env.step(action)
                self.on_env_step(env, episode, timestep)
                total_tracking_cost += info["cost_tracking"]
                total_fuel_cost += info["cost_fuel"]
                # total_solver_time += self.run_time
                performance += info["cost_fuel"] + 0.0025 * info["cost_tracking"]

                returns[episode] += r
                timestep += 1
                self.on_timestep_end(env, episode, timestep)
                
            end_time = time.time()
            self.on_episode_end(env, episode, returns[episode])
            total_runtime = end_time - start_time
            # runtimes.append(sum(self.mpc.solver_time))
            runtimes.append(total_runtime)
            tracking_cost_list.append(total_tracking_cost)
            fuel_cost_list.append(total_fuel_cost)
            performance_list.append(performance)
            print(f"Episode {episode}, performance: {performance}, tracking cost: {total_tracking_cost}, fuel cost: {total_fuel_cost}, runtime: {total_runtime}")    
        self.on_validation_end(env, returns)

        average_runtime = np.mean(runtimes)
        average_tracking_cost = np.mean(tracking_cost_list)
        average_fuel_cost = np.mean(fuel_cost_list)
        average_performance = np.mean(performance_list)
        logging.info("Fuel penalize parameter: %s", self.mpc.fuel_penalize)
        logging.info("Average performance: %s", average_performance)
        logging.info("Average tracking cost: %s", average_tracking_cost)
        logging.info("Average fuel cost: %f", average_fuel_cost)
        logging.info("Average runtime: %s", average_runtime)
        return returns
    
def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    n_episodes: int = 1,
    n_fuel_params: int = 1,
    seed: int = 1,
    thread_limit: int | None = None,
    leader_index=0,
):
    n = 1  # num cars
    N = sim.N_mpc  # controller horizon
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
    # set up logging
    # empty the output.log file
    with open(f'output_horizon_{N}.log', "w"):
        pass
    logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=[
        logging.FileHandler(f'output_horizon_{N}.log'),
        logging.StreamHandler()
    ])

    # # mpc = SolverTimeRecorder(MpcMldCentNew(N, systems[0]))
    # mpc = SolverTimeRecorder(FuelMpcCentNew(N, systems[0], fuel_penalize = sim.fuel_penalize))
    # # agent
    # agent = TrackingCentralizedAgentNew(mpc, ep_len, N, leader_x)

    # agent.evaluate(env=env, episodes = n_episodes, seed=seed, raises=True)

    for i in range(n_fuel_params):
        # mpc = SolverTimeRecorder(FuelMpcCentNew(N, systems[0], fuel_penalize = i + 1))
        mpc = SolverTimeRecorder(FuelMpcCentNew(N, systems[0], fuel_penalize = n_fuel_params-i))
        agent = TrackingCentralizedAgentNew(mpc, ep_len, N, leader_x)
        agent.evaluate(env=env, episodes = n_episodes, seed=seed, raises=True)

    if len(env.observations) > 0:
        X = env.observations[0].squeeze()
        U = env.actions[0].squeeze()
        R = env.rewards[0]
    else:
        X = np.squeeze(env.ep_observations)
        U = np.squeeze(env.ep_actions)
        R = np.squeeze(env.ep_rewards)

    r_tracking = env.unwrapped.cost_tracking_list
    r_fuel = env.unwrapped.cost_fuel_list
    acc = env.unwrapped.acc_list
    r_tracking = np.array(r_tracking).squeeze()
    r_fuel = np.array(r_fuel).squeeze()
    acc = np.array(acc).squeeze()

    # print(f"Return = {sum(R.squeeze())}")
    # print(f"Violations = {env.unwrapped.viol_counter}")
    # print(f"Run_times_sum: {sum(mpc.solver_time)}")

    if plot and n_episodes == 1:
        plot_fleet(
            1,
            X,
            U,
            R,
            r_tracking,
            r_fuel,
            agent.leader_x,
            violations=env.unwrapped.viol_counter[0],
            )
        plt.show()     

    if save:
        with open(
            f"cent_{sim.id}_seed_{seed}" + ".pkl",
            "wb",
        ) as file:
            pickle.dump(X, file)
            pickle.dump(U, file)
            pickle.dump(R, file)
            pickle.dump(mpc.solver_time, file)
            pickle.dump(env.unwrapped.viol_counter[0], file)
            pickle.dump(leader_x, file)


if __name__ == "__main__":
    simulate(Sim(), save=False, plot = False, n_episodes = 100, n_fuel_params = 4, seed=Sim.seed, leader_index=0)
