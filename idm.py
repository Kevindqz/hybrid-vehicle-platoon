import pickle
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag
from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon
from mpcs.cent_mld import MpcMldCent
from mpcs.mpc_gear import MpcGear, MpcNonlinearGear
# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet
import matplotlib.pyplot as plt

import numpy as np

class IdmAgent:
    def __init__(self, leader_x, platoon) -> None:
        self.acc_max = 4
        self.leader_x = leader_x
        self.epsillon = 4
        self.d_des = 1
        self.platoon = platoon
        self.s0 = 0  # Minimum headway distance
        self.T = 0  # Safe time headway
        self.b = 4  # Comfortable deceleration

    def get_control(self, state, timestep, platoon):
        # Intelligent Driver Model
        v_des = self.leader_x[1, timestep]
        v = state[1]
        d = self.leader_x[0, timestep] - state[0]
        # print(d)
        delta_v = self.leader_x[1, timestep] - v  # Speed difference with the lead vehicle

        # Calculate desired headway distance
        s_star = self.s0 + v * self.T - (v * delta_v) / (2 * np.sqrt(self.acc_max * self.b))

        follow_error = d - s_star
        print(follow_error)
        # Calculate desired acceleration
        acc_des = self.acc_max * (1 - (v / v_des) ** self.epsillon - (s_star / d) ** 2)

        # Get vehicle information
        vehicles = platoon.get_vehicles()
        gear = vehicles[0].get_gear_from_velocity(v.item())
        traction_force = platoon.get_traction_from_vehicle_gear(0, gear)
        m = vehicles[0].m
        c_fric = vehicles[0].c_fric
        mu = vehicles[0].mu

        # Calculate control input
        u = (m * acc_des + c_fric * v ** 2 + mu * m * 9.8) / traction_force
        u = np.clip(u, -1, 1)
        action = np.array([[u.item()], [gear]])

        return action
        
    def evaluate(self, env: Env, num_episode: int = 1, seed: int = None):
        tracking_cost_list = []
        fuel_cost_list = []
        total_reward = 0
        seeds = map(int, np.random.SeedSequence(seed).generate_state(num_episode))

        for episode, current_seed in zip(range(num_episode), seeds):
            total_tracking_cost = 0
            total_fuel_cost = 0
            state, info = env.reset(seed=current_seed)
            self.leader_x = info['leader_trajectory']
            truncated, terminated, timestep = False, False, 0
            episode_reward = 0
            while not (truncated or terminated):
                action = self.get_control(state, timestep, env.unwrapped.platoon)
                state, reward, truncated, terminated, rewards = env.step(action)
                total_tracking_cost += rewards["cost_tracking"]
                total_fuel_cost += rewards["cost_fuel"]
                episode_reward += reward
                timestep += 1
            total_reward += episode_reward

            tracking_cost_list.append(total_tracking_cost)
            fuel_cost_list.append(total_fuel_cost)
            print(f"Episode {episode}, tracking cost: {total_tracking_cost}, fuel cost: {total_fuel_cost}")
        average_reward = total_reward / num_episode
        average_tracking_cost = np.mean(tracking_cost_list)
        average_fuel_cost = np.mean(fuel_cost_list)
        print(f"Average tracking cost: {average_tracking_cost}")
        print(f"Average fuel cost: {average_fuel_cost}")
        return average_reward

def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    num_episode: int = 1,
    seed: int = 1,
    thread_limit: int | None = None,
    leader_index=0,
):
    n = sim.n  # num cars
    ep_len = sim.ep_len  # length of episode (sim len)
    ts = Params.ts
    masses = sim.masses

    spacing_policy = sim.spacing_policy
    leader_trajectory = sim.leader_trajectory
    leader_x = leader_trajectory.get_seeded_leader_trajectory(seed)
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

    # agent
    agent = IdmAgent(leader_x, platoon)

    agent.evaluate(env, num_episode, seed)
    leader_x = agent.leader_x
    if len(env.observations) > 0:
        X = env.observations[0].squeeze()
        U = env.actions[0].squeeze()
        R = env.rewards[0]
    else:
        X = np.squeeze(env.ep_observations)
        U = np.squeeze(env.ep_actions)
        R = np.squeeze(env.ep_rewards)

    # print(f"Return = {sum(R.squeeze())}")
    # print(f"Violations = {env.unwrapped.viol_counter}")
    # print(f"Run_times_sum: {sum(agent.solve_times)}")
    # print(f"average_bin_vars: {sum(agent.bin_var_counts)/len(agent.bin_var_counts)}")

    # grab individual costs
    r_tracking = env.unwrapped.cost_tracking_list
    r_fuel = env.unwrapped.cost_fuel_list
    acc = env.unwrapped.acc_list
    r_tracking = np.array(r_tracking).squeeze()
    r_fuel = np.array(r_fuel).squeeze()
    acc = np.array(acc).squeeze()

    if plot:
        plot_fleet(n, X, U, R, r_tracking, r_fuel, leader_x, violations=env.unwrapped.viol_counter[0])
        plt.show()

    # if save:
    #     with open(
    #         f"cent_{sim.id}_seed_{seed}" + ".pkl",
    #         "wb",
    #     ) as file:
    #         pickle.dump(X, file)
    #         pickle.dump(U, file)
    #         pickle.dump(R, file)
    #         pickle.dump(r_tracking, file)
    #         pickle.dump(r_fuel, file)
    #         pickle.dump(agent.solve_times, file)
    #         pickle.dump(agent.node_counts, file)
    #         pickle.dump(env.unwrapped.viol_counter[0], file)
    #         pickle.dump(leader_x, file)

    purempc_data = {
        "X": X,
        "U": U,
        "R": R,
        "r_tracking": r_tracking,
        "r_fuel": r_fuel,
    }

    if save:
        with open(
            'idm_data.pkl',
            "wb",
        ) as file:
            pickle.dump(purempc_data, file)


if __name__ == "__main__":
    simulate(Sim(), save=True, plot = True, num_episode = 1 , seed= Sim().seed, leader_index=0)