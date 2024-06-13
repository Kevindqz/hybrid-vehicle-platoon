import pickle
import gurobipy as gp
import numpy as np
from dmpcpwa.agents.mld_agent import MldAgent
from dmpcpwa.mpc.mpc_mld import MpcMld
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup
from gymnasium import Env
from gymnasium.wrappers import TimeLimit
from mpcrl.wrappers.envs import MonitorEpisodes
from scipy.linalg import block_diag
from models import Vehicle, Platoon
from env import PlatoonEnv
from misc.common_controller_params import Params, Sim
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon
from mpcs.fuel_mpc import FuelMpcCent
from mpcs.mpc_gear import MpcGear, MpcNonlinearGear

# from mpcs.mpc_gear import MpcGear
from plot_fleet import plot_fleet

np.random.seed(2)


class MpcGearCent(FuelMpcCent, MpcMldCentDecup, MpcGear):
    def __init__(
        self,
        n: int,
        N: int,
        systems: list[dict],
        platoon: Platoon,
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
        fuel_penalize: float = 0.0,
    ) -> None:
        self.n = n
        MpcMldCentDecup.__init__(
            self, systems, n, N, thread_limit=thread_limit, constrain_first_state=False
        )  # use the MpcMld constructor
        F = block_diag(*[systems[i]["F"] for i in range(n)])
        G = np.vstack([systems[i]["G"] for i in range(n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u_g,
            platoon,
            spacing_policy,
            leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
            fuel_penalize,
        )

    def setup_cost_and_constraints(self,
                                   u,
                                   platoon: Platoon,
                                   spacing_policy: SpacingPolicy = ...,
                                   leader_index: int = 0,
                                   quadratic_cost: bool = True,
                                   accel_cnstr_tightening: float = 0,
                                   real_vehicle_as_reference: bool = False,
                                   fuel_penalize: float = 0.0):
        # overriding the method fuelMPCGear because for discrete input model delta don't exist.
        """Set up  cost and constraints for platoon tracking. Penalises the u passed in."""
        if quadratic_cost:
            self.cost_func = self.min_2_norm
        else:
            self.cost_func = self.min_1_norm

        if leader_index != 0 and real_vehicle_as_reference:
            raise NotImplementedError(
                f"Not implemented for real vehicle with leader not 0."
            )

        nx_l = Vehicle.nx_l
        nu_l = Vehicle.nu_l

        # slack vars for soft constraints
        self.s = self.mpc_model.addMVar(
            (self.n, self.N + 1), lb=0, ub=float("inf"), name="s"
        )

        # cost func
        # leader_traj - gets updated each time step
        self.leader_traj = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="leader_traj"
        )
        cost = 0
        x_l = [self.x[i * nx_l : (i + 1) * nx_l, :] for i in range(self.n)]
        u_l = [u[i * nu_l : (i + 1) * nu_l, :] for i in range(self.n)]

        # creating fuel variables and constraints
        acc = self.mpc_model.addMVar(
            (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="Acc"
        )
        vehicles = platoon.vehicles
        self.mpc_model.addConstrs(
            acc[i, k]
            == -vehicles[i].c_fric * x_l[i][1, k] ** 2 / vehicles[i].m
            - vehicles[i].mu * vehicles[i].grav
            + self.u[i,k] / vehicles[i].m
            for i in range(self.n)
            for k in range(self.N)
        )

        acc_abs = self.mpc_model.addMVar((self.n, self.N), lb=0, ub=float("inf"), name="Acc_abs")
        self.mpc_model.addConstrs(acc_abs[i, k] >= acc[i, k] for i in range(self.n) for k in range(self.N))
        self.mpc_model.addConstrs(acc_abs[i, k] >= -acc[i, k] for i in range(self.n) for k in range(self.N))

        coefficients_b = [
            0.1569,
            2.450 * 10 ** (-2),
            -7.415 * 10 ** (-4),
            5.975 * 10 ** (-5),
        ]

        coefficients_c = [0.0724, 9.681 * 10 ** (-2) , 1.075 * 10 ** (-3)]
        f = self.mpc_model.addMVar(
            (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="Fuel"
        )
        x_l2 = self.mpc_model.addMVar(
            (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="x_l2"
        )
        for i in range(self.n):
            for k in range(self.N):
                poly_b = (
                    coefficients_b[0]
                    + coefficients_b[1] * x_l[i][1, k]
                    + coefficients_b[2] * x_l2[i, k]
                    + coefficients_b[3] * x_l2[i, k] * x_l[i][1, k]
                )
                poly_c = (
                    coefficients_c[0]
                    + coefficients_c[1] * x_l[i][1, k]
                    + coefficients_c[2] * x_l2[i, k]
                )

                self.mpc_model.addConstr(x_l2[i, k] == x_l[i][1, k] ** 2)
                self.mpc_model.addConstr(f[i, k] == poly_b + poly_c *acc_abs[i, k])

        # tracking cost
        if not real_vehicle_as_reference:
            cost += sum(
                [
                    self.cost_func(
                        x_l[leader_index][:, [k]] - self.leader_traj[:, [k]], self.Q_x
                    )
                    for k in range(self.N + 1)
                ]
            )
        else:
            cost += sum(
                [
                    self.cost_func(
                        x_l[0][:, [k]]
                        - self.leader_traj[:, [k]]
                        - spacing_policy.spacing(x_l[0][:, [k]]),
                        self.Q_x,
                    )
                    for k in range(self.N + 1)
                ]
            )
        cost += sum(
            [
                self.cost_func(
                    x_l[i][:, [k]]
                    - x_l[i - 1][:, [k]]
                    - spacing_policy.spacing(x_l[i][:, [k]]),
                    self.Q_x,
                )
                for i in range(1, self.n)
                for k in range(self.N + 1)
            ]
        )
        # control effort cost
        cost += sum(
            [
                self.cost_func(u_l[i][:, [k]], self.Q_u)
                for i in range(self.n)
                for k in range(self.N)
            ]
        )
        # control variation cost
        cost += sum(
            [
                self.cost_func(u_l[i][:, [k + 1]] - u_l[i][:, [k]], self.Q_du)
                for i in range(self.n)
                for k in range(self.N - 1)
            ]
        )
        # slack variable cost
        cost += sum(
            [self.w * self.s[i, k] for i in range(self.n) for k in range(self.N + 1)]
        )

        # add fuel consumption cost
        cost += sum(
            fuel_penalize * f[i, k] for i in range(self.n) for k in range(self.N)
        )

        self.mpc_model.setObjective(cost, gp.GRB.MINIMIZE)

        # add extra constraints
        # acceleration constraints
        self.mpc_model.addConstrs(
            (
                self.a_dec * self.ts
                <= x_l[i][1, k + 1] - x_l[i][1, k] - k * accel_cnstr_tightening
                for i in range(self.n)
                for k in range(self.N)
            ),
            name="dec",
        )
        self.mpc_model.addConstrs(
            (
                x_l[i][1, k + 1] - x_l[i][1, k]
                <= self.a_acc * self.ts - k * accel_cnstr_tightening
                for i in range(self.n)
                for k in range(self.N)
            ),
            name="acc",
        )
        # safe distance behind follower vehicle
        if real_vehicle_as_reference:
            self.mpc_model.addConstrs(
                (
                    x_l[0][0, k] <= self.leader_traj[0, k] - self.d_safe + self.s[0, k]
                    for k in range(self.N + 1)
                ),
                name="safe_leader",
            )
        self.mpc_model.addConstrs(
            (
                x_l[i][0, k] <= x_l[i - 1][0, k] - self.d_safe + self.s[i, k]
                for i in range(1, self.n)
                for k in range(self.N + 1)
            ),
            name="safe",
        )


class MpcNonlinearGearCent(FuelMpcCent, MpcNonlinearGear):
    def __init__(
        self,
        n: int,
        N: int,
        nl_systems: list[dict],
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        thread_limit: int | None = None,
        real_vehicle_as_reference: bool = False,
    ) -> None:
        MpcNonlinearGear.__init__(self, nl_systems, N, thread_limit=thread_limit)
        F = block_diag(*[nl_systems[i]["F"] for i in range(n)])
        G = np.vstack([nl_systems[i]["G"] for i in range(n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u_g,
            spacing_policy,
            leader_index,
            quadratic_cost,
            real_vehicle_as_reference,
        )


class TrackingCentralizedAgent(MldAgent):
    def __init__(self, mpc: MpcMld, ep_len: int, N: int, leader_x: np.ndarray) -> None:
        self.ep_len = ep_len
        self.N = N
        self.leader_x = leader_x

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        self.bin_var_counts = np.zeros((ep_len, 1))
        super().__init__(mpc)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(self.leader_x[:, timestep: (timestep + self.N + 1)])
        self.solve_times[env.step_counter - 1, :] = self.run_time
        self.node_counts[env.step_counter - 1, :] = self.node_count
        self.bin_var_counts[env.step_counter - 1, :] = self.num_bin_vars
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        self.mpc.set_leader_traj(self.leader_x[:, 0: self.N + 1])
        return super().on_episode_start(env, episode, state)


def simulate(
    sim: Sim,
    save: bool = False,
    plot: bool = True,
    seed: int = 1,
    thread_limit: int | None = None,
    leader_index=0,
):
    n = sim.n  # num cars
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
                fuel_penalize=sim.fuel_penalize,
            ),
            max_episode_steps=ep_len,
        )
    )

    # mpcs
    if sim.vehicle_model_type == "pwa_gear":
        mpc = FuelMpcCent(
            n,
            N,
            platoon,
            systems,
            spacing_policy=spacing_policy,
            leader_index=leader_index,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
            quadratic_cost=sim.quadratic_cost,
            fuel_penalize=sim.fuel_penalize,
        )
    elif sim.vehicle_model_type == "pwa_friction":
        mpc = MpcGearCent(
            n,
            N,
            systems,
            platoon,
            spacing_policy=spacing_policy,
            leader_index=leader_index,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
            quadratic_cost=sim.quadratic_cost,
            fuel_penalize=sim.fuel_penalize,
        )
    elif sim.vehicle_model_type == "nonlinear":
        mpc = MpcNonlinearGearCent(
            n,
            N,
            systems,
            spacing_policy=spacing_policy,
            leader_index=leader_index,
            thread_limit=thread_limit,
            real_vehicle_as_reference=sim.real_vehicle_as_reference,
            quadratic_cost=sim.quadratic_cost,
        )
    else:
        raise ValueError(f"{sim.vehicle_model_type} is not a valid vehicle model type.")

    # agent
    agent = TrackingCentralizedAgent(mpc, ep_len, N, leader_x)

    agent.evaluate(env=env, episodes=1, seed=seed, open_loop=sim.open_loop)

    if len(env.observations) > 0:
        X = env.observations[0].squeeze()
        U = env.actions[0].squeeze()
        R = env.rewards[0]  # grabs only total cost
    else:
        X = np.squeeze(env.ep_observations)
        U = np.squeeze(env.ep_actions)
        R = np.squeeze(env.ep_rewards)

    # now grab individual costs
    r_tracking = env.unwrapped.cost_tracking_list
    r_fuel = env.unwrapped.cost_fuel_list
    acc = env.unwrapped.acc_list
    r_tracking = np.array(r_tracking).squeeze()
    r_fuel = np.array(r_fuel).squeeze()
    acc = np.array(acc).squeeze()

    print(f"Return = {sum(R.squeeze())}")
    print(f"Violations = {env.unwrapped.viol_counter}")
    print(f"Run_times_sum: {sum(agent.solve_times)}")
    print(f"average_bin_vars: {sum(agent.bin_var_counts)/len(agent.bin_var_counts)}")

    if plot:
        plot_fleet(n, X, acc, U, R, r_tracking, r_fuel, leader_x, violations=env.unwrapped.viol_counter[0])

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
