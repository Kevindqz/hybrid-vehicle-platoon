import pickle
import gurobipy as gp
from gurobipy import GRB
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

class RlMpcCent(FuelMpcCent, MpcMldCentDecup, MpcGear):
    """A centralized MPC controller interfacing with RL agent."""
    def __init__(
    self,
    n: int,
    N: int,
    platoon: Platoon,
    pwa_systems: list[dict],
    spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
    leader_index: int = 0,
    quadratic_cost: bool = True,
    thread_limit: int | None = None,
    accel_cnstr_tightening: float = 0.0,
    real_vehicle_as_reference: bool = False,
    ) -> None:
        MpcMldCentDecup.__init__(
            self, pwa_systems, n, N, thread_limit=thread_limit, constrain_first_state=False, verbose=True
        ) # creates the state and control variables, sets the dynamics, and creates the MLD constraints for PWA dynamics
        self.n = n
        self.N = N
        F = block_diag(*[pwa_systems[i]["F"] for i in range(n)])
        G = np.vstack([pwa_systems[i]["G"] for i in range(n)])
        self.setup_gears(N, F, G)
        self.setup_cost_and_constraints(
            self.u,
            platoon,
            pwa_systems,
            spacing_policy,
            leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
        )
    
    #override the setup_cost_and_constraints method, with pre-decided gear choice from upstream RL agent
    def setup_cost_and_constraints(
        self,
        u,
        platoon: Platoon,
        pwa_systems: list[dict],
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        leader_index: int = 0,
        quadratic_cost: bool = True,
        accel_cnstr_tightening: float = 0.0,
        real_vehicle_as_reference: bool = False,
    ):
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

        # Assign gear choice given by the RL agent to the control variable
        # for i in range(len(Vehicle.b)):
        #     for k in range(self.N):
        #         self.sigma[i, 0, k].ub = gear_choice[i, k]
        #         self.sigma[i, 0, k].lb = gear_choice[i, k]

        # cost func
        # leader_traj - gets updated each time step
        self.leader_traj = self.mpc_model.addMVar(
            (nx_l, self.N + 1), lb=0, ub=0, name="leader_traj"
        )
        cost = 0
        x_l = [self.x[i * nx_l : (i + 1) * nx_l, :] for i in range(self.n)]
        u_l = [u[i * nu_l : (i + 1) * nu_l, :] for i in range(self.n)]

        # creating fuel variables and constraints

        # acc = self.mpc_model.addMVar(
        #     (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="Acc"
        # )
        # vehicles = platoon.vehicles
        # self.mpc_model.addConstrs(
        #     acc[i, k]
        #     == -vehicles[i].c_fric * x_l[i][1, k] ** 2 / vehicles[i].m
        #     - vehicles[i].mu * vehicles[i].grav
        #     + self.u[i, k] / vehicles[i].m
        #     for i in range(self.n)
        #     for k in range(self.N)
        # )

        # Add a binary variable, when u is negative, the binary variable is 0, otherwise 1
        
        # epsilon = self.mpc_model.addMVar(
        #     (self.n, self.N), vtype=gp.GRB.BINARY, name="epsilon"
        # )
        # Add constraints
        # self.mpc_model.addConstrs(
        #     u[i, k] <= epsilon[i, k] for i in range(self.n) for k in range(self.N)
        # )
        # self.mpc_model.addConstrs(
        #     u[i, k] >= epsilon[i, k] - 1 for i in range(self.n) for k in range(self.N)
        # )

        # coefficients_b = [
        #     0.1569,
        #     2.450 * 10 ** (-2),
        #     -7.415 * 10 ** (-4),
        #     5.975 * 10 ** (-5),
        # ]

        # coefficients_c = [0.0724, 9.681 * 10 ** (-2), 1.075 * 10 ** (-3)]
        # f = self.mpc_model.addMVar(
        #     (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="Fuel"
        # )
        # x_l2 = self.mpc_model.addMVar(
        #     (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="x_l2"
        # )
        # for i in range(self.n):
        #     for k in range(self.N):
        #         poly_b = (
        #             coefficients_b[0]
        #             + coefficients_b[1] * x_l[i][1, k]
        #             + coefficients_b[2] * x_l2[i, k]
        #             + coefficients_b[3] * x_l2[i, k] * x_l[i][1, k]
        #         )
        #         poly_c = (
        #             coefficients_c[0]
        #             + coefficients_c[1] * x_l[i][1, k]
        #             + coefficients_c[2] * x_l2[i, k]
        #         )

        #         self.mpc_model.addConstr(x_l2[i, k] == x_l[i][1, k] ** 2)
        #         self.mpc_model.addConstr(f[i, k] == poly_b + poly_c * acc[i, k])

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

        # store fuel cost as list but not used in the cost function
        # cost += sum(
        #     epsilon[i, k] * fuel_penalize * f[i, k]
        #     for i in range(self.n)
        #     for k in range(self.N)
        # )

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

    def set_leader_traj(self, leader_traj):
        for k in range(self.N + 1):
            self.leader_traj[:, [k]].ub = leader_traj[:, [k]]
            self.leader_traj[:, [k]].lb = leader_traj[:, [k]]        

class RlMpcAgent(MldAgent):
    """An agent that solves the mpc given the gear choice from the RL agent."""
    def __init__(self, mpc: RlMpcCent, ep_len: int, N: int, leader_x: np.ndarray) -> None:
        self.ep_len = ep_len
        self.N = N
        self.leader_x = leader_x

        self.solve_times = np.zeros((ep_len, 1))
        self.node_counts = np.zeros((ep_len, 1))
        self.bin_var_counts = np.zeros((ep_len, 1))
        super().__init__(mpc)

    def on_timestep_end(self, env: Env, episode: int, timestep: int) -> None:
        # time step starts from 1, so this will set the cost accurately for the next time-step
        self.mpc.set_leader_traj(self.leader_x[:, timestep : (timestep + self.N + 1)])
        self.solve_times[env.step_counter - 1, :] = self.run_time
        self.node_counts[env.step_counter - 1, :] = self.node_count
        self.bin_var_counts[env.step_counter - 1, :] = self.num_bin_vars
        return super().on_timestep_end(env, episode, timestep)

    def on_episode_start(self, env: Env, episode: int, state) -> None:
        self.mpc.set_leader_traj(self.leader_x[:, 0 : self.N + 1])
        return super().on_episode_start(env, episode, state)
    