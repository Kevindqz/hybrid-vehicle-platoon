from typing import Any, Tuple

import gymnasium as gym
import numpy as np
import numpy.typing as npt

from misc.leader_trajectory import ConstantVelocityLeaderTrajectory, LeaderTrajectory
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Platoon


class PlatoonEnv(gym.Env[npt.NDArray[np.floating], npt.NDArray[np.floating]]):
    """An env for a platoon of non-linear hybrid vehicles who track each other."""

    # tracking costs
    Q_x = np.diag([1, 0.1])  # penalty of state tracking error
    Q_u = 1 * np.eye(1)  # penalty on control effort
    Q_du = 0 * np.eye(1)  # penalty on variation in control effort

    # local state and control dimension of vehicles in the platoon
    nx_l = Platoon.nx_l
    nu_l = Platoon.nu_l

    step_counter = 0
    viol_counter = []  # count constraint violations

    def __init__(
        self,
        n: int,
        platoon: Platoon,
        ep_len: int,
        leader_index: int = 0,
        ts: float = 1,
        leader_trajectory: LeaderTrajectory = ConstantVelocityLeaderTrajectory(
            p=3000, v=20, trajectory_len=150, ts=1
        ),
        spacing_policy: SpacingPolicy = ConstantSpacingPolicy(50),
        d_safe: float = 25,
        start_from_platoon: bool = False,
        quadratic_cost: bool = True,
        real_vehicle_as_reference: bool = False,
        fuel_penalize: float = 0.0,
    ) -> None:
        super().__init__()

        self.leader_index = leader_index
        self.platoon = platoon
        self.ts = ts
        self.n = n
        self.ep_len = ep_len
        self.d_safe = d_safe
        self.start_from_platoon = start_from_platoon
        self.leader_trajectory = leader_trajectory
        self.spacing_policy = spacing_policy
        self.real_vehicle_as_reference = real_vehicle_as_reference
        self.fuel_penalize = fuel_penalize
        if quadratic_cost:
            self.cost_func = self.quad_cost
        else:
            self.cost_func = self.lin_cost

        if leader_index != 0 and real_vehicle_as_reference:
            raise NotImplementedError(
                f"Not implemented for real vehicle with leader not 0."
            )

        self.previous_action: np.ndarray | None = (
            None  # store previous action to penalise variation
        )
        self.previous_state: np.ndarray | None = None  # store previous state

    def reset(
        self,
        *,
        seed: int = None,
        options: dict[str, Any] = None,
    ) -> tuple[npt.NDArray[np.floating], dict[str, Any]]:
        """Resets the state of the platoon."""
        super().reset(seed=seed, options=options)
        # self.leader_x = self.leader_trajectory.get_leader_trajectory()
        self.leader_x = self.leader_trajectory.get_seeded_leader_trajectory(seed)
        self.x = np.tile(np.array([[0], [0]]), (self.n, 1))

        # create lists to store fuel and tracking costs seperately
        self.cost_tracking_list: list[float] = []
        self.cost_fuel_list: list[float] = []
        self.acc_list: list[float] = []
        np.random.seed(seed)
        # create once 100 random starting states
        # starting_velocities = [
        #     30 * np.random.random() + 5 for i in range(100)
        # ]  
        starting_velocities = [20 * np.random.random() + 5 for i in range(100)]
        # # starting velocities between 5-35 ms-1
        # starting positions between 0-1000 meters, with some forced spacing
        # front_pos = 3000.0
        # the front_pos is random around 3000, can be higher or lower
        front_pos = 3000.0 - 100 + 200 * np.random.random()

        spread = 100
        spacing = 60
        starting_positions = [front_pos]
        for i in range(1, 100):
            starting_positions.append(
                -spread * np.random.random() + starting_positions[-1] - spacing
            )
        if not self.start_from_platoon:
            # order the agents by starting distance
            for i in range(self.n):
                init_pos = max(starting_positions)
                self.x[i * self.nx_l, :] = init_pos
                self.x[i * self.nx_l + 1, :] = starting_velocities[i]
                starting_positions.remove(init_pos)

        else:  # if not random, the vehicles start in perfect platoon with leader on trajectory
            for i in range(self.n):
                if not self.real_vehicle_as_reference:
                    self.x[i * self.nx_l : self.nx_l * (i + 1), :] = self.leader_x[
                        :, [0]
                    ] + i * self.spacing_policy.spacing(self.leader_x[:, [0]])
                else:
                    self.x[i * self.nx_l : self.nx_l * (i + 1), :] = self.leader_x[
                        :, [0]
                    ] + (i + 1) * self.spacing_policy.spacing(self.leader_x[:, [0]])

        self.step_counter = 0
        self.viol_counter.append(np.zeros(self.ep_len))
        info = {"leader_trajectory": self.leader_x}
        return self.x, info

    def quad_cost(self, x: np.ndarray, Q: np.ndarray):
        """Returns x'Qx"""
        return x.T @ Q @ x

    def lin_cost(self, x: np.ndarray, Q: np.ndarray):
        """Return sum_i |Q_i x_i|."""
        return np.linalg.norm(Q @ x, ord=1)

    def get_stage_cost(
        self,
        state: npt.NDArray[np.floating],
        action: npt.NDArray[np.floating],
        gear: npt.NDArray[np.integer],
    ) -> Tuple[float, float, float]:
        """Computes the tracking stage cost."""
        if (
            self.previous_action is None
        ):  # for the first time step the variation penalty will be zero
            self.previous_action = action

        # split global vars into list of local
        x = np.split(state, self.n, axis=0)
        u = np.split(action, self.n, axis=0)
        u_p = np.split(self.previous_action, self.n, axis=0)

        cost_tracking = 0
        cost_fuel = 0
        # tracking cost
        if not self.real_vehicle_as_reference:
            cost_tracking += self.cost_func(
                x[self.leader_index] - self.leader_x[:, [self.step_counter]], self.Q_x
            )  # first vehicle tracking leader trajectory
        else:
            cost_tracking += self.cost_func(
                x[0]
                - self.leader_x[:, [self.step_counter]]
                - (self.spacing_policy.spacing(x[0])),
                self.Q_x,
            )
        cost_tracking += sum(
            [
                self.cost_func(
                    x[i] - x[i - 1] - (self.spacing_policy.spacing(x[i])),
                    self.Q_x,
                )
                for i in range(1, self.n)
            ]
        )
        # control effort cost
        cost_tracking += sum([self.cost_func(u[i], self.Q_u) for i in range(self.n)])
        # control variation cost
        cost_tracking += sum(
            [self.cost_func(u[i] - u_p[i], self.Q_du) for i in range(self.n)]
        )

        cost_tracking = cost_tracking.item()

        # fuel consumption cost
        vehicles = self.platoon.get_vehicles()
        coefficients_b = [
            0.1569,
            2.450 * 10 ** (-2),
            -7.415 * 10 ** (-4),
            5.975 * 10 ** (-5),
        ]
        coefficients_c = [0.0724, 9.681 * 10 ** (-2), 1.075 * 10 ** (-3)]
        for i in range(self.n):
            acc: float = 0

            # gear = self.platoon.get_gear_from_vehicle_velocity(i, x[i][1])
            traction = self.platoon.get_traction_from_vehicle_gear(i, int(gear[i][0]))
            acc = (
                -vehicles[i].c_fric * x[i][1] ** 2 / vehicles[i].m
                - vehicles[i].mu * vehicles[i].grav
                + traction * u[i][0] / vehicles[i].m
            )
            self.acc_list.append(acc)

            poly_b = (
                coefficients_b[0]
                + coefficients_b[1] * x[i][1]
                + coefficients_b[2] * x[i][1] ** 2
                + coefficients_b[3] * x[i][1] ** 3
            )
            poly_c = (
                coefficients_c[0]
                + coefficients_c[1] * x[i][1]
                + coefficients_c[2] * x[i][1] ** 2
            )

            # no fuel consumption if vehicle is decelerating
            if u[i][0] < 0:
                cost_fuel += 0
            else:
                cost_fuel += sum(poly_b + poly_c * acc)


            total_cost = cost_fuel + 0.0025 * cost_tracking
        # check for constraint violations
        if (
            self.real_vehicle_as_reference
            and self.leader_x[0, self.step_counter] - x[0][0, 0] < self.d_safe
        ):
            self.viol_counter[-1][self.step_counter] = 100
        elif any(
            [x[i][0, 0] - x[i + 1][0, 0] < self.d_safe for i in range(self.n - 1)]
        ):
            self.viol_counter[-1][self.step_counter] = 100

        self.previous_action = action
        self.previous_state = state
        return total_cost, cost_tracking, cost_fuel

    def step(
        self, action: np.ndarray
    ) -> tuple[npt.NDArray[np.floating], float, bool, bool, dict[str, Any]]:
        """Steps the platoon system."""
        if action.shape != (self.n * self.nu_l, 1) and action.shape != (
            2 * self.n * self.nu_l,
            1,
        ):
            raise ValueError(
                f"Expected action of size {(self.n*self.nu_l, 1)} (no gears) or {(2*self.n*self.nu_l, 1)} (with gears). Got {action.shape}"
            )
        if (
            action.shape[0] == 2 * self.n * self.nu_l
        ):  # this action contains n gear choices aswell as continuous throttle vals
            u = action[: self.n * self.nu_l, :]
            j = action[self.n * self.nu_l :, :]
        else:
            u = action
            j = np.zeros((self.n, 1))
            for i in range(self.n):
                j[i, :] = self.platoon.get_gear_from_vehicle_velocity(
                    i, self.x[2 * i + 1, 0]
                )

        r_total, r_tracking, r_fuel = self.get_stage_cost(self.x, u, j)
        self.cost_tracking_list.append(r_tracking)
        self.cost_fuel_list.append(r_fuel)
        x_new = self.platoon.step_platoon(self.x, u, j, self.ts)
        self.x = x_new

        self.step_counter += 1
        # print(f"step {self.step_counter}")
        return x_new, r_total, False, False, {"cost_tracking": r_tracking, "cost_fuel": r_fuel}

    def get_state(self):
        return self.x

    def get_previous_state(self):
        """Returns state of platoon at previous time step. Uses current state for first time-step."""
        return self.previous_state if self.previous_state is not None else self.x

    def set_leader_x(self, leader_x: np.ndarray):
        self.leader_x = leader_x