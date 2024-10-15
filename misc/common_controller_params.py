from typing import Literal

import numpy as np

from misc.leader_trajectory import (
    ConstantVelocityLeaderTrajectory,
    StopAndGoLeaderTrajectory,
    VolatileTrajectory,
    RandomVolatileTrajectory
)
from misc.spacing_policy import ConstantSpacingPolicy, ConstantTimePolicy

np.random.seed(None)


class Params:
    Q_x = np.diag([1, 0.1])  # penalty of state tracking error
    Q_u = 1 * np.eye(1)  # penalty on control effort
    q_du = 0
    Q_du = q_du * np.eye(1)  # penalty on variation in control effort
    w = 1e4  # penalty on slack variables
    ts = 1
    a_acc = 2.5  # acceleration limit
    a_dec = -2  # deceleration limit
    d_safe = 25


class Sim:
    open_loop = False
    real_vehicle_as_reference = False
    vehicle_model_type: Literal[
        "nonlinear", "pwa_friction", "pwa_gear"
    ] = "pwa_friction"  # MLD form (model II) is pwa_friction
    start_from_platoon: bool = False
    quadratic_cost: bool = True
    fuel_penalize: float = 0
    seed = 4
    n = 1
    N_dqn = 5
    N_mpc = 2
    ep_len = N_dqn if open_loop else 60
    spacing_policy = ConstantSpacingPolicy(50)
    # leader_trajectory = ConstantVelocityLeaderTrajectory(
    #     p=3000, v=20, trajectory_len=ep_len + 50, ts=Params.ts
    # )
    leader_trajectory = RandomVolatileTrajectory(
        p=3000, trajectory_len=ep_len + 50, ts=Params.ts, seed=None
    )
    masses = None
    id = f"default_n_{n}_N_{N_dqn}"


class Sim_n_task_1(Sim):
    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.id = f"task_1_n_{n}_N_{self.N}"
        self.spacing_policy = ConstantSpacingPolicy(50)
        self.leader_trajectory = ConstantVelocityLeaderTrajectory(
            p=3100, v=20, trajectory_len=self.ep_len + 50, ts=Params.ts
        )


class Sim_n_task_2(Sim):
    def __init__(
        self, n: int, seed: int, leader_index: int | None = None, N: int = 6
    ) -> None:
        super().__init__()
        self.n = n
        self.N = N
        if Params.q_du == 0:
            self.id = f"task_2_n_{n}_N_{self.N}"
        else:
            self.id = f"task_2_n_{n}_N_{self.N}_q_{Params.q_du}"
        if leader_index is not None:
            self.id += f"_lead_{leader_index}"
        self.spacing_policy = ConstantTimePolicy(10, 3)
        self.leader_trajectory = StopAndGoLeaderTrajectory(
            p=3000,
            vh=20,
            vl=10,
            vf=30,
            v_change_steps=[30, 50],
            trajectory_len=self.ep_len + 50,
            ts=Params.ts,
        )
        np.random.seed(seed)
        self.masses = np.random.uniform(700, 1000, n).tolist()
