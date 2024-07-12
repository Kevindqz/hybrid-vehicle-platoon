import gurobipy as gp
from dmpcpwa.mpc.mpc_mld_cent_decup import MpcMldCentDecup

from misc.common_controller_params import Params
from misc.spacing_policy import ConstantSpacingPolicy, SpacingPolicy
from models import Vehicle, Platoon


class FuelMpcCent(MpcMldCentDecup):
    """A centralized MPC controller for the platoon using mixed-integer MLD approach."""

    Q_x = Params.Q_x
    Q_u = Params.Q_u
    Q_du = Params.Q_du
    w = Params.w
    a_acc = Params.a_acc
    a_dec = Params.a_dec
    ts = Params.ts
    d_safe = Params.d_safe

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
        fuel_penalize: float = 0.0,
    ) -> None:
        super().__init__(
            pwa_systems, n, N, thread_limit=thread_limit, constrain_first_state=False, verbose=True
        )  # creates the state and control variables, sets the dynamics, and creates the MLD constraints for PWA dynamics
        self.n = n
        self.N = N

        self.setup_cost_and_constraints(
            self.u,
            platoon,
            pwa_systems,
            spacing_policy,
            leader_index,
            quadratic_cost,
            accel_cnstr_tightening,
            real_vehicle_as_reference,
            fuel_penalize,
        )

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
        fuel_penalize: float = 0.0,
    ):
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

        # calculate acceleration using dynamics
        b = self.mpc_model.addMVar(
            (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="b"
        )

        self.mpc_model.addConstrs(
            (
                b[i, k]
                == sum(
                    (
                        self.delta[i][j][k] * pwa_systems[i]["B"][j][1]
                        for j in range(len(pwa_systems[i]["B"]))
                    )
                )
                for i in range(self.n)
                for k in range(self.N)
            ),
            name="b",
        )

        acc = self.mpc_model.addMVar(
            (self.n, self.N), lb=-float("inf"), ub=float("inf"), name="Acc"
        )
        vehicles = platoon.vehicles
        self.mpc_model.addConstrs(
            acc[i, k]
            == -vehicles[i].c_fric * x_l[i][1, k] ** 2 / vehicles[i].m
            - vehicles[i].mu * vehicles[i].grav
            + b[i, k] * u_l[i][0, k] / vehicles[i].m
            for i in range(self.n)
            for k in range(self.N)
        )

        # Add a binary variable, when u is negative, the binary variable is 0, otherwise 1
        epsilon = self.mpc_model.addMVar(
            (self.n, self.N), vtype=gp.GRB.BINARY, name="epsilon"
        )
        # Add constraints
        self.mpc_model.addConstrs(
            u[i, k] <= epsilon[i, k] for i in range(self.n) for k in range(self.N)
        )
        self.mpc_model.addConstrs(
            u[i, k] >= epsilon[i, k] - 1 for i in range(self.n) for k in range(self.N)
        )

        # omega  = self.mpc_model.addMVar((self.n, self.N), vtype=gp.GRB.BINARY, name="omega")
        # self.mpc_model.addConstrs(
        #     acc[i,k] >= 100000 * (omega[i,k] - 1)
        #     for i in range(self.n)
        #     for k in range(self.N)
        # )

        # if_fuel  = self.mpc_model.addMVar((self.n, self.N), vtype=gp.GRB.BINARY, name="if_fuel")
        # self.mpc_model.addConstrs(
        #     if_fuel[i,k] == epsilon[i,k] * omega[i,k]
        #     for i in range(self.n)
        #     for k in range(self.N)
        # )

        coefficients_b = [
            0.1569,
            2.450 * 10 ** (-2),
            -7.415 * 10 ** (-4),
            5.975 * 10 ** (-5),
        ]

        coefficients_c = [0.0724, 9.681 * 10 ** (-2), 1.075 * 10 ** (-3)]
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
                self.mpc_model.addConstr(f[i, k] == poly_b + poly_c * acc[i, k])

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
        # contral variation cost
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
            epsilon[i, k] * fuel_penalize * f[i, k]
            for i in range(self.n)
            for k in range(self.N)
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

    def set_leader_traj(self, leader_traj):
        for k in range(self.N + 1):
            self.leader_traj[:, [k]].ub = leader_traj[:, [k]]
            self.leader_traj[:, [k]].lb = leader_traj[:, [k]]
