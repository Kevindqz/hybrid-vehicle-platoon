import numpy as np


class LeaderTrajectory:
    """Class for leader trajectories in the platoon."""

    def __init__(self, trajectory_len: int, ts: float) -> None:
        self.trajectory_len = trajectory_len
        self.ts = ts

    def get_leader_trajectory(self) -> np.ndarray:
        raise NotImplementedError(
            "get_leader_trajectory only implemented in sub-classes of LeaderTrajectory."
        )


class ConstantVelocityLeaderTrajectory(LeaderTrajectory):
    """A leader trajectory with a constant velocity."""

    def __init__(self, p: float, v: float, trajectory_len: int, ts: float) -> None:
        super().__init__(trajectory_len, ts)
        self.p0 = p
        self.v = v

    def get_leader_trajectory(self) -> np.ndarray:
        x = np.zeros((2, self.trajectory_len))
        x[:, [0]] = np.array([[self.p0], [self.v]])
        for k in range(self.trajectory_len - 1):
            x[:, [k + 1]] = x[:, [k]] + self.ts * np.array([[self.v], [0]])
        return x


class StopAndGoLeaderTrajectory(LeaderTrajectory):
    """A leader trajectory that slows down and then reaccelerates to initial speed."""

    def __init__(
        self,
        p: float,
        vh: float,
        vl: float,
        v_change_steps: list[int],
        trajectory_len: int,
        ts: float,
        vf: float | None = None,
    ) -> None:
        super().__init__(trajectory_len, ts)
        self.p0 = p
        self.vh = vh
        self.vl = vl
        self.vf = vf
        if len(v_change_steps) != 2:
            raise ValueError(
                f"v_change_steps should have 2 items, received {len(v_change_steps)}"
            )
        self.v_change_steps = v_change_steps

    def get_leader_trajectory(self) -> np.ndarray:
        x = np.zeros((2, self.trajectory_len))
        x[:, [0]] = np.array([[self.p0], [self.vh]])
        v = self.vh
        for k in range(self.trajectory_len - 1):
            x[:, [k + 1]] = x[:, [k]] + self.ts * np.array([[v], [0]])
            if k >= self.v_change_steps[0] and k < self.v_change_steps[1]:
                v = self.vl
                x[1, [k + 1]] = v
            elif k >= self.v_change_steps[1]:
                v = self.vh if self.vf is None else self.vf
                x[1, [k + 1]] = v
        return x


class VolatileTrajectory(LeaderTrajectory):
    """A leader trajectory for a volatile human driven vehicle."""

    def __init__(self, p: float, trajectory_len: int, ts: float) -> None:
        self.p0 = p
        super().__init__(trajectory_len, ts)

    def get_leader_trajectory(self) -> np.ndarray:
        x = np.zeros((2, self.trajectory_len))
        v = 30
        x[:, [0]] = np.array([[self.p0], [v]])
        for k in range(20):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        v = 20
        for k in range(20, 30):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(30, 50):
            v = v + 1
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        v = 10
        for k in range(50, 70):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        v = 20
        for k in range(70, self.trajectory_len - 1):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        return x
    
class RandomVolatileTrajectory(LeaderTrajectory):
    """A leader trajectory for a volatile human driven vehicle, with random acceleration changes."""
    
    def __init__(self, p: float, trajectory_len: int, ts: float, seed: int) -> None:
        self.p0 = p
        self.seed = seed
        super().__init__(trajectory_len, ts)

    def get_leader_trajectory(self) -> np.ndarray:      
        np.random.seed(self.seed)
        x = np.zeros((2, self.trajectory_len))
        v = 20 + np.random.uniform(-5, 5)
        x[:, [0]] = np.array([[self.p0], [v]])


        # 随机化速度变化的时间点和斜率
        change_points = np.sort(np.random.randint(-5, 5, size = 4))
        slopes = np.random.uniform(-0.6, 0.6, size = 3)  # 随机斜率范围 [-5, 5]
    
        # 初始速度

        # current_slope = 0
    
        # for k in range(self.trajectory_len - 1):
        #     if k in change_points:
        #         current_slope = slopes[np.where(change_points == k)[0][0]]
        #     v = max(10, v + current_slope)
        #     x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(20 + change_points[0]):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(20 + change_points[0], 35 + change_points[1]):
            v = max(min(35, v + slopes[0]), 5)
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(35 + change_points[1], 50 + change_points[2]):
            v = max(min(35, v + slopes[1]),5)
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(50 + change_points[2], 70 + change_points[3]):
            v = max(min(35, v + slopes[2]),5)
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(70 + change_points[3], self.trajectory_len - 1):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])

        return x
    
    def get_seeded_leader_trajectory(self, seed: int) -> np.ndarray:
        np.random.seed(seed)
        x = np.zeros((2, self.trajectory_len))
        v = 15 + np.random.uniform(-5, 5)
        x[:, [0]] = np.array([[self.p0], [v]])


        # 随机化速度变化的时间点和斜率
        change_points = np.sort(np.random.randint(-5, 5, size = 4))
        slopes = np.random.uniform(-0.6, 0.6, size = 3)  # 随机斜率范围 [-5, 5]
    
        # 初始速度

        # current_slope = 0
    
        # for k in range(self.trajectory_len - 1):
        #     if k in change_points:
        #         current_slope = slopes[np.where(change_points == k)[0][0]]
        #     v = max(10, v + current_slope)
        #     x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(20 + change_points[0]):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(20 + change_points[0], 35 + change_points[1]):
            v = max(min(35, v + slopes[0]), 5)
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(35 + change_points[1], 50 + change_points[2]):
            v = max(min(35, v + slopes[1]),5)
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(50 + change_points[2], 70 + change_points[3]):
            v = max(min(35, v + slopes[2]),5)
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        for k in range(70 + change_points[3], self.trajectory_len - 1):
            x[:, [k + 1]] = np.array([[x[0, k] + self.ts * v], [v]])
        return x