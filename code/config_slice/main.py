import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from setup import *
from CBFPathFollower import *
from slice_creator import *


def score(r):
    return np.sum(np.power(np.abs(r), 3 / 2))


class SimulationRunner:
    def __init__(self, robot, sim, all_obs, q_goal, htm_tg, dt=0.01, max_steps=2000):
        self.robot = robot
        self.sim = sim
        self.all_obs = all_obs
        self.q_goal = q_goal
        self.htm_tg = htm_tg
        self.dt = dt
        self.max_steps = max_steps

        self.qdot_hist = []
        self.r_norm = []
        self.r_hist = []
        self.t_hist = []

        self.cbf = CBFPathFollower(
            robot=self.robot,
            obstacles=self.all_obs,
            htm_tg=self.htm_tg,
            dj_lim=np.deg2rad(2),
            d_lim=0.01,
            dlim_auto=0.002,
            eta_obs=0.3,
            eta_auto=0.3,
            eta_joint=0.3,
            eps=1e-2,
            kp=1.0
        )

        self.i = 0  # global step counter

    def run(self, joint_pairs, slice_folder):
        t = 0
        for j1, j2 in joint_pairs:
            print(f"\nExecutando controle para juntas {j1 + 1} e {j2 + 1}")
            trajectory_q1, trajectory_q2 = self._run_joint_control(j1, j2, t)
            self._create_slice(j1, j2, slice_folder, trajectory_q1, trajectory_q2)

    def _run_joint_control(self, j1, j2, t):
        j = 0
        trajectory_q1 = []
        trajectory_q2 = []
        cond = True
        hist_error = []

        while cond:
            hist_error.append(np.linalg.norm(self.q_goal - self.robot.q))
            t = self.i * self.dt
            curr_q = self.robot.q.copy()

            self.cbf.compute_control_task_func(j1=j1, j2=j2)
            u = self.cbf.u.reshape(-1, 1)

            set_configuration_speed(robot=self.robot, q_dot=u, t=t, dt=self.dt)

            self.qdot_hist.append(u)
            self.r_norm.append(score(self.cbf.r))
            self.r_hist.append(self.cbf.r)
            self.t_hist.append(t)

            j += 1
            self.i += 1

            trajectory_q1.append(curr_q[j1, 0])
            trajectory_q2.append(curr_q[j2, 0])

            if j > 50:
                recent = hist_error[-self.i:]
                max_diff = (max(recent) - min(recent))
                cond = max_diff > 0.005
            cond = cond and (j < self.max_steps)

        return trajectory_q1, trajectory_q2

    def _create_slice(self, j1, j2, slice_folder, trajectory_q1, trajectory_q2):
        print(f"Criando mapa p/ as juntas {j1 + 1} e {j2 + 1}")
        filename = os.path.join(slice_folder, f"juntas_{j1 + 1}_e_{j2 + 1}")
        q_fix = self.robot.q

        draw_slice(
            robot=self.robot,
            obstacles=self.all_obs,
            q=q_fix,
            indexes=[j1, j2],
            value=0.003,
            score_fun=lambda _q: score(self.robot.task_function(q=_q, htm_tg=self.htm_tg)[0][0:3, :]),
            path_q=[],
            eps_to_obs=0.0,
            h_to_obs=0.0,
            h_merge=1e-3,
            save_path=filename,
            track=[trajectory_q1, trajectory_q2]
        )


if __name__ == "__main__":
    base_folder = os.path.join("testes", f"problema")
    os.makedirs(base_folder, exist_ok=True)
    slice_folder = os.path.join(base_folder, "config_slice")
    os.makedirs(slice_folder, exist_ok=True)


    pares_de_juntas = [(0, 1),(2,3),(4,5)]

    robot, sim, all_obs, q0, htm_tg, htm_init = create_scenario()
    q_goal = robot.ikm(htm_tg=htm_tg, htm=robot.htm, no_iter_max=1000, no_tries=100)

    runner = SimulationRunner(robot, sim, all_obs, q_goal, htm_tg)
    runner.run(pares_de_juntas, slice_folder)

    store_info(sim, runner.qdot_hist, runner.r_norm, runner.r_hist, runner.t_hist, folder_path=base_folder)