import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from setup import *
from CBFPathFollower import *
from slice_creator import *


def score( r ):
    return  np.sum( np.power( np.abs(r), 3/2 ) )


def plot_result_matrix(matrix, q1_vals, q2_vals, joint_idx_1, joint_idx_2, q_goal, filename=None, traj_q1=None, traj_q2=None, show=False):
    cmap = ListedColormap(["#d82626", "#115711"])  # Colisão / Livre

    plt.figure(figsize=(7, 6))
    plt.imshow(
        matrix.astype(int),
        origin='lower',
        extent=[q1_vals[0], q1_vals[-1], q2_vals[0], q2_vals[-1]],
        cmap=cmap,
        aspect='auto',
        vmin=0,
        vmax=1
    )

    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Ocupado (0)', 'Livre (1)'])

    if traj_q1 is not None and traj_q2 is not None:
        plt.plot(traj_q1, traj_q2, color='blue', linewidth=2.5, label='Trajetória')
        plt.plot(traj_q1[0], traj_q2[0], 'ko', markersize=8, label='Início')
        plt.plot(traj_q1[-1], traj_q2[-1], 'kX', markersize=10, label='Fim')

    q_goal_1 = q_goal[joint_idx_1]
    q_goal_2 = q_goal[joint_idx_2]
    plt.plot(q_goal_1, q_goal_2, 'b*', markersize=14, label='Objetivo')

    plt.legend(loc='upper right')
    plt.xlabel(f'Junta {joint_idx_1 + 1} (rad)')
    plt.ylabel(f'Junta {joint_idx_2 + 1} (rad)')
    plt.title(f'Fatia do Espaço de Configuração: Juntas {joint_idx_1 + 1} × {joint_idx_2 + 1}')
    plt.grid(True)
    plt.tight_layout()

    if not show:
        plt.savefig(filename, dpi=600)
        print(f"Figura salva em: {filename}")
        plt.close()
    if show:
        plt.show()


def get_q_goal(robot, all_obs, htm_tg, htm_base, max_iter=300):
    for _ in range(round(max_iter)):
        try:
            q_goal = robot.ikm(htm_target=htm_tg, htm=htm_base, no_iter_max=1000)
        except Exception:
            return None
        isfree, _, _ = robot.check_free_configuration(q=q_goal, htm=htm_base, obstacles=all_obs)
        if isfree:
            return q_goal
    return None

def run_simulation(problem_number, joint_pairs):

    base_folder = os.path.join("testes", f"prob_{problem_number}")
    os.makedirs(base_folder, exist_ok=True)
    slice_folder = os.path.join(base_folder, "config_slice")
    os.makedirs(slice_folder, exist_ok=True)
    
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(problem_number)
    q_goal = get_q_goal(robot, all_obs, htm_tg, htm_base)


    if q_goal is None:
        print("Não foi possível encontrar uma configuração final válida.")
        return

    dt = 0.01
    max_steps = 2000
    i = 0

    qdot_hist = []
    r_norm    = []
    r_hist    = []
    t_hist    = []

    cbf = CBFPathFollower(
        robot=robot,
        obstacles=all_obs,
        htm_tg=htm_tg,
        dj_lim=np.deg2rad(2),
        d_lim=0.01,
        dlim_auto=0.002,
        eta_obs=0.3,
        eta_auto=0.3,
        eta_joint=0.3,
        eps=1e-2,
        kp=1.0
    )
    t = 0
    for j1, j2 in joint_pairs:

        print(f"\nExecutando controle para juntas {j1 + 1} e {j2 + 1}")

        j = 0
        trajectory_q1 = []
        trajectory_q2 = []
        cond = True
        hist_error = []

        while cond:
            hist_error.append(np.linalg.norm(q_goal - robot.q))
            t = i * dt
            curr_q = robot.q.copy()

            cbf.compute_control_task_func(j1=j1, j2=j2)

            u = cbf.u.reshape(-1, 1)

            set_configuration_speed(robot=robot, q_dot=u, t=t, dt=dt)
            
            qdot_hist.append(u)
            r_norm.append(score( cbf.r ))    
            r_hist.append(cbf.r )    
            t_hist.append(t)


            j += 1
            i += 1

            trajectory_q1.append(curr_q[j1, 0])
            trajectory_q2.append(curr_q[j2, 0])

            if j>50:
                recent = hist_error[-i:]
                max_diff = (max(recent) - min(recent))
                cond = max_diff > 0.005
            cond = cond and (j < max_steps)

        print(f"Criando mapa p/ as juntas {j1 + 1} e {j2 + 1}")

        filename = os.path.join(slice_folder, f"juntas_{j1 + 1}_e_{j2 + 1}")
        q_fix = robot.q
        
        draw_slice(
            robot      = robot,
            obstacles  = all_obs,
            q          = q_fix,
            indexes    = [j1, j2],
            value      = 0.003,
            score_fun  = lambda _q: score( robot.task_function(q=_q, htm_des=htm_tg)[0][0 : 3, : ] ),
            path_q     = [],
            eps_to_obs = 0.0,
            h_to_obs   = 0.0,
            h_merge    = 1e-3,
            save_path  = filename,
            track      = [trajectory_q1 , trajectory_q2]
        )


    sim.save(base_folder, "simulacao_completa")
    store_info(sim, qdot_hist, r_norm, r_hist, t_hist, folder_path=base_folder)

if __name__ == "__main__": # 1798 434 734 114
    problema = 734
    pares_de_juntas = [ ] # (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)
    run_simulation(problema, pares_de_juntas)
    