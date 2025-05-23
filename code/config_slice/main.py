import os
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from setup import *
from CBFPathFollower import *

def create_config_slice(   robot, joint_idx_1, joint_idx_2, num_points, htm_base, all_obs, q_fix):
    q_min = -np.pi
    q_max = np.pi

    q1_vals = np.linspace(q_min, q_max, num_points)
    q2_vals = np.linspace(q_min, q_max, num_points)



    result_matrix = np.zeros((num_points, num_points), dtype=bool)

    for i, q1 in enumerate(q1_vals):
        for j, q2 in enumerate(q2_vals):
            config = q_fix.copy()
            config[joint_idx_1] = q1
            config[joint_idx_2] = q2

            verify , _ , _= robot.check_free_configuration(q=config, htm=htm_base, obstacles=all_obs)
            
            result_matrix[j, i] = verify

    return result_matrix, q1_vals, q2_vals

def plot_result_matrix(
    matrix, q1_vals, q2_vals, joint_idx_1, joint_idx_2, q_goal,
    filename=None, traj_q1=None, traj_q2=None, show=False
):
    from matplotlib.colors import ListedColormap
    import matplotlib.pyplot as plt

    cmap = ListedColormap(["#d82626", "#115711"])  # Vermelho (colisão), Verde (livre)

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

    # Barra de cores
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(['Ocupado (0)', 'Livre (1)'])

    # Trajetória
    if traj_q1 is not None and traj_q2 is not None:
        plt.plot(traj_q1, traj_q2, color='blue', linewidth=2.5, label='Trajetória')
        plt.plot(traj_q1[0], traj_q2[0], 'ko', markersize=8, label='Início')  # ponto preto no início
        plt.plot(traj_q1[-1], traj_q2[-1], 'kX', markersize=10, label='Fim')  # X preto no fim

    # Ponto objetivo (q_goal)
    q_goal_1 = q_goal[joint_idx_1]
    q_goal_2 = q_goal[joint_idx_2]
    plt.plot(q_goal_1, q_goal_2, 'b*', markersize=14, label='Objetivo')  # estrela vermelha

    plt.legend(loc='upper right')

    # Rótulos e título
    plt.xlabel(f'Junta {joint_idx_1 + 1} (rad)')
    plt.ylabel(f'Junta {joint_idx_2 + 1} (rad)')
    plt.title(f'Fatia do Espaço de Configuração: Juntas {joint_idx_1 + 1} × {joint_idx_2 + 1}')
    plt.grid(True)
    plt.tight_layout()

    # Salvamento ou exibição
    if not show:
        plt.savefig(filename, dpi=600)
        print(f"Figura salva em: {filename}")
        plt.close()
    if show:
        plt.show()

def get_q_goal(robot, all_obs, htm_tg, htm_base, max_iter = 300):
    for i in range(round(max_iter)):
        try:
            q_goal = robot.ikm(htm_target=htm_tg, htm=htm_base, no_iter_max = 1000)
        except Exception as e:
            return None
        isfree, message, info = robot.check_free_configuration(q=q_goal, htm=htm_base, obstacles=all_obs)
        if isfree:
            return q_goal
    return None

def run_all_joint_pairs(robot, num_points, htm_base, all_obs):
    dof = robot.q.shape[0]
    q_fix = robot.q.copy()

    for i in range(dof - 1):
        j1 = i
        j2 = i + 1
        print(f"\nAnalisando pares de juntas: {j1 + 1} e {j2 + 1}")
        result_matrix, q1_vals, q2_vals = create_config_slice(
            robot=robot,
            joint_idx_1=j1,
            joint_idx_2=j2,
            num_points=num_points,
            htm_base=htm_base,
            all_obs=all_obs,
            q_fix=q_fix
        )
        print(result_matrix)
        plot_result_matrix(result_matrix, q1_vals, q2_vals, j1, j2, "/home/pedro/uaibot_files/code/config_slice/" + f"juntas { j1 + 1 } e { j2 + 1 }")

if __name__ == "__main__":
    # 1798 257 434 734
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(1798)

    q_goal = get_q_goal(robot=robot, all_obs=all_obs, htm_tg=htm_tg, htm_base=htm_base)
    if q_goal is None:
        print("Não foi possível encontrar uma configuração final válida.")
        exit()

    dof = robot.q.shape[0]
    dt = 0.01
    max_steps = 1000
    i = 0

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

    joint_pairs = [(0, 1), (1, 2), (2, 3), (3, 4),(4, 5),(5, 6)]
    for j1, j2 in joint_pairs:
        print(f"\nExecutando controle para juntas {j1 + 1} e {j2 + 1}")

        j = 0
        t = 0
        trajectory_q1 = []
        trajectory_q2 = []

        while j < max_steps:
            curr_q = robot.q.copy()
            cbf.compute_control(j1=j1, j2=j2, qd=q_goal)

            u = cbf.u.reshape(-1, 1)
            set_configuration_speed(robot=robot, q_dot=u, t=t, dt=dt)

            t = i * dt
            j += 1
            i += 1

            trajectory_q1.append(curr_q[j1, 0])
            trajectory_q2.append(curr_q[j2, 0])
       
        print(f"Criando mapa p/ as juntas {j1 + 1} e {j2 + 1}")
        q_fix = robot.q.copy()
        result_matrix, q1_vals, q2_vals = create_config_slice(
            robot=robot,
            joint_idx_1=j1,
            joint_idx_2=j2,
            num_points=50,
            htm_base=htm_base,
            all_obs=all_obs,
            q_fix=q_fix
        )

        filename = f"/home/pedro/uaibot_files/code/config_slice/juntas_{j1 + 1}_e_{j2 + 1}.png"
        plot_result_matrix(
            result_matrix,
            q1_vals,
            q2_vals,
            joint_idx_1=j1,
            joint_idx_2=j2,
            q_goal=q_goal,
            filename=filename,
            traj_q1=trajectory_q1,
            traj_q2=trajectory_q2
        )


    # Salva o estado final da simulação
    sim.save("/home/pedro/uaibot_files/code/config_slice/", "simulacao_completa")
