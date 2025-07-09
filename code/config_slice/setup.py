import os
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt

def store_info(sim, qdot_hist, r_norm ,r_hist, t_hist, folder_path):
    os.makedirs(folder_path, exist_ok=True)

    sim.save(folder_path, "simulacao_completa")
    save_plots(qdot_hist, r_hist, r_norm ,t_hist, folder_path=folder_path)

def save_plots(q_dot_hist, r_hist, r_norm_hist , t_hist, folder_path):
    q_dot_hist = np.array(q_dot_hist).squeeze()
    r_hist = np.array(r_hist).squeeze()
    t_hist = np.array(t_hist).squeeze()

    def _plot_and_save(data, ylabel, title, filename):
        plt.figure()
        plt.plot(t_hist, data, label=ylabel)
        plt.xlabel("t (s)")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.savefig(os.path.join(folder_path, filename))
        plt.close()
        print(f"Gráfico salvo em: {os.path.join(folder_path, filename)}")

    _plot_and_save(q_dot_hist, "q_dot (deg/s)", "q_dot x t", "q_dot.png")
    _plot_and_save(r_hist, "r", "r x t", "r.png")
    _plot_and_save(r_norm_hist, "r_norm", "||r|| x t", "r_norm.png")

def create_scenario():
    robot = ub.Robot.create_kuka_kr5()
    q = np.array([[np.pi/2], [0.2], [-1.8], [ 0 ], [-np.pi/2], [0.0]])
    robot.add_ani_frame(time=0, q=q)


    obs1 = ub.Box(name="obs1", htm = ub.Utils.trn([0 , 0.85, 0.55]), width=1, depth=1, height=0.05)

    htm_init = robot.fkm()
    htm_tg = htm_init * ub.Utils.trn([0.3, 0, 0.2]) * ub.Utils.roty(-np.pi)

    frame_tg = ub.Frame(htm=htm_tg, size=0.1)
    all_obs = [obs1]

    sim = ub.Simulation()
    sim.add(all_obs)
    sim.add(robot)
    sim.add(frame_tg)    

    ok, message, _ = robot.check_free_config(obstacles=all_obs)
    if ok:
        print("Configuração inicial livre de colisões")
    else:
        print(f"Configuração inicial em colisão: {message}")

    try:
        q_goal = robot.ikm(htm_tg=htm_tg, q0=q, obstacles=all_obs, no_tries = 200, no_iter_max=1000)
        print("IK encontrada com sucesso!")
    except Exception as e:
        print(f"Erro na IK: {e}")

    return robot, sim, all_obs, robot.q, htm_tg, htm_init

def setup_motion_planning_simulation(problem_index):

    filename = "fishbotics_mp_problems.npz"
    raw_url = "https://raw.githubusercontent.com/viniciusmgn/uaibot_content/master/contents/MotionPlanningProblems/" + filename

    if not os.path.isfile(filename):
        print(f"Baixando {filename}...")
        urllib.request.urlretrieve(raw_url, filename)
        print("Finalizado!")
    else:
        print("Arquivos já detectados")

    allproblems = np.load("fishbotics_mp_problems.npz", allow_pickle=True)
    allproblems = allproblems['arr_0'].item()
    name_problems = list(allproblems.keys())

    #Escolha um dos 2600 problemas pelo nome. Por exemplo, vamos pegar o primeiro
    name = name_problems[problem_index]
    prob = allproblems[name]

    #Extrai as informações
    all_obs = prob['all_obs']
    q0 = prob['q0']
    htm_tg = prob['htm_tg']
    htm_base = prob['htm_base']
    
    frame_tg = ub.Frame(htm=htm_tg, size=0.1)
    robot = ub.Robot.create_franka_emika_3(htm=htm_base)
    robot.add_ani_frame(time=0, q=q0)
    robot.update_col_object(0)
    # for obs in all_obs:
    #     obs._mesh_material = ub.MeshMaterial(color='magenta')

    sim = ub.Simulation()
    sim.add(all_obs)
    sim.add(robot)
    sim.add(frame_tg)

    # for link in robot.links:
    #     for col_obj in link.col_objects:
    #         sim.add(col_obj[0])

    return robot, sim, all_obs, q0, htm_tg, htm_base

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)

