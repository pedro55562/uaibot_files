import os
import random
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt


def save_plots(q_dot_hist, r_hist, q_hist, t_hist, folder_path="/home/pedro55562/uaibot_files/code/"):
    q_dot_hist = np.array(q_dot_hist).squeeze()
    r_hist = np.array(r_hist).squeeze()
    q_hist = np.array(q_hist).squeeze()
    t_hist = np.array(t_hist).squeeze()

    def _plot_and_save(data, ylabel, title, filename):
        plt.figure()
        if data.ndim == 1:
            plt.plot(t_hist, data, label=ylabel)
        else:
            for i in range(data.shape[1]):
                plt.plot(t_hist, data[:, i], label=f"{ylabel}[{i+1}]")

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
    _plot_and_save(q_hist, "q (deg)", "q x t", "q.png")


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
    for obs in all_obs:
        obs._mesh_material = ub.MeshMaterial(color='magenta')

    sim = ub.Simulation()
    sim.add(all_obs)
    sim.add(robot)
    sim.add(frame_tg)

    return robot, sim, all_obs, q0, htm_tg, htm_base

def get_configuration(robot):
    return robot.q

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)

def fun_F(r):
    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j, 0] = np.sign(r[j, 0]) * np.sqrt(np.abs(r[j, 0]))
    return f

def constrained_control(robot, obstacles, htm_tg):
    n = len(robot.links)
    u = robot.q*0
    q = robot.q 
    r, Jr = robot.task_function(htm_des = htm_tg, q = q)

    k_alpha = 4
    dj_lim  = (np.pi/180)*2
    d_lim   = 0.05
    dlim_auto = 0.005
 
    eta_obs   = 0.3
    eta_auto  = 0.3
    eta_joint = 0.3
    
    # Limite de junta
    q_min = robot.joint_limit[:,0]
    q_max = robot.joint_limit[:,1]

    Aj_min =  np.identity(n)
    Aj_max = -np.identity(n)

    Bj_min = - eta_joint*( (q - q_min) - dj_lim)
    Bj_max = - eta_joint*( (q_max - q) - dj_lim)


    # colisao com objetos
    Ad_obj = np.matrix(np.zeros((0,n)))
    Bd_obj =np.matrix(np.zeros((0,1)))

    for ob in obstacles:
        ds = robot.compute_dist(ob)
        Ad_obj = np.matrix(np.vstack((Ad_obj, ds.jac_dist_mat      )))
        Bd_obj = np.matrix(np.vstack((Bd_obj, ds.dist_vect - d_lim )))
    Bd_obj = - eta_obs*Bd_obj

    # auto colisao
    dist_auto = robot.compute_dist_auto()
    A_auto = dist_auto.jac_dist_mat
    B_auto = -eta_auto*(dist_auto.dist_vect - dlim_auto)
    
    #Create the optimization problem
    A =    np.matrix(np.vstack( (Aj_min, Aj_max, Ad_obj, A_auto) ) )
    b =    np.matrix(np.vstack( (Bj_min, Bj_max, Bd_obj, B_auto) ) )
    H = 2*(Jr.transpose() * Jr + 1e-3*np.identity(n))
    f = Jr.transpose() * k_alpha * fun_F(r)

    u = ub.Utils.solve_qp(H, f, A, b)


    return u, r




tmax = 20
index = 414
dt = 0.01
t = 0

q_hist    = []
qdot_hist = []
r_hist    = []
t_hist    = []
robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)

for i in range(round(0/dt),round(tmax/dt)):
    t = dt*i
    qdot , r= constrained_control(robot = robot, obstacles = all_obs, htm_tg = htm_tg)
    
    qdot_hist.append( np.degrees(qdot))
    r_hist.append(r)
    t_hist.append(t)
    q_hist.append(np.degrees(robot.q))

    set_configuration_speed(robot, qdot, t, dt)
    

save_plots(qdot_hist, r_hist, q_hist, t_hist)
sim.run()
sim.save("/home/pedro55562/uaibot_files/html/path_planning", "constrained" + str(index))