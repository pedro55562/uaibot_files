import os
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt


def store_info(sim, qdot_hist, r_norm ,r_hist, t_hist, scenario_code):
    folder = os.path.basename(os.path.dirname(os.path.abspath(__file__)))
    path = "/home/pedro/uaibot_files/info_storage/" + folder
    os.makedirs(path, exist_ok=True)

    path = path + "/" + str(scenario_code)
    os.makedirs(path, exist_ok=True)

    sim.save(path , str(scenario_code))
    save_plots(qdot_hist, r_hist, r_norm ,t_hist, folder_path=path)

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
    temp_obs = []
    for obs in all_obs:
        obs._mesh_material = ub.MeshMaterial(color='magenta')
        temp_obs.append( obs.to_point_cloud(disc=0.06) )

    sim = ub.Simulation()
    sim.add(temp_obs)
    sim.add(robot)
    sim.add(frame_tg)

    return robot, sim, temp_obs, q0, htm_tg, htm_base

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)

def draw_balls(pathhh_, robot, sim, color="cyan", radius = 0.01):
        sl = [ ]
        for q_c in pathhh_:
            fkm = robot.fkm(q = q_c)
            sl.append( fkm[ 0 : 3 , 3] )            
        balls = []
        for s in sl:
            balls.append( ub.Ball(htm = ub.Utils.trn(s), radius = radius, color = color))
        sim.add(balls)

def draw_pc(pathhh_, robot, sim, color="cyan", radius = 0.01):
    sl = [ ]
    for q_c in pathhh_:
        fkm = robot.fkm(q_c)
        sl.append( fkm[ 0 : 3 , 3] ) 
    pc = ub.PointCloud(size = radius, color = color, points = sl)
    sim.add(pc)


def move_to_configuration (robot, goal_q, base_frame, t, dt, consider_orientation = True ,task_tol = 0.05):
    n = robot.q.shape[0]
    goal_H = robot.fkm(q = goal_q)

    goal_s = goal_H[0:3 , 3]
    goal_x = goal_H[0:3 , 0]
    goal_y = goal_H[0:3 , 1]
    goal_z = goal_H[0:3 , 2]


    task_complete = False

    while task_complete == False:
        r   = np.matrix(np.zeros((6,1)))

        current_fkm = robot.fkm()

        x_eef = current_fkm[0:3 , 0]
        y_eef = current_fkm[0:3 , 1]
        z_eef = current_fkm[0:3 , 2]
        s_eef = current_fkm[0:3 , 3]
        
        
        r[0 : 3] = s_eef - goal_s
        r[3] = 1 - goal_x.T*x_eef 
        r[4] = 1 - goal_y.T*y_eef
        r[5] = 1 - goal_z.T*z_eef        


        u = np.matrix(np.zeros((n,1)))
        u = 1*(goal_q - robot.q)

        set_configuration_speed(robot , u,t,dt)

        t = t + dt
        if ( np.linalg.norm(r) < task_tol):
            return t
    

def move_robot_through_path(robot, q_list, base_frame, t, dt, task_tol=0.05, consider_path_orientation = True):

    path_lenght = len(q_list)
    for i in range(len(q_list) - 2):
        q_initial = q_list[i]
        q_target = q_list[i + 1]
        
        print(f"Moving from configuration {i} to {i+1}...")
        t = move_to_configuration(robot, q_target, base_frame, t, dt, consider_orientation = consider_path_orientation)

    print(f"Moving from configuration {path_lenght - 2} to {path_lenght - 1}...")
    t = move_to_configuration(robot, q_list[path_lenght - 1], base_frame, t, dt, consider_orientation = True , task_tol = task_tol)

    print("Path execution completed.")
    return t



