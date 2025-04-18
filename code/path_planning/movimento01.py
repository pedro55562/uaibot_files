import os
import urllib.request
import uaibot as ub
import numpy as np
import networkx as nx
from scipy.spatial import KDTree

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

def distance(q1, q2):
    return np.linalg.norm(q1 - q2)

def get_random_configuration(robot, q_goal, iteration, n, base_goal_bias=0.25, decay_rate= 0.001):
    goal_bias = base_goal_bias * np.exp(-decay_rate * iteration)
    if np.random.uniform(0, 1) < goal_bias:
        return q_goal
    return np.matrix(np.random.uniform(-np.pi, np.pi, (n, 1)))

def nearest_node(q_rand, nodes_list, n):
    kd_tree = KDTree(np.array(nodes_list))
    _, idx = kd_tree.query(q_rand.reshape(1, -1))
    nearest = nodes_list[idx[0]]
    return nearest.reshape(n, 1)

def get_new_node(q_near, q_rand, n, step_min=0.4, step_max=2):
    dist_val = distance(q_rand, q_near)
    step_size = np.random.uniform(step_min, step_max)
    direction = (q_rand - q_near) / dist_val
    q_new = q_near + step_size * direction  
    return q_new, step_size

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

def is_path_free(q_near, q_new, robot, htm_base, all_obs, sample_rate = 0.1):
    # verifica se o caminho entre q_near e q_new é livre
    # utilizando a interpolação linear
    # coleta amostras a cada 0.3 de distancia no espaço de configuração.
    num_samples = int(distance(q_near, q_new) / sample_rate)
    if num_samples < 1:
        num_samples = 5
    for i in range(1, num_samples + 1):
        alpha = i / num_samples  
        q_interp = (1 - alpha) * q_near + alpha * q_new 
        verify, _, _ = robot.check_free_configuration(q=q_interp, htm=htm_base, obstacles=all_obs)
        if verify == False:
            return False 
    return True  

def backpropagation(graph, q_goal):
    path = []
    current_node = tuple(q_goal.flat)
    while current_node is not None:
        path.append(np.array(current_node).reshape(-1, 1))
        current_node = graph.nodes[current_node]['parent']
    path.reverse()
    return path


def get_path(graph, q_goal_node):
    graph_path = []
    current_node = q_goal_node
    while current_node is not None:
        graph_path.append(current_node)
        current_node = graph.nodes[current_node]['parent']

    graph_path.reverse()
    return graph_path



def refine_path(graph, q_goal_node, robot, htm_base, all_obs):
    graph_path = get_path(graph, q_goal_node)
    if len(graph_path) < 2:
        return graph
    size = len(graph_path)
    # itera o caminho encontrado do final até o começo
    for j in range(round(size - 1), 0, -1): # do final 
        for i in range(0 , round(size - 1)): # do começo
            print(size)
            if i >= size or j >= size:
                continue
            # verifica se o caminho entre os nos esta livre
            free = is_path_free(np.array(graph_path[i]).reshape(-1, 1) , np.array(graph_path[j]).reshape(-1, 1) ,robot, htm_base, all_obs)
            
            # se o caminho esta livre, troca o parent deste nó
            if free == True:
                if j > 0:
                    graph.remove_edge(graph_path[j], graph_path[j - 1])
                graph.add_edge(graph_path[j], graph_path[i])
                graph.nodes[ graph_path[j] ]['parent'] = graph_path[i]
                graph_path = get_path(graph, q_goal_node)
                size = len(graph_path)
                j = size - 1
                break
    return graph




def rrt_path_planning(robot, all_obs, q0, htm_base, htm_tg, max_iter= 500, tolerance=0.15, goal_bias=0.35, bias_decay_rate = 0.0001, num_of_trys = 10):
    n = robot.q.shape[0]
    path = []

    # inicia o grafo com a configuração inicial q0
    graph = nx.Graph()
    graph.add_node(tuple(q0.flat), parent=None, cost=0)
    nodes_list = [np.array(q0).flatten()]

    # encontrar q_goal usando a fkm, considerando restrições
    q_goal = get_q_goal(robot, all_obs, htm_tg, htm_base)
    if q_goal is None:
        print("\n\n The goal is not reachable! \n\n")
        return []

    print("Initial distance :", distance(q0, q_goal))



    for i in range(round(max_iter)):
        # gera uma configuração aleatória
        q_rand = get_random_configuration(robot=robot, q_goal = q_goal,iteration=i, n = n,base_goal_bias=goal_bias, decay_rate= bias_decay_rate)
        
        # encontra o nó mais próximo do grafo em relação a q_rand - utiliza a distancia euclidiana para verificar a distancia   
        q_near = nearest_node(q_rand, nodes_list, n)
        
        # a partir de q_near da um passo de tamanho aleatorio em direção a q_rand 
        q_new, stepsize = get_new_node(q_near, q_rand, n)

        # verifica se a configuração q_new é valida e se o caminho entre q_near e q_new é livre
        isfree, message, info = robot.check_free_configuration(q=q_new, htm=htm_base, obstacles=all_obs)
        if isfree and is_path_free(q_near, q_new, robot, htm_base, all_obs):

            q_new_tuple = tuple(q_new.flat)
            q_near_tuple = tuple(q_near.flat)
            cost = graph.nodes[q_near_tuple]['cost'] + distance(q_new, q_near)

            # adiciona q_new ao grafo, o liganado a q_near
            graph.add_node(q_new_tuple, parent=q_near_tuple, cost=cost)
            graph.add_edge(q_near_tuple, q_new_tuple)
            nodes_list.append(np.array(q_new).flatten())
            # se a distancia entre q_new e q_goal for menor que a tolerancia, o objetivo foi atingido
            if distance(q_new, q_goal) < tolerance:
                # adiciona q_goal ao grafo, o liganado a q_new
                q_goal_tuple = tuple(q_goal.flat)
                graph.add_node(q_goal_tuple, parent=q_new_tuple, cost=cost)
                graph.add_edge(q_new_tuple, q_goal_tuple)
                # pós processamento

                # monta uma lista com todos os nós do grafo, de q_goal até q0
                graph = refine_path(graph, q_goal_tuple, robot, htm_base, all_obs)
                # volta de q_goal até q0 para encontrar o caminho        
                path = backpropagation(graph, q_goal)
                print(f"Goal reached with {i} iterations!")
                print(f"Final distance: {distance(q_new, q_goal)}")
                print(f"Number of nodes: {len(nodes_list)}")
                print(f"Path length: {len(path)}")
                break


    if path == []:
        print("Failed at reching the goal!")
        if num_of_trys == 0:
            return []
        if num_of_trys > 0:
           print("Trying again...\n")
           path = rrt_path_planning(robot, all_obs, q0, htm_base, htm_tg, max_iter, tolerance, goal_bias, num_of_trys = num_of_trys - 1)

    return path

def get_configuration(robot):
  return robot.q

def set_configuration_speed(robot, d_dot, t, dt):
  q_next = robot.q + d_dot*dt
  robot.add_ani_frame(time = t+dt, q = q_next)


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

scenario_index = [412]
for index in scenario_index:
    print(f"==============================\n\tScenario {index}\n==============================\n")
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)
    pathh = []
    pathh = rrt_path_planning(robot, all_obs, q0, htm_base, htm_tg, max_iter = 500, tolerance = 0.35 , goal_bias = 0.35, bias_decay_rate = 0.0001, num_of_trys = 50)
    if pathh != []:

    # coloca bola nas posicoes do caminho
        sl = [ ]
        for q_c in pathh:
            fkm = robot.fkm(q_c)
            print(fkm[ 0 : 3 , 3].T)
            sl.append( fkm[ 0 : 3 , 3] )            
        balls = []
        for s in sl:
            balls.append( ub.Ball(htm = ub.Utils.trn(s), radius=0.02, color="cyan"))

        sim.add(balls)

        t = move_robot_through_path(robot=robot, q_list=pathh , base_frame=htm_base, t = 0, dt = 0.01, task_tol = 0.01 ,consider_path_orientation = True)
        sim.run()
        sim.save("/home/pedro55562/uaibot_files/html/path_planning", "mov" + str(index))


print("\nSimulation completed.\n")

