import os
import random
import urllib.request
import uaibot as ub
import numpy as np
import networkx as nx
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

q_dot_max = np.radians([150, 150, 150, 150, 301, 301, 301])

def save_plot(q_dot_hist, t_hist, folder_path = "/home/pedro55562/uaibot_files/code/", file_name="q_dot.png"):
    q_dot_hist = np.array(q_dot_hist).squeeze()

    plt.figure()
    if q_dot_hist.ndim == 1:
        plt.plot(t_hist, q_dot_hist, label="q_dot")
    else:
        for i in range(q_dot_hist.shape[1]):
            plt.plot(t_hist, q_dot_hist[:, i], label=f"q_dot[{i+1}]")

    plt.xlabel("t(s)")
    plt.ylabel("q_dot(deg/s)")
    plt.title("q_dot x t")
    plt.legend()
    plt.grid()

    plt.savefig(folder_path + file_name)
    plt.close()
    print(f"Gráfico salvo em: {folder_path + file_name}")


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
            q = robot.ikm(htm_target=htm_tg, htm=htm_base, no_iter_max = 1000)
        except Exception as e:
            continue
        isfree, message, info = robot.check_free_configuration(q=q, htm=htm_base, obstacles=all_obs)
        if isfree:
            return q
    return None

def is_path_free(q_near, q_new, robot, htm_base, all_obs, sample_rate = 0.1):
    # verifica se o caminho entre q_near e q_new é livre
    # utilizando a interpolação linear
    num_samples = int(distance(q_near, q_new) / sample_rate)
    if num_samples < 1:
        num_samples = 5
    for i in range(1, num_samples):
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
        q = np.array(current_node).reshape(-1, 1)
        if q is not None:
            path.append(q)
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



def refine_path(graph, q_goal_node, robot, htm_base, all_obs, sample_d = 0.1):
    # adicionar pontos intermediarios, para melhorar a fase de refino
    current_node = q_goal_node
    parent_node  = q_goal_node
    while current_node is not None and parent_node is not None:
        if parent_node is None or current_node is None:
            break
        
        parent_node = graph.nodes[current_node]['parent']
        current_q = np.array(current_node, dtype=float).reshape(-1, 1)
        parent_q  = np.array(parent_node , dtype=float).reshape(-1, 1)

        if distance( current_q, parent_q) > 0.5:
            # adicionar pontos 
            dist = np.linalg.norm( np.array(current_q) - np.array(parent_q) )
            alpha = 0
            l = 1
            aux_node = current_node
            while alpha < 1:
                alpha = l*sample_d/dist
                l = l + 1
                # adicionar pontos
                new_q = (alpha)*(parent_q) + (1 - alpha)*(current_q)
                new_tuple = tuple(new_q.flat)
                graph.add_node(new_tuple, parent= None)
                graph.nodes[ aux_node ][ 'parent' ] = new_tuple
                graph.add_edge(aux_node, new_tuple)

                aux_node = graph.nodes[aux_node]['parent']

            graph.nodes[ new_tuple ][ 'parent' ] = parent_node
            graph.add_edge( new_tuple, parent_node)
        
        current_node = parent_node
        parent_node  = graph.nodes[current_node]['parent']


    ext_path = backpropagation(graph , np.array(q_goal_node, dtype=float).reshape(-1, 1) )

    # lista dos nós que fazem parte do caminho
    graph_path = get_path(graph, q_goal_node)
    if len(graph_path) < 2:
        return graph
    size = len(graph_path)
    # itera pelos nós que fazem parte do caminho em dois sentidos
    for j in range(round(size - 1), 0, -1): # do final p/ o inicio
        for i in range(0 , round(size - 1)): # do inicio p/ o final
            if i >= size or j >= size:
                continue
            # verifica se o caminho dado por uma reta esta livre
            free = is_path_free(np.array(graph_path[i]).reshape(-1, 1) , np.array(graph_path[j]).reshape(-1, 1) ,robot, htm_base, all_obs)
            if free == True: # seo caminho está livre, conecta os respectivos nós no grafo.
                if j > 0:
                    graph.remove_edge(graph_path[j], graph_path[j - 1])
                graph.add_edge(graph_path[j], graph_path[i])
                graph.nodes[ graph_path[j] ]['parent'] = graph_path[i]
                graph_path = get_path(graph, q_goal_node)
                size = len(graph_path)
                j = size - 1
                break
    return graph, ext_path


def is_goal_configuration(q_new, q_goal, tolerance):
    for q in q_goal:
        if distance(q, q_new) < tolerance:
            return True, q
    return False, q_new

def rrt_path_planning(robot, all_obs, q0, htm_base, htm_tg, max_iter= 500, tolerance=0.15, goal_bias=0.35, bias_decay_rate = 0.0001, num_of_trys = 10):
    n = robot.q.shape[0]
    path = []
    path_found = False
    # inicia o grafo com a configuração inicial q0
    graph = nx.Graph()
    graph.add_node(tuple(q0.flat), parent=None, cost=0)
    nodes_list = [np.array(q0).flatten()]

    # encontrar q_goal usando a fkm, considerando restrições
    q_goal = get_q_goal(robot, all_obs, htm_tg, htm_base)
    if q_goal is None:
        print("\n\n The goal is not reachable! \n\n")
        return [], []




    for i in range(round(max_iter)):
        # gera uma configuração aleatória
        q_rand = get_random_configuration(robot=robot, q_goal = q_goal,iteration=i, n = n,base_goal_bias=goal_bias, decay_rate= bias_decay_rate)
        
        # encontra o nó mais próximo do grafo em relação a q_rand   
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
            if distance(q_goal, q_new) < tolerance:
                # adiciona q_goal ao grafo, o liganado a q_new
                q_goal_tuple = tuple(q_goal.flat)
                graph.add_node(q_goal_tuple, parent=q_new_tuple, cost=cost)
                graph.add_edge(q_new_tuple, q_goal_tuple)
                # pós processamento

                # monta uma lista com todos os nós do grafo, de q_goal até q0
                graph , ext_path = refine_path(graph, q_goal_tuple, robot, htm_base, all_obs)
                # volta de q_goal até q0 para encontrar o caminho   
                path = backpropagation(graph, q_goal)
                path_found = True
                break


    if path == []:
        if num_of_trys == 0:
            return [], []
        if num_of_trys > 0:
           path , ext_path = rrt_path_planning(robot, all_obs, q0, htm_base, htm_tg, max_iter, tolerance, goal_bias, num_of_trys = num_of_trys - 1)


    return path, ext_path

def add_intermediate_points(path, interp_distance = 0.4):
    
    # IDEIA:
    #   Colocar interp_distance como uma porcentagem da distancia
    #   entre path[i] e path[i + 1]
    # 0 < interp_distance < 0.5 
    # para que nao ter conflito
    # itera pelos elementos intermediarios e salva os intermediarios
    intermediate_q = []
    new_path = []
    new_path.append(path[0])
    for i in range(1, round(len(path) - 1)):
        previous_direction = (path[i - 1] - path[i])/distance(path[i - 1], path[i])
        next_direction     = (path[i + 1] - path[i])/distance(path[i + 1], path[i])
        previous_q = previous_direction * interp_distance * distance(path[i - 1], path[i]) + path[i]
        next_q     = next_direction     * interp_distance * distance(path[i + 1], path[i]) + path[i]
        new_path.append(previous_q)
        new_path.append(path[i])  
        new_path.append(next_q)
        intermediate_q.append(previous_q)
        intermediate_q.append(next_q)
    new_path.append(path[len(path) - 1])
    return new_path, intermediate_q

def get_configuration(robot):
  return robot.q

def set_configuration_speed(robot, u, q_dot, t, dt):
  tau = 0.1

  q_dot_next = q_dot + (dt / tau) * (u - q_dot)
  q_next = robot.q + q_dot*dt

  robot.add_ani_frame(time = t+dt, q = q_next)
  return q_dot_next

def move_robot_through_path(robot, q_list, base_frame, t, dt, task_tol=0.1, consider_path_orientation = True):
    # move o robo pelo caminho amostrado
    task_complete = False
    n = robot.q.shape[0]
    i = 0
    q_dot_hist = []
    t_hist     = []
    q_dot = 0*robot.q
    while task_complete == False:
        u = np.matrix(np.zeros((n,1)))
        # calcula a velocidade q_dot através de um campo vetorial no espaco de configuracao
        u = my_vector_field(robot, q_list, q_dot_max)
        
        q_dot = set_configuration_speed(robot , u, q_dot,t,dt)
        q_dot_hist.append(np.degrees(u))
        t_hist.append(t)
        t = t + dt
        i = i + 1
        dist = ( np.linalg.norm(robot.q - q_list[-1])) 
        print(dist)
        if ( dist < task_tol):
            task_complete = True
            return q_dot_hist, t_hist
    return q_dot_hist, t_hist

def my_vector_field(robot, curve, q_dot_max ,alpha= 1, const_vel= 1.7, track_vel = 0.5):
    n = np.shape(robot.q)[0]

    # Encontrar o ponto mais próximo na curva
    index = -1
    dmin = float('inf')
    for i in range(len(curve)):
        d = np.linalg.norm(robot.q - curve[i])
        if d < dmin:
            dmin = d
            index = i

    # define as funcoes usadas como peso para os vetores
    f_g = 0.63 * np.arctan(alpha * dmin)
    f_h = track_vel * np.sqrt(max(1 - f_g**2, 0))

    # calcula o vetor tangente
    if index < len(curve) - 1:
        diff = curve[index + 1] - curve[index]
    else: 
        diff = curve[index] - curve[index - 1]
    T = diff / (np.linalg.norm(diff) + 1e-8)
    
    # calcula o vetor normal
    N = (curve[index] - robot.q) / (np.linalg.norm(curve[index] - robot.q) + 1e-8)


    q_dot = abs(const_vel) * (f_g * N + f_h * T)
    if index < len(curve) - 1:
        dq_ds = (curve[index + 1] - curve[index])/( 1/(len(curve) - 1) )
    else:
        dq_ds = (curve[index] - curve[index - 1])/( 1/(len(curve) - 1) )

    s = []
    for i in range(0, round(len(q_dot))):
        s.append( q_dot_max[i]/( abs(dq_ds[i]) + 1e-8 ) )
    s_dot = np.min(s)
    alpha = np.linalg.norm(dq_ds)*s_dot/np.linalg.norm(q_dot)
    #print(alpha)
    return q_dot*min(1,alpha)

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
   

def bezier_curve(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def trajectory_polynomial(q0, q1, q2, q_dot0, q_dot1, t):
    """
    Polinomio:
        P(t) = e + d*t + c*t^2 + b*t^3 + a*t^4

    Restricoes:
        P(0)   = q0
        P(1/2) = q1
        P(1)   = q2
        P'(0)  = q_dot0
        P'(1)  = q_dot1

        a = -8*q0  + 16*q1 - 8*q2  - 2*q_dot0 + 2*q_dot1   
        b =  18*q0 - 32*q1 + 14*q2 + 5*q_dot0 - 3*q_dot1   
        c = -11*q0 + 16*q1 - 5*q2  - 4*q_dot0 + q_dot1     
        d = q_dot0                                        
        e = q0                                            

    """
    # Coeficientes do polinomio
    a = -8*q0 + 16*q1 - 8*q2 - 2*q_dot0 + 2*q_dot1     
    b = 18*q0 - 32*q1 + 14*q2 + 5*q_dot0 - 3*q_dot1    
    c = -11*q0 + 16*q1 - 5*q2 - 4*q_dot0 + q_dot1      
    d = q_dot0                                       
    e = q0                                           
    
    P_t = e + d*t + c*t**2 + b*t**3 + a*t**4
    return P_t

def generate_path_samples( path, sample_distance, htm_base, all_obs, robot):
    new_path, intermediate_q = add_intermediate_points(path)
    sampled_path = []
    # coleta amostras na reta entre os pontos path[0] e intermediate_q[0] 
    bezier_list = []
    interp_list = []

    for i in range(1,len(path) - 1):
        k = 2*i - 2

        
        is_curve_free = False
        while is_curve_free == False:
            D = distance(intermediate_q[k], intermediate_q[k+1])
            l = 1
            alpha = 0
            temp = []
            while alpha < 1:
                # coleta amostras entre k e k+1
                temp.append( bezier_curve(intermediate_q[k] , path[i] , intermediate_q[k+1] , alpha) )
                alpha = (l*sample_distance)/D
                l = l + 1
            
            for q in temp:
                # verifica se o caminho encontrado pela interpolacao esta livre
                free = True
                temp_free, message, _ = robot.check_free_configuration(q=q, htm=htm_base, obstacles=all_obs)
                free = free and temp_free

                # se o caminho nao esta livre, reduz a distancia entre os pontos adicionais e tenta novamente
                if temp_free == False:
                    prev_dir = (path[i] - intermediate_q[k])  /np.linalg.norm(path[i] - intermediate_q[k]  )
                    next_dir = (path[i] - intermediate_q[k+1])/np.linalg.norm(path[i] - intermediate_q[k+1])

                    intermediate_q[k]   = intermediate_q[k]   + prev_dir*0.05
                    intermediate_q[k+1] = intermediate_q[k+1] + next_dir*0.05
                    break
            
            # caminho livre
            if free:
                bezier_list.append(temp)
                is_curve_free = True
                break

    for i in range(1,len(path) - 1):
        k = 2*i - 2
        temp = []
        # coleatar amostras das retas entre k+1 e k+2
        if ( k < (2*len(path) - 6)):
            D = distance(intermediate_q[k + 1], intermediate_q[k+2])
            l = 0
            alpha = 0
            while alpha < 1:
                temp.append( (1-alpha)*(intermediate_q[k + 1]) + alpha*intermediate_q[k + 2] )
                l = l + 1
                alpha = (l*sample_distance)/D
        interp_list.append(temp)




    for l in range(0, round(len(bezier_list)-1)):
        sampled_path.extend(bezier_list[l])
        sampled_path.extend(interp_list[l])
    sampled_path.extend(bezier_list[-1])
    # amostras da primeira reta
    temp = []
    D = distance(path[0], intermediate_q[0])
    l = 0
    alpha = 0
    while alpha < 1:
        temp.append( (1-alpha)*(path[0]) + alpha*intermediate_q[0] )
        l = l + 1
        alpha = (l*sample_distance)/D  
    temp.extend(sampled_path)
    sampled_path = temp

    # coleta amostras da ultima reta
    D = distance(intermediate_q[2*len(path) - 5], path[ len(path) - 1])
    l = 0
    alpha = 0
    while alpha < 1:
        sampled_path.append( (1-alpha)*(intermediate_q[2*len(path) - 5]) + alpha*path[ len(path) - 1] )
        l = l + 1
        alpha = (l*sample_distance)/D

    new_path = [ ]
    new_path.append(sampled_path[0])
    for i in range( 0 , int(len(sampled_path)  - 2 )):
        if np.linalg.norm(new_path[-1] - sampled_path[i + 1]) > 1e-7:
            new_path.append(sampled_path[i+1])


    free = True
    
    # antes de retornar, verifica se o caminho final é seguro
    for q_c in new_path:
        # verifica todas as amostras do caminho final
        tempfree ,_ ,_ = robot.check_free_configuration(q=q, htm=htm_base, obstacles=all_obs)
        free = free and tempfree
        if free == False:
            print("\n\n\nO Caminho não está livre!\n\n\n")
            break
    if free:
        print("\n\n\nO Caminho está livre!\n\n\n")
        
    return new_path, intermediate_q


scenario_index = np.random.uniform(0, 2599, 0).astype(int)
scenario_index = np.append(scenario_index, [734]) #412, 734

for index in scenario_index:
    print(f"==============================\n\tScenario {index}\n==============================\n")
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)
    pathh = []
    pathh , ext_path = rrt_path_planning(robot, all_obs, q0, htm_base, htm_tg, max_iter = 500, tolerance = 0.35 , goal_bias = 0.35, bias_decay_rate = 0.0001, num_of_trys = 50)
    sample_path, intermediate_q = generate_path_samples(pathh,sample_distance=0.01, htm_base=htm_base, all_obs=all_obs,robot=robot)

    if pathh != []:
        draw_balls(ext_path      , robot, sim, color="green" , radius = 0.01)
        draw_balls(sample_path   , robot, sim, color="red"  , radius = 0.01)
        draw_balls(pathh         , robot, sim, color="white", radius = 0.02)
        draw_balls(intermediate_q, robot, sim, color="cyan" , radius = 0.02)

        q_dot_hist, t_hist = move_robot_through_path(robot=robot, q_list=sample_path , base_frame=htm_base, t = 0, dt = 0.01, task_tol = 0.01 ,consider_path_orientation = True)
        sim.run()
        sim.save("/home/pedro/uaibot_files/info_storage/", "mov" + str(index))



print("\nSimulation completed.\n")