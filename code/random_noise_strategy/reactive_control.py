import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
from setup import *

def get_configuration(robot):
    return robot.q

def set_configuration_speed(robot, q_dot, t, dt):
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)

def fun_F(r):
    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        # suaviza o resultado da task function => suaviza o controle quando r proximo de 0
        f[j, 0] = np.sign(r[j, 0]) * np.sqrt(np.abs(r[j, 0]))
    return f


def constrained_control(robot, obstacles, htm_tg, qq = None):
    # Número de juntas do robô
    n = len(robot.links)
    u = robot.q*0
    
    # Define a configuração atual (usa qq se fornecida, senão usa robot.q)
    if qq is None:
        q = robot.q
    else:
        q = qq
    
    # Calcula o erro da tarefa e o Jacobiano
    r, Jr = robot.task_function(htm_des = htm_tg, q = q)

    # distancia minima
    dj_lim  = (np.pi/180)*2
    d_lim   = 0.005
    dlim_auto = 0.002
    

    # quanto maior o valor, mais o controlador vai ser agressivo em relacao a essa restricao
    eta_obs   = 0.3
    eta_auto  = 0.6
    eta_joint = 0.6

    # peso no prob de otimização minimizar a norma de qdot
    eps = 1e-3
    kp  = 1

    # Limite de junta
    q_min = robot.joint_limit[:,0]
    q_max = robot.joint_limit[:,1]
    Aj_min =  np.identity(n)
    Aj_max = -np.identity(n)
    Bj_min = - eta_joint*( (q - q_min) - dj_lim)
    Bj_max = - eta_joint*( (q_max - q) - dj_lim)

    # Configuração das restrições de colisão com obstáculos
    Ad_obj = np.matrix(np.zeros((0,n)))
    Bd_obj = np.matrix(np.zeros((0,1)))
    for ob in obstacles: 
        ds = robot.compute_dist(ob, q=q,h=0.1, eps=0.1)
        Ad_obj = np.matrix(np.vstack((Ad_obj, ds.jac_dist_mat)))
        Bd_obj = np.matrix(np.vstack((Bd_obj, ds.dist_vect - d_lim)))
    Bd_obj = - eta_obs*Bd_obj

    # Configuração das restrições de auto-colisão
    dist_auto = robot.compute_dist_auto(q=q)
    A_auto = dist_auto.jac_dist_mat
    B_auto = -eta_auto*(dist_auto.dist_vect - dlim_auto)
    
    # Monta o problema de otimização quadrática
    A = np.matrix(np.vstack((Aj_min, Aj_max, Ad_obj, A_auto)))
    b = np.matrix(np.vstack((Bj_min, Bj_max, Bd_obj, B_auto)))
    H = 2*(Jr.transpose() * Jr + eps*np.identity(n))
    f = kp * Jr.transpose() * fun_F(r)

    # Converte para o formato adequado para o solver
    H = np.array(H, dtype=np.float64)
    f = np.array(f, dtype=np.float64)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    # Resolve o problema de otimização para obter as velocidades de junta
    try:
        u = ub.Utils.solve_qp(H, f, A, b)
    except:
        v =0

    return u, r

def constrained_control_joint(robot, obstacles, goal_tg, htm_tg):
    n = len(robot.links)
    u = robot.q*0
    q = robot.q 

    r = robot.q - goal_tg
    Jr = np.identity(n)
    r_temp , _ = robot.task_function(htm_des = htm_tg, q = q)
    # distância mínima
    dj_lim  = (np.pi/180)*2
    d_lim   = 0.005
    dlim_auto = 0.002
    

    # quanto maior o valor, mais o controlador vai ser agressivo em relacao a essa restricao
    eta_obs   = 0.3
    eta_auto  = 0.6
    eta_joint = 0.6

    # peso no prob de otimização minimizar a norma de qdot
    eps = 1e-3
    kp = 2
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
        # calcula distancia entre o robot e o objeto, e cria a restrição
        ds = robot.compute_dist(ob, h=0.1, eps=0.1)
        Ad_obj = np.matrix(np.vstack((Ad_obj, ds.jac_dist_mat      )))
        Bd_obj = np.matrix(np.vstack((Bd_obj, ds.dist_vect - d_lim )))
    Bd_obj = - eta_obs*Bd_obj

    # auto colisao
    dist_auto = robot.compute_dist_auto()
    A_auto = dist_auto.jac_dist_mat
    B_auto = -eta_auto*(dist_auto.dist_vect - dlim_auto)
    
    #cria o prob. de otimizacao
    A =    np.matrix(np.vstack( (Aj_min, Aj_max, Ad_obj, A_auto) ) )
    b =    np.matrix(np.vstack( (Bj_min, Bj_max, Bd_obj, B_auto) ) )
    H = 2*(Jr.transpose() * Jr + eps*np.identity(n))
    f = kp * Jr.transpose() * fun_F(r)

    H = np.array(H, dtype=np.float64)
    f = np.array(f, dtype=np.float64)
    A = np.array(A, dtype=np.float64)
    b = np.array(b, dtype=np.float64)

    u = ub.Utils.solve_qp(H, f, A, b)


    return u, r_temp

# Função para encontrar uma configuração livre de colisão que alcance o alvo 
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

# Função para simular a evolução do sistema a partir de uma configuração inicial
# e calcular um score baseado no erro residual
def sim_reactive(robot, q_init, obstacles, htm_tg):
    dt = 0.01
    q_temp = q_init
    q_dot_list = []
    for k in range(0, 500):
        qdot , r = constrained_control(robot, obstacles, htm_tg, qq= q_temp)
        q_dot_list.append(qdot)
        q_temp = q_temp + qdot*dt
    return np.linalg.norm(r) , q_temp , q_dot_list

# Verifica se o caminho entre duas configurações está livre de colisões
def is_path_free(q_near, q_new, robot, htm_base, all_obs, sample_rate = 0.03):
    # verifica se o caminho entre q_near e q_new é livre
    # utilizando a interpolação linear
    num_samples = int(np.linalg.norm(q_near - q_new) / sample_rate)
    if num_samples < 1:
        num_samples = 5
    for i in range(1, num_samples):
        alpha = i / num_samples  
        q_interp = (1 - alpha) * q_near + alpha * q_new 
        verify, _, _ = robot.check_free_configuration(q=q_interp, htm=htm_base, obstacles=all_obs)
        if verify == False:
            return False 
    return True 

def plot_and_save(data, t_hist ,ylabel, title, filename):
    plt.figure()
    if data.ndim == 1:
        plt.plot(t_hist, data, label=ylabel)
    else:
        for i in range(data.shape[1]):
            plt.plot(t_hist, data[:, i], label=f"{ylabel}[{i+1}]")

# vai com controle reativo para tentar alcançar o alvo
# detecta quando o controle reativo esta travado
# gera configs aleatorias para escapar do minimo
# escolhe a melhor config simulando o controle reativo
# vai para a melhor configuracao e volta para o reativo 
def reactive_test(index):
    dt = 0.01                          
    t = 0
    qdot_hist = []                     
    r_hist = []                       
    t_hist = []                        
    min_found = False                 
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)
    n = robot.q.shape[0]              
    r, Jr = robot.task_function(htm_des = htm_tg, q = robot.q)
    i = 0 
    m = 0   
    
    # Loop principal da simulação (até 2000 passos)
    while i < 3000:
        t = dt*i
        i = i + 1
        r_dot = 1
        # calcula a media das ultimas derivadas de r
        if len(r_hist) > 15:
            r_dot = 0
            for k in range(1,11):
                r_dot += np.linalg.norm(r_hist[-k] - r_hist[-k-1])
            r_dot = r_dot/10
        
        # Detecta se o controle reativo falhou
        if r_dot < 1e-3 and np.linalg.norm(r) > 0.08 and m > 50:
            print("\n\n REATIVO FALHOU \n\n")
            print("||r|| = " + str( np.linalg.norm(r) ))
            min_found = True
            
            eps = .25
            q_temp_list = []
            k = 0
            q_temp = robot.q * 0
            while k < 10:
                # config aleatoria
                rand = np.random.uniform(-np.pi, np.pi, size=(n,1))
                print(k)
                # da um passo na direcao aleatoria
                q_temp = robot.q + np.random.uniform(eps, 2) * rand/np.linalg.norm(rand)

                # Verifica se o caminho para a configuração é livre de colisão
                # assim o controle reativo no espaço de juntas consegue chegar nela sem dificuldades 
                if is_path_free(robot.q, q_temp, robot, htm_base, all_obs, sample_rate=0.03) and robot.check_free_configuration(q=q_temp, htm=htm_base, obstacles=all_obs):
                    q_temp_list.append(q_temp)
                    k += 1
                

            # Avalia as configurações candidatas e seleciona a melhor
            best_r = 1000
            best_q = robot.q
            for q_test in q_temp_list:
                # Simula o controle a partir desta configuração
                # para verificar se melhora a funcao de tarefa
                norm_r, qqte, q_dot_list = sim_reactive(robot=robot, q_init=q_test, obstacles=all_obs, htm_tg=htm_tg)
                if norm_r < best_r:
                    best_q = q_test
                    best_r = norm_r
                    best_qqte = qqte
                    if best_r < 0.2:
                        break
            # verifica se a config encontrada pelo sim_reactive é realmente melhor
            rtemp, _ = robot.task_function(htm_des = htm_tg, q = best_qqte)     
            if np.linalg.norm(r_hist[-1]) < np.linalg.norm(rtemp):
                continue
            print("||best_r|| = " + str(np.linalg.norm(rtemp)))
            print(" best_qqte = " + str(best_qqte.T))
            
        # move o robo para escapar do min. local
        if min_found:
            k = 0
            # Move para a configuração melhor usando o controle reativo no espaço de configuracao
            while np.linalg.norm(best_q - robot.q) > .001 and k < 1500:   
                qdot, r = constrained_control_joint(robot=robot, obstacles=all_obs, goal_tg=best_q, htm_tg=htm_tg)
                k +=1
                
                set_configuration_speed(robot, qdot, t, dt)
                qdot_hist.append(np.degrees(qdot))
                r_hist.append(r)
                t_hist.append(t)
                t = i*dt
                i+= 1
                
            m = 0
            min_found = False
            # executa o controle reativo normal para chegar a best_qqte
            for qd in q_dot_list:
                set_configuration_speed(robot, qd, t, dt)
                current_q = robot.q
                current_r, _ = robot.task_function(htm_des = htm_tg, q = current_q) 
                qdot_hist.append(np.degrees(qd))
                r_hist.append(current_r)
                t_hist.append(t)
                t = i*dt
                i+= 1
            test_q = robot.q
            rtemp, _ = robot.task_function(htm_des = htm_tg, q = test_q)     
            print("||reached r|| = " + str(np.linalg.norm(rtemp)))
            print("  reached q   = " + str(test_q.T))
        else:
            # Controle reativo normal
            #try:
            qdot, r = constrained_control(robot=robot, obstacles=all_obs, htm_tg=htm_tg)
            print("||r|| = " + str(np.linalg.norm(r)))
            #except ValueError:
            #    continue

        r, _ = robot.task_function(htm_des=htm_tg)
        if np.linalg.norm(r) < 0.005:
            print("goal reached!")
            break


        m += 1
        set_configuration_speed(robot, qdot, t, dt)
        qdot_hist.append(np.degrees(qdot))
        r_hist.append(r)
        t_hist.append(t)

    r_norm = []
    for r in r_hist:
        r_norm.append(np.linalg.norm(r))
    store_info(sim, qdot_hist, r_norm ,r_hist, t_hist, "benchmark" + str(index))

scenario_index = np.random.uniform(0, 2599, size = 0).astype(int)
scenario_index = np.append(scenario_index, [1148]) 
for index in scenario_index:
    print(index)
    reactive_test(index)



