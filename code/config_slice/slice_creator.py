import multiprocessing as mp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

def master_dist(q, robot, obstacles, eps_to_obs=0.015, h_to_obs=0.02, h_merge = 0.3, compute_grad=False):


    q_min = robot.joint_limit[:,0]
    q_max = robot.joint_limit[:,1]
    n = np.shape(q_min)[0]
       
    D_tot = np.matrix(np.zeros((0,1)))
    grad_tot = np.matrix(np.zeros((0,n)))
    

    
    for obs in obstacles:
        dr_obj =robot.compute_dist(obj = obs, q = q, eps=eps_to_obs, h=h_to_obs, no_iter_max=2000, tol=1e-5)
        D_tot = np.vstack( (D_tot, np.sqrt(dr_obj.dist_vect)) )
        grad_tot = np.vstack( (grad_tot,  dr_obj.jac_dist_mat) )
    

    # dr_auto =robot.compute_dist_auto(q = q, eps=eps_auto, h=h_auto, no_iter_max=300)    
    # D_tot = np.vstack( (D_tot, 10*dr_auto.dist_vect) )
    # grad_tot = np.vstack( (grad_tot, 10*dr_auto.jac_dist_mat) )


    A_joint = np.matrix(np.vstack(  (np.identity(n), -np.identity(n))  ))
    b_joint = np.matrix(np.vstack(  (q-(q_min) , (q_max) - q)  ))  
    D_tot = np.vstack( (D_tot, b_joint) )
    grad_tot = np.vstack( (grad_tot, A_joint) )
    
    
    ############
    # Ensure D_tot is column vector
    D_tot[D_tot<0] = 0
    D_tot = D_tot.reshape(-1, 1)  # shape: (m, 1)
    D_min = np.min(D_tot)
    D_tot = D_tot+1e-6

    # Step 1: Compute s_i = 1 / D_i^(1/h)
    s_i = np.power(D_min/D_tot, 1/h_merge)  # shape: (m, 1)

    # Step 2: Compute S = sum_i s_i
    S = np.sum(s_i)+1e-6  # scalar
    


    # Step 3: Compute F = S^(-h)
    F = D_min*(S ** (-h_merge))

    grad_F = []
    
    if compute_grad:
        # Step 4: Compute each term in the gradient sum
        # Each grad D_i is in grad_tot[i, :]
        coeffs = np.power(D_min/D_tot, (1/h_merge + 1))  # shape: (m, 1)
        weighted_grads = coeffs.T * grad_tot  # shape: (m, n), row-wise multiplication

        # Sum over i (rows)
        grad_F = (1 / S) ** (h_merge + 1) * np.sum(weighted_grads, axis=0)  # shape: (n,)    
    

    return F, grad_F



def worker_main(task_queue, result_queue, robot, obstacles, eps_to_obs, h_to_obs, h_merge):
    """
    Worker process: creates robot and obstacles ONCE and then listens for tasks.
    """


    while True:
        task = task_queue.get()

        if task == 'STOP':
            #print("[Worker] Stopping.")
            break

        (START_IDX, FINAL_IDX, NO_DIV, q_base, q_min, q_max, i1, i2) = task
        
        mat = np.zeros((FINAL_IDX-START_IDX, NO_DIV+1))
        
        for id_2 in range(START_IDX, FINAL_IDX, 1):
            q = np.matrix(q_base)
            q[i2,0] = q_max[i2, 0] - (id_2/NO_DIV)*(q_max[i2, 0] - q_min[i2, 0])
            
            for id_1 in range(NO_DIV+1):
                q[i1,0] = q_min[i1, 0] + (id_1/NO_DIV)*(q_max[i1, 0] - q_min[i1, 0])
                
                fun_val, _ = master_dist(q, robot, obstacles, eps_to_obs, h_to_obs, h_merge)
                mat[id_2-START_IDX,id_1] = fun_val

 

        result_queue.put([START_IDX, mat])

def plot_heatmap(mat_stacked, q_min, q_max, i1, i2):
    total_rows, total_cols = mat_stacked.shape

    y_vals = np.linspace(q_min[i1, 0], q_max[i1, 0], total_rows)
    x_vals = np.linspace(q_min[i2, 0], q_max[i2, 0], total_cols)

    extent = [x_vals[0], x_vals[-1], y_vals[-1], y_vals[0]]  # Note: Flip y to match image orientation

    plt.imshow(mat_stacked, extent=extent, aspect='auto', origin='upper', cmap='viridis')
    plt.colorbar(label='Value')
    plt.xlabel(f'q[{i2}]')
    plt.ylabel(f'q[{i1}]')
    plt.title('Heatmap of mat_stacked')
    plt.show()
   
   
def upsample_contour(contour, N):
    if len(contour) == 0:
        return []  # Retorna uma lista vazia caso o contorno esteja vazio
    
    contour = np.array(contour)
    upsampled = []

    for i in range(len(contour) - 1):
        p1 = contour[i]
        p2 = contour[i + 1]

        xs = np.linspace(p1[0], p2[0], N + 1)[:-1]  # exclude last point to avoid duplicates
        ys = np.linspace(p1[1], p2[1], N + 1)[:-1]

        upsampled.extend(list(zip(xs, ys)))

    # Add the final point
    upsampled.append(tuple(contour[-1]))

    return [list(p) for p in upsampled]
 
 
def worker_deform(task_queue, result_queue, robot, obstacles, eps_to_obs, h_to_obs, h_merge):
    """
    Worker process: creates robot and obstacles ONCE and then listens for tasks.
    """


    while True:
        task = task_queue.get()

        if task == 'STOP':
            #print("[Worker] Stopping.")
            break

        (points, q_base, val, i1, i2) = task
        
        corrected_points = []
        for k in range(len(points)):
            
            q = np.matrix(q_base)
            q[i1,0] = points[k][0]
            q[i2,0] = points[k][1]
            F, grad_F = master_dist(q, robot, obstacles, eps_to_obs, h_to_obs, h_merge, compute_grad=True)
            
            error_ant = abs(F-val)
            if error_ant>=0.001:
                
                cont = True
                iterations = 0
                while cont:
                    error = F-val
                    eta = 0.1
                    q[i1,0] += -eta*error*grad_F[0,i1]
                    q[i2,0] += -eta*error*grad_F[0,i2]

                    F, grad_F = master_dist(q, robot, obstacles, eps_to_obs, h_to_obs, h_merge, compute_grad=True)
                    error = F-val
                    
                    iterations+=1
                    
                    cont = iterations<5 and abs(error)<error_ant
                    error_ant = abs(error)
                    
                corrected_points.append([q[i1,0], q[i2,0]])    
            else:
                corrected_points.append(points[k])

 
        result_queue.put(corrected_points)
        
def find_level_curve(robot, obstacles, val, q_base, index, q_min, q_max, eps_to_obs, h_to_obs, h_merge):
    """
    Runs the parallel computation using persistent processes with shared robot/obstacle state.
    """

    
    i1 = index[0]
    i2 = index[1]

    
    NO_CPU = mp.cpu_count()
    NO_DIV = 55
    DIV_PER_CPU = NO_DIV//NO_CPU
    
    task_list = []
    
    
    for i in range(NO_CPU):
        start_idx = i*DIV_PER_CPU
        final_idx = (i+1)*DIV_PER_CPU if i < NO_CPU-1 else NO_DIV+1
        task_list.append((start_idx, final_idx, NO_DIV, q_base, q_min, q_max, i1, i2))

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Spawn workers
    workers = []

    for _ in range(NO_CPU):
        p = mp.Process(target=worker_main, args=(
            task_queue, result_queue, robot, obstacles, eps_to_obs, h_to_obs, h_merge))
        p.start()
        workers.append(p)

    # Dispatch tasks
    for task in task_list:
        task_queue.put(task)

    # Collect results
    list_results = []
    for _ in task_list:
        list_results.append(result_queue.get())


    # Stop workers
    for _ in workers:
        task_queue.put('STOP')
    for p in workers:
        p.join()


    list_sorted = sorted(list_results, key=lambda x: x[0])

    matrices = [item[1] for item in list_sorted]
    mat_stacked = np.vstack(matrices)  


    # plot_heatmap(mat_stacked, q_min, q_max, i1, i2)
    
    total_rows, total_cols = mat_stacked.shape

    y_vals = np.linspace(q_max[i2, 0], q_min[i2, 0], total_rows)
    x_vals = np.linspace(q_min[i1, 0], q_max[i1, 0], total_cols)

    X, Y = np.meshgrid(x_vals, y_vals)


    fig, ax = plt.subplots()
    CS = ax.contour(X, Y, mat_stacked, levels=[val])


    contours_list = []

    # colors = plt.cm.tab10.colors  # A set of 10 distinct colors; can be changed to any colormap


    # fig, ax = plt.subplots()

    # color_idx = 0

    for i in range(len(CS.allsegs)):
        seg_group = CS.allsegs[i]
        for seg in seg_group:
            contour = [[x, y] for x, y in seg]
            contours_list += upsample_contour(contour,2)


    #Correct the points
    task_list = []
    
    DIV_PER_CPU = len(contours_list)//NO_CPU
    for i in range(NO_CPU):
        first_index = i*DIV_PER_CPU
        last_index = (i+1)*DIV_PER_CPU if i<NO_CPU-1 else len(contours_list)
        task_list.append((contours_list[first_index:last_index], q_base, val, i1, i2))    
    

    # Create queues
    task_queue = mp.Queue()
    result_queue = mp.Queue()

    # Spawn workers
    workers = []

    for _ in range(NO_CPU):
        p = mp.Process(target=worker_deform, args=(
            task_queue, result_queue, robot, obstacles, eps_to_obs, h_to_obs, h_merge))
        p.start()
        workers.append(p)

    # Dispatch tasks
    for task in task_list:
        task_queue.put(task)

    # Collect results
    list_results = []
    for _ in task_list:
        list_results+=result_queue.get()


    # # Stop workers
    # for _ in workers:
    #     task_queue.put('STOP')
    # for p in workers:
    #     p.join()
        
            
    # corrected_points = []
    # for k in range(len(contours_list)):
        
 
    #     q = np.matrix(q_base)
    #     q[i1,0] = contours_list[k][0]
    #     q[i2,0] = contours_list[k][1]
    #     F, grad_F = master_dist(q, robot, obstacles, eps_to_obs, h_to_obs, h_merge, compute_grad=True)
        
    #     error_ant = abs(F-val)
    #     if error_ant>=0.001:
            
    #         cont = True
    #         iterations = 0
    #         while cont:
    #             error = F-val
    #             eta = 0.1
    #             q[i1,0] += -eta*error*grad_F[0,i1]
    #             q[i2,0] += -eta*error*grad_F[0,i2]

    #             F, grad_F = master_dist(q, robot, obstacles, eps_to_obs, h_to_obs, h_merge, compute_grad=True)
    #             error = F-val
                
    #             iterations+=1
                
    #             cont = iterations<5 and abs(error)<error_ant
    #             error_ant = abs(error)
                
    #         corrected_points.append([q[i1,0], q[i2,0]])    
    #     else:
    #         corrected_points.append(contours_list[k])
            
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Contours with Different Colors')
    # plt.show()


    
    return [p[0] for p in list_results], [p[1] for p in list_results]


def draw_slice(robot, obstacles, q, indexes, value, score_fun, path_q, 
               eps_to_obs=0.015, h_to_obs=0.02, h_merge=0.3, save_path=None, track=None):
    """
    Main entry point — unchanged from the user perspective, with options to save the plot as PNG and plot track.
    
    Parameters:
    - track: Optional. Uma lista ou tupla com duas listas/arrays: track[0] com valores para q[indexes[0]] e track[1] para q[indexes[1]].
    """
    q_min = robot.joint_limit[:, 0]
    q_max = robot.joint_limit[:, 1]

    X_obs, Y_obs = find_level_curve(robot, obstacles, value, q, indexes, q_min, q_max, eps_to_obs, h_to_obs, h_merge)

    xmin, xmax = q_min[indexes[0],0]-0.05, q_max[indexes[0],0]+0.05
    ymin, ymax = q_min[indexes[1],0]-0.05, q_max[indexes[1],0]+0.05

    n_points = 200
    x = np.linspace(xmin, xmax, n_points)
    y = np.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(x, y)

    def score_fun_sliced(_x,_y):
        q_temp = np.matrix(q)
        q_temp[indexes[0],0] = _x
        q_temp[indexes[1],0] = _y
        return score_fun(q_temp)

    Z = np.vectorize(score_fun_sliced)(X, Y)

    Z_min = np.min(Z)
    Z = Z_min + np.sqrt(Z - Z_min)
    
    plt.contour(X, Y, Z, levels=10, colors='red', linewidths=0.8)
    cf = plt.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(cf)

    plt.xlim([xmin, xmax])
    plt.ylim([ymin, ymax])

    plt.xlabel('q_'+str(indexes[0]))
    plt.ylabel('q_'+str(indexes[1]))

    # Ponto atual
    plt.scatter([q[indexes[0], 0]], [q[indexes[1], 0]], s=15, color='cyan', zorder=20)

    # Caminho (se existir)
    if path_q:
        plt.plot(
            [_q[indexes[0], 0] for _q in path_q],
            [_q[indexes[1], 0] for _q in path_q],
            color='cyan', linewidth=1, zorder=20
        )

    # Obstáculos
    plt.scatter(X_obs, Y_obs, s=4, color='black', zorder=10)

    # Track (trajetória)
    if track is not None:
        traj_x, traj_y = track

        # Linha da trajetória
        plt.plot(traj_x, traj_y, color='yellow', linewidth=1, zorder=25)

        # Pontos da trajetória
        plt.scatter(traj_x, traj_y, color='yellow', s=8, zorder=30)

        # Ponto inicial (círculo)
        plt.scatter(traj_x[0], traj_y[0], color='green', s=25, zorder=35, label='Início')

        # Ponto final (x)
        plt.scatter(traj_x[-1], traj_y[-1], color='red', s=40, marker='x', zorder=35, label='Fim')

        plt.legend()


    if save_path is not None:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

