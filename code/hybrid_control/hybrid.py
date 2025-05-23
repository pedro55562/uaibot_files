import uaibot as ub
import numpy as np
from multiprocessing import Process, Event, Queue, Manager
from collections import deque
import time

from CBFPathFollower import *
from deliberative_control import *
from ShadowReactivePlanner import *
from setup import *

class ReactivePlannerHybrid:
    def __init__(self, robot, all_obs, q0, htm_tg, htm_base, eps_task=1e-2, tmax = 5 , ghost_time_advance = 3 , dt = 0.01):
        self.robot = robot
        self.all_obs = all_obs
        self.htm_tg = htm_tg
        self.htm_base = htm_base
        self.q0 = q0
        self.n = robot.q.shape[0]
        self.stucked = False
        self.stop = False
        self.path = []  

        self.eps_task = eps_task

        self.t = 0
        self.ghost_time_advance = ghost_time_advance
        self.tmax = tmax
        self.dt = dt

        self.qdot_hist = []                     
        self.r_hist = []
        self.rnorm_hist = []                       
        self.t_hist = [] 

        self.path_follower = CBFPathFollower(
            robot=robot,
            obstacles=all_obs,
            htm_tg=htm_tg,
            dj_lim=np.deg2rad(2),
            d_lim=0.01,
            dlim_auto=0.002,
            eta_obs=0.2,
            eta_auto=0.6,
            eta_joint=0.6,
            eps=1e-3,
            kp=2
        )

        self.deliberative_planner = RRTStar(
            robot=robot,
            all_obs=all_obs,
            htm_base=htm_base,
            htm_tg=htm_tg,
            goal_bias=0.4,
            gamma=2,
            d_max=1.2,
            d_min=0.2,
            max_iter=2000
        )

        self.GhostPlanner = GhostReactivePlanner(
            robot=robot,
            obstacles=all_obs,
            htm_tg=htm_tg,
            htm_base=htm_base,
            dj_lim=np.deg2rad(2),
            d_lim=0.01,
            dlim_auto=0.002,
            eta_obs=0.2,
            eta_auto=0.6,
            eta_joint=0.6,
            eps=1e-3,
            kp=2,
            dt=self.dt,
            ghost_time_advance=self.ghost_time_advance
        )

        self.__stopRRT = Event()
        self.local_minimum_detected = Event()

    def __process_deliberative(self, queue_out, queue_in):
        while not self.__stopRRT.is_set():
            self.deliberative_planner.expand_tree()

            if self.local_minimum_detected.is_set():
                print("local_minimum_detected")
                best_node = self.deliberative_planner.best_node
                buffer_q, buffer_u, buffer_r, buffer_t = queue_in.get()
                path_to_best = self.deliberative_planner.find_path(buffer_q, buffer_r, best_node)
                queue_out.put(path_to_best)

        queue_out.put(self.deliberative_planner.nodes)
        print("__process_deliberative finalizado corretamente")

    

    def run(self):
        queue_out = Queue()    
        queue_in  = Queue()
        process_delib = Process(target=self.__process_deliberative, args=(queue_out,queue_in))
        process_delib.start()
        time.sleep(5)
        
        i = 0
        u_real = self.path_follower.u
        r_real = self.path_follower.r
        while self.t <= self.tmax:
            #################################
            # Início da lógica de SimGhost  #
            #################################

            self.GhostPlanner.ghost_sim(self.t)

            # se detectou um min. local
            if  self.GhostPlanner.detect_local_minimum():
                self.local_minimum_detected.set()
                buffers = (self.GhostPlanner.buffer_q, self.GhostPlanner.buffer_u, self.GhostPlanner.buffer_r, self.GhostPlanner.buffer_t)
                queue_in.put( buffers )
                path_to_best = queue_out.get()
                self.ghost_time_advance = self.t + self.ghost_time_advance
                self.GhostPlanner.follow_path(path_to_best)
            #################################
            # Fim da lógica de SimGhost     #
            #################################

            #################################
            # Início da lógica de controle  #
            #################################

            if self.t >= self.ghost_time_advance:
                # segue o caminho com um cont. proporcional + restricao da CBF
                self.path_follower.compute_control(self.GhostPlanner.buffer_q[0])
                u_real = self.path_follower.u
                r_real = self.path_follower.r
                set_configuration_speed(
                    robot = self.path_follower.robot,
                    q_dot = u_real, 
                    t     = self.t,
                    dt    = self.dt
                )


            #################################
            # Fim da lógica de controle     #
            #################################

            self.qdot_hist.append(np.rad2deg(u_real))                   
            self.r_hist.append(r_real)
            self.rnorm_hist.append(np.linalg.norm(r_real))                    
            self.t_hist.append(self.t)
            i += 1
            self.t = i*self.dt

        self.__stopRRT.set()
        self.local_minimum_detected.clear()
        temp = queue_out.get()
        process_delib.join()
        self.deliberative_planner.nodes = temp



def hybrid_test(index):
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)
    
    controller = ReactivePlannerHybrid(
        robot=robot,
        all_obs=all_obs,
        q0=q0,
        htm_tg=htm_tg,
        htm_base=htm_base
    )
    controller.run()
    sim.add(controller.GhostPlanner.ghost_robot)
    store_info(sim, controller.qdot_hist, controller.rnorm_hist, controller.r_hist, controller.t_hist, "benchmark" + str(index))

hybrid_test(734) #1148