import uaibot as ub
import numpy as np
from multiprocessing import Process, Event, Queue, Manager
from collections import deque

from CBFPathFollower import *
from reactive_control import * 
from deliberative_control import *
from ShadowReactivePlanner import *
from setup import *

class ReactivePlannerHybrid:
    def __init__(self, robot, all_obs, q0, htm_tg, htm_base, eps_task=1e-2, tmax = 15 , ghost_time_advance = 2 , dt = 0.01):
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
            eta_obs=1,
            eta_auto=0.6,
            eta_joint=0.6,
            eps=1e-3,
            kp=1
        )

        self.deliberative_planner = RRTStar(
            robot=robot,
            all_obs=all_obs,
            htm_base=htm_base,
            htm_tg=htm_tg,
            gamma=3.5,
            d_max=0.75,
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
            eta_obs=1.0,
            eta_auto=0.6,
            eta_joint=0.6,
            eps=1e-3,
            kp=1.5,
            dt=self.dt,
            ghost_time_advance=self.ghost_time_advance
        )

        self.__stopRRT = Event()


    def __process_deliberative(self, queue):
        while not self.__stopRRT.is_set():
            self.deliberative_planner.expand_tree()

        queue.put(self.deliberative_planner.nodes)  

    

    def run(self):
        queue = Queue()    
        process_delib = Process(target=self.__process_deliberative, args=(queue,))
        process_delib.start()
    
        
        i = 0
        u_real = self.path_follower.u
        r_real = self.path_follower.r
        while self.t <= self.tmax:
            #################################
            # Início da lógica de SimGhost  #
            #################################

            self.GhostPlanner.ghost_sim(self.t)

            #################################
            # Fim da lógica de SimGhost     #
            #################################

            #################################
            # Início da lógica de controle  #
            #################################

            t_real = self.t - self.ghost_time_advance
            if t_real > 0:
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
        temp = queue.get()
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
    print(f"size {len(controller.deliberative_planner.nodes)}")
    sim.add(controller.GhostPlanner.ghost_robot)
    store_info(sim, controller.qdot_hist, controller.rnorm_hist, controller.r_hist, controller.t_hist, "benchmark" + str(index))

hybrid_test(434)