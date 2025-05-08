import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
from setup import *


class CBFPathFollower:
    def __init__(self, robot, obstacles, htm_tg,
                 dj_lim=np.deg2rad(2), d_lim=0.01, dlim_auto=0.002,
                 eta_obs=0.6, eta_auto=0.6, eta_joint=0.6,
                 eps=1e-3, kp=1.0):
        
        self.htm_tg = htm_tg
        self.robot = robot
        self.obstacles = obstacles
        self.dj_lim = dj_lim
        self.d_lim = d_lim
        self.dlim_auto = dlim_auto
        self.eta_obs = eta_obs
        self.eta_auto = eta_auto
        self.eta_joint = eta_joint

        self.eps = eps
        self.kp = kp

        self.u = robot.q * 0
        self.r , _ = self.robot.task_function(htm_des=self.htm_tg, q=self.robot.q)
        self.n = len(self.robot.links)

    def compute_control(self, qd):
        '''
        Resolve o QP:

        min ||u - un||^2 + eps * ||u||^2
        com un = kp * (qd - q), controle proporcional.

        Forma quadrática:
        min u^T H u + f^T u
        H = 2 * (1 + eps) * I, f = -2 * un
        '''


        u = self.robot.q * 0

        q = self.robot.q
        r, Jr = self.robot.task_function(htm_des=self.htm_tg, q=q)

        q_min = self.robot.joint_limit[:, 0]
        q_max = self.robot.joint_limit[:, 1]

        Aj_min = np.identity(self.n)
        Aj_max = -np.identity(self.n)

        Bj_min = -self.eta_joint * ((q - q_min) - self.dj_lim)
        Bj_max = -self.eta_joint * ((q_max - q) - self.dj_lim)

        Ad_obj = np.zeros((0,  self.n))
        Bd_obj = np.zeros((0, 1))

        for ob in self.obstacles:
            ds = self.robot.compute_dist(ob, q=q)
            Ad_obj = np.vstack((Ad_obj, ds.jac_dist_mat))
            Bd_obj = np.vstack((Bd_obj, ds.dist_vect - self.d_lim))
        Bd_obj = -self.eta_obs * Bd_obj

        dist_auto = self.robot.compute_dist_auto(q=q)
        A_auto = dist_auto.jac_dist_mat
        B_auto = -self.eta_auto * (dist_auto.dist_vect - self.dlim_auto)

        A = np.vstack((Aj_min, Aj_max, Ad_obj, A_auto))
        b = np.vstack((Bj_min, Bj_max, Bd_obj, B_auto))

        un = self.kp*(qd - self.robot.q)

        H = 2*(1 + self.eps)*np.identity(self.n)
        f = -2*np.array(un).flatten() 

        u = ub.Utils.solve_qp(
            np.array(H, dtype=np.float64),
            np.array(f, dtype=np.float64),
            np.array(A, dtype=np.float64),
            np.array(b, dtype=np.float64)
        )

        self.u = u
        self.r = r






