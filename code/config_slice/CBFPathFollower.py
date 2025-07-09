import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
from setup import *

def set_configuration_speed(robot, q_dot, t, dt):
    q_dot = q_dot.reshape(-1, 1)
    q_next = robot.q + q_dot*dt
    robot.add_ani_frame(time = t+dt, q = q_next)

def score(r):
    return np.sum(np.power(np.abs(r), 3/2))

def fun_G(_r, _param_k):
    m = np.shape(_r)[0]
    out = np.matrix(np.zeros((m,1)))
    for i in range(m):
        out[i,0] = -_param_k * np.sign(_r[i,0]) * np.sqrt(np.abs(_r[i,0]))
    return out

def fun_F(r, kp):
    f = np.matrix(r)
    for j in range(np.shape(r)[0]):
        f[j, 0] = kp * np.sign(r[j, 0]) * np.sqrt(np.abs(r[j, 0]))
    return f

class CBFPathFollower:
    def __init__(self, robot : ub.Robot, obstacles, htm_tg,
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
        self.r, _ = self.robot.task_function(htm_tg = self.htm_tg, q = self.robot.q)
        self.n = len(self.robot.links)

    def _build_joint_constraints(self, q):
        q_min = self.robot.joint_limit[:, 0]
        q_max = self.robot.joint_limit[:, 1]
        Aj_min = np.identity(self.n)
        Aj_max = -np.identity(self.n)
        Bj_min = -self.eta_joint * ((q - q_min) - self.dj_lim)
        Bj_max = -self.eta_joint * ((q_max - q) - self.dj_lim)

        return Aj_min, Aj_max, Bj_min, Bj_max

    def _build_obstacle_constraints(self, q):
        Ad_obj = np.zeros((0, self.n))
        Bd_obj = np.zeros((0, 1))
        for obj in self.obstacles:

            ds = self.robot.compute_dist(obj=obj)
            
            Ad_obj = np.vstack((Ad_obj, ds.jac_dist_mat))
            Bd_obj = np.vstack((Bd_obj, ds.dist_vect - self.d_lim))
        Bd_obj = -self.eta_obs * Bd_obj
        return Ad_obj, Bd_obj

    def _build_auto_constraints(self, q):
        dist_auto = self.robot.compute_dist_auto(q=q)
        A_auto = dist_auto.jac_dist_mat
        B_auto = -self.eta_auto * (dist_auto.dist_vect - self.dlim_auto)
        return A_auto, B_auto

    def _build_barrier_constraints(self, q):
        Aj_min, Aj_max, Bj_min, Bj_max = self._build_joint_constraints(q)
        Ad_obj, Bd_obj = self._build_obstacle_constraints(q)
        A_auto, B_auto = self._build_auto_constraints(q)
        A = np.vstack((Aj_min, Aj_max, Ad_obj, A_auto))
        b = np.vstack((Bj_min, Bj_max, Bd_obj, B_auto))
        return A, b

    def compute_control(self, j1, j2, qd):
        '''
        Resolve o QP:

        min ||u - un||^2 + eps * ||u||^2
        com un = kp * (qd - q)

        Forma quadrática:
        min 0.5 u^T H u + f^T u
        H = 2 * (1 + eps) * I 
        f = -2 * un
        '''
        u = self.robot.q * 0
        q = self.robot.q
        r, Jr = self.robot.task_function(htm_tg = self.htm_tg, q = q)
        A, b = self._build_barrier_constraints(q)

        # calcula un - cont. proporcional
        un = fun_G(self.robot.q - qd, self.kp)
        H_full = 2 * (1 + self.eps) * np.identity(self.n)
        f_full = -2 * np.array(un).flatten()

        indices = [j1, j2]
        H_red = H_full[np.ix_(indices, indices)]
        f_red = f_full[indices]
        A_red = A[:, indices]

        try:
            u_red = ub.Utils.solve_qp(
                np.array(H_red, dtype=np.float64),
                np.array(f_red, dtype=np.float64),
                np.array(A_red, dtype=np.float64),
                np.array(b, dtype=np.float64)
            )
        except Exception as e:
            u_red = [0.0, 0.0]

        u_full = np.zeros(self.n)
        u_full[j1] = u_red[0]
        u_full[j2] = u_red[1]
        self.u = u_full
        self.r = r

    def compute_control_task_func(self, j1, j2, qq=None):
        q = self.robot.q if qq is None else qq

        r, Jr = self.robot.task_function(htm_tg = self.htm_tg, q = q)
        
        r  =  r[0:3, :]
        Jr = Jr[0:3, :]
        
        A, b = self._build_barrier_constraints(q)        
        H_full = 2 * (Jr.T @ Jr + self.eps * np.identity(self.n))
        f_full = Jr.T @ fun_F(r, self.kp)

        indices = [j1, j2]
        H_red = H_full[np.ix_(indices, indices)]
        f_red = f_full[indices]
        A_red = A[:, indices]

        try:
            u_red = ub.Utils.solve_qp(
                np.array(H_red, dtype=np.float64),
                np.array(f_red, dtype=np.float64),
                np.array(A_red, dtype=np.float64),
                np.array(b, dtype=np.float64)
            )
        except Exception as e:
            u_red = [0.0, 0.0]

        u_full = np.zeros(self.n)
        u_full[j1] = u_red[0]
        u_full[j2] = u_red[1]
        self.u = u_full
        self.r = r




if "__main__" == __name__:
    robot, sim, all_obs, q0, htm_tg, htm_init = create_scenario()

    controller = CBFPathFollower(
        robot=robot,
        obstacles=all_obs,
        htm_tg=htm_tg,
        dj_lim=np.deg2rad(2),
        d_lim=1e-2,
        dlim_auto=5e-3,
        eta_obs=0.6,
        eta_auto=0.6,
        eta_joint=0.6,
        eps=1e-3,
        kp=1.0
    )

    t = 0
    dt = 0.01
    tmax_joint = 4


    pares_de_juntas = [ (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    for ji, jj in pares_de_juntas:
        t_junta = 0
        while t_junta < tmax_joint:

            controller.compute_control_task_func(j1=ji, j2=jj)
            u1 = controller.u
            set_configuration_speed(robot, u1, t, dt)
            t = t + dt
            t_junta = t_junta + dt

    sim.save(file_name = "teste_CBF")
