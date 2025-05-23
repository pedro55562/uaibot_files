import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt
from setup import *
from collections import deque

class GhostReactivePlanner:
    def __init__(self, robot, obstacles, htm_tg, htm_base,
                 dj_lim=np.deg2rad(2), d_lim=0.01, dlim_auto=0.002,
                 eta_obs=1.0, eta_auto=0.6, eta_joint=0.6,
                 eps=1e-3, kp=1.0, dt=0.01, ghost_time_advance = 1):


        self.htm_base = htm_base
        # Cria uma copia do robo real
        self.ghost_robot = ub.Robot.create_franka_emika_3(htm=htm_base,opacity=0.55)
        self.ghost_robot.add_ani_frame(time=0, q=robot.q)                           
        self.obstacles = obstacles
        self.htm_tg = htm_tg
        self.dj_lim = dj_lim
        self.d_lim = d_lim
        self.dlim_auto = dlim_auto
        self.eta_obs = eta_obs
        self.eta_auto = eta_auto
        self.eta_joint = eta_joint
        self.eps = eps
        self.kp = kp
        self.dt = dt
        self.ghost_time_advance = ghost_time_advance

        self.u = self.ghost_robot.q * 0
        self.r , _ = self.ghost_robot.task_function(htm_des=self.htm_tg, q=self.ghost_robot.q)
        self.n = len(self.ghost_robot.links)

        # Buffers para armazenar histórico de simulaço dos ultimos segundos
        buffer_size   = int(self.ghost_time_advance / self.dt)
        self.buffer_q = deque(maxlen=buffer_size)
        self.buffer_u = deque(maxlen=buffer_size)
        self.buffer_r = deque(maxlen=buffer_size)
        self.buffer_t = deque(maxlen=buffer_size)

        self.buffer_r_dot = deque(maxlen=10)
        self.r_dot_avg = 1

        self.path_to_follow = []
        self.follow = False
        self.q_goal = self.ghost_robot.q
        self.index  = 0

    def fun_F(self, r):
        f = np.copy(r)
        for j in range(f.shape[0]):
            f[j, 0] = np.sign(f[j, 0]) * np.sqrt(np.abs(f[j, 0]))
        return f

    def compute_control(self, qq=None):
        u = self.ghost_robot .q * 0

        q = self.ghost_robot .q if qq is None else qq
        r, Jr = self.ghost_robot .task_function(htm_des=self.htm_tg, q=q)

        q_min = self.ghost_robot .joint_limit[:, 0]
        q_max = self.ghost_robot .joint_limit[:, 1]

        Aj_min = np.identity(self.n)
        Aj_max = -np.identity(self.n)

        Bj_min = -self.eta_joint * ((q - q_min) - self.dj_lim)
        Bj_max = -self.eta_joint * ((q_max - q) - self.dj_lim)

        Ad_obj = np.zeros((0,  self.n))
        Bd_obj = np.zeros((0, 1))

        for ob in self.obstacles:
            ds = self.ghost_robot .compute_dist(ob, q=q)
            Ad_obj = np.vstack((Ad_obj, ds.jac_dist_mat))
            Bd_obj = np.vstack((Bd_obj, ds.dist_vect - self.d_lim))
        Bd_obj = -self.eta_obs * Bd_obj

        dist_auto = self.ghost_robot .compute_dist_auto(q=q)
        A_auto = dist_auto.jac_dist_mat
        B_auto = -self.eta_auto * (dist_auto.dist_vect - self.dlim_auto)

        A = np.vstack((Aj_min, Aj_max, Ad_obj, A_auto))
        b = np.vstack((Bj_min, Bj_max, Bd_obj, B_auto))

        H = 2 * (Jr.T @ Jr + self.eps * np.identity(self.n))
        f = self.kp * Jr.T @ self.fun_F(r)

        try:
            u = ub.Utils.solve_qp(
                np.array(H, dtype=np.float64),
                np.array(f, dtype=np.float64),
                np.array(A, dtype=np.float64),
                np.array(b, dtype=np.float64)
            )
        except ValueError as e:
            print(f"Erro ao resolver QP: {e}")
            print(f"Limites Bj_min: {Bj_min}")
            print(f"Limites Bj_max: {Bj_max}")
            print(f"Limites Bd_obj: {Bd_obj}")
            print(f"Limites B_auto: {B_auto}")
            raise

        self.u = u
        self.r = r


    def compute_control_follow(self):
        q = self.ghost_robot.q
        q_goal = self.q_goal.reshape(-1, 1)
        print(np.linalg.norm(q - q_goal))
        if (np.linalg.norm(q - q_goal) < 0.08):
            self.index = self.index + 1

            if self.index < len(self.path_to_follow):
                self.q_goal = self.path_to_follow[self.index]
                q_goal = self.q_goal.reshape(-1, 1)

        if (np.linalg.norm(q - self.path_to_follow[-1]) < 0.01):
            self.follow = False

        qd = q_goal
        u = self.ghost_robot.q * 0
        r, Jr = self.ghost_robot.task_function(htm_des=self.htm_tg, q=q)

        q_min = self.ghost_robot.joint_limit[:, 0]
        q_max = self.ghost_robot.joint_limit[:, 1]

        Aj_min = np.identity(self.n)
        Aj_max = -np.identity(self.n)

        Bj_min = -self.eta_joint * ((q - q_min) - self.dj_lim)
        Bj_max = -self.eta_joint * ((q_max - q) - self.dj_lim)

        Ad_obj = np.zeros((0,  self.n))
        Bd_obj = np.zeros((0, 1))

        for ob in self.obstacles:
            ds = self.ghost_robot.compute_dist(ob, q=q)
            Ad_obj = np.vstack((Ad_obj, ds.jac_dist_mat))
            Bd_obj = np.vstack((Bd_obj, ds.dist_vect - self.d_lim))
        Bd_obj = -self.eta_obs * Bd_obj

        dist_auto = self.ghost_robot.compute_dist_auto(q=q)
        A_auto = dist_auto.jac_dist_mat
        B_auto = -self.eta_auto * (dist_auto.dist_vect - self.dlim_auto)

        A = np.vstack((Aj_min, Aj_max, Ad_obj, A_auto))
        b = np.vstack((Bj_min, Bj_max, Bd_obj, B_auto))


        # calcula un - cont. proporcional
        un = self.kp*(qd - self.ghost_robot.q)

        # def. H e f do QP
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

    # Executa um passo da simulação do robô fantasma
    def ghost_sim(self, t_sim):
        if self.follow == False:
            self.compute_control()
            u_ghost = self.u
            q_ghost = self.ghost_robot.q
            r_ghost = np.linalg.norm(self.r)**2

        if self.follow == True:
            self.compute_control_follow()
            u_ghost = self.u
            q_ghost = self.ghost_robot.q
            r_ghost = np.linalg.norm(self.r)**2

        # aplica a acao de controle no robo simulado
        set_configuration_speed(
            robot = self.ghost_robot,
            q_dot = u_ghost, 
            t     = t_sim,
            dt    = self.dt
        )

        # Atualiza buffers 
        self.buffer_u.append(u_ghost)
        self.buffer_q.append(q_ghost)
        self.buffer_r.append(r_ghost)
        self.buffer_t.append(t_sim)
        
        # atualiza r_dot_avg - media da derivada da funcao de tarefa
        if len(self.buffer_r) > 2:
            r_dot = (self.buffer_r[-1] - self.buffer_r[-2])/self.dt
            self.buffer_r_dot.append( r_dot )
            self.r_dot_avg = sum(self.buffer_r_dot)/len(self.buffer_r_dot)


    def follow_path(self, path):
        self.path_to_follow = path
        self.follow = True
        self.index  = 0
        self.q_goal = path[self.index]

    # detecta se o robo ficou preso em um min. local
    def detect_local_minimum(self):
        if self.follow == True:
            return False
        if  self.r_dot_avg > -1e-2 and self.r_dot_avg < 0 and self.buffer_r[-1] > 0.01:
            return True
        return False





