import os
import random
import urllib.request
import uaibot as ub
import numpy as np
import matplotlib.pyplot as plt

def funF(r):
    f = np.matrix(r)
    for j in range(r.shape[0]):
        f[j, 0] = np.sign(r[j, 0]) * np.sqrt(np.abs(r[j, 0]))
    return f

def constrained_control(robot, obstacles, htm_tg):
    n = len(robot.links)
    q = robot.q
    q_min = robot.joint_limit[:, 0]
    q_max = robot.joint_limit[:, 1]

    # Hiperparâmetros fixos
    eta_obs = 0.3
    eta_auto = 0.3
    eta_joint = 0.3
    d_safe_obs = 0.02
    d_safe_auto = 0.002
    d_safe_jl = (np.pi / 180) * 5
    eps_obs = 0.003
    h_obs = 0.003
    eps_auto = 0.02
    h_auto = 0.05
    eps_reg = 0.01
    kp = 2.0

    if not isinstance(obstacles, list):
        obstacles = [obstacles]

    # Obstáculos externos
    A_obs = np.zeros((0, n))
    b_obs_raw = np.zeros((0, 1))
    for ob in obstacles:
        dist = robot.compute_dist(ob, q=q, eps=eps_obs, h=h_obs)
        A_obs = np.vstack((A_obs, dist.jac_dist_mat))
        b_obs_raw = np.vstack((b_obs_raw, dist.dist_vect))
    b_obs = -eta_obs * (b_obs_raw - d_safe_obs)

    # Autocolisão
    dist_auto = robot.compute_dist_auto(q=q, eps=eps_auto, h=h_auto)
    A_auto = dist_auto.jac_dist_mat
    b_auto = -eta_auto * (dist_auto.dist_vect - d_safe_auto)

    # Limites das juntas
    A_joint = np.vstack((np.identity(n), -np.identity(n)))
    b_joint = -eta_joint * (np.vstack((q - q_min, q_max - q)) - d_safe_jl)

    # Função de tarefa
    r, Jr = robot.task_function(htm_tg, q=q)

    # Otimização
    A = np.vstack((A_obs, A_auto, A_joint))
    b = np.vstack((b_obs, b_auto, b_joint))
    H = 2 * (Jr.T @ Jr + eps_reg * np.identity(n))
    f = Jr.T @ kp * funF(r)

    qdot = ub.Utils.solve_qp(H, f, A, b)

    return qdot
