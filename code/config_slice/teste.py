import uaibot as ub
from setup import *
from slice_creator import *
from CBFPathFollower import *


#robot, sim, all_obs, q0, htm_tg, htm_init = create_scenario()


# draw_slice(
#     robot      = robot,
#     obstacles  = all_obs,
#     q          = robot.q,
#     indexes    = [0, 1],
#     value      = 0.003,
#     score_fun  = lambda _q: score( robot.task_function(q=_q, htm_tg=htm_tg)[0][0 : 3, : ] ),
#     path_q     = [],
#     eps_to_obs = 0.0,
#     h_to_obs   = 0.0,
#     h_merge    = 1e-3,
#     save_path  = "/home/pedro/uaibot_files/code/config_slice/level",
# )



def compute_soft_min(D, grad_D, r, eps=1e-8):
    D = np.array(D)
    grad_D = np.array(grad_D)
    if D.ndim == 1:
        D = D[:, None]
    if grad_D.ndim == 2:
        grad_D = grad_D[:, :, None]

    n_obs, k = D.shape
    n_joints = grad_D.shape[2]
    soft_Dmin = np.zeros(k)
    soft_Dmin_grad = np.zeros((k, n_joints))

    for i in range(k):
        Di = D[:, i]  # (n_obs,)
        grad_Di = grad_D[:, i, :]  # (n_obs, n_joints)
        Di_safe = np.clip(Di, eps, None)  # evita zero
        S = np.sum(np.power(Di_safe, -1/r))
        soft_Dmin[i] = S ** (-r)
        weights = Di_safe ** (-1 / r - 1)
        weighted_grads = weights[:, None] * grad_Di
        soft_Dmin_grad[i, :] = (S ** (-r - 1)) * np.sum(weighted_grads, axis=0)

    return soft_Dmin, soft_Dmin_grad


# Teste
D = [[0, 1, 2], [1, 0, 3]]
grad_D = np.random.rand(2, 3, 6)
soft_Dmin, soft_Dmin_grad = compute_soft_min(D, grad_D, r=0.8)
print(soft_Dmin)