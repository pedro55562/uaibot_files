import uaibot as ub
from setup import *
from slice_creator import *
from CBFPathFollower import *


robot, sim, all_obs, q0, htm_tg, htm_init = create_scenario()


draw_slice(
    robot      = robot,
    obstacles  = all_obs,
    q          = robot.q,
    indexes    = [3, 4],
    value      = 0.003,
    score_fun  = lambda _q: score( robot.task_function(q=_q, htm_tg=htm_tg)[0][0 : 3, : ] ),
    path_q     = [],
    eps_to_obs = 0.0,
    h_to_obs   = 0.0,
    h_merge    = 1e-3,
    save_path  = "/home/pedro/uaibot_files/code/config_slice/level",
)





    