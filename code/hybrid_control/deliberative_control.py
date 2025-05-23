import uaibot as ub
import numpy as np
from scipy.spatial import cKDTree
from setup import *

class Node:
    def __init__(self, q, r = 1000):
        self.q = np.array(q).flatten()
        self.parent = None
        self.cost = 0.0
        self.r    = r

class RRTStar:
    def __init__(self, robot, all_obs, htm_base, htm_tg, goal_bias = 0.4 ,gamma = 2, d_max = 1.2, d_min = 0.2 ,max_iter = 2000):
        self.htm_tg = htm_tg
        self.robot = robot
        self.n = self.robot.q.shape[0]
        self.all_obs = all_obs
        self.htm_base = htm_base
        self.goal_bias = goal_bias
        self.max_iter = max_iter
        self.q_start = Node(robot.q , self.task_function(robot.q))
        self.nodes = [self.q_start]
        self.tree = None
        self.d_max = d_max
        self.d_min = d_min
        self.gamma = gamma
        q_min = robot.joint_limit[:,0]
        q_max = robot.joint_limit[:,1]
        self.cspace_limits = list(zip(q_min, q_max))

        self.goal_list_q = []
        for k in range(0,100):
            goal_q = self.get_q_goal(300)
            if goal_q is not None:
                goal_q = np.array( goal_q ).flatten()
                self.goal_list_q.append(goal_q)
                
        self.best_node = self.q_start

    def get_q_goal(self, max_iter):
        for i in range(round(max_iter)):
            try:
                q = self.robot.ikm(htm_target=self.htm_tg, htm=self.htm_base, no_iter_max = 1000)
            except Exception as e:
                continue
            isfree, message, info = self.robot.check_free_configuration(q=q, htm=self.htm_base, obstacles=self.all_obs)
            if isfree:
                return q
        return None

    def random_sample(self):
        if np.random.uniform(0,1) < self.goal_bias:
            index = np.random.randint(low=0, high=len(self.goal_list_q) - 1)
            return self.goal_list_q[index]
        return np.array([
            np.random.uniform(low, high) for (low, high) in self.cspace_limits
        ]).flatten()

    def nearest(self, q_rand):
        tree = cKDTree([node.q for node in self.nodes])
        _, idx = tree.query(q_rand)
        return self.nodes[idx]

    def adaptive_near_radius(self):
        r = self.gamma * (np.log(len(self.nodes)) / len(self.nodes)) ** (1 / self.n )
        return r

    def near(self, q_new, radius):
        tree = cKDTree([node.q for node in self.nodes])
        idxs = tree.query_ball_point(q_new, radius)
        return [self.nodes[i] for i in idxs]

    def cost(self, node):
        return node.cost
    
    def is_path_free(self, q_near, q_new, sample_rate = 0.1):
        dist = np.linalg.norm(q_new - q_near)
        num_samples = int(dist / sample_rate)
        if num_samples < 1:
            num_samples = 5

        for i in range(1, num_samples):
            alpha = i / num_samples  
            q_interp = (1 - alpha) * q_near + alpha * q_new 
            verify, _, _ = self.robot.check_free_configuration(q=q_interp, htm=self.htm_base, obstacles=self.all_obs)
            if not verify:
                return False 
        return True  

    def steer(self, q_nearest, q_rand):
        direction = q_rand - q_nearest.q
        dist = np.linalg.norm(direction)
        direction = direction/dist
        stepsize = np.random.uniform(self.d_min , self.d_max)
        q_new = q_nearest.q + direction * stepsize

        return q_new

    def expand_tree(self):
        q_rand = self.random_sample()
        nearest_node = self.nearest(q_rand)
        q_new = self.steer(nearest_node, q_rand)
        if self.is_path_free(nearest_node.q, q_new):
            new_node = Node(q_new)
            radius = self.adaptive_near_radius()
            neighbors = self.near(q_new, radius)
            best_parent = nearest_node
            min_cost = self.cost(nearest_node) + np.linalg.norm(q_new - nearest_node.q)
            # verifica qual é o melhor pai para o novo nó, vendo o nó que minimiza o custo
            # busca qual dos vizinhos, dentro de um raio, tem o menor custo ate q_new
            for node in neighbors:
                if self.is_path_free(node.q, q_new):
                    c = self.cost(node) + np.linalg.norm(q_new - node.q)
                    if c < min_cost:
                        best_parent = node
                        min_cost = c    
            new_node.parent = best_parent
            new_node.cost = min_cost
            new_node.r = self.task_function(q_new)
            self.nodes.append(new_node)

            if new_node.r < self.best_node.r:
                self.best_node = new_node
            
            # Rewire
            # verifica se algum vizinho tem um caminho mais barato passando por new_node
            for node in neighbors:
                if node == best_parent:
                    continue
                new_cost = new_node.cost + np.linalg.norm(node.q - new_node.q)
                if new_cost < node.cost:
                    if self.is_path_free(node.q, new_node.q):
                        node.parent = new_node
                        node.cost = new_cost
                        
    def task_function(self, q):

        p_des = self.htm_tg[0:3, 3]
        x_des = self.htm_tg[0:3, 0]
        y_des = self.htm_tg[0:3, 1]
        z_des = self.htm_tg[0:3, 2]

        htm_eef = self.robot.fkm(q=q , htm = self.htm_base)
        p_eef = htm_eef[0:3, 3]
        x_eef = htm_eef[0:3, 0]
        y_eef = htm_eef[0:3, 1]
        z_eef = htm_eef[0:3, 2]

        r = np.matrix(np.zeros((6,1)))
        r[0:3,0] = p_eef - p_des
        r[3] = max(1 - x_des.T * x_eef, 0)
        r[4] = max(1 - y_des.T * y_eef, 0)
        r[5] = max(1 - z_des.T * z_eef, 0)

        return np.linalg.norm(r)**2                     

    def extract_path_between_any_nodes(self, node_a, node_b):
        # Caminho de node_a ate a raiz
        path_a = []
        ancestors_a = set()
        curr = node_a
        while curr is not None:
            path_a.append(curr)
            ancestors_a.add(curr)
            curr = curr.parent

        # Caminho de node_b ate a raiz
        path_b = []
        curr = node_b
        common_ancestor = None
        while curr is not None:
            path_b.append(curr)
            if curr in ancestors_a:
                common_ancestor = curr
                break
            curr = curr.parent


        if common_ancestor is None:
            return []

        # Monta o caminho completo
        path = []
        # de node_a até o ancestral comum (inverso)
        for node in path_a:
            path.append(node.q)
            if node == common_ancestor:
                break

        #de node_b até o ancestral comum
        for node in reversed(path_b[:-1]):
            path.append(node.q)

        return path


            

    def find_path(self, buffer_q, buffer_r, best_node):
        for i in range(len(self.nodes)):
            qp_node = self.nodes[i]
            qp     = qp_node.q
            for j in range(len(buffer_q)):
                q_start = np.array(buffer_q[j]).flatten()
                if self.is_path_free(q_start, qp):
                    path = self.extract_path_between_any_nodes(qp_node, best_node)
                    return path

        return []



def RRTstar_test():
    index = 1148
    robot, sim, all_obs, q0, htm_tg, htm_base = setup_motion_planning_simulation(index)
    rrt_star = RRTStar(robot, all_obs, htm_base, htm_tg)
    for i in range(500):
        rrt_star.expand_tree()
    print("Número de nós: " + str(len(rrt_star.nodes)))

    print(f"best r : {rrt_star.best_node.r }")
    print(f"||q|| = {np.linalg.norm( rrt_star.best_node.q - rrt_star.goal_list_q[0] )} ")


    q_list = []
    for node in rrt_star.nodes:
        q_list.append(node.q)
    qdot_hist = [0]
    r_norm    = [0]
    r_hist    = [0]
    t_hist    = [0]
    draw_balls(pathhh_=q_list, robot=robot, sim=sim)
    draw_balls(pathhh_=[rrt_star.best_node.q], robot=robot, sim=sim, color="red")

    store_info(sim, qdot_hist, r_norm ,r_hist, t_hist, "benchmark" + str(index))

if __name__ == "__main__":
    RRTstar_test()

