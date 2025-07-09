from random import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import uaibot as ub
import random


R_plus90 = np.array([[0, -1],
                     [1,  0]])

R_minus90 = np.array([[0, 1],
                      [-1, 0]])





def fun_beta(h, eta=1):
    return eta * h

def compute_soft_min(D, grad_D, r):
    D = np.array(D, dtype=np.float64)
    grad_D = np.array(grad_D, dtype=np.float64)

    S = np.sum(np.power(D, -1/r))
    soft_Dmin = S ** (-r)

    weights = D ** (-1 / r - 1)
    weighted_grads = weights[:, None] * grad_D 
    soft_Dmin_grad = (S ** (-r - 1)) * np.sum(weighted_grads, axis=0)

    return soft_Dmin, soft_Dmin_grad



class CBFSimulator:
    def __init__(self, kp, eta, tangential_threshold, tangential_eta, R_mat,
                 dt, tmax, delta, eps,
                 q0, qF, centers, radii,
                 plot_limits,
                 enable_simulation=True, 
                 rotation_enabled=True  
                 ):

        # Parâmetros gerais
        self.kp = kp
        self.eta = eta
        self.dt = dt
        self.tmax = tmax
        self.delta = delta
        self.eps = eps

        # Estados iniciais
        self.q = q0.copy()  
        self.q0 = q0.copy()  
        self.qF = qF
        self.centers = centers
        self.radii = radii

        # Parâmetros de circulação
        self.tangential_threshold = tangential_threshold
        self.plot_limits = plot_limits
        self.tangential_eta = tangential_eta  
        self.R_mat = R_mat

        # Estado interno
        self.hist_q = []
        self.hist_e = []
        self.hist_circulation = []  
        self.enable_simulation = enable_simulation   
        self.rotation_enabled = rotation_enabled  
        self.circulation_active = False  


    def _build_barrier_constraints(self):
        A_full = np.zeros((0, 2))
        b_full = np.zeros((0, 1))

        D = []
        grad_D = []

        self.circulation_active = False  

        for i in range(len(self.centers)):
            center = self.centers[i]
            r = self.radii[i]

            h = np.linalg.norm(self.q - center) - r - self.delta
            grad_h_q = (self.q - center).T / np.linalg.norm(self.q - center)

            D.append(h)
            grad_D.append(grad_h_q)

        grad_D = [np.asarray(g).flatten() for g in grad_D]

        soft_Dmin, soft_Dmin_grad = compute_soft_min(D, grad_D, r= 0.8)
        
        A_full = np.vstack((A_full, soft_Dmin_grad))
        b_full = np.vstack((b_full, -fun_beta( soft_Dmin , self.eta)))


        if self.rotation_enabled: 

            tangential_unit = (self.R_mat @ soft_Dmin_grad.T).reshape(1, 2)
            b_tangential = - fun_beta(soft_Dmin - self.tangential_threshold , self.tangential_eta)

            A_full = np.vstack((A_full, tangential_unit))
            b_full = np.vstack((b_full, b_tangential))

            self.circulation_active = True

        return A_full, b_full


    def run(self):
        if not self.enable_simulation:
            print("Simulação desativada. Apenas exibindo o mapa.")
            return

        steps = round(self.tmax / self.dt)
        for i in range(steps):
            self.hist_q.append(self.q.copy())  
            self.hist_e.append(np.linalg.norm(self.q - self.qF))

            H = 2 * (1 + self.eps) * np.identity(2)
            f = 2 * self.kp * (self.q - self.qF)

            A, b = self._build_barrier_constraints()

            try:
                u = ub.Utils.solve_qp(H, f, A, b)
                self.q += u * self.dt  

                if self.circulation_active:
                    self.hist_circulation.append(self.q.copy())
            except ValueError as e:
                print(f"Error: {e}. Simulation stopped.")
                break

        print("Simulation completed. Proceeding to generate plot or animation.")

    def plot(self, save=False, filename="simulacao_plot.png", dpi=300):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect('equal')

        ax.plot([q[0, 0] for q in self.hist_q], [q[1, 0] for q in self.hist_q], color='#81d41a', label='Trajetória')

        if self.hist_circulation:
            ax.scatter([q[0, 0] for q in self.hist_circulation], 
                       [q[1, 0] for q in self.hist_circulation], 
                       color='#ff5733', s=10, label='Circulação Ativa')

        ax.scatter(self.qF[0, 0], self.qF[1, 0], color='magenta', s=40, label='qF (Final)')
        ax.scatter(self.q0[0, 0], self.q0[1, 0], color='blue', s=40, label='q0 (Inicial)')

        # Obstáculos sem legenda
        for i in range(len(self.centers)):
            circle = patches.Circle((self.centers[i][0, 0], self.centers[i][1, 0]),
                                    radius=self.radii[i], color='#5983b0', alpha=0.5)
            ax.add_patch(circle)

        ax.set_xlim(self.plot_limits[0], self.plot_limits[1])
        ax.set_ylim(self.plot_limits[2], self.plot_limits[3])
        ax.legend()
        plt.title("Simulação de Navegação com Barreiras")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True)
        if save:
            plt.savefig(filename, dpi=dpi, bbox_inches='tight')
            print(f"Figura salva em {filename}")
            plt.close(fig)
        else:
            plt.show()


    def animate(self, interval=5, speed=1.0, save=False, filename="simulacao.mp4", frame_step=5, fps=20):
        """
        interval: tempo base entre frames em ms (default 5)
        speed: fator multiplicativo para o intervalo (ex: 2.0 = 2x mais devagar)
        save: salva o vídeo se True
        filename: nome do arquivo de saída
        frame_step: quantos frames pular (ex: 5 = mostra 1 a cada 5 frames)
        fps: frames por segundo do vídeo salvo
        """
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        for i in range(len(self.centers)):
            circle = patches.Circle((self.centers[i][0], self.centers[i][1]),
                                    radius=self.radii[i], color='#5983b0')
            ax.add_patch(circle)

        ax.scatter([self.qF[0, 0]], [self.qF[1, 0]], color='magenta', s=40, label='qF (Final)')
        ax.scatter([self.q0[0, 0]], [self.q0[1, 0]], color='blue', s=40, label='q0 (Inicial)')

        ax.set_xlim(self.plot_limits[0], self.plot_limits[1])
        ax.set_ylim(self.plot_limits[2], self.plot_limits[3])
        ax.legend()  

        path_line, = ax.plot([], [], lw=2, color='#81d41a')
        robot_dot, = ax.plot([], [], 'ro')

        # Reduz o número de frames processados
        frames = list(range(0, len(self.hist_q), frame_step))
        if frames[-1] != len(self.hist_q) - 1:
            frames.append(len(self.hist_q) - 1)  # garante que o último frame seja mostrado

        def init():
            path_line.set_data([], [])
            robot_dot.set_data([], [])
            return path_line, robot_dot

        def update(frame):
            x_vals = [q[0, 0] for q in self.hist_q[:frame+1]]
            y_vals = [q[1, 0] for q in self.hist_q[:frame+1]]
            path_line.set_data(x_vals, y_vals)
            robot_dot.set_data([x_vals[-1]], [y_vals[-1]])
            return path_line, robot_dot

        real_interval = int(interval * speed)
        ani = animation.FuncAnimation(fig, update, frames=frames,
                                      init_func=init, blit=True, interval=real_interval)

        if save:
            print(f"Salvando animação em {filename} ...")
            try:
                ani.save(filename, writer='ffmpeg', fps=fps)
                print("Animação salva com sucesso!")
            except Exception as e:
                print(f"Erro ao salvar animação: {e}")
        else:
            plt.show()


def get_obstacle_config_by_number(config_number):

    if config_number == 1:
        qF = np.matrix([1.5, 0.0]).T
        q0 = np.matrix([-1.5, 0.2]).T
        centers = [
            np.matrix([-0.5,  0.7]).T,
            np.matrix([-0.5, -0.7]).T,
            np.matrix([ 0.0,  0.0]).T
        ]
        radius = [0.5, 1, 0.5]

    elif config_number == 2:
        # Dois círculos sobrepostos bloqueando o caminho direto
        qF = np.matrix([2.0, 0.0]).T
        q0 = np.matrix([-2.0, 0.0]).T
        centers = [
            np.matrix([0.0,  0.6]).T,
            np.matrix([0.0, -0.6]).T
        ]
        radius = [0.7, 0.7]

    elif config_number == 3:
        # U invertido bloqueando o centro – só circulando em volta é possível passar
        qF = np.matrix([2.0, 0.0]).T
        q0 = np.matrix([-2.0, 0.0]).T
        centers = [
            np.matrix([0.0,  0.0]).T,
            np.matrix([0.0,  1.0]).T,
            np.matrix([0.0, -1.0]).T
        ]
        radius = [0.6, 0.6, 0.6]

    elif config_number == 4:
        # Labirinto circular com pequenas aberturas em “S”
        qF = np.matrix([2.0, 0.5]).T
        q0 = np.matrix([-2.0, -0.5]).T
        centers = [
            np.matrix([-0.5,  0.0]).T,
            np.matrix([ 0.5,  0.0]).T,
            np.matrix([ 0.0,  0.5]).T,
            np.matrix([ 0.0, -0.5]).T
        ]
        radius = [0.5, 0.5, 0.3, 0.3]

    elif config_number == 5:
        # Dois anéis concêntricos de círculos bloqueando quase todo o caminho
        qF = np.matrix([0.0, 0.0]).T
        q0 = np.matrix([3.0, 0.0]).T
        centers = []
        radius = []
        # Anel interno (6 círculos)
        for k in range(6):
            theta = 2*np.pi*k/6
            centers.append(np.matrix([1.0*np.cos(theta), 1.0*np.sin(theta)]).T)
            radius.append(0.4)
        # Anel externo (8 círculos)
        for k in range(8):
            theta = 2*np.pi*k/8 + np.pi/8  # deslocado do interno
            centers.append(np.matrix([2.0*np.cos(theta), 2.0*np.sin(theta)]).T)
            radius.append(0.4)

    elif config_number == 6:
        qF = np.matrix([3.0, 0.0]).T
        q0 = np.matrix([-3.0, 0.0]).T
        centers = []
        radius = []
        num_obstacles = 7
        for i in range(num_obstacles):
            x = -2.5 + i * 1.0
            y = 0.4 if i % 2 == 0 else -0.4
            centers.append(np.matrix([x, y]).T)
            radius.append(0.45) 

    elif config_number == 7:
        q0 = np.matrix([-3.0, 0.0]).T
        qF = np.matrix([3.0, 0.0]).T
        centers = []
        radius = []

        # Muralha central
        for theta in np.linspace(0, 2*np.pi, 12, endpoint=False):
            x = 0.0 + 1.0 * np.cos(theta)
            y = 0.0 + 1.0 * np.sin(theta)
            if not (np.isclose(theta, np.pi/2, atol=0.3)):  # Deixe uma abertura no topo
                centers.append(np.matrix([x, y]).T)
                radius.append(0.4)

        # Guardas no caminho até a entrada
        centers.append(np.matrix([-1.5, 0.8]).T)
        centers.append(np.matrix([-1.0, 1.2]).T)
        radius += [0.4, 0.4]
    else:
        raise ValueError("Número de configuração inválido")

    return q0, qF, centers, radius


if __name__ == "__main__":
    # 1 3 4 5 6
    config_number = 6
    q0, qF, centers, radii = get_obstacle_config_by_number(config_number)


    simulator = CBFSimulator(
        kp=3.0,                    
        eta=0.6,                      
        tangential_threshold=0.05,  
        tangential_eta=0.6,        
        R_mat=R_minus90,
        dt=0.01,                  
        tmax=50,                   
        delta=0.05,                
        eps=0.01,                  
        q0=q0,
        qF=qF,
        centers=centers,
        radii=radii,
        plot_limits=(-4.0, 4.0, -4.0, 4.0),
        enable_simulation=True,
        rotation_enabled=True
    )

    simulator.run()
    simulator.plot(save=False, filename="trajetoria.png")
    #simulator.animate()
