import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
import uaibot as ub

R_plus90 = np.array([[0, -1],
                     [1,  0]])

R_minus90 = np.array([[0, 1],
                      [-1, 0]])



class CBFSimulator:
    def __init__(self, kp, eta, dt, tmax, delta, eps,
                 q0, qF, centers, radii,
                 last_T, tol_stuck, decay_rate,
                 perturb_norm, min_perturb_norm, min_error_start_perturb,
                 perturbation_enabled=True):

        # Parâmetros gerais
        self.kp = kp
        self.eta = eta
        self.dt = dt
        self.tmax = tmax
        self.delta = delta
        self.eps = eps

        # Estados iniciais
        self.q = q0
        self.q0 = q0
        self.qF = qF
        self.centers = centers
        self.radii = radii

        # Parâmetros de perturbação
        self.last_T = last_T
        self.tol_stuck = tol_stuck
        self.decay_rate = decay_rate
        self.perturb_norm = perturb_norm
        self.min_perturb_norm = min_perturb_norm
        self.min_error_start_perturb = min_error_start_perturb
        self.perturbation_enabled = perturbation_enabled

        # Estado interno
        self.hist_q = []
        self.hist_e = []
        self.perturbation = np.zeros((2, 1))
        self.perturbation_mode = False

    def _build_barrier_constraints(self):
        A_full = np.zeros((0, 2))
        b_full = np.zeros((0, 1))

        Ak = []
        bk = []

        for i in range(len(self.centers)):
            center = self.centers[i]
            r = self.radii[i]

            h        = np.linalg.norm(self.q - center) - r - self.delta
            grad_h_q = ( self.q - center )/np.linalg.norm( self.q - center )

            Ak.append(   grad_h_q    )
            bk.append( - self.kp * h )





        A_full = np.vstack((A_full , ))
        b_full = np.vstack((b_full , ))

        return A_full, b_full


    def _check_stuck_and_perturb(self, i):
        if i > self.last_T and self.hist_e[-1] > self.min_error_start_perturb and not self.perturbation_mode:
            recent = self.hist_e[-self.last_T:]
            max_diff = max(recent) - min(recent)
            if max_diff < self.tol_stuck:
                self.perturbation = np.random.randn(2, 1)
                self.perturbation *= self.perturb_norm / np.linalg.norm(self.perturbation)
                self.perturbation_mode = True
                print("PERTURBATION MODE")

        if self.perturbation_mode:
            print(self.perturbation.T)
            self.perturbation *= self.decay_rate
            if np.linalg.norm(self.perturbation) <= self.min_perturb_norm:
                self.perturbation = np.zeros((2, 1))
                self.perturbation_mode = False

    def run(self):
        steps = round(self.tmax / self.dt)
        for i in range(steps):
            self.hist_q.append(np.matrix(self.q))
            self.hist_e.append(np.linalg.norm(self.q - self.qF))

            if self.perturbation_enabled:
                self._check_stuck_and_perturb(i)
            else:
                self.perturbation = 0

            H = 2 * (1 + self.eps) * np.identity(2)
            f = 2 * self.kp * (self.q - self.qF) + self.perturbation

            A, b = self._build_barrier_constraints()

            u = ub.Utils.solve_qp(H, f, A, b)
            self.q += u * self.dt

    def plot(self):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')
        ax.plot([q[0, 0] for q in self.hist_q], [q[1, 0] for q in self.hist_q], color='#81d41a')
        ax.scatter([self.qF[0, 0]], [self.qF[1, 0]], color='magenta', s=40)

        for i in range(len(self.centers)):
            circle = patches.Circle((self.centers[i][0], self.centers[i][1]),
                                    radius=self.radii[i], color='#5983b0')
            ax.add_patch(circle)

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.5, 1.5)
        plt.show()

    def animate(self, interval=5, save=False, filename="simulacao.mp4"):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        # Obstáculos
        for i in range(len(self.centers)):
            circle = patches.Circle((self.centers[i][0], self.centers[i][1]),
                                    radius=self.radii[i], color='#5983b0')
            ax.add_patch(circle)

        # Alvo final
        ax.scatter([self.qF[0, 0]], [self.qF[1, 0]], color='magenta', s=40)

        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.5, 1.5)

        # Elementos da animação
        path_line, = ax.plot([], [], lw=2, color='#81d41a')
        robot_dot, = ax.plot([], [], 'ro')

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


        ani = animation.FuncAnimation(fig, update, frames=len(self.hist_q),
                                    init_func=init, blit=True, interval=interval)

        if save:
            ani.save(filename, writer='ffmpeg')
        else:
            plt.show()
            
if __name__ == "__main__":
    simulator = CBFSimulator(
        kp=0.5,
        eta=0.3,
        dt=0.01,
        tmax=60,
        delta=0.05,
        eps=0.01,
        q0=np.matrix([-1.5, 0.2]).T,
        qF=np.matrix([1.5, 0]).T,
        centers=[np.matrix([-0.25, 0.5]).T, np.matrix([0.0, 0.0]).T],
        radii=[0.3, 0.5],
        last_T=50,
        tol_stuck=0.01,
        decay_rate=0.99,
        perturb_norm=3.0,
        min_perturb_norm=0.3,
        min_error_start_perturb=0.05,
        perturbation_enabled=False  
    )

    simulator.run()
    simulator.plot()
    #simulator.animate(interval=5, save=False)