import numpy as np
import matplotlib.pyplot as plt

def E_A(x, x0, L):
    d = abs(x - x0)
    if d > L / 2:
        return (d - L/2)**2
    else:
        return -(d - L/2)**2

def dE_A_dx(x, x0, L):
    d = abs(x - x0)
    sgn = np.sign(x - x0)
    if d > L / 2:
        return 2 * (d - L/2) * sgn
    else:
        return -2 * (d - L/2) * sgn



def projection_grad_iter(x, x0, L, tol=1e-8, max_iter=10):
    for k in range(max_iter):
        grad = dE_A_dx(x, x0, L)
        if abs(grad) < tol:
            break
        x = x - E_A(x, x0, L) / grad
        print(f"Iteração: {k}  x: {x}  gradiente: {grad}")
    return x


x0 = 0         # centro do segmento
L = 4          # comprimento do segmento
x = .3         # ponto que queremos projetar

pi_grad = projection_grad_iter(x, x0, L)
dist = E_A(x, x0, L)
print(f"Projeção de {x} no segmento A: {pi_grad}")
print(f"Distância ao segmento A: {dist}")


fig, ax = plt.subplots(figsize=(8, 2))
segment_start = x0 - L/2
segment_end = x0 + L/2
ax.plot([segment_start, segment_end], [0, 0], 'k-', lw=4, label='Segmento A')
ax.plot(x, 0, 'ro', label='x')
ax.plot(pi_grad, 0, 'go', label=r'$\pi^A(x)$')
ax.plot([x, pi_grad], [0.1, 0.1], 'r--', lw=1)
ax.set_ylim(-0.5, 0.5)
ax.set_xlim(x0 - L - 1, x0 + L + 1)
ax.set_yticks([])
ax.set_xlabel('x')
ax.legend()
ax.set_title('Projeção de $x$ no segmento $A$')

plt.grid(True)
plt.tight_layout()
plt.show()
