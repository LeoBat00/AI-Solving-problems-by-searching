import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

#-------------------------------------------------------------------------------
def f(x1, x2):
    return (x1**2 - 10 * np.cos(2 * np.pi * x1) + 10) + (x2**2 - 10 * np.cos(2 * np.pi * x2) + 10)

def perturb(x, e, bounds):
    x_new = x + np.random.uniform(-e, e, size=x.shape)
    return np.clip(x_new, bounds[:, 0], bounds[:, 1])

def hill_climbing(e, max_it, max_viz, bounds):
    x_opt = np.array([-2, -2])  
    f_opt = f(x_opt[0], x_opt[1])
    i = 0
    melhoria = True
    while i < max_it and melhoria:
        melhoria = False
        for j in range(max_viz):
            x_cand = perturb(x_opt, e, bounds)
            f_cand = f(x_cand[0], x_cand[1])
            if f_cand < f_opt: 
                x_opt = x_cand
                f_opt = f_cand
                melhoria = True
                break
        i += 1
    return x_opt, f_opt

#-------------------------------------------------------------------------------
resultados = []
e = 0.1  
max_it = 10000  
max_viz = 20  
bounds = np.array([[-5.12, 5.12], [5.12, 5.12]])  
#-------------------------------------------------------------------------------

for rodada in range(100):
    x_best, f_best = hill_climbing(e, max_it, max_viz, bounds)
    resultados.append((np.round(x_best, 3), np.round(f_best, 5)))

for i, (x_best, f_best) in enumerate(resultados):
    print(f'Rodada {i+1}: Ponto ótimo = ({x_best[0]:.3f}, {x_best[1]:.3f}), f(x1, x2) = {f_best:.5f}')

#-------------------------------------------------------------------------------
pontos_otimos = [tuple(x_best) for x_best, _ in resultados]  
valores_f = [f_best for _, f_best in resultados] 
moda_pontos_otimos = Counter(pontos_otimos).most_common(1)[0] 
moda_valores_f = Counter(valores_f).most_common(1)[0] 
print("\nTabela de Moda das Soluções:")
print(f"Moda dos Pontos Ótimos: {moda_pontos_otimos[0]}, Frequência: {moda_pontos_otimos[1]} vezes")
print(f"Moda dos Valores da Função: {moda_valores_f[0]}, Frequência: {moda_valores_f[1]} vezes")

#-------------------------------------------------------------------------------
x1_vals = np.linspace(-5.12, 5.12, 400)
x2_vals = np.linspace(-5.12, 5.12, 400)
x1, x2 = np.meshgrid(x1_vals, x2_vals)
z = f(x1, x2)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
ax.scatter(x_best[0], x_best[1], f_best, color='r', s=100, label='Ponto Ótimo')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Maximização de f(x1, x2)')

plt.legend()
plt.show()
#-------------------------------------------------------------------------------
