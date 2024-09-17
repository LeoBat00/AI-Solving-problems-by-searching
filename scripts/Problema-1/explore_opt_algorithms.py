import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter

# Parâmetros
epsilon = 0.1
sigma = 0.1
max_it = 10000
max_viz = 20
R = 100
generate_plots = True
objective_type = 'min'  # Escolha entre 'min' ou 'max'

bounds = np.array([[-100, 100], [-100, 100]])
x1_inferior, x2_inferior = bounds[:, 0]
x1_superior, x2_superior = bounds[:, 1]

# Função objetivo
def f(x1, x2):
    return x1**2 + x2**2 


def compare(f_new, f_opt, objective_type):
    if objective_type == 'max':
        return f_new > f_opt
    elif objective_type == 'min':
        return f_new < f_opt

# Perturbação para Hill Climbing
def hill_climbing_perturb(x, epsilon, bounds):
    x_new = x + np.random.uniform(-epsilon, epsilon, size=x.shape)
    return np.clip(x_new, bounds[:, 0], bounds[:, 1])

# Perturbação para Local Random Search
def lrs_perturb(x, sigma, bounds):
    x_new = x + np.random.normal(0, sigma, size=x.shape)
    return np.clip(x_new, bounds[:, 0], bounds[:, 1])

# Perturbação para Global Random Search
def grs_perturb(bounds):
    return np.random.uniform(bounds[:, 0], bounds[:, 1])

# Hill Climbing
def hill_climbing(epsilon, max_it, max_viz, bounds, generate_plots):
    x_opt = bounds[:, 0]  # Ponto inicial no limite inferior do domínio
    f_opt = f(x_opt[0], x_opt[1])
    x_history, f_history = [x_opt.copy()], [f_opt]
    
    if generate_plots:
        fig = plt.figure(figsize=(6, 6))
        ax1 = fig.add_subplot(111, projection='3d')
        x1_vals = np.linspace(x1_inferior, x1_superior, 400)
        x2_vals = np.linspace(x2_inferior, x2_superior, 400)
        x1, x2 = np.meshgrid(x1_vals, x2_vals)
        z = f(x1, x2)
        ax1.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
        ax1.set_xlabel('x1')
        ax1.set_ylabel('x2')
        ax1.set_zlabel('f(x1, x2)')
        ax1.set_title('Hill Climbing')
        ax1.set_xlim([x1_inferior, x1_superior])
        ax1.set_ylim([x2_inferior, x2_superior])
        ax1.set_zlim([z.min(), z.max()])

    for it in range(max_it):
        melhoria = False
        for _ in range(max_viz):
            x_cand = hill_climbing_perturb(x_opt, epsilon, bounds)
            f_cand = f(x_cand[0], x_cand[1])
            if compare(f_cand, f_opt, objective_type):
                x_opt = x_cand
                f_opt = f_cand
                x_history.append(x_opt.copy())
                f_history.append(f_opt)
                melhoria = True
                if generate_plots:
                    ax1.clear()
                    ax1.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
                    ax1.scatter(np.array(x_history)[:, 0], np.array(x_history)[:, 1], np.array(f_history), color='r', s=20, marker='x')
                    ax1.set_xlabel('x1')
                    ax1.set_ylabel('x2')
                    ax1.set_zlabel('f(x1, x2)')
                    ax1.set_title('Hill Climbing')
                    ax1.set_xlim([x1_inferior, x1_superior])
                    ax1.set_ylim([x2_inferior, x2_superior])
                    ax1.set_zlim([z.min(), z.max()])
                    plt.pause(0.01)
                break
        if not melhoria:
            break
    
    if generate_plots:
        plt.show()

    return round(f_opt, 5), np.array(x_history), np.array(f_history),

# LRS
def local_random_search(sigma, max_it, bounds, generate_plots):
    x_opt = np.random.uniform(bounds[:, 0], bounds[:, 1])
    f_opt = f(x_opt[0], x_opt[1])
    x_history, f_history = [x_opt.copy()], [f_opt]
    
    if generate_plots:
        fig = plt.figure(figsize=(6, 6))
        ax2 = fig.add_subplot(111, projection='3d')
        x1_vals = np.linspace(x1_inferior, x1_superior, 400)
        x2_vals = np.linspace(x2_inferior, x2_superior, 400)
        x1, x2 = np.meshgrid(x1_vals, x2_vals)
        z = f(x1, x2)
        ax2.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
        ax2.set_xlabel('x1')
        ax2.set_ylabel('x2')
        ax2.set_zlabel('f(x1, x2)')
        ax2.set_title('Local Random Search')
        ax2.set_xlim([x1_inferior, x1_superior])
        ax2.set_ylim([x2_inferior, x2_superior])
        ax2.set_zlim([z.min(), z.max()])

    for it in range(max_it):
        x_cand = lrs_perturb(x_opt, sigma, bounds)
        f_cand = f(x_cand[0], x_cand[1])
        if compare(f_cand, f_opt, objective_type):
            x_opt = x_cand
            f_opt = f_cand
            x_history.append(x_opt.copy())
            f_history.append(f_opt)
            if generate_plots:
                ax2.clear()
                ax2.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
                ax2.scatter(np.array(x_history)[:, 0], np.array(x_history)[:, 1], np.array(f_history), color='b', s=20, marker='x')
                ax2.set_xlabel('x1')
                ax2.set_ylabel('x2')
                ax2.set_zlabel('f(x1, x2)')
                ax2.set_title('Local Random Search')
                ax2.set_xlim([x1_inferior, x1_superior])
                ax2.set_ylim([x2_inferior, x2_superior])
                ax2.set_zlim([z.min(), z.max()])
                plt.pause(0.01)
    
    if generate_plots:
        plt.show()

    return round(f_opt, 5), np.array(x_history), np.array(f_history), x_opt

# GRS
def global_random_search(max_it, bounds, generate_plots):
    x_opt = np.random.uniform(bounds[:, 0], bounds[:, 1])
    f_opt = f(x_opt[0], x_opt[1])
    x_history, f_history = [x_opt.copy()], [f_opt]
    
    if generate_plots:
        fig = plt.figure(figsize=(6, 6))
        ax3 = fig.add_subplot(111, projection='3d')
        x1_vals = np.linspace(x1_inferior, x1_superior, 400)
        x2_vals = np.linspace(x2_inferior, x2_superior, 400)
        x1, x2 = np.meshgrid(x1_vals, x2_vals)
        z = f(x1, x2)
        ax3.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
        ax3.set_xlabel('x1')
        ax3.set_ylabel('x2')
        ax3.set_zlabel('f(x1, x2)')
        ax3.set_title('Global Random Search')
        ax3.set_xlim([x1_inferior, x1_superior])
        ax3.set_ylim([x2_inferior, x2_superior])
        ax3.set_zlim([z.min(), z.max()])

    for it in range(max_it):
        x_cand = grs_perturb(bounds)
        f_cand = f(x_cand[0], x_cand[1])
        if compare(f_cand, f_opt, objective_type):
            x_opt = x_cand
            f_opt = f_cand
            x_history.append(x_opt.copy())
            f_history.append(f_opt)
            if generate_plots:
                ax3.clear()
                ax3.plot_surface(x1, x2, z, cmap='viridis', alpha=0.6)
                ax3.scatter(np.array(x_history)[:, 0], np.array(x_history)[:, 1], np.array(f_history), color='g', s=20, marker='x')
                ax3.set_xlabel('x1')
                ax3.set_ylabel('x2')
                ax3.set_zlabel('f(x1, x2)')
                ax3.set_title('Global Random Search')
                ax3.set_xlim([x1_inferior, x1_superior])
                ax3.set_ylim([x2_inferior, x2_superior])
                ax3.set_zlim([z.min(), z.max()])
                plt.pause(0.01)
    
    if generate_plots:
        plt.show()

    return round(f_opt, 5), np.array(x_history), np.array(f_history)

# Função para rodar o algoritmo R vezes e calcular a moda
def run_algorithm(algorithm, *args):
    results = [algorithm(*args)[0] for _ in range(R)]
    most_common_result = Counter(results).most_common(1)[0][0]  # Obtém o resultado mais frequente
    
    return most_common_result

if generate_plots:
    # Executa e plote cada algoritmo com gráficos atualizados
    print("Executando Hill Climbing...")
    hill_climbing_result = hill_climbing(epsilon, max_it, max_viz, bounds, generate_plots)
    
    print("Executando Local Random Search...")
    local_random_search_result = local_random_search(sigma, max_it, bounds, generate_plots)
    
    print("Executando Global Random Search...")
    global_random_search_result = global_random_search(max_it, bounds, generate_plots)

else:
    # Executa os algoritmos R vezes e calcula a moda dos resultados
    print("Executando o hill_climbing")
    hill_climbing_result = run_algorithm(hill_climbing, epsilon, max_it, max_viz, bounds, generate_plots)
    print("Executando o lrs")
    lrs_result = run_algorithm(local_random_search, sigma, max_it, bounds, generate_plots)
    print("Executando o grs")
    grs_result = run_algorithm(global_random_search, max_it, bounds, generate_plots)

    # Resultados finais
    print("\nTabela de Resultados:")
    print(f"{'Algoritmo':<20} {'Resultado':<10}")
    print(f"{'-'*30}")
    print(f"{'Hill Climbing':<20} {hill_climbing_result:<10}")
    print(f"{'Local Random Search':<20} {lrs_result:<10}")
    print(f"{'Global Random Search':<20} {grs_result:<10}")
