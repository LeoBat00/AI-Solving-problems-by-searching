import numpy as np

# Hill Climbing
def hill_climbing(f, bounds, epsilon, Nmax):
    """
    Executa o algoritmo Hill Climbing.
    
    :param f: Função objetivo.
    :param bounds: Limites das variáveis (matriz 2D com limites inferiores e superiores).
    :param epsilon: Tamanho do passo para gerar vizinhos.
    :param Nmax: Número máximo de iterações.
    :return: Melhor solução encontrada e seu valor de função objetivo.
    """
    # Inicializar a solução melhor no limite inferior
    xbest = bounds[:, 0]
    fbest = f(xbest)
    
    for _ in range(Nmax):
        # Gerar um novo candidato dentro da vizinhança
        xcand = xbest + np.random.uniform(-epsilon, epsilon, size=xbest.shape)
        xcand = np.clip(xcand, bounds[:, 0], bounds[:, 1])
        
        # Avaliar o novo candidato
        fcand = f(xcand)
        
        # Atualizar a solução melhor se o candidato for melhor
        if fcand < fbest:
            xbest = xcand
            fbest = fcand
    
    return xbest, fbest

# Local Random Search (LRS)
def perturb(xbest, sigma, bounds):
    """
    Gera uma nova solução candidata adicionando ruído gaussiano à solução atual.
    
    :param xbest: Solução atual (vetor de variáveis).
    :param sigma: Desvio padrão do ruído gaussiano.
    :param bounds: Limites inferiores e superiores para as variáveis.
    :return: Nova solução candidata.
    """
    # Gerar ruído gaussiano
    noise = np.random.normal(0, sigma, size=xbest.shape)
    
    # Gerar nova solução candidata
    xcand = xbest + noise
    
    # Aplicar restrições de caixa (garantir que a solução esteja dentro dos limites)
    xcand = np.clip(xcand, bounds[:, 0], bounds[:, 1])
    
    return xcand

def local_random_search(f, bounds, sigma, Nmax):
    """
    Executa o algoritmo de Busca Local Aleatória (LRS).
    
    :param f: Função objetivo.
    :param bounds: Limites das variáveis (matriz 2D com limites inferiores e superiores).
    :param sigma: Desvio padrão para a perturbação.
    :param Nmax: Número máximo de iterações.
    :return: Melhor solução encontrada e seu valor de função objetivo.
    """
    # Inicializar a solução melhor
    xbest = np.random.uniform(bounds[:, 0], bounds[:, 1])
    fbest = f(xbest)
    
    for _ in range(Nmax):
        # Gerar uma nova solução candidata
        xcand = perturb(xbest, sigma, bounds)
        
        # Avaliar a nova solução candidata
        fcand = f(xcand)
        
        # Atualizar a solução melhor se a candidata for melhor
        if fcand < fbest:
            xbest = xcand
            fbest = fcand
    
    return xbest, fbest

# Definindo os parâmetros
epsilon = 0.1
sigma = epsilon * 0.5  # Definido como uma fração de epsilon
bounds = np.array([[-100, 100], [-100, 100]])  # Exemplo para duas variáveis

# Função objetivo exemplo
def objective_function(x):
    return np.sum(x**2)  # Exemplo simples: minimizar a soma dos quadrados

# Executando Hill Climbing
xbest_hc, fbest_hc = hill_climbing(objective_function, bounds, epsilon, Nmax=1000)
print(f"Hill Climbing - Melhor solução: {xbest_hc}, Valor: {fbest_hc}")

# Executando Local Random Search (LRS)
xbest_lrs, fbest_lrs = local_random_search(objective_function, bounds, sigma, Nmax=1000)
print(f"LRS - Melhor solução: {xbest_lrs}, Valor: {fbest_lrs}")
