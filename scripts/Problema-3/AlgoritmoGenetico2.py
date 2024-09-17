import numpy as np
import random

# Parâmetros
A = 10
p = 20  # Número de dimensões do problema
tam_populacao = 100  # Tamanho da população
geracoes = 100  # Número de gerações
taxa_mutacao = 0.01  # Taxa de mutação
taxa_recombinacao = 0.85
limite_inferior = -10  # Limite inferior para as variáveis contínuas
limite_superior = 10  # Limite superior para as variáveis contínuas

# Função de Rastrigin
def rastrigin(x):
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão: rastrigin(x) + 1
def fitness(individuo):
    return rastrigin(individuo) + 1

# Inicializando a população aleatoriamente (com números flutuantes entre os limites)
def inicializar_populacao(tam_populacao, p, limite_inferior, limite_superior):
    return np.random.uniform(low=limite_inferior, high=limite_superior, size=(tam_populacao, p))

# Seleção por torneio
def selecao_torneio(populacao, aptidoes, tam_torneio=3):
    indices_torneio = np.random.choice(len(populacao), tam_torneio, replace=False)
    aptidoes_torneio = [aptidoes[i] for i in indices_torneio]
    indice_vencedor = indices_torneio[np.argmin(aptidoes_torneio)]
    return populacao[indice_vencedor]

# Recombinação via SBX (Simulated Binary Crossover)
def recombinacao_sbx(pai1, pai2, eta=2):
    if random.random() < taxa_recombinacao:
        filho1, filho2 = np.zeros_like(pai1), np.zeros_like(pai2)
        for i in range(len(pai1)):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u)**(1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u)))**(1 / (eta + 1))
            filho1[i] = 0.5 * ((1 + beta) * pai1[i] + (1 - beta) * pai2[i])
            filho2[i] = 0.5 * ((1 - beta) * pai1[i] + (1 + beta) * pai2[i])
        return filho1, filho2
    return pai1, pai2

# Mutação Gaussiana
def mutacao_gaussiana(individuo, taxa_mutacao, limite_inferior, limite_superior):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            valor_mutacao = np.random.normal(0, 1)  # Média 0 e desvio padrão 1
            individuo[i] += valor_mutacao
            # Limita os valores ao intervalo permitido
            individuo[i] = np.clip(individuo[i], limite_inferior, limite_superior)
    return individuo

# Algoritmo genético
def algoritmo_genetico():
    populacao = inicializar_populacao(tam_populacao, p, limite_inferior, limite_superior)
    melhores_aptidoes = []
    
    for geracao in range(geracoes):
        aptidoes = [fitness(individuo) for individuo in populacao]
        proxima_geracao = []

        # Seleção, recombinação e mutação
        for _ in range(tam_populacao // 2):
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)
            filho1, filho2 = recombinacao_sbx(pai1, pai2)
            proxima_geracao.append(mutacao_gaussiana(filho1, taxa_mutacao, limite_inferior, limite_superior))
            proxima_geracao.append(mutacao_gaussiana(filho2, taxa_mutacao, limite_inferior, limite_superior))
        
        populacao = np.array(proxima_geracao)
        # Armazena a melhor aptidão de cada geração
        melhores_aptidoes.append(min(aptidoes))

    return melhores_aptidoes

# Executando 100 rodadas
todas_aptidoes = [] 
for _ in range(100):
    melhores_aptidoes = algoritmo_genetico()
    todas_aptidoes.append(min(melhores_aptidoes))

# Analisando os resultados
melhor_aptidao = min(todas_aptidoes)
pior_aptidao = max(todas_aptidoes)
media_aptidao = np.mean(todas_aptidoes)
desvio_padrao_aptidao = np.std(todas_aptidoes)

print(f"Melhor aptidão: {melhor_aptidao}")
print(f"Pior aptidão: {pior_aptidao}")
print(f"Média da aptidão: {media_aptidao}")
print(f"Desvio padrão da aptidão: {desvio_padrao_aptidao}")
