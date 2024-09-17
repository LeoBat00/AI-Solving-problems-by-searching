import numpy as np
import random

# Parâmetros
A = 10
p = 20  # Número de dimensões do problema
tam_populacao = 100  # Tamanho da população
geracoes = 100  # Número de gerações
taxa_mutacao = 0.01  # Taxa de mutação
taxa_recombinacao = 0.85
tam_bits = 8  # Número de bits para representar cada variável
limite_inferior = -10  # Limite inferior para as variáveis contínuas
limite_superior = 10  # Limite superior para as variáveis contínuas

# Função de Rastrigin
def rastrigin(x):
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função para converter binário em valor contínuo
def decodificar(binario, inf, sup):
    s = 0
    for i in range(len(binario)):
        s += binario[len(binario) - i - 1] * 2**i
    return inf + (sup - inf) / (2**len(binario) - 1) * s

# Inicializando a população aleatoriamente com bits (0 ou 1)
def inicializar_populacao(tam_populacao, p, tam_bits):
    # A população é uma matriz de 0s e 1s, onde cada variável é representada por `tam_bits` bits
    return np.random.uniform(low=0, high=2, size=(tam_populacao, p * tam_bits)).astype(int)

# Decodificando um indivíduo de bits para valores contínuos
def decodificar_individuo(individuo, p, tam_bits, limite_inferior, limite_superior):
    decodificado = []
    for i in range(p):
        var_binaria = individuo[i * tam_bits:(i + 1) * tam_bits]
        decodificado.append(decodificar(var_binaria, limite_inferior, limite_superior))
    return decodificado

# Função de aptidão: rastrigin(x) + 1
def fitness(individuo):
    individuo_decodificado = decodificar_individuo(individuo, p, tam_bits, limite_inferior, limite_superior)
    return rastrigin(individuo_decodificado) + 1

# Seleção por roleta
def selecao_roleta(populacao, aptidoes):
    aptidao_total = sum(aptidoes)
    escolha = random.uniform(0, aptidao_total)
    atual = 0
    for i, apt in enumerate(aptidoes):
        atual += apt
        if atual > escolha:
            return populacao[i]

# Crossover de um ponto (opera em bits)
def crossover(pai1, pai2):
    if random.random() < taxa_recombinacao:
        ponto = random.randint(1, len(pai1) - 1)
        filho1 = np.concatenate((pai1[:ponto], pai2[ponto:]))
        filho2 = np.concatenate((pai2[:ponto], pai1[ponto:]))
        return filho1, filho2
    return pai1, pai2

# Mutação (opera em bits)
def mutacao(individuo):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            individuo[i] = 1 - individuo[i]  # Inverte o bit
    return individuo

# Algoritmo genético
def algoritmo_genetico():
    populacao = inicializar_populacao(tam_populacao, p, tam_bits)
    melhores_aptidoes = []
    
    for geracao in range(geracoes):
        aptidoes = [fitness(individuo) for individuo in populacao]
        proxima_geracao = []

        # Seleção, recombinação e mutação
        for _ in range(tam_populacao // 2):
            pai1 = selecao_roleta(populacao, aptidoes)
            pai2 = selecao_roleta(populacao, aptidoes)
            filho1, filho2 = crossover(pai1, pai2)
            proxima_geracao.append(mutacao(filho1))
            proxima_geracao.append(mutacao(filho2))
        
        populacao = proxima_geracao
        
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
