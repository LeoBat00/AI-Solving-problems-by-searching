import numpy as np
import random

# Parâmetros
A = 10
p = 20  # Número de dimensões do problema
pop_size = 100  # Tamanho da população
generations = 100  # Número de gerações
mutation_rate = 0.01  # Taxa de mutação
recombination_rate = 0.85
bit_length = 8  # Número de bits para representar cada variável
lower_bound = -10  # Limite inferior para as variáveis contínuas
upper_bound = 10  # Limite superior para as variáveis contínuas

# Função de Rastrigin
def rastrigin(x):
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função para converter binário em valor contínuo
def decode(binary, inf, sup):
    s = 0
    for i in range(len(binary)):
        s += binary[len(binary) - i - 1] * 2**i
    return inf + (sup - inf) / (2**len(binary) - 1) * s

# Inicializando a população aleatoriamente com bits (0 ou 1)
def initialize_population(pop_size, p, bit_length):
    # A população é uma matriz de 0s e 1s, onde cada variável é representada por `bit_length` bits
    return np.random.uniform(low=0, high=2, size=(pop_size, p * bit_length)).astype(int)

# Decodificando um indivíduo de bits para valores contínuos
def decode_individual(individual, p, bit_length, lower_bound, upper_bound):
    decoded = []
    for i in range(p):
        binary_var = individual[i * bit_length:(i + 1) * bit_length]
        decoded.append(decode(binary_var, lower_bound, upper_bound))
    return decoded

# Função de aptidão: rastrigin(x) + 1
def fitness(individual):
    decoded_individual = decode_individual(individual, p, bit_length, lower_bound, upper_bound)
    return rastrigin(decoded_individual) + 1

# Seleção por roleta
def roulette_selection(population, fitnesses):
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for i, fit in enumerate(fitnesses):
        current += fit
        if current > pick:
            return population[i]

# Crossover de um ponto (opera em bits)
def crossover(parent1, parent2):
    if random.random() < recombination_rate:
        point = random.randint(1, len(parent1) - 1)
        child1 = np.concatenate((parent1[:point], parent2[point:]))
        child2 = np.concatenate((parent2[:point], parent1[point:]))
        return child1, child2
    return parent1, parent2

# Mutação (opera em bits)
def mutate(individual):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = 1 - individual[i]  # Inverte o bit
    return individual

# Algoritmo genético
def genetic_algorithm():
    population = initialize_population(pop_size, p, bit_length)
    best_fitnesses = []
    
    for generation in range(generations):
        fitnesses = [fitness(individual) for individual in population]
        next_generation = []

        # Seleção, recombinação e mutação
        for _ in range(pop_size // 2):
            parent1 = roulette_selection(population, fitnesses)
            parent2 = roulette_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            next_generation.append(mutate(child1))
            next_generation.append(mutate(child2))
        
        population = next_generation
        
        # Armazena a melhor aptidão de cada geração
        best_fitnesses.append(min(fitnesses))

    return best_fitnesses

# Executando 100 rodadas
all_fitnesses = [] 
for _ in range(100):
    best_fitnesses = genetic_algorithm()
    all_fitnesses.append(min(best_fitnesses))

# Analisando os resultados
best_fitness = min(all_fitnesses)
worst_fitness = max(all_fitnesses)
mean_fitness = np.mean(all_fitnesses)
std_fitness = np.std(all_fitnesses)

print(f"Melhor aptidão: {best_fitness}")
print(f"Pior aptidão: {worst_fitness}")
print(f"Média da aptidão: {mean_fitness}")
print(f"Desvio padrão da aptidão: {std_fitness}")
