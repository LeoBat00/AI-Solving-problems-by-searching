import numpy as np
import random

# Parâmetros
A = 10
p = 20  # Número de dimensões do problema
pop_size = 100  # Tamanho da população
generations = 100  # Número de gerações
mutation_rate = 0.01  # Taxa de mutação
recombination_rate = 0.85
lower_bound = -10  # Limite inferior para as variáveis contínuas
upper_bound = 10  # Limite superior para as variáveis contínuas

# Função de Rastrigin
def rastrigin(x):
    return A * p + sum([(xi**2 - A * np.cos(2 * np.pi * xi)) for xi in x])

# Função de aptidão: rastrigin(x) + 1
def fitness(individual):
    return rastrigin(individual) + 1

# Inicializando a população aleatoriamente (com números flutuantes entre os limites)
def initialize_population(pop_size, p, lower_bound, upper_bound):
    return np.random.uniform(low=lower_bound, high=upper_bound, size=(pop_size, p))

# Seleção por torneio
def tournament_selection(population, fitnesses, tournament_size=3):
    tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
    tournament_fitnesses = [fitnesses[i] for i in tournament_indices]
    winner_index = tournament_indices[np.argmin(tournament_fitnesses)]
    return population[winner_index]

# Recombinação via SBX (Simulated Binary Crossover)
def sbx_crossover(parent1, parent2, eta=2):
    if random.random() < recombination_rate:
        child1, child2 = np.zeros_like(parent1), np.zeros_like(parent2)
        for i in range(len(parent1)):
            u = random.random()
            if u <= 0.5:
                beta = (2 * u)**(1 / (eta + 1))
            else:
                beta = (1 / (2 * (1 - u)))**(1 / (eta + 1))
            child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
            child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        return child1, child2
    return parent1, parent2

# Mutação Gaussiana
def gaussian_mutation(individual, mutation_rate, lower_bound, upper_bound):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            mutation_value = np.random.normal(0, 1)  # Média 0 e desvio padrão 1
            individual[i] += mutation_value
            # Limita os valores ao intervalo permitido
            individual[i] = np.clip(individual[i], lower_bound, upper_bound)
    return individual

# Algoritmo genético
def genetic_algorithm():
    population = initialize_population(pop_size, p, lower_bound, upper_bound)
    best_fitnesses = []
    
    for generation in range(generations):
        fitnesses = [fitness(individual) for individual in population]
        next_generation = []

        # Seleção, recombinação e mutação
        for _ in range(pop_size // 2):
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = sbx_crossover(parent1, parent2)
            next_generation.append(gaussian_mutation(child1, mutation_rate, lower_bound, upper_bound))
            next_generation.append(gaussian_mutation(child2, mutation_rate, lower_bound, upper_bound))
        
        population = np.array(next_generation)
        
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
