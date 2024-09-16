import numpy as np
import matplotlib.pyplot as plt
import time

def montar_tabuleiro(rainhas):
    n = len(rainhas)
    tabuleiro = np.zeros((n, n), dtype=str)
    tabuleiro[:, :] = '.'  

    for i, rainha in enumerate(rainhas):
        tabuleiro[rainha, i] = 'R'  

    tabuleiro_exibicao = np.flip(tabuleiro, axis=0)

    for linha in tabuleiro_exibicao:
        print(" ".join(linha))
    print("\n")

    tabuleiro_representacao = []
    for i, rainha in enumerate(rainhas):
        tabuleiro_representacao.append((8 - rainha, i + 1))  

    return tabuleiro_representacao

def calcular_ataques(rainhas):
    ataques = 0
    n = len(rainhas)

    for i in range(n):
        for j in range(i + 1, n):
            if rainhas[i] == rainhas[j]:
                ataques += 1
            if abs(rainhas[i] - rainhas[j]) == abs(i - j):
                ataques += 1

    return ataques

def aptidao(rainhas):
    return 28 - calcular_ataques(rainhas)

def perturb(rainhas, sigma=0.2):
    nova_solucao = np.copy(rainhas)
    idx = np.random.choice(len(rainhas)) 
    nova_posicao = (nova_solucao[idx] + int(np.random.normal(0, sigma) * 8)) % 8  
    nova_solucao[idx] = nova_posicao
    return nova_solucao

def resfriar(T, metodo, iteracao, iter_max):
    if metodo == 'escalonamento 1':
        return T * 0.98  
    elif metodo == 'escalonamento 2':
        return T / np.log(iteracao + 2)  
    elif metodo == 'escalonamento 3':
        return T - (T / iter_max) 
    return T

def tempera_simulada(metodo_resfriamento):
    rainhas = np.random.permutation(8) 
    temperatura = 10000
    iter_max = 1000
    historico_conflitos = []
    menor_conflito = aptidao(rainhas)
    melhor_rainhas = rainhas.copy()

    for iteracao in range(iter_max):
        nova_solucao = perturb(rainhas, sigma=0.2)  
        apt_nova = aptidao(nova_solucao)
        apt_atual = aptidao(rainhas)

        if apt_nova > apt_atual or np.exp((apt_nova - apt_atual) / temperatura) > np.random.rand():
            rainhas = nova_solucao

        apt_atual = aptidao(rainhas)
        if apt_atual > menor_conflito:
            menor_conflito = apt_atual
            melhor_rainhas = rainhas.copy()

        historico_conflitos.append(menor_conflito)

        temperatura = resfriar(temperatura, metodo_resfriamento, iteracao, iter_max)

        if menor_conflito == 28:
            break

    return menor_conflito, historico_conflitos, melhor_rainhas

def executar_temperas():
    metodos = ['escalonamento 1', 'escalonamento 2', 'escalonamento 3']
    resultados = {}

    for metodo in metodos:
        inicio = time.time()  

        menor_conflito, conflitos, melhor_rainhas = tempera_simulada(metodo)

        fim = time.time()  
        tempo_gasto = fim - inicio  

        resultados[metodo] = {
            'tempo': tempo_gasto,
            'menor_conflito': menor_conflito,
            'conflitos': conflitos,
            'melhor_rainhas': melhor_rainhas
        }

        plt.plot(conflitos, label=f'{metodo}')

    plt.xlabel('Iteração')
    plt.ylabel('Conflitos (quanto menor, melhor)')
    plt.title('Evolução dos conflitos para diferentes métodos de resfriamento')
    plt.legend()
    plt.show()

    for metodo, resultado in resultados.items():
        print(f"--- Método de Resfriamento: {metodo} ---")
        print(f"Número de conflitos: {28 - resultado['menor_conflito']}")
        print(f"Tempo de Execução: {resultado['tempo']:.4f} segundos")
        print("Tabuleiro final:")
        montar_tabuleiro(resultado['melhor_rainhas'])
        print(resultado['melhor_rainhas'] + 1)

executar_temperas()
