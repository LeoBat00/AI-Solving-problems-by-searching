import numpy as np
import matplotlib.pyplot as plt
import time

def montar_tabuleiro(rainhas):
    """Monta e exibe o tabuleiro com a posição das rainhas."""
    n = len(rainhas)
    tabuleiro = np.zeros((n, n), dtype=str)
    tabuleiro[:, :] = '.'  # Preenche o tabuleiro com pontos

    for i, rainha in enumerate(rainhas):
        tabuleiro[rainha, i] = 'R'  # Coloca a rainha na posição correta

    tabuleiro_exibicao = np.flip(tabuleiro, axis=0)  # Inverte para a exibição ficar correta

    for linha in tabuleiro_exibicao:
        print(" ".join(linha))  # Exibe o tabuleiro
    print("\n")

    tabuleiro_representacao = []
    for i, rainha in enumerate(rainhas):
        tabuleiro_representacao.append((8 - rainha, i + 1))  # Converte para formato representacional

    return tabuleiro_representacao

def calcular_ataques(rainhas):
    """Calcula o número de pares de rainhas que se atacam."""
    ataques = 0
    n = len(rainhas)

    for i in range(n):
        for j in range(i + 1, n):
            if rainhas[i] == rainhas[j]:  # Verifica se estão na mesma linha
                ataques += 1
            if abs(rainhas[i] - rainhas[j]) == abs(i - j):  # Verifica se estão na mesma diagonal
                ataques += 1

    return ataques

def aptidao(rainhas):
    """Função de aptidão: quanto maior, melhor. O valor máximo é 28 (sem ataques)."""
    return 28 - calcular_ataques(rainhas)

def perturb(rainhas, sigma=0.2):
    """Perturbação da solução atual alterando a posição de uma rainha aleatoriamente."""
    nova_solucao = np.copy(rainhas)
    idx = np.random.choice(len(rainhas))  # Seleciona uma coluna aleatória
    nova_posicao = (nova_solucao[idx] + int(np.random.normal(0, sigma) * 8)) % 8  # Nova posição
    nova_solucao[idx] = nova_posicao
    return nova_solucao

def resfriar(T, metodo, iteracao, iter_max):
    """Define as diferentes formas de resfriamento da temperatura."""
    if metodo == 'escalonamento 1':
        return T * 0.98  # Decaimento de 2% a cada iteração
    elif metodo == 'escalonamento 2':
        return T / (1 + 0.02 * np.sqrt(T))  # Resfriamento adaptativo baseado em T
    elif metodo == 'escalonamento 3':
        delta_T = (T - 1e-3) / iter_max  # Resfriamento linear até uma temperatura muito baixa
        return T - delta_T
    return T

def tempera_simulada(metodo_resfriamento, solucoes_distintas):
    """Executa a têmpera simulada até encontrar uma nova solução distinta e válida."""
    iter_max = 1000
    rainhas = np.random.permutation(8)  # Solução inicial aleatória
    temperatura = 1000
    historico_conflitos = []
    menor_conflito = aptidao(rainhas)
    melhor_rainhas = rainhas.copy()

    for iteracao in range(iter_max):
        nova_solucao = perturb(rainhas, sigma=0.2)  # Gera uma nova solução
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

        # Se encontrarmos uma solução ótima (28 pares não atacantes)
        if menor_conflito == 28:
            sol_tuple = tuple(melhor_rainhas)
            if sol_tuple not in solucoes_distintas:
                solucoes_distintas.add(sol_tuple)  # Adiciona ao conjunto de soluções
                print(f"Nova solução encontrada ({len(solucoes_distintas)} de 92): {melhor_rainhas}")
                montar_tabuleiro(melhor_rainhas)
            break

    return menor_conflito, historico_conflitos, melhor_rainhas

def executar_temperas():
    """Executa a têmpera simulada até encontrar as 92 soluções distintas."""
    metodos = ['escalonamento 1', 'escalonamento 2', 'escalonamento 3']
    resultados = {}

    for metodo in metodos:
        inicio = time.time()  # Marca o tempo de início
        solucoes_distintas = set()

        while len(solucoes_distintas) < 92:
            _, conflitos, melhor_rainhas = tempera_simulada(metodo, solucoes_distintas)

        fim = time.time()  # Marca o tempo de término
        tempo_gasto = fim - inicio  # Calcula o tempo de execução

        resultados[metodo] = {
            'tempo': tempo_gasto,
            'solucoes_distintas': len(solucoes_distintas),
            'melhor_rainhas': melhor_rainhas
        }

        print(f"--- Método {metodo} completado ---")
        print(f"Tempo gasto para encontrar 92 soluções distintas: {tempo_gasto:.4f} segundos\n")

    return resultados

resultados = executar_temperas()
