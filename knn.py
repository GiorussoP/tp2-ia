import pandas as pd
import numpy as np

def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
  # Quanto menor a distância, mais próximos os pontos, mais semelhantes são os jogadores.
  
# Calcula a distância euclidiana entre a amostra e cada ponto de treino, ordena e seleciona os k pontos mais próximos. 
def classificar_knn(amostra, X_treino, y_treino, k):
    distancias = []
    for i in range(len(X_treino)):
        dist = distancia_euclidiana(amostra, X_treino[i])
        distancias.append((dist, y_treino[i]))
    distancias.sort(key=lambda x: x[0])
    k_mais_proximos = distancias[:k]
    votos = [0, 0]
    for _, classe in k_mais_proximos:
        votos[classe] += 1
    return 0 if votos[0] > votos[1] else 1
# Conta os votos para cada classe e retorna a classe com mais votos.

def avaliar_knn(k, X_treino, y_treino, X_teste, y_teste):
    predicoes = []
    for i in range(len(X_teste)):
        pred = classificar_knn(X_teste[i], X_treino, y_treino, k)
        predicoes.append(pred)
    predicoes = np.array(predicoes)

    # Matriz de confusão:
    # [ [VP, FP],
    #   [FN, VN] ]
    VP = np.sum((predicoes == 1) & (y_teste == 1))
    FP = np.sum((predicoes == 1) & (y_teste == 0))
    FN = np.sum((predicoes == 0) & (y_teste == 1))
    VN = np.sum((predicoes == 0) & (y_teste == 0))

    acuracia = (VP + VN) / (VP + FP + FN + VN)
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0
    recall = VP / (VP + FN) if (VP + FN) > 0 else 0
    f1 = 2 * (precisao * recall) / (precisao + recall) if (precisao + recall) > 0 else 0

    return acuracia, precisao, recall, f1, (VP, FP, FN, VN)
