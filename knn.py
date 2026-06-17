

import pandas as pd
import numpy as np

def load_data():
    treino = pd.read_csv('nba_treino.csv')
    teste = pd.read_csv('nba_teste.csv')
    
    #ler os dados e separar em X e y para treino e teste
    X_treino = treino.drop('TARGET_5Yrs', axis=1).values
    y_treino = treino['TARGET_5Yrs'].values
    X_teste = teste.drop('TARGET_5Yrs', axis=1).values
    y_teste = teste['TARGET_5Yrs'].values
    return X_treino, y_treino, X_teste, y_teste

def normalizacao(X_treino, X_teste):
    media = X_treino.mean(axis=0)
    desvio = X_treino.std(axis=0)
    desvio[desvio == 0] = 1
    X_treino_norm = (X_treino - media) / desvio
    X_teste_norm = (X_teste - media) / desvio
    return X_treino_norm, X_teste_norm

def distancia_euclidiana(a, b):
    return np.sqrt(np.sum((a - b) ** 2))
  #quanto menor a distância, mais próximos os pontos, mais semelhantes são os jogadores.
  
# calcula a distância euclidiana entre a amostra e cada ponto de treino, ordena e seleciona os k pontos mais próximos. 
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
# conta os votos para cada classe e retorna a classe com mais votos.


def avaliar_knn(k, X_treino, y_treino, X_teste, y_teste):
    predicoes = []
    for i in range(len(X_teste)):
        pred = classificar_knn(X_teste[i], X_treino, y_treino, k)
        predicoes.append(pred)
    predicoes = np.array(predicoes)
    
    #Matriz de confusão:
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

def main():
    #carregando
    X_treino, y_treino, X_teste, y_teste = load_data()
    #normalizando
    X_treino, X_teste = normalizacao(X_treino, X_teste)
    
    valores_k = [2, 10, 50, 99]
    for k in valores_k:
        
        print(f"\n k = {k} ")
        
        acuracia, precisao, recall, f1, (VP, FP, FN, VN) = avaliar_knn(k, X_treino, y_treino, X_teste, y_teste)
        print("Matriz de confusão:")
        print(f"        Previsto 1   Previsto 0")
        print(f"Real 1:  {VP:6d}       {FN:6d}")
        print(f"Real 0:  {FP:6d}       {VN:6d}")
        print(f"Acurácia: {acuracia:.4f}")
        print(f"Precisão: {precisao:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1: {f1:.4f}")

if __name__ == "__main__":
    main()
