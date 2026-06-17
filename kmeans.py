
import numpy as np
import pandas as pd


# kmeans, o algoritmo para quando nenhum ponto mudou de grupo
def kmeans(k: int, df: pd.DataFrame, n_init: int = 10, max_iter: int = 3000):
    if k <= 1:
        raise ValueError("O número de clusters deve ser maior que um.")
    
    X = df.values.astype(float)
    n_samples, n_features = X.shape
    
    best_silhouette_score = -1
    best_labels = None

    # n_init dita quantas vezes vamos tentar inicializações diferentes
    for _ in range(n_init):
        # Inicialização aleatória dos centroides
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        labels = np.zeros(n_samples, dtype=int)
        
        # Loop de convergência do K-Means
        for _ in range(max_iter):
            # 1. Atribuição dos pontos ao centroide mais próximo
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)
            
            # Se as labels não mudaram, o algoritmo convergiu
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels
            
            # 2. Atualização dos centroides
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)
        
        # ---- CÁLCULO DA SILHOUETTE SCORE CORRIGIDO ----
        silhouette_scores = np.zeros(n_samples)
        for i, point in enumerate(X):
            own_cluster = labels[i]
            cluster_mask = labels == own_cluster
            
            # Se o cluster só tem 1 ponto, o score por definição é 0
            if np.sum(cluster_mask) <= 1:
                silhouette_scores[i] = 0
                continue
                
            # 'a': distância média para os OUTROS pontos do mesmo cluster
            # Removemos o próprio ponto da média filtrando distâncias > 0
            intra_distances = np.linalg.norm(X[cluster_mask] - point, axis=1)
            a = np.mean(intra_distances[intra_distances > 0]) 
            
            # 'b': menor distância média para pontos de outro cluster
            b = np.inf
            for j in range(k):
                if j != own_cluster:
                    other_cluster_points = X[labels == j]
                    if len(other_cluster_points) > 0:
                        mean_dist = np.mean(np.linalg.norm(other_cluster_points - point, axis=1))
                        b = min(b, mean_dist)
            
            # Cálculo final do ponto
            silhouette_scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

        current_silhouette = silhouette_scores.mean()
        
        # Salva o melhor resultado baseado no Silhouette Score
        if current_silhouette > best_silhouette_score:
            best_silhouette_score = current_silhouette
            best_labels = labels.copy()
            
    return best_labels, best_silhouette_score