import numpy as np
import pandas as pd


# K-means, o algoritmo para quando nenhum ponto mudou de grupo
def kmeans(k: int, df: pd.DataFrame, n_init: int = 100, max_iter: int = 3000):
    if k <= 1:
        # Todos no mesmo cluster, silhouette score zero. Centróide é o ponto médio
        return np.zeros(len(df), dtype=int), 0.0, df.mean().values.reshape(1, -1)

    X = df.values.astype(float)
    n_samples, n_features = X.shape

    best_silhouette_score = -1
    best_centroids = None
    best_labels = None

    # n_init diz quantas vezes o algoritmo irá sortear os clusters iniciais
    for _ in range(n_init):
        # Sorteio dos centroides iniciais
        centroids = X[np.random.choice(n_samples, k, replace=False)]
        labels = np.zeros(n_samples, dtype=int)

        # Loop de convergência do K-Means, quase sempre converge antes de atingir max_iter
        for _ in range(max_iter):
            # Classificação dos pontos
            distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
            new_labels = np.argmin(distances, axis=1)

            # Se as labels são as mesmas, o algoritmo convergiu, podemos sair do loop
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Atualizando os centroides com a média dos pontos de cada cluster
            for i in range(k):
                cluster_points = X[labels == i]
                if len(cluster_points) > 0:
                    centroids[i] = cluster_points.mean(axis=0)

        # Sillhouette Score: Avaliando a qualidade da clusterização
        silhouette_scores = np.zeros(n_samples)
        for i, point in enumerate(X):
            own_cluster = labels[i]
            cluster_mask = labels == own_cluster

            # Cluster de 1 ponto = score zero
            if np.sum(cluster_mask) <= 1:
                silhouette_scores[i] = 0
                continue

            # 'a': distância média para os OUTROS pontos do mesmo cluster
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

            # Cálculo da sillhouette score para o ponto atual
            silhouette_scores[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0

        current_silhouette = silhouette_scores.mean()

        # Salva o melhor resultado encontrado, entre os sorteios de centroides iniciais
        if current_silhouette > best_silhouette_score:
            best_silhouette_score = current_silhouette
            best_labels = labels.copy()
            best_centroids = centroids.copy()

    return best_labels, best_silhouette_score, best_centroids
