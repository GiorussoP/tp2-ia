# Numpy e pandas, para manipulação de dados
import numpy as np
import pandas as pd

# Nossos módulos
import kmeans
import knn
import util

# Módulos do Scikit-Learn para comparação
from sklearn.cluster import KMeans as SKKMeans
from sklearn.neighbors import KNeighborsClassifier as SKKNN
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix



# ----------------------
# Análise kNN
# ----------------------

# Carregando e normalizando os dados
X_treino, y_treino, X_teste, y_teste = util.load_data_knn()
X_treino, X_teste = util.normalizacao(X_treino), util.normalizacao(X_teste)

# Rodando para os diferentes valores de k e coletando os resultados
valores_k = [2, 10, 50, 99]
resultados = []
matrizes = []
for k in valores_k:
    acuracia, precisao, recall, f1, (VP, FP, FN, VN) = knn.avaliar_knn(
        k, X_treino, y_treino, X_teste, y_teste
    )
    resultados.append({
        'k': k,
        'Acurácia': acuracia,
        'Precisão': precisao,
        'Recall': recall,
        'F1': f1
    })
    matrizes.append({
        'k': k,
        'VP': VP, 'FP': FP, 'FN': FN, 'VN': VN
    })
df_metricas = pd.DataFrame(resultados)

# Plotando os gráficos
util.plot_metricas_knn(df_metricas)
util.plot_matrizes_knn(matrizes)
util.plot_accuracia_knn(df_metricas, valores_k)


# Comparando com o knn do scikit-learn, usando os mesmos k's e os mesmos dados
sk_resultados = []
sk_matrizes = []
for k in valores_k:
    # KNN do scikit-learn
    knn_sk = SKKNN(n_neighbors=k)
    knn_sk.fit(X_treino, y_treino)
    y_pred_sk = knn_sk.predict(X_teste)
    
    # Métricas
    acuracia_sk = accuracy_score(y_teste, y_pred_sk)
    precisao_sk = precision_score(y_teste, y_pred_sk, zero_division=0)
    recall_sk = recall_score(y_teste, y_pred_sk, zero_division=0)
    f1_sk = f1_score(y_teste, y_pred_sk, zero_division=0)

    sk_resultados.append({
        'k': k,
        'Acurácia': acuracia_sk,
        'Precisão': precisao_sk,
        'Recall': recall_sk,
        'F1': f1_sk
    })
    
    tn, fp, fn, tp = confusion_matrix(y_teste, y_pred_sk).ravel()
    sk_matrizes.append({
        'k': k,
        'VP': tp, 'FP': fp, 'FN': fn, 'VN': tn
    })
df_metricas_sk = pd.DataFrame(sk_resultados)
df_metricas['Fonte'] = ' Nossa implementação'
df_metricas_sk['Fonte'] = 'Scikit-Learn'
df_combinado = pd.concat([df_metricas, df_metricas_sk], ignore_index=True)
df_melted_comp = df_combinado.melt(id_vars=['k', 'Fonte'], var_name='Métrica', value_name='Valor')

util.plot_comparacao_acuracia(df_melted_comp)
util.plot_sk_matrizes(sk_matrizes)



# ----------------------
# Análise k-means
# ----------------------

# Carregando e normalizando os dados
df_kmeans, y_real = util.load_data_kmeans()
df_kmeans_norm = util.normalizacao(df_kmeans)

# Rodando pra k=2 e k=3
for k in [2, 3]:
    # Rodando o algoritmo
    clusters, silhouette_score, centroids = kmeans.kmeans(k, df_kmeans_norm)

    # Plotando os clusters encontrados, com PCA para visualização
    util.plot_kmeans_clusters(df_kmeans_norm, clusters, k, silhouette_score, centroids)

# Rodando o K-means para vários k's diferentes e coletando os silhouette scores para comparação
silhouette_scores = []
centroids_list = []
for k in range(1, 10):
    clusters, silhouette_score, centroids = kmeans.kmeans(k, df_kmeans_norm)
    silhouette_scores.append(silhouette_score)
    centroids_list.append(centroids)

# Escrevendo os silhouette scores e centróides para cada k
print("Silhouette Scores para diferentes valores de k:")
for k, score in enumerate(silhouette_scores, start=1):
    print(f"k={k}: Silhouette Score = {score:.4f}")

for k, centroids in enumerate(centroids_list, start=1):
    print(f"\nCentróides para k={k}:")
    print(centroids)


# Plotar silhouette scores
util.plot_silhouette_scores(range(1, 10), silhouette_scores)

# Plotando os clusters da melhor k encontrada
melhor_k = np.argmax(silhouette_scores) + 1  # +1  porque começamos a contar de 1
clusters, silhouette_score, centroids = kmeans.kmeans(melhor_k, df_kmeans_norm)
util.plot_kmeans_clusters(df_kmeans_norm, clusters, melhor_k, silhouette_score, centroids)

# Comparando com os rótulos reais
util.plot_real_clusters(df_kmeans_norm, y_real)

# Comparaando com o K-means do Scikit-Learn, usando 2 e 3 encontrado pelo nosso algoritmo
sk_kmeans = SKKMeans(n_clusters=2, random_state=42, n_init=10)
sk_clusters = sk_kmeans.fit_predict(df_kmeans_norm)
sk_centroids = sk_kmeans.cluster_centers_
util.plot_sk_kmeans_clusters(df_kmeans_norm, sk_clusters, 2, sk_centroids)

sk_kmeans_3 = SKKMeans(n_clusters=3, random_state=42, n_init=10)
sk_clusters_3 = sk_kmeans_3.fit_predict(df_kmeans_norm)
sk_centroids_3 = sk_kmeans_3.cluster_centers_
util.plot_sk_kmeans_clusters(df_kmeans_norm, sk_clusters_3, 3, sk_centroids_3)
