import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

# Importa cada modulo
import kmeans
from knn import load_data, normalizacao, avaliar_knn


df_teste = pd.read_csv('nba_teste.csv')
df_treino = pd.read_csv('nba_treino.csv')


#Analise Kmeans

# Preparar dados para o k-means (sem a coluna TARGET_5Yrs)
df_treino_kmeans = df_treino.drop(columns=['TARGET_5Yrs'])
df_treino_kmeans = (df_treino_kmeans - df_treino_kmeans.mean()) / df_treino_kmeans.std()


# Plotando os clusters com PCA para visualização 2D
def plot_kmeans_clusters(df, clusters, k, silhouette_score):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.values)

    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')

    plt.title(f'{k}-means Clusters (PCA), silhouette score: {silhouette_score:.2f}')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')

    labels_legenda = [f'Grupo {i}' for i in range(k)]
    handles, _ = scatter.legend_elements()

    plt.legend(handles, labels_legenda, title="Clusters Identificados")
    plt.savefig(f'images/kmeans_clusters_k{k}.png', bbox_inches='tight', dpi=300)
    plt.show()
     
     

# 0 = Carreira durou menos de 5 anos, 1 = Carreira durou 5 anos ou mais, são os clusters anotados
def plot_real_clusters(df, real_clusters):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.values)

    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=real_clusters, cmap='viridis')

    plt.title('Carreira durou 5 anos ou mais? (PCA)')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')

    labels_legenda = ['Menos de 5 anos', '5 anos ou mais']
    handles, _ = scatter.legend_elements()

    plt.legend(handles, labels_legenda, title="Status da Carreira")
    plt.savefig('images/clusters_reais.png', bbox_inches='tight', dpi=300)
    plt.show()



# Rodar para k=2 e k=3
for k in [2, 3]:
    clusters, silhouette_score = kmeans.kmeans(k, df_treino_kmeans)
    print(f"\nK = {k} - Silhouette Score: {silhouette_score:.4f}")
    print(f"Centróides:")
    for i in range(k):
        cluster_points = df_treino_kmeans.values[clusters == i]
        if len(cluster_points) > 0:
            centroid = cluster_points.mean(axis=0)
            print(f"  Cluster {i}: {centroid[:5]}...") 
    plot_kmeans_clusters(df_treino_kmeans, clusters, k, silhouette_score)


plot_real_clusters(df_treino_kmeans, df_treino['TARGET_5Yrs'])

#Análise kNN

# Carregar e normalizar os dados - funções do knn
X_treino, y_treino, X_teste, y_teste = load_data()
X_treino, X_teste = normalizacao(X_treino, X_teste)

valores_k = [2, 10, 50, 99]
resultados = []
matrizes = []

for k in valores_k:
    acuracia, precisao, recall, f1, (VP, FP, FN, VN) = avaliar_knn(
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
print("\n📊 RESULTADOS NUMÉRICOS:")
print(df_metricas.to_string(index=False))

# Barras comparativas
df_melted = df_metricas.melt(id_vars='k', var_name='Métrica', value_name='Valor')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_melted, x='k', y='Valor', hue='Métrica')
plt.title('Comparação das métricas do kNN para diferentes k', fontsize=14)
plt.ylim(0, 1)
plt.ylabel('Valor')
plt.xlabel('k (número de vizinhos)')
plt.legend(loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('images/knn_metricas.png', dpi=300, bbox_inches='tight')
plt.show()

# Matrizes de Confusão
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for i, m in enumerate(matrizes):
    matriz_conf = np.array([[m['VP'], m['FP']],
                            [m['FN'], m['VN']]])
    
    sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Previsto 1', 'Previsto 0'],
                yticklabels=['Real 1', 'Real 0'],
                linewidths=0.5, linecolor='gray',
                ax=axes[i], cbar=False)
    axes[i].set_title(f'k = {m["k"]}', fontsize=12)

plt.suptitle('Matrizes de Confusão para diferentes valores de k', fontsize=14)
plt.tight_layout()
plt.savefig('images/matrizes_confusao_knn.png', dpi=300, bbox_inches='tight')
plt.show()

# evolução da acurácia
plt.figure(figsize=(8, 5))
plt.plot(df_metricas['k'], df_metricas['Acurácia'], marker='o', linestyle='-', linewidth=2, color='green')
plt.title('Evolução da Acurácia do kNN')
plt.xlabel('k')
plt.ylabel('Acurácia')
plt.grid(True)
plt.xticks(valores_k)
plt.ylim(0.5, 0.8)
plt.tight_layout()
plt.savefig('images/evolucao_acuracia_knn.png', dpi=300, bbox_inches='tight')
plt.show()