

import pandas as pd
import kmeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Aqui vamos fazer as comparações entre algoritmos e com as versões das bibliotecas

df_teste = pd.read_csv('nba_teste.csv')
df_treino = pd.read_csv('nba_treino.csv')

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



# retirando 'TARGET_5Yrs' e normalizando
df_treino_kmeans = df_treino.drop(columns=['TARGET_5Yrs'])
df_treino_kmeans = (df_treino_kmeans - df_treino_kmeans.mean()) / df_treino_kmeans.std()

plot_real_clusters(df_treino_kmeans, df_treino['TARGET_5Yrs'])

k = 3
clusters, silhouette_score = kmeans.kmeans(k, df_treino_kmeans)
print(f"Silhouette Score: {silhouette_score}")
plot_kmeans_clusters(df_treino_kmeans, clusters, k,silhouette_score)




