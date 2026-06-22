import pandas as pd
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Carrega os dados para o KNN, separando em treino e teste
def load_data_knn():
    treino = pd.read_csv('./data/nba_treino.csv')
    teste = pd.read_csv('./data/nba_teste.csv')
    
    X_treino = treino.drop('TARGET_5Yrs', axis=1).values
    y_treino = treino['TARGET_5Yrs'].values
    X_teste = teste.drop('TARGET_5Yrs', axis=1).values
    y_teste = teste['TARGET_5Yrs'].values
    return X_treino, y_treino, X_teste, y_teste


# Carrega os dados para o K-means, retornando o DataFrame completo e os rótulos reais para comparação
def load_data_kmeans():
    df_treino = pd.read_csv('./data/nba_treino.csv')
    df_teste = pd.read_csv('./data/nba_teste.csv')
    df_todos = pd.concat([df_treino, df_teste], ignore_index=True)
    
    # Retirando os rótulos reais do dataframe
    y_real = df_todos['TARGET_5Yrs'].values
    df_features = df_todos.drop(columns=['TARGET_5Yrs'])
    return df_features, y_real


# Normalização para os dados
def normalizacao(df):

    # Funciona tanto para DataFrames quanto para Arrays do NumPy
    media = df.mean(axis=0)
    desvio = df.std(axis=0)
    
    if isinstance(desvio, np.ndarray):
        desvio[desvio == 0] = 1
    else:
        desvio = desvio.replace(0, 1)
        
    return (df - media) / desvio



def plot_kmeans_clusters(df, clusters, k, silhouette_score, centroids):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.values)

    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')
    reduced_centroids = pca.transform(centroids)
    m_centroids = plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], c='red', marker='X', s=100, label='Centroides')

    plt.title(f'{k}-means Clusters (PCA), silhouette score: {silhouette_score:.2f}')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')

    labels_legenda = [f'Grupo {i}' for i in range(k)]
    handles, _ = scatter.legend_elements()

    handles.append(m_centroids)
    labels_legenda.append('Centroides')

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

def plot_silhouette_scores(k_values, silhouette_scores):
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o', linestyle='-', linewidth=2, color='blue')
    plt.title('Silhouette Score para diferentes valores de k')
    plt.xlabel('k (número de clusters)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('images/silhouette_scores.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_metricas_knn(df_metricas):
    
    print("Resultados numéricos:")
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

def plot_matrizes_knn(matrizes):
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

def plot_accuracia_knn(df_metricas,valores_k):
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

def plot_sk_kmeans_clusters(df, clusters, k, centroids):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(df.values)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clusters, cmap='viridis')

    reduced_centroids = pca.transform(centroids)
    m_centroids = plt.scatter(reduced_centroids[:, 0], reduced_centroids[:, 1], 
                              c='red', marker='X', s=100, label='Centroides')

    plt.title(f'Scikit-Learn {k}-means Clusters (PCA)')
    plt.xlabel('Componente principal 1')
    plt.ylabel('Componente principal 2')

    labels_legenda = [f'Grupo {i}' for i in range(k)]
    handles, _ = scatter.legend_elements()

    handles.append(m_centroids)
    labels_legenda.append('Centroides')

    plt.legend(handles, labels_legenda, title="Clusters Scikit-Learn")
    plt.savefig(f'images/sk_kmeans_clusters_k{k}.png', bbox_inches='tight', dpi=300)
    plt.show()

def plot_comparacao_acuracia(df_combinado):

    # Gráfico comparativo de Acurácia para cada k
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_combinado[df_combinado['Métrica'] == 'Acurácia'], x='k', y='Valor', hue='Fonte', palette='Set2')
    plt.title('Comparação Direta de Acurácia: Nossa implementação vs Scikit-Learn', fontsize=14)
    plt.ylim(0, 1)
    plt.ylabel('Acurácia')
    plt.xlabel('k (número de vizinhos)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('images/knn_comparacao_acuracia.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_sk_matrizes(sk_matrizes):

    # Matrizes de Confusão do Scikit-Learn
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for i, m in enumerate(sk_matrizes):
        matriz_conf = np.array([[m['VP'], m['FP']],
                                [m['FN'], m['VN']]])
        
        sns.heatmap(matriz_conf, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=['Previsto 1', 'Previsto 0'],
                    yticklabels=['Real 1', 'Real 0'],
                    linewidths=0.5, linecolor='gray',
                    ax=axes[i], cbar=False)
        axes[i].set_title(f'SkLearn k = {m["k"]}', fontsize=12)

    plt.suptitle('Matrizes de Confusão - Scikit-Learn', fontsize=14)
    plt.tight_layout()
    plt.savefig('images/matrizes_confusao_sklearn.png', dpi=300, bbox_inches='tight')
    plt.show()