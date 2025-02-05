from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats
import seaborn as sns

df = pd.read_csv('tweets.csv')
df.dataframeName = 'tweets.csv'
df.head(100)

def  verificar_dados():
  print(f"Tipos de dados:\n{df.dtypes}\n")
  print(f"Descrição dos dados:\n{df.describe()}\n")
  print(f"Verificar percentual de valores nulos:\n{df.isnull().mean() * 100}\n")
  print(f"Verificar percentual de valores nulos:\n{df.notnull().sum()}\n")
  print(f"Valores duplicados: {df.duplicated().sum()}\n")

  intervalos_valores = {col: (df[col].min(), df[col].max()) for col in df.select_dtypes(include=[np.number]).columns}
  print(f"Intervalos de valores:\n{intervalos_valores}\n")
  print(f"Dados do tipo numerico:\n{df.select_dtypes(include=['number']).head(100)}\n")

verificar_dados()

# Limpeza dos dados
def limpar_dados(df):
    # Remover colunas irrelevantes
    for col in ['country', 'latitude', 'longitude', 'id']:
        if col in df.columns:
            df = df.drop(col, axis=1)
    # Transformar variáveis categóricas
    df = pd.get_dummies(df, columns=['language'], drop_first=True)
    return df

df = limpar_dados(df)

# Correlation matrix
df.dataframeName = 'tweets.csv'
def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName

    # Remover colunas com valores NaN
    df = df.dropna(axis=1, how='all')
    print(df.head())

    # Filtrar apenas colunas numéricas
    df = df.select_dtypes(include=['number'])

    # Remover colunas com apenas 1 valor único (constantes)
    df = df[[col for col in df if df[col].nunique() > 1]]

    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return

    # Calcular a matriz de correlação
    corr = df.corr()

    # Plotando a matriz de correlação
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum=1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


plotCorrelationMatrix(df, 8)

# Identificar e Remover Outliers
def remover_outliers(df):
    z_scores = stats.zscore(df.select_dtypes(include=['number']))
    outliers = (z_scores > 3) | (z_scores < -3)
    df_cleaned = df[(~outliers).all(axis=1)]
    print(outliers.sum())
    return df_cleaned

df = remover_outliers(df)

# Visualizações iniciais
def visualizar_dados(df):
    df.hist(figsize=(12, 8), bins=20)
    plt.suptitle("Distribuições das Variáveis", fontsize=16)
    plt.show()

    sns.boxplot(data=df, orient="h", palette="Set2")
    plt.title("Boxplot das Variáveis", fontsize=16)
    plt.show()

visualizar_dados(df)

# Normalizar dados numéricos
def normalizar_dados(df):
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df.select_dtypes(include=['number']))
    return pd.DataFrame(df_scaled, columns=df.select_dtypes(include=['number']).columns)

df = normalizar_dados(df)
df

# Clusterização
def determinar_numero_clusters(df, features):
    inertia = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df[features])
        inertia.append(kmeans.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, inertia, marker='o')
    plt.title("Método Elbow")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inércia")
    plt.show()

def aplicar_kmeans(df, features, k):
    kmeans = KMeans(n_clusters=k, random_state=42) #random_state=0
    df['cluster'] = kmeans.fit_predict(df[features])

    # Avaliar a qualidade da clusterização
    silhouette_avg = silhouette_score(df[features], df['cluster'])
    db_score = davies_bouldin_score(df[features], df['cluster'])
    ch_score = calinski_harabasz_score(df[features], df['cluster'])

    print(f"Coeficiente de Silhouette: {silhouette_avg:.2f}")
    print(f"Índice Davies-Bouldin: {db_score:.2f}")
    print(f"Índice Calinski-Harabasz: {ch_score:.2f}")

    return df

features = ['number_of_likes', 'number_of_shares']
determinar_numero_clusters(df, features)

k_optimal = 3  # Escolha baseada no método Elbow
df = aplicar_kmeans(df, features, k_optimal)

# Visualizar os Clusters
def visualizar_clusters(df, features):
    plt.figure(figsize=(8, 5))
    for cluster in df['cluster'].unique():
        cluster_data = df[df['cluster'] == cluster]
        plt.scatter(cluster_data[features[0]], cluster_data[features[1]], label=f'Cluster {cluster}')

    plt.title("Clusterização com K-Means")
    plt.xlabel(features[0])
    plt.ylabel(features[1])
    plt.legend()
    plt.show()

visualizar_clusters(df, features)

# Finalização
def resumo_clusters(df):
    resumo = df.groupby('cluster')[features].mean()
    print("Resumo dos Clusters:")
    print(resumo)

resumo_clusters(df)

df

# Métricas Complementares
# Davies-Bouldin Index
db_index = davies_bouldin_score(X, df['cluster'])
print(f"Davies-Bouldin Index: {db_index}")
print("")
# Calinski-Harabasz Index
ch_index = calinski_harabasz_score(X, df['cluster'])
print(f"Calinski-Harabasz Index: {ch_index}")