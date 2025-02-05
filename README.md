```md
# Análise e Clusterização de Tweets

Este repositório contém um código para análise, limpeza, visualização e clusterização de um conjunto de dados de tweets. O pipeline permite explorar as informações contidas no dataset e segmentar os tweets em grupos distintos.

## Etapas de Execução

### 1. Preparar o Ambiente

Antes de iniciar a análise, configure o ambiente corretamente:

1. **Instalar o Python:**  
   Certifique-se de ter o Python instalado. Baixe-o em [python.org](https://www.python.org/).

2. **Instalar as Bibliotecas Necessárias:**  
   Execute o seguinte comando para instalar as dependências:
   ```sh
   pip install pandas numpy matplotlib seaborn scipy scikit-learn
   ```

3. **Escolher o Ambiente de Desenvolvimento:**  
   Use um editor de código de sua preferência, como:
   - [PyCharm](https://www.jetbrains.com/pycharm/download/)
   - [VS Code](https://code.visualstudio.com/download)
   - [Jupyter Notebook](https://jupyter.org/) ou [Google Colab](https://colab.research.google.com/) para uma experiência interativa.

### 2. Carregar o Dataset

Certifique-se de que o arquivo `tweets.csv` esteja no caminho especificado no código. Caso esteja em outro local, ajuste o caminho na linha:
```python
df = pd.read_csv('sample_data/tweets.csv')
```

### 3. Verificar a Qualidade dos Dados

Execute a função `verificar_dados(df)` para obter informações sobre:
- Tipos de dados.
- Resumo estatístico (médias, desvios, mínimos e máximos).
- Percentual de valores nulos e duplicados.
- Intervalos de valores para colunas numéricas.

### 4. Limpeza dos Dados

A função `limpar_dados(df)` realiza:
- Remoção de colunas irrelevantes (`country`, `latitude`, `longitude`, `id`).
- Conversão de variáveis categóricas em variáveis dummy (ex.: `language`).
- Retorna um DataFrame limpo e pronto para análise.

### 5. Visualizações Iniciais

Use a função `visualizar_dados(df)` para:
- Gerar histogramas das variáveis numéricas.
- Exibir boxplots para identificar outliers e distribuições.

### 6. Análise de Correlação

A função `plotCorrelationMatrix(df, graphWidth)` exibe uma matriz de correlação:
- Remove colunas com valores nulos ou constantes.
- Gera um mapa de calor das correlações.

### 7. Identificação e Remoção de Outliers

Utilize a função `remover_outliers(df)` para:
- Identificar outliers com base no **Z-Score**.
- Filtrar o DataFrame removendo os valores atípicos.

### 8. Normalização dos Dados

A função `normalizar_dados(df)` utiliza `StandardScaler` para:
- Normalizar variáveis numéricas, tornando os dados com média 0 e desvio padrão 1.

### 9. Clusterização dos Dados

#### 9.1 Determinar o Número de Clusters

A função `determinar_numero_clusters(df, features)` aplica o **método Elbow**:
- Calcula a inércia para diferentes valores de `k`.
- Gera um gráfico para ajudar na escolha do número ideal de clusters.

#### 9.2 Aplicar o Algoritmo K-Means

A função `aplicar_kmeans(df, features, k)`:
- Realiza a clusterização com o `k` escolhido.
- Avalia a qualidade da clusterização utilizando:
  - **Coeficiente de Silhouette**
  - **Índice Davies-Bouldin**
  - **Índice Calinski-Harabasz**

#### 9.3 Visualizar os Clusters

A função `visualizar_clusters(df, features)` gera um scatter plot para visualizar os grupos formados.

### 10. Resumo dos Clusters

A função `resumo_clusters(df)` exibe a média das variáveis para cada cluster, permitindo a interpretação dos grupos formados.

## Considerações Finais

Este pipeline analítico oferece um processo estruturado para explorar, limpar, visualizar e interpretar dados de tweets, além de realizar a clusterização para segmentação dos dados. Sinta-se à vontade para modificar o dataset e os parâmetros conforme suas necessidades analíticas.

---

Feito com ❤️ para análise de dados!
```
