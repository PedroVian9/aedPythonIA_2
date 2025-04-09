import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    features = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    df_normalized = pd.DataFrame(features_normalized, 
                                 columns=['Age', 'AnnualIncome', 'SpendingScore'])
    return df, df_normalized, scaler

def select_optimal_k(df_normalized):
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(df_normalized)
        inertia.append(kmeans.inertia_)
        score = silhouette_score(df_normalized, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K={k} -> Silhouette Score: {score:.4f}")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia, marker='o')
    plt.title('Método do Cotovelo')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')

    plt.subplot(1, 2, 2)
    plt.plot(k_range, silhouette_scores, marker='o', color='green')
    plt.title('Silhouette Score')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Score')

    plt.tight_layout()
    plt.show()

    k_ideal = silhouette_scores.index(max(silhouette_scores)) + 2
    print(f"\nK ideal sugerido pelo Silhouette Score: {k_ideal}")
    return k_ideal

def train_kmeans(df_normalized, df_original, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(df_normalized)

    df_with_clusters = df_original.copy()
    df_with_clusters['Cluster'] = cluster_labels

    centroids_normalized = kmeans.cluster_centers_
    return df_with_clusters, centroids_normalized, kmeans

def show_centroids(centroids_normalized, scaler):
    centroids_original = scaler.inverse_transform(centroids_normalized)
    df_centroids = pd.DataFrame(centroids_original, columns=['Age', 'Annual Income (k$)', 'Spending Score (1-100)'])

    print("\nCentroides dos clusters (valores reais):")
    print(df_centroids.round(2))
    return df_centroids

def plot_clusters(df_with_clusters, centroids_original):
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        df_with_clusters['Annual Income (k$)'],
        df_with_clusters['Spending Score (1-100)'],
        c=df_with_clusters['Cluster'],
        cmap='Set1',
        s=60,
        alpha=0.8,
        label='Clientes'
    )

    # Centroides
    plt.scatter(
        centroids_original[:, 1],
        centroids_original[:, 2],
        c='red',
        marker='X',
        s=200,
        label='Centroides'
    )

    plt.title('Segmentação de clientes - KMeans (2D)')
    plt.xlabel('Renda anual (k$)')
    plt.ylabel('Pontuação de gasto (1-100)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    file_path = 'Mall_Customers.csv'
    print("Escolha a questão:")
    print("1 - Pré-processamento dos dados")
    print("2 - Seleção do número de clusters")
    print("3 - Modelagem com K-Means")
    print("4 - Visualização e análise")

    choice = input("Digite o número da questão: ")

    if choice == "1":
        print("\nExecutando Questão 1: Pré-processamento...")
        df_original, df_normalized, scaler = preprocess_data(file_path)
        print("\nDados pré-processados (5 primeiras linhas):")
        print(df_normalized.head())

    elif choice == "2":
        print("\nExecutando Questão 2: Seleção do número de clusters...")
        df_original, df_normalized, scaler = preprocess_data(file_path)
        k_ideal = select_optimal_k(df_normalized)
        print(f"\nK ideal: {k_ideal}")

    elif choice == "3":
        print("\nExecutando Questão 3: Modelagem com K-Means...")
        df_original, df_normalized, scaler = preprocess_data(file_path)
        k = int(input("Informe o valor de K (determinado na questão 2): "))
        df_clusters, centroids_normalized, kmeans_model = train_kmeans(df_normalized, df_original, k)
        show_centroids(centroids_normalized, scaler)
        df_clusters.to_csv("clientes_com_clusters.csv", index=False)
        print("\nArquivo exportado: clientes_com_clusters.csv")

    elif choice == "4":
        print("\nExecutando Questão 4: Visualização dos Clusters...")
        df_original, df_normalized, scaler = preprocess_data(file_path)
        k = int(input("Informe o valor de K: "))
        df_clusters, centroids_normalized, kmeans_model = train_kmeans(df_normalized, df_original, k)
        centroids_original = scaler.inverse_transform(centroids_normalized)
        plot_clusters(df_clusters, centroids_original)

    else:
        print("Questão inválida!")

if __name__ == "__main__":
    main()
