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


def main():
    file_path = 'Mall_Customers.csv'
    print("Escolha a questão:")
    print("1 - Pré-processamento dos dados")
    
    choice = input("Digite o número da questão: ")

    if choice == "1":
        print("\nExecutando Questão 1: Pré-processamento...")
        df_original, df_normalized, scaler = preprocess_data(file_path)
        print("\nDados pré-processados (5 primeiras linhas):")
        print(df_normalized.head())

    else:
        print("Questão inválida!")

if __name__ == "__main__":
    main()
