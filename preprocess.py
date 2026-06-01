import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def main():
    input_path = r"c:\Users\nicol\Documentos\Arquivos - Estudo\Projetos\algoritmo-artigo\reels_attention_span_dataset_12000.csv"
    output_path = r"c:\Users\nicol\Documentos\Arquivos - Estudo\Projetos\algoritmo-artigo\dataset_clean.csv"

    print("Iniciando Pré-processamento dos Dados...")

    df = pd.read_csv(input_path)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")

    missing_values = df.isnull().sum()
    print("\n--- Valores Ausentes por Coluna ---")
    for col, val in missing_values.items():
        print(f"{col}: {val}")

    print("\n--- Plataformas únicas antes da correção ---")
    print(df['platform'].value_counts())

    df['platform'] = df['platform'].replace({
        "Inatagram Reels": "Instagram Reels",
        "Instagram Reels": "Instagram Reels"
    })

    print("\n--- Plataformas únicas após a correção ---")
    print(df['platform'].value_counts())

    duplicates_all = df.duplicated().sum()
    duplicates_id = df.duplicated(subset=['user_id']).sum()
    print(f"\nDuplicatas exatas: {duplicates_all}")
    print(f"Duplicatas pelo 'user_id': {duplicates_id}")

    if duplicates_id > 0:
        df = df.drop_duplicates(subset=['user_id'], keep='first')
        print(f"Registros duplicados removidos. Novo tamanho: {df.shape[0]}")

    num_cols = [
        'age', 'reels_watch_time_hours', 'daily_screen_time_hours',
        'sleep_hours', 'attention_span_score', 'focus_level', 'task_completion_rate'
    ]

    print("\n--- Identificação de Outliers (Método IQR com Limite 1.5) ---")
    outliers_summary = {}
    for col in num_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_summary[col] = len(outliers)
        print(f"{col}: Limites [{lower_bound:.2f}, {upper_bound:.2f}] | Outliers detectados: {len(outliers)}")

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[num_cols])

    for i, col in enumerate(num_cols):
        df[f"{col}_scaled"] = scaled_features[:, i]

    print("\nVariáveis numéricas padronizadas com sucesso (média=0, desvio=1).")

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nPré-processamento concluído com sucesso!")
    print(f"Tamanho final do dataset limpo: {df.shape}")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
