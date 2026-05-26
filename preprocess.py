import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def main():
    input_path = r"c:\Users\nicol\Documentos\Arquivos - Estudo\Projetos\algoritmo-artigo\reels_attention_span_dataset_12000.csv"
    output_path = r"c:\Users\nicol\Documentos\Arquivos - Estudo\Projetos\algoritmo-artigo\dataset_clean.csv"
    
    print("Iniciando Pré-processamento dos Dados...")
    
    # 1. Carregar os dados
    df = pd.read_csv(input_path)
    print(f"Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    
    # 2. Identificar valores ausentes
    missing_values = df.isnull().sum()
    print("\n--- Valores Ausentes por Coluna ---")
    for col, val in missing_values.items():
        print(f"{col}: {val}")
        
    # Decisão: Como não há valores ausentes na base extraída, não há necessidade de imputação.
    # Se houvesse, a imputação pela mediana (numéricas) ou moda (categóricas) seria o ideal.
    
    # 3. Tratar inconsistências de texto (como "Inatagram Reels" -> "Instagram Reels")
    print("\n--- Plataformas únicas antes da correção ---")
    print(df['platform'].value_counts())
    
    # Corrigindo a grafia de "Inatagram Reels" para "Instagram Reels"
    df['platform'] = df['platform'].replace({
        "Inatagram Reels": "Instagram Reels",
        "Instagram Reels": "Instagram Reels"  # Garante uniformidade
    })
    
    print("\n--- Plataformas únicas após a correção ---")
    print(df['platform'].value_counts())
    
    # 4. Remover duplicatas
    duplicates_all = df.duplicated().sum()
    duplicates_id = df.duplicated(subset=['user_id']).sum()
    print(f"\nDuplicatas exatas: {duplicates_all}")
    print(f"Duplicatas pelo 'user_id': {duplicates_id}")
    
    # Decisão: Remover duplicatas de 'user_id' se existirem, mantendo a primeira ocorrência
    if duplicates_id > 0:
        df = df.drop_duplicates(subset=['user_id'], keep='first')
        print(f"Registros duplicados removidos. Novo tamanho: {df.shape[0]}")
        
    # 5. Identificar outliers utilizando o método IQR (Intervalo Interquartil)
    # Definir colunas numéricas de interesse
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
        
        # Contar outliers
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outliers_summary[col] = len(outliers)
        print(f"{col}: Limites [{lower_bound:.2f}, {upper_bound:.2f}] | Outliers detectados: {len(outliers)}")
        
    # Decisão: Os dados estão em faixas biológicas e comportamentais extremamente plausíveis
    # (Ex: idade de 15 a 44, tempo de tela de 2 a 12, etc.). Não há outliers extremos causados 
    # por erros de medição ou digitação que exijam remoção. Manter todos os registros preserva 
    # a integridade estatística da amostra populacional para mídias digitais.
    
    # 6. Normalização/Padronização
    # Faremos a padronização das variáveis preditoras para uso no K-Means e Random Forest
    # Criaremos colunas com o sufixo '_scaled' para manter as variáveis originais intactas
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[num_cols])
    
    for i, col in enumerate(num_cols):
        df[f"{col}_scaled"] = scaled_features[:, i]
        
    print("\nVariáveis numéricas padronizadas com sucesso (média=0, desvio=1).")
    
    # 7. Salvar o arquivo pré-processado
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nPré-processamento concluído com sucesso!")
    print(f"Tamanho final do dataset limpo: {df.shape}")
    print(f"Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
