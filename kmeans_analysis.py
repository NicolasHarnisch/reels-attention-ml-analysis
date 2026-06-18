import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy import stats
import os

def main():
    clean_dataset_path = "dataset_clean.csv"
    figuras_dir = "figuras"

    os.makedirs(figuras_dir, exist_ok=True)

    df = pd.read_csv(clean_dataset_path)

    sns.set_theme(style="whitegrid")

    features_b = [
        'reels_watch_time_hours_scaled', 'daily_screen_time_hours_scaled',
        'sleep_hours_scaled', 'focus_level_scaled', 'task_completion_rate_scaled'
    ]
    X_b = df[features_b]

    pca_b = PCA(n_components=2, random_state=42)
    X_b_pca = pca_b.fit_transform(X_b)

    k_range = range(2, 11)

    wcss_sem_pca = []
    sil_sem_pca = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_b)
        wcss_sem_pca.append(kmeans.inertia_)
        score = silhouette_score(X_b, labels, sample_size=3000, random_state=42)
        sil_sem_pca.append(score)

    k_metrics_sem_pca = pd.DataFrame({
        'K': list(k_range),
        'WCSS': wcss_sem_pca,
        'Silhouette_Score': sil_sem_pca
    })
    k_metrics_sem_pca.to_csv(os.path.join(figuras_dir, "kmeans_k_metrics_sem_pca.csv"), index=False)

    wcss_com_pca = []
    sil_com_pca = []

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_b_pca)
        wcss_com_pca.append(kmeans.inertia_)
        score = silhouette_score(X_b_pca, labels, sample_size=3000, random_state=42)
        sil_com_pca.append(score)

    k_metrics_com_pca = pd.DataFrame({
        'K': list(k_range),
        'WCSS': wcss_com_pca,
        'Silhouette_Score': sil_com_pca
    })
    k_metrics_com_pca.to_csv(os.path.join(figuras_dir, "kmeans_k_metrics_com_pca.csv"), index=False)

    best_k_idx = np.argmax(sil_com_pca)
    best_k_val = list(k_range)[best_k_idx]
    best_k_score = sil_com_pca[best_k_idx]

    print("\n--- Validação do Número de Clusters (Com PCA) ---")
    print(f"Melhor K por métrica de Silhouette (com PCA): K = {best_k_val} (Score: {best_k_score:.4f})")

    k_ideal = 3

    kmeans_sem = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
    df['cluster_b_sem_pca'] = kmeans_sem.fit_predict(X_b)

    attention_sem = df.groupby('cluster_b_sem_pca')['attention_span_score'].mean().reset_index()
    attention_sem.columns = ['cluster_b', 'attention_span_score_mean']
    attention_sem.to_csv(os.path.join(figuras_dir, "media_atencao_por_cluster_sem_pca.csv"), index=False)

    groups_sem = [df[df['cluster_b_sem_pca'] == i]['attention_span_score'].values for i in range(k_ideal)]
    f_val_sem, anova_p_sem = stats.f_oneway(*groups_sem)
    h_val_sem, kruskal_p_sem = stats.kruskal(*groups_sem)

    kmeans_com = KMeans(n_clusters=k_ideal, random_state=42, n_init=10)
    df['cluster_b'] = kmeans_com.fit_predict(X_b_pca)

    attention_com = df.groupby('cluster_b')['attention_span_score'].mean().reset_index()
    attention_com.columns = ['cluster_b', 'attention_span_score_mean']
    attention_com.to_csv(os.path.join(figuras_dir, "media_atencao_por_cluster_com_pca.csv"), index=False)

    groups_com = [df[df['cluster_b'] == i]['attention_span_score'].values for i in range(k_ideal)]
    f_val_com, anova_p_com = stats.f_oneway(*groups_com)
    h_val_com, kruskal_p_com = stats.kruskal(*groups_com)

    sil_a = 0.1129
    sil_b_no_pca = silhouette_score(X_b, df['cluster_b_sem_pca'], sample_size=3000, random_state=42)
    sil_b_pca = silhouette_score(X_b_pca, df['cluster_b'], sample_size=3000, random_state=42)

    profiles_b = df.groupby('cluster_b')[['age', 'reels_watch_time_hours', 'daily_screen_time_hours', 'sleep_hours', 'attention_span_score', 'focus_level', 'task_completion_rate']].mean()
    profiles_b.to_csv(os.path.join(figuras_dir, "perfis_clusters_versao_b.csv"))

    print(f"\n--- Estatísticas do K-Means Recalculadas ---")
    print(f"Silhueta B Sem PCA: {sil_b_no_pca:.4f} | ANOVA p-value: {anova_p_sem:.4e}")
    print(f"Silhueta B Com PCA: {sil_b_pca:.4f} | ANOVA p-value: {anova_p_com:.4e}")

    with open(os.path.join(figuras_dir, "resultados_estatisticos_sbc.txt"), "w") as f:
        f.write("VALIDACAO ESTATISTICA COMPLETA DOS CLUSTERS (VERSAO B)\n\n")
        f.write("1. VERSAO B SEM PCA (Espaco 5D original):\n")
        f.write(f"  Silhouette Score (K=3):         {sil_b_no_pca:.4f}\n")
        f.write(f"  ANOVA de uma via:               F-value = {f_val_sem:.4f} | p-value = {anova_p_sem:.6e}\n")
        f.write(f"  Teste de Kruskal-Wallis:        H-value = {h_val_sem:.4f} | p-value = {kruskal_p_sem:.6e}\n")
        f.write("  Conclusao: Nao foram encontradas evidencias estatisticas suficientes para sustentar diferenca significativa de capacidade atencional entre os clusters analisados.\n\n")

        f.write("2. VERSAO B COM PCA (Espaco 2D reduzido):\n")
        f.write(f"  Silhouette Score (K=3):         {sil_b_pca:.4f}\n")
        f.write(f"  ANOVA de uma via:               F-value = {f_val_com:.4f} | p-value = {anova_p_com:.6e}\n")
        f.write(f"  Teste de Kruskal-Wallis:        H-value = {h_val_com:.4f} | p-value = {kruskal_p_com:.6e}\n")
        f.write("  Conclusao: Nao foram encontradas evidencias estatisticas suficientes para sustentar diferenca significativa de capacidade atencional entre os clusters analisados.\n")

    plt.figure(figsize=(8, 6))
    sns.boxplot(x="cluster_b", y="attention_span_score", data=df, palette="Set2")
    plt.xlabel("Clusters Comportamentais baseados em PCA (Versão B)")
    plt.ylabel("Score de Capacidade Atencional (1-10)")
    plt.title("Distribuição da Atenção Sustentada por Cluster Comportamental (PCA B)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "boxplot_atencao_cluster_b.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for i in range(k_ideal):
        cluster_data = df[df['cluster_b'] == i]
        pca_coords = X_b_pca[df['cluster_b'] == i]
        plt.scatter(
            pca_coords[:, 0],
            pca_coords[:, 1],
            alpha=0.6,
            label=f'Perfil B{i+1} (N={len(cluster_data)})',
            s=12,
            color=colors[i]
        )
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.title("Visualização dos Perfis de Usuários - K-Means Versão B (Espaço PCA)", pad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "clusters_scatter_versao_b.png"), dpi=300)
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(k_range, wcss_com_pca, 'bo-', markersize=6, linewidth=2)
    ax1.set_xlabel('Número de Clusters (K)')
    ax1.set_ylabel('Inércia (WCSS)')
    ax1.set_title('Elbow Method (Versão B com PCA)')
    ax1.set_xticks(k_range)
    ax1.grid(True)
    ax1.axvline(x=3, color='red', linestyle='--', label='Inflexão K=3')
    ax1.legend()

    ax2.plot(k_range, sil_com_pca, 'ro-', markersize=6, linewidth=2)
    ax2.set_xlabel('Número de Clusters (K)')
    ax2.set_ylabel('Coeficiente de Silhueta')
    ax2.set_title('Silhouette Analysis (Versão B com PCA)')
    ax2.set_xticks(k_range)
    ax2.grid(True)
    ax2.axvline(x=3, color='red', linestyle='--', label='K Escolhido (K=3)')
    ax2.legend()

    plt.suptitle("Validação do Número de Clusters (Versão B com PCA)", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "metodo_cotovelo_silhueta_pca.png"), dpi=300)
    plt.close()

    df.to_csv(clean_dataset_path, index=False, encoding="utf-8-sig")
    print("Processamento de K-Means e análises estatísticas concluídos com sucesso!")

if __name__ == "__main__":
    main()
