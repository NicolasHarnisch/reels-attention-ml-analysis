import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def main():
    clean_dataset_path = r"c:\Users\nicol\Documentos\Arquivos - Estudo\Projetos\algoritmo-artigo\dataset_clean.csv"
    figuras_dir = r"c:\Users\nicol\Documentos\Arquivos - Estudo\Projetos\algoritmo-artigo\figuras"

    os.makedirs(figuras_dir, exist_ok=True)

    df = pd.read_csv(clean_dataset_path)

    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18
    })

    num_cols = [
        'age', 'reels_watch_time_hours', 'daily_screen_time_hours',
        'sleep_hours', 'attention_span_score', 'focus_level', 'task_completion_rate'
    ]

    desc = df[num_cols].describe()
    print("\n--- Estatísticas Descritivas Consolidadas ---")
    print(desc)

    desc.to_csv(os.path.join(figuras_dir, "estatisticas_descritivas.csv"))

    correlation_pearson = df[num_cols].corr(method='pearson')
    correlation_spearman = df[num_cols].corr(method='spearman')

    correlation_pearson.to_csv(os.path.join(figuras_dir, "pearson_correlation.csv"))
    correlation_spearman.to_csv(os.path.join(figuras_dir, "spearman_correlation.csv"))

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(correlation_pearson, dtype=bool))

    sns.heatmap(
        correlation_pearson,
        mask=mask,
        annot=True,
        cmap="coolwarm",
        fmt=".3f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de Correlação Linear de Pearson", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "heatmap_correlacao.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_spearman,
        mask=mask,
        annot=True,
        cmap="viridis",
        fmt=".3f",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8}
    )
    plt.title("Matriz de Correlação de Postos de Spearman", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "heatmap_correlacao_spearman.png"), dpi=300)
    plt.close()

    print("\nMatrizes de Correlação de Pearson e Spearman salvas em CSV com sucesso!")
    print("Heatmaps de correlação Pearson e Spearman salvos com sucesso!")

    plt.figure(figsize=(12, 6))
    df_melted = pd.melt(df[num_cols])
    sns.boxplot(x="variable", y="value", data=df_melted, palette="Set2")
    plt.xticks(rotation=30)
    plt.xlabel("Variáveis")
    plt.ylabel("Escala de Valores")
    plt.title("Dispersão e Amplitude das Variáveis Numéricas", pad=20)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "boxplot_distribuicao.png"), dpi=300)
    plt.close()

    print("Boxplot geral salvo com sucesso!")

    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], color="skyblue", bins=30)
        axes[i].set_title(f"Distribuição de: {col}")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Frequência")

    for j in range(len(num_cols), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle("Distribuição Empírica das Variáveis do Estudo", y=0.98)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "histogramas_distribuicao.png"), dpi=300)
    plt.close()

    print("Histogramas de distribuição salvos com sucesso!")

    plt.figure(figsize=(8, 6))
    sns.regplot(
        x="reels_watch_time_hours",
        y="attention_span_score",
        data=df,
        scatter_kws={"alpha": 0.3, "color": "darkblue", "s": 10},
        line_kws={"color": "red", "linewidth": 2}
    )
    plt.xlabel("Consumo de Vídeos Curtos (Horas/Dia)")
    plt.ylabel("Score de Capacidade Atencional (1-10)")
    plt.title("Relação entre Consumo de Vídeos Curtos e Capacidade Atencional", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "scatter_reels_atencao.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.regplot(
        x="daily_screen_time_hours",
        y="attention_span_score",
        data=df,
        scatter_kws={"alpha": 0.3, "color": "indigo", "s": 10},
        line_kws={"color": "orange", "linewidth": 2}
    )
    plt.xlabel("Tempo de Tela Total (Horas/Dia)")
    plt.ylabel("Score de Capacidade Atencional (1-10)")
    plt.title("Relação entre Tempo de Tela Total e Capacidade Atencional", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "scatter_tela_atencao.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.regplot(
        x="sleep_hours",
        y="attention_span_score",
        data=df,
        scatter_kws={"alpha": 0.3, "color": "teal", "s": 10},
        line_kws={"color": "crimson", "linewidth": 2}
    )
    plt.xlabel("Horas de Sono por Noite")
    plt.ylabel("Score de Capacidade Atencional (1-10)")
    plt.title("Relação entre Horas de Sono e Capacidade Atencional", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "scatter_sono_atencao.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x="stress_level",
        y="attention_span_score",
        data=df,
        order=["Low", "Medium", "High"],
        palette="crest"
    )
    plt.xlabel("Nível de Estresse")
    plt.ylabel("Score de Capacidade Atencional (1-10)")
    plt.title("Distribuição do Score de Atenção por Nível de Estresse", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "scatter_estresse_atencao.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(8, 6))
    sns.regplot(
        x="focus_level",
        y="task_completion_rate",
        data=df,
        scatter_kws={"alpha": 0.3, "color": "darkgreen", "s": 10},
        line_kws={"color": "magenta", "linewidth": 2}
    )
    plt.xlabel("Nível de Foco Reportado (1-10)")
    plt.ylabel("Taxa de Conclusão de Tarefas (%)")
    plt.title("Relação entre Nível de Foco e Desempenho (Taxa de Conclusão)", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(figuras_dir, "scatter_foco_desempenho.png"), dpi=300)
    plt.close()

    print("Gráficos de dispersão e boxplots de relações salvos com sucesso!")
    print(f"\nTodos os artefatos de EDA foram gerados e salvos em: {figuras_dir}")

if __name__ == "__main__":
    main()
