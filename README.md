# Análise Computacional da Relação entre Consumo de Vídeos Curtos e Capacidade Atencional

Este projeto realiza uma análise exploratória baseada em estatística e aprendizado de máquina para investigar associações entre o consumo de vídeos curtos, hábitos digitais e capacidade atencional dos usuários.

Foram aplicadas técnicas de análise exploratória de dados, correlação, clusterização com K-Means, redução de dimensionalidade com PCA, modelagem supervisionada com Random Forest e análises complementares com testes estatísticos e modelos alternativos.

## Base de Dados

A análise foi conduzida sobre uma base pública disponibilizada na plataforma Kaggle.

* **Dataset:** `reels_attention_span_dataset_12000.csv`
* **Fonte:** Reels/Shorts Consumption vs Attention Span
* **Link:** https://www.kaggle.com/datasets/jayjoshi37/reelsshorts-consumption-vs-attention-span

A base contém aproximadamente 12.000 registros com variáveis relacionadas a idade, tempo diário de consumo de vídeos curtos, tempo total de tela, horas de sono, score de atenção, nível de foco, taxa de conclusão de tarefas, nível de estresse e plataforma utilizada.

## Aviso Importante

Este projeto possui caráter exploratório e científico. Os resultados indicam associações estatísticas, ausência de associações ou limitações preditivas observadas na base analisada.

As análises não permitem afirmar causalidade entre consumo de vídeos curtos e capacidade atencional. Portanto, termos como “causa”, “comprova” ou “determina” devem ser evitados na interpretação dos resultados.

## Como instalar as dependências

Certifique-se de ter o Python 3.8 ou superior instalado. Recomenda-se o uso de um ambiente virtual.

```bash
pip install -r requirements.txt
```

## Como executar os scripts

Execute os scripts na seguinte ordem, a partir da raiz do projeto:

```bash
python preprocess.py
python eda.py
python kmeans_analysis.py
python random_forest_analysis.py
python analises_complementares.py
```

## Etapas do Pipeline

1. **Pré-processamento**

   * Verificação de valores ausentes.
   * Verificação de duplicatas.
   * Correção de inconsistências textuais.
   * Padronização das variáveis numéricas por Z-score.
   * Geração de `dataset_clean.csv`.

2. **Análise Exploratória**

   * Estatísticas descritivas.
   * Matrizes de correlação de Pearson e Spearman.
   * Gráficos de dispersão e distribuição.

3. **Clusterização**

   * Aplicação do K-Means.
   * Comparação entre versões com e sem score de atenção.
   * Uso de PCA para visualização em duas dimensões.
   * Avaliação por coeficiente de silhueta, método do cotovelo, ANOVA e Kruskal-Wallis.

4. **Modelagem Supervisionada**

   * Random Forest Regressor para prever o score contínuo de atenção.
   * Random Forest Classifier para classificar atenção baixa e alta.
   * Comparação com Dummy Regressor e Dummy Classifier.
   * Validação cruzada 5-Fold.

5. **Análises Complementares**

   * Criação da variável `short_video_ratio`.
   * Comparação por quartis de consumo.
   * Comparação por plataforma.
   * Correção FDR de Benjamini-Hochberg para múltiplos testes.
   * Modelos alternativos de regressão e classificação.
   * Clusterizações alternativas sem uso da variável-alvo.

## Arquivos Gerados

Durante a execução, são gerados os seguintes arquivos:

* `dataset_clean.csv`: base tratada.
* `resultados_modelos_final.txt`: relatório principal dos modelos e métricas.
* `resultados_complementares.txt`: relatório das análises complementares.
* `metricas_modelos_complementares.csv`: métricas dos modelos complementares.
* `testes_estatisticos_complementares.csv`: resultados dos testes estatísticos complementares.
* `figuras/`: pasta contendo os gráficos utilizados no artigo e análises complementares.

## Principais Resultados

Os resultados indicaram ausência de correlação relevante entre o tempo diário de consumo de vídeos curtos e o score de atenção. A clusterização com K-Means permitiu formar perfis comportamentais exploratórios, mas os grupos não apresentaram diferenças estatisticamente significativas na capacidade atencional.

Os modelos supervisionados apresentaram desempenho próximo aos modelos baseline, indicando baixo poder preditivo das variáveis disponíveis para estimar o score de atenção. As análises complementares reforçaram a ausência de evidência estatística ou preditiva robusta para uma relação direta entre consumo de vídeos curtos e capacidade atencional na base analisada.

## Reprodutibilidade

Todos os scripts foram organizados para execução sequencial. As figuras, tabelas e relatórios utilizados no artigo podem ser reproduzidos a partir dos arquivos presentes neste repositório.

## Autores

* Nícolas G. Harnisch
* Samuel S. Beserra
* José Levi Gonçalves de Lima Menezes
