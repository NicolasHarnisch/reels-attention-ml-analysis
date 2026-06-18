import os
import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.model_selection import StratifiedKFold, KFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns

# Garantir a criação da pasta para figuras
os.makedirs('figuras', exist_ok=True)

report_lines = []
def log_report(text):
    print(text)
    report_lines.append(text)

log_report("=== ANÁLISES COMPLEMENTARES EXPLORATÓRIAS ===")
log_report("Este script realiza análises complementares na relação entre consumo de vídeos curtos e atenção.")
log_report("Todas as conclusões a seguir são de caráter exploratório, sugerindo padrões estatísticos e não implicando causalidade.\n")

if not os.path.exists('dataset_clean.csv'):
    log_report("ERRO: 'dataset_clean.csv' não encontrado. Por favor, execute o script 'preprocess.py' primeiro.")
    with open('resultados_complementares.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    exit(1)

# 2. Carregamento e Preparação de Variáveis
df = pd.read_csv('dataset_clean.csv')

# Variáveis derivadas
df['short_video_ratio'] = df['reels_watch_time_hours'] / df['daily_screen_time_hours']
df['short_video_ratio'] = df['short_video_ratio'].fillna(0)

df['reels_quartile'] = pd.qcut(df['reels_watch_time_hours'], q=4, labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto'])
df['ratio_quartile'] = pd.qcut(df['short_video_ratio'], q=4, labels=['Q1 (Baixo)', 'Q2', 'Q3', 'Q4 (Alto)'])

def categorize_sleep(h):
    if h < 6: return '< 6h'
    elif h <= 8: return '6-8h'
    else: return '> 8h'
df['sleep_category'] = df['sleep_hours'].apply(categorize_sleep)

# Exploratórias
df['high_consum_low_sleep'] = ((df['reels_quartile'] == 'Alto') & (df['sleep_category'] == '< 6h')).astype(int)
df['high_consum_high_stress'] = ((df['reels_quartile'] == 'Alto') & (df['stress_level'] == 'High')).astype(int)

# 3. Testes Estatísticos
pvals_to_correct = []
test_names = []
results_stats = []

# T-test e Mann-Whitney (Baixo vs Alto)
baixo = df[df['reels_quartile'] == 'Baixo']['attention_span_score']
alto = df[df['reels_quartile'] == 'Alto']['attention_span_score']
t_stat, p_t = stats.ttest_ind(baixo, alto, equal_var=False)
u_stat, p_u = stats.mannwhitneyu(baixo, alto)

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / dof)

d_val = cohen_d(baixo, alto)

pvals_to_correct.extend([p_t, p_u])
test_names.extend(['T-test: Baixo vs Alto consumo', 'Mann-Whitney: Baixo vs Alto consumo'])
results_stats.append({'Test': 'T-test: Baixo vs Alto consumo', 'Statistic': t_stat, 'Original_PValue': p_t, 'Cohen_d': d_val})
results_stats.append({'Test': 'Mann-Whitney: Baixo vs Alto consumo', 'Statistic': u_stat, 'Original_PValue': p_u, 'Cohen_d': None})

# ANOVA e Kruskal para Plataformas, Quartis Consumo e Quartis Ratio
def evaluate_groups(group_col, test_prefix):
    groups = [group['attention_span_score'].values for name, group in df.groupby(group_col, observed=False)]
    f_stat, p_f = stats.f_oneway(*groups)
    h_stat, p_h = stats.kruskal(*groups)
    pvals_to_correct.extend([p_f, p_h])
    results_stats.extend([
        {'Test': f'ANOVA: {test_prefix}', 'Statistic': f_stat, 'Original_PValue': p_f, 'Cohen_d': None},
        {'Test': f'Kruskal: {test_prefix}', 'Statistic': h_stat, 'Original_PValue': p_h, 'Cohen_d': None}
    ])

evaluate_groups('platform', 'Plataforma')
evaluate_groups('reels_quartile', 'Quartis Consumo')
evaluate_groups('ratio_quartile', 'Quartis Ratio')

# Correção de Benjamini-Hochberg (Implementação manual)
def fdr_bh(pvals):
    pvals = np.asarray(pvals)
    n = len(pvals)
    sorted_ind = np.argsort(pvals)
    sorted_pvals = pvals[sorted_ind]
    
    # Calculando os p-values ajustados
    adjusted = np.empty(n)
    min_adj = 1.0
    for i in range(n - 1, -1, -1):
        # p_adj = p_i * n / rank
        val = sorted_pvals[i] * n / (i + 1)
        min_adj = min(min_adj, val)
        adjusted[i] = min_adj
    
    # Restaurando a ordem original
    orig_order = np.empty(n)
    orig_order[sorted_ind] = adjusted
    return np.minimum(orig_order, 1.0)

pvals_corrected = fdr_bh(pvals_to_correct)

for i, test in enumerate(results_stats):
    test['Corrected_PValue_FDR'] = pvals_corrected[i]
    test['Significant_after_FDR'] = bool(pvals_corrected[i] < 0.05)

stats_df = pd.DataFrame(results_stats)
stats_df.to_csv('testes_estatisticos_complementares.csv', index=False)

log_report("--- Testes Estatísticos (Target: attention_span_score) ---")
for i, row in stats_df.iterrows():
    log_report(f"{row['Test']}: P-val original = {row['Original_PValue']:.4f}, P-val corrigido (FDR) = {row['Corrected_PValue_FDR']:.4f}")

if stats_df['Significant_after_FDR'].any():
    log_report("\nResultado Estatístico: A análise exploratória sugere a presença de diferenças estatisticamente significativas após o controle de falsas descobertas (FDR). Isso indica padrões que podem ser interessantes para estudos futuros, mas as evidências não confirmam causas diretas.")
else:
    log_report("\nResultado Estatístico: A análise complementar exploratória não encontrou evidências suficientes para apontar diferenças significativas no nível de atenção sob as variáveis estudadas, após aplicação do rigor da correção FDR. Isso sugere ausência de relação forte e direta nestes cortes.")

# 4. Modelagem Supervisionada
log_report("\n--- Modelagem Supervisionada (CV 5-Fold) ---")
target_reg = 'attention_span_score'

# Para Classificação, binarizar attention_span_score baseado na mediana.
median_att = df['attention_span_score'].median()
df['attention_span_bin'] = (df['attention_span_score'] > median_att).astype(int)
target_clf = 'attention_span_bin'

log_report("NOTA: O target de classificação ('attention_span_bin') foi criado dividindo a variável contínua pela mediana apenas de forma exploratória, mantendo a regressão como abordagem principal.")

# Garantindo que a variável alvo original e anteriores vazamentos de clusters sejam excluídos das preditivas.
drop_cols = ['user_id', 'attention_span_score', 'attention_span_bin', 'attention_span_score_scaled', 'cluster_a', 'cluster_b', 'cluster_b_sem_pca']
features_df = df.drop(columns=[col for col in drop_cols if col in df.columns])

numeric_features = ['age', 'reels_watch_time_hours', 'daily_screen_time_hours', 'sleep_hours', 'focus_level', 'task_completion_rate', 'short_video_ratio']
numeric_features = [f for f in numeric_features if f in features_df.columns]
categorical_features = ['stress_level', 'platform', 'sleep_category', 'reels_quartile', 'ratio_quartile']
categorical_features = [f for f in categorical_features if f in features_df.columns]

X = features_df[numeric_features + categorical_features]
y_reg = df[target_reg]
y_clf = df[target_clf]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ])

# Regressão (KFold shuffle)
reg_models = {
    'Dummy Regressor': DummyRegressor(strategy='mean'),
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(random_state=42),
    'Random Forest Regressor': RandomForestRegressor(random_state=42, n_estimators=50),
    'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42, n_estimators=50)
}

reg_cv = KFold(n_splits=5, shuffle=True, random_state=42)
ml_metrics = []

for name, model in reg_models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scoring = {'mae': 'neg_mean_absolute_error', 'mse': 'neg_mean_squared_error', 'rmse': 'neg_root_mean_squared_error', 'r2': 'r2'}
    cv_res = cross_validate(pipe, X, y_reg, cv=reg_cv, scoring=scoring)
    ml_metrics.append({
        'Model_Type': 'Regression',
        'Model_Name': name,
        'MAE': -cv_res['test_mae'].mean(),
        'MSE': -cv_res['test_mse'].mean(),
        'RMSE': -cv_res['test_rmse'].mean(),
        'R2': cv_res['test_r2'].mean()
    })

# Classificação (Stratified KFold)
clf_models = {
    'Dummy Classifier': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest Classifier': RandomForestClassifier(random_state=42, n_estimators=50),
    'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42, n_estimators=50)
}

clf_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in clf_models.items():
    pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']
    cv_res = cross_validate(pipe, X, y_clf, cv=clf_cv, scoring=scoring)
    ml_metrics.append({
        'Model_Type': 'Classification',
        'Model_Name': name,
        'Accuracy': cv_res['test_accuracy'].mean(),
        'Precision': cv_res['test_precision_macro'].mean(),
        'Recall': cv_res['test_recall_macro'].mean(),
        'F1_Score': cv_res['test_f1_macro'].mean()
    })

metrics_df = pd.DataFrame(ml_metrics)
metrics_df.to_csv('metricas_modelos_complementares.csv', index=False)

dummy_r2 = metrics_df[metrics_df['Model_Name']=='Dummy Regressor']['R2'].values[0]
best_reg_r2 = metrics_df[metrics_df['Model_Type']=='Regression']['R2'].max()

if best_reg_r2 > dummy_r2 + 0.05:
    log_report(f"Resultado Regressão: Os modelos treinados sugerem um poder preditivo levemente superior ao baseline, indicando potencial relação multivariada (Best R2: {best_reg_r2:.4f} vs Dummy: {dummy_r2:.4f}).")
else:
    log_report(f"Resultado Regressão: Os modelos não apresentaram melhorias expressivas em comparação ao Dummy Regressor (Best R2: {best_reg_r2:.4f} vs Dummy: {dummy_r2:.4f}). Isso sugere que as variáveis não possuem sinal preditivo robusto.")

dummy_acc = metrics_df[metrics_df['Model_Name']=='Dummy Classifier']['Accuracy'].values[0]
best_clf_acc = metrics_df[metrics_df['Model_Type']=='Classification']['Accuracy'].max()

if best_clf_acc > dummy_acc + 0.05:
    log_report(f"Resultado Classificação: O algoritmo exploratório sugere melhoria leve na separação das categorias de atenção acima da mediana (Best Acc: {best_clf_acc:.4f} vs Dummy: {dummy_acc:.4f}).")
else:
    log_report(f"Resultado Classificação: Os modelos de classificação alternativa apresentaram performance virtualmente idêntica à de um adivinhador cego da maioria (Dummy Classifier), sugerindo forte limitação no poder preditivo para essa divisão (Best Acc: {best_clf_acc:.4f} vs Dummy: {dummy_acc:.4f}).")

# Gráfico de Importância (RF Regressor)
rf_pipe = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestRegressor(random_state=42, n_estimators=100))])
rf_pipe.fit(X, y_reg)
importances = rf_pipe.named_steps['model'].feature_importances_
feature_names = numeric_features + list(rf_pipe.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features))
imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(15)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='Importance', y='Feature', palette='viridis', hue='Feature', legend=False)
plt.title('Top 15 Features mais importantes (RF Regressor)')
plt.tight_layout()
plt.savefig('figuras/complementar_importancia_modelos.png')
plt.close()

# 5. Clusterização Alternativa (Totalmente Cega para attention_span)
log_report("\n--- Clusterização sem a Variável Alvo ---")
X_clust = preprocessor.fit_transform(X)
X_dense = X_clust.toarray() if hasattr(X_clust, 'toarray') else X_clust

clusters_results = []
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['cluster_kmeans_new'] = kmeans.fit_predict(X_dense)
clusters_results.append(('K-Means', df['cluster_kmeans_new']))

gmm = GaussianMixture(n_components=3, random_state=42)
df['cluster_gmm_new'] = gmm.fit_predict(X_dense)
clusters_results.append(('GMM', df['cluster_gmm_new']))

agg = AgglomerativeClustering(n_clusters=3)
df['cluster_agg_new'] = agg.fit_predict(X_dense)
clusters_results.append(('Agglomerative', df['cluster_agg_new']))

dbs = DBSCAN(eps=2.5, min_samples=10)
df['cluster_dbscan_new'] = dbs.fit_predict(X_dense)
if len(set(df['cluster_dbscan_new'])) > 1:
    clusters_results.append(('DBSCAN', df['cluster_dbscan_new']))

for name, labels in clusters_results:
    unique_labels = set(labels)
    if len(unique_labels) > 1:
        valid_idx = labels != -1
        if sum(valid_idx) > len(labels) * 0.1: # At least 10% not noise for DBscan
            sil = silhouette_score(X_dense[valid_idx], labels[valid_idx])
            db = davies_bouldin_score(X_dense[valid_idx], labels[valid_idx])
            ch = calinski_harabasz_score(X_dense[valid_idx], labels[valid_idx])
            log_report(f"Cluster {name}: Silhouette={sil:.4f}, Davies-Bouldin={db:.4f}, Calinski-Harabasz={ch:.4f}")
            
            valid_groups = [group['attention_span_score'].values for g, group in df[valid_idx].groupby(labels[valid_idx], observed=False)]
            if len(valid_groups) > 1:
                f_c, p_f_c = stats.f_oneway(*valid_groups)
                h_c, p_h_c = stats.kruskal(*valid_groups)
                log_report(f"  -> Comparação Attention Score por Cluster ({name}): ANOVA p_val={p_f_c:.4f}, Kruskal p_val={p_h_c:.4f}")

# 6. Salvar boxplots exploratórios
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='reels_quartile', y='attention_span_score', palette='Set2', hue='reels_quartile', legend=False)
plt.title('Attention Span Score por Quartil de Consumo de Reels')
plt.tight_layout()
plt.savefig('figuras/complementar_boxplot_quartis_reels.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='platform', y='attention_span_score', palette='Set3', hue='platform', legend=False)
plt.title('Attention Span Score por Plataforma')
plt.tight_layout()
plt.savefig('figuras/complementar_boxplot_plataforma.png')
plt.close()

plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='ratio_quartile', y='attention_span_score', palette='Pastel1', hue='ratio_quartile', legend=False)
plt.title('Attention Span Score por Quartil de Proporção (Ratio)')
plt.tight_layout()
plt.savefig('figuras/complementar_boxplot_ratio.png')
plt.close()

log_report("\nRelatório finalizado. Análises e figuras salvas com sucesso.")

with open('resultados_complementares.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(report_lines))
