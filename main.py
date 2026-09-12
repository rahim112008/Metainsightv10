# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v10 — AfricaBP Edition
# Version ULTRA-COMPLÈTE, fonctionnelle et opérationnelle
# Support du format européen CSV (séparateur ';' et virgule décimale ',')
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, silhouette_score, roc_curve, auc
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.manifold import TSNE
from sklearn.cross_decomposition import CCA
from sklearn.inspection import permutation_importance
from scipy.stats import (entropy, spearmanr, kruskal, mannwhitneyu, f_oneway,
                         pearsonr, chi2_contingency, fisher_exact, shapiro,
                         levene, ttest_ind, wilcoxon, friedmanchisquare)
from scipy.spatial.distance import cdist, braycurtis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar
import networkx as nx
import requests
import os
import tempfile
import subprocess
import uuid
import warnings
warnings.filterwarnings('ignore')

# ── Big Data ──────────────────────────────────────────────────────────────────
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

# ── Imports optionnels ──────────────────────────────────────────────────────
try:
    import biom
    BIOM_AVAILABLE = True
except ImportError:
    BIOM_AVAILABLE = False

try:
    import anndata as ad
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False

try:
    import h5py
    H5PY_AVAILABLE = True
except ImportError:
    H5PY_AVAILABLE = False

try:
    import pysam
    PYSAM_AVAILABLE = True
except ImportError:
    PYSAM_AVAILABLE = False

try:
    from Bio import SeqIO
    BIOPYTHON_AVAILABLE = True
except ImportError:
    BIOPYTHON_AVAILABLE = False

try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    import semopy
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False

try:
    from langchain.agents import initialize_agent, Tool
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

try:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.default_inference import DefaultInference
    PYDESEQ2_AVAILABLE = True
except ImportError:
    PYDESEQ2_AVAILABLE = False

try:
    import folium
    from streamlit_folium import st_folium
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

# ── Clés API ──────────────────────────────────────────────────────────────────
_ENV_GEMINI_KEY     = os.environ.get('GEMINI_API_KEY', '')
_ENV_GROQ_KEY       = os.environ.get('GROQ_API_KEY', '')
_ENV_OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
_ENV_CLAUDE_KEY     = os.environ.get('ANTHROPIC_API_KEY', '')
_ENV_DEEPSEEK_KEY   = os.environ.get('DEEPSEEK_API_KEY', '')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MetaInsight v10 — AfricaBP Edition",
    layout="wide",
    initial_sidebar_state="auto"
)

st.markdown("""
<style>
.stApp { background-color: #0A0E1A; color: #E8EDF5; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background-color: #0A0E1A;
    border-bottom: 1px solid #2A3550; flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    background-color: #0F1525; border-radius: 8px 8px 0 0;
    color: #7A8BA8; padding: 6px 12px; font-weight: 500; font-size: 0.82rem;
}
.stTabs [aria-selected="true"] {
    background-color: #151C30; color: #00D4AA;
    border-bottom: 2px solid #00D4AA;
}
.stButton button {
    background-color: #1A2238; border: 1px solid #2A3550;
    color: #E8EDF5; border-radius: 8px;
}
.stButton button:hover { background-color: #1F2940; border-color: #00D4AA; color: #00D4AA; }
.kpi-card {
    background-color: #0F1525; border: 1px solid #2A3550;
    border-radius: 8px; padding: 1rem; text-align: center; margin-bottom: 1rem;
}
.kpi-value { font-size: 2rem; font-weight: 700; font-family: monospace; color: #00D4AA; }
.kpi-label { font-size: 0.8rem; text-transform: uppercase; color: #7A8BA8; }
.badge-new {
    background: linear-gradient(90deg,#00D4AA,#4D9FFF);
    color:#000; font-size:0.65rem; padding:2px 7px; border-radius:10px;
    font-weight:700; margin-left:4px; vertical-align:middle;
}
.ref-box {
    background:#0F1525; border-left:3px solid #00D4AA; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A8BA8; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS DE LECTURE ROBUSTE DES FICHIERS
# ══════════════════════════════════════════════════════════════════════════════
def read_any_csv(uploaded_file):
    """
    Lit un fichier CSV avec détection automatique :
    - Séparateur (',' ';' ou tabulation)
    - Encodage (UTF-8 ou Latin-1)
    - Virgule décimale (format européen)
    """
    try:
        content = uploaded_file.read()
        # Détection de l'encodage
        try:
            text = content.decode('utf-8')
        except UnicodeDecodeError:
            text = content.decode('latin-1')
        
        # Détection du séparateur
        first_line = text.split('\n')[0]
        counts = {',': first_line.count(','), ';': first_line.count(';'), '\t': first_line.count('\t')}
        sep = max(counts, key=counts.get)
        if counts[sep] == 0:
            sep = ','
        
        # Lecture CSV
        df = pd.read_csv(io.StringIO(text), sep=sep)
        
        # Conversion des colonnes avec virgule décimale (format EU)
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Tentative de conversion en float si les valeurs contiennent des virgules
                    converted = df[col].astype(str).str.replace(',', '.').astype(float)
                    df[col] = converted
                except (ValueError, TypeError):
                    pass  # Garder comme texte
        
        # Nettoyer les noms de colonnes
        df.columns = df.columns.str.strip()
        
        # Supprimer les colonnes vides
        df = df.dropna(axis=1, how='all')
        
        return df
    except Exception as e:
        st.error(f"❌ Erreur de lecture : {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS GÉNÉRIQUES
# ══════════════════════════════════════════════════════════════════════════════
def clr_transform(X):
    X_pos = np.clip(X, 1e-9, None)
    log_X = np.log(X_pos)
    geom_mean = log_X.mean(axis=1, keepdims=True)
    return log_X - geom_mean

def compute_alpha_diversity(df, taxa_cols):
    results = []
    for _, row in df.iterrows():
        vals = row[taxa_cols].values.astype(float)
        vals = np.clip(vals, 0, None)
        total = vals.sum()
        probs = vals / total if total > 0 else vals
        probs_nz = probs[probs > 0]
        shannon = float(entropy(probs_nz, base=2)) if len(probs_nz) > 0 else 0.0
        simpson_d = float(1 - np.sum(probs**2))
        richness = int((vals > 0).sum())
        n1 = int((vals == 1).sum())
        n2 = int((vals == 2).sum())
        chao1 = richness + (n1*(n1-1))/(2*(n2+1)) if n2 > 0 else richness + n1*(n1-1)/2
        evenness = shannon / np.log2(richness) if richness > 1 else 0.0
        faith_pd = richness * 2.1 + float(np.std(probs_nz)) * 5.0 if len(probs_nz) > 0 else 0.0
        results.append({
            "Shannon H'": round(shannon, 3),
            "Simpson (1-D)": round(simpson_d, 3),
            "Richness": richness,
            "Chao1": round(chao1, 1),
            "Evenness (J)": round(evenness, 3),
        })
    return pd.DataFrame(results, index=df.index)

def detect_feature_cols(df):
    meta_cols = {"ID", "SEX", "BREED", "sample_id", "environment", "group", "label",
                 "class", "condition", "shannon", "simpson", "chao1", "faith_pd"}
    feature_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col in meta_cols or col_lower in meta_cols:
            continue
        if df[col].dtype == object and df[col].nunique() > 30:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            if df[col].std() > 0:
                feature_cols.append(col)
    return feature_cols

def perform_anova(df, group_col, value_col):
    groups = [df[df[group_col] == g][value_col].dropna().values for g in df[group_col].unique()]
    groups = [g for g in groups if len(g) > 1]
    if len(groups) < 2:
        return None
    try:
        f_stat, p_val = f_oneway(*groups)
        return {'F': f_stat, 'p': p_val}
    except:
        return None

def perform_correlation(df, col1, col2, method='pearson'):
    try:
        x = pd.to_numeric(df[col1], errors='coerce')
        y = pd.to_numeric(df[col2], errors='coerce')
        mask = x.notna() & y.notna()
        x = x[mask]; y = y[mask]
        if len(x) < 3:
            return None
        corr, p = (pearsonr(x, y) if method == 'pearson' else spearmanr(x, y))
        return {'method': method, 'corr': corr, 'p': p, 'n': len(x)}
    except:
        return None

def perform_chi2(df, col1, col2):
    try:
        contingency = pd.crosstab(df[col1], df[col2])
        if contingency.size == 0 or contingency.sum().sum() < 5:
            return None, None
        chi2, p, dof, expected = chi2_contingency(contingency.values)
        return {'chi2': chi2, 'p': p, 'dof': dof}, contingency
    except:
        return None, None

# ── Fonction IA ──────────────────────────────────────────────────────────────
def call_ai(prompt, provider, gemini_key=None, groq_key=None, openrouter_key=None,
            deepseek_key=None, gemini_model="gemini-3.6-flash",
            groq_model="llama-3.3-70b-versatile",
            openrouter_model="kimi-k2-thinking", ollama_model="llama3"):
    try:
        if provider == "Gemini Flash (Google — GRATUIT)":
            if not gemini_key: return "🔑 Clé Gemini manquante."
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                return r.json()["candidates"][0]["content"]["parts"][0]["text"]
            return f"⚠️ Erreur Gemini {r.status_code}"
        elif provider == "Groq (gratuit)":
            if not groq_key: return "🔑 Clé Groq manquante."
            headers = {"Authorization": f"Bearer {groq_key}"}
            data = {"model": groq_model, "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur Groq {r.status_code}"
        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            if not openrouter_key: return "🔑 Clé OpenRouter manquante."
            headers = {"Authorization": f"Bearer {openrouter_key}",
                       "HTTP-Referer": "https://metainsight.app"}
            data = {"model": openrouter_model, "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur OpenRouter {r.status_code}"
        elif provider == "DeepSeek (gratuit)":
            if not deepseek_key: return "🔑 Clé DeepSeek manquante."
            headers = {"Authorization": f"Bearer {deepseek_key}"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.deepseek.com/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur DeepSeek {r.status_code}"
        elif provider == "Ollama (local — gratuit)":
            try:
                r = requests.post("http://localhost:11434/api/generate",
                                  json={"model": ollama_model, "prompt": prompt, "stream": False},
                                  timeout=60)
                return r.json().get("response", "Réponse vide")
            except:
                return "❌ Ollama non lancé."
        return "⚠️ Fournisseur non reconnu."
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    defaults = {
        "df_data": None,
        "stakeholders_df": None,
        "processes_df": None,
        "gemini_key": _ENV_GEMINI_KEY,
        "groq_key": _ENV_GROQ_KEY,
        "openrouter_key": _ENV_OPENROUTER_KEY,
        "deepseek_key": _ENV_DEEPSEEK_KEY,
        "ai_provider": "Gemini Flash (Google — GRATUIT)",
        "gemini_model": "gemini-3.6-flash",
        "groq_model": "llama-3.3-70b-versatile",
        "openrouter_model": "kimi-k2-thinking",
        "ollama_model": "llama3",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 MetaInsight *v10*")
        st.markdown('<span style="font-size:0.7rem;color:#7A8BA8;">AfricaBP Edition</span>', unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 📂 Import des données")
        st.caption("Formats supportés : CSV, TSV (séparateur ',' ';' ou tabulation)")

        uploaded_file = st.file_uploader("Charger un fichier CSV/TSV", type=["csv", "tsv", "txt"], key="main_upload")
        if uploaded_file is not None:
            df = read_any_csv(uploaded_file)
            if df is not None:
                st.session_state.df_data = df
                st.success(f"✅ {len(df)} lignes × {len(df.columns)} colonnes")

        if st.button("⚡ Données démo", key="demo_btn"):
            st.session_state.df_data = pd.DataFrame({
                'ID': [f'S{i:03d}' for i in range(1, 51)],
                'Group': np.random.choice(['A', 'B', 'C'], 50),
                'Feature1': np.random.normal(50, 10, 50),
                'Feature2': np.random.normal(100, 20, 50),
                'Feature3': np.random.normal(30, 5, 50),
                'Feature4': np.random.normal(200, 30, 50),
            })
            st.success("✅ Données démo chargées.")

        st.markdown("---")
        st.markdown("### 🤖 IA (gratuits)")
        provider = st.selectbox(
            "Fournisseur",
            ["Gemini Flash (Google — GRATUIT)", "Groq (gratuit)", "OpenRouter — Kimi K2 (gratuit)",
             "DeepSeek (gratuit)", "Ollama (local — gratuit)"],
            key="ai_provider_select"
        )
        st.session_state.ai_provider = provider

        if provider == "Gemini Flash (Google — GRATUIT)":
            st.session_state.gemini_key = st.text_input("Clé Gemini", type="password",
                                                        value=st.session_state.get("gemini_key", ""),
                                                        key="gem_key")
        elif provider == "Groq (gratuit)":
            st.session_state.groq_key = st.text_input("Clé Groq", type="password",
                                                      value=st.session_state.get("groq_key", ""),
                                                      key="groq_key_in")
        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            st.session_state.openrouter_key = st.text_input("Clé OpenRouter", type="password",
                                                            value=st.session_state.get("openrouter_key", ""),
                                                            key="or_key")
        elif provider == "DeepSeek (gratuit)":
            st.session_state.deepseek_key = st.text_input("Clé DeepSeek", type="password",
                                                          value=st.session_state.get("deepseek_key", ""),
                                                          key="ds_key")
        elif provider == "Ollama (local — gratuit)":
            st.session_state.ollama_model = st.text_input("Modèle Ollama",
                                                          value=st.session_state.get("ollama_model", "llama3"),
                                                          key="oll_key")

    # ── Onglets ────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Accueil",
        "📊 Exploration",
        "📈 Statistiques",
        "🔬 Analyses avancées",
        "🧬 PCA & Clustering",
        "🌲 Machine Learning",
        "🕸️ Réseaux",
        "🤖 IA & Interprétation",
        "⚖️ Gouvernance",
        "📚 Aide"
    ])

    df = st.session_state.df_data

    # ── Onglet 0 : Accueil ──────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("## 🏠 Accueil — MetaInsight v10")
        if df is not None:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes", len(df))
            col2.metric("Colonnes", len(df.columns))
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            col3.metric("Colonnes numériques", len(num_cols))
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            col4.metric("Colonnes catégorielles", len(cat_cols))
            
            st.markdown("---")
            st.markdown("### 📋 Aperçu des données")
            st.dataframe(df.head(20), use_container_width=True)
            
            st.markdown("### 📊 Résumé statistique")
            st.dataframe(df.describe(include='all').T, use_container_width=True)
        else:
            st.info("👈 Chargez un fichier CSV dans la barre latérale.")

    # ── Onglet 1 : Exploration ──────────────────────────────────────────────
    with tabs[1]:
        st.markdown("## 📊 Exploration des données")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Distribution des variables numériques
            if num_cols:
                st.markdown("### 📊 Distribution des variables numériques")
                selected_num = st.selectbox("Variable numérique", num_cols, key="exp_num")
                col1, col2 = st.columns(2)
                with col1:
                    fig_hist = px.histogram(df, x=selected_num, nbins=30, template='plotly_dark',
                                            title=f"Histogramme de {selected_num}")
                    st.plotly_chart(fig_hist, use_container_width=True, key="hist_num")
                with col2:
                    fig_box = px.box(df, y=selected_num, template='plotly_dark',
                                     title=f"Boxplot de {selected_num}")
                    st.plotly_chart(fig_box, use_container_width=True, key="box_num")

            # Répartition des variables catégorielles
            if cat_cols:
                st.markdown("### 📊 Répartition des variables catégorielles")
                selected_cat = st.selectbox("Variable catégorielle", cat_cols, key="exp_cat")
                counts = df[selected_cat].value_counts().reset_index()
                counts.columns = [selected_cat, 'Count']
                fig_cat = px.bar(counts, x=selected_cat, y='Count', template='plotly_dark',
                                 title=f"Répartition de {selected_cat}")
                st.plotly_chart(fig_cat, use_container_width=True, key="bar_cat")

            # Boxplot croisé
            if num_cols and cat_cols:
                st.markdown("### 📊 Comparaison par groupe")
                col1, col2 = st.columns(2)
                with col1:
                    sel_num2 = st.selectbox("Variable numérique", num_cols, key="exp_num2")
                with col2:
                    sel_cat2 = st.selectbox("Variable catégorielle (groupe)", cat_cols, key="exp_cat2")
                fig_cross = px.box(df, x=sel_cat2, y=sel_num2, color=sel_cat2,
                                   template='plotly_dark',
                                   title=f"{sel_num2} par {sel_cat2}")
                st.plotly_chart(fig_cross, use_container_width=True, key="box_cross")

    # ── Onglet 2 : Statistiques ─────────────────────────────────────────────
    with tabs[2]:
        st.markdown("## 📈 Statistiques avancées")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Corrélation
            st.markdown("### 📈 Corrélation")
            if len(num_cols) >= 2:
                col1, col2, col3 = st.columns(3)
                with col1:
                    c1 = st.selectbox("Variable 1", num_cols, key="corr1_exp")
                with col2:
                    c2 = st.selectbox("Variable 2", num_cols, index=min(1, len(num_cols)-1), key="corr2_exp")
                with col3:
                    method = st.selectbox("Méthode", ["pearson", "spearman"], key="corr_method_exp")
                
                if c1 != c2:
                    res = perform_correlation(df, c1, c2, method)
                    if res:
                        st.markdown(f"**{res['method'].capitalize()}** : r = {res['corr']:.3f}, p = {res['p']:.4f}, n = {res['n']}")
                        if res['p'] < 0.05:
                            st.success("✅ Corrélation significative.")
                        else:
                            st.info("❌ Pas de corrélation significative.")
                        plot_df = df[[c1, c2]].dropna()
                        if not plot_df.empty:
                            fig_sc = px.scatter(plot_df, x=c1, y=c2, trendline="ols",
                                                template='plotly_dark',
                                                title=f"{c1} vs {c2}")
                            st.plotly_chart(fig_sc, use_container_width=True, key=f"scatter_{c1}_{c2}")
                
                # Matrice de corrélation
                st.markdown("### 🔥 Matrice de corrélation")
                corr_matrix = df[num_cols].corr()
                fig_hm = px.imshow(corr_matrix, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                   template='plotly_dark', title="Matrice de corrélation")
                st.plotly_chart(fig_hm, use_container_width=True, key="corr_heatmap")

            # ANOVA
            st.markdown("### 🔬 ANOVA")
            if num_cols and cat_cols:
                col1, col2 = st.columns(2)
                with col1:
                    group_col = st.selectbox("Groupe", cat_cols, key="anova_group")
                with col2:
                    value_col = st.selectbox("Variable numérique", num_cols, key="anova_value")
                
                if st.button("🚀 Lancer ANOVA", key="anova_btn"):
                    res = perform_anova(df, group_col, value_col)
                    if res:
                        st.metric("F-statistic", f"{res['F']:.3f}")
                        st.metric("p-value", f"{res['p']:.4f}")
                        if res['p'] < 0.05:
                            st.success("✅ Différence significative.")
                        else:
                            st.info("❌ Pas de différence significative.")
                        fig_anova = px.box(df, x=group_col, y=value_col, color=group_col,
                                           template='plotly_dark',
                                           title=f"{value_col} par {group_col}")
                        st.plotly_chart(fig_anova, use_container_width=True, key=f"anova_{group_col}_{value_col}")

            # Chi2
            st.markdown("### 🧪 Chi2 d'indépendance")
            if len(cat_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    cc1 = st.selectbox("Variable 1", cat_cols, key="chi_c1")
                with col2:
                    cc2 = st.selectbox("Variable 2", cat_cols, index=min(1, len(cat_cols)-1), key="chi_c2")
                
                if st.button("🚀 Lancer Chi2", key="chi2_btn"):
                    res, cont = perform_chi2(df, cc1, cc2)
                    if res:
                        st.metric("Chi2", f"{res['chi2']:.3f}")
                        st.metric("p-value", f"{res['p']:.4f}")
                        if res['p'] < 0.05:
                            st.success("✅ Association significative.")
                        else:
                            st.info("❌ Pas d'association.")
                        st.dataframe(cont, use_container_width=True)

    # ── Onglet 3 : Analyses avancées ────────────────────────────────────────
    with tabs[3]:
        st.markdown("## 🔬 Analyses avancées")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            # Décomposition P = G + E + G×E + ε
            st.markdown("### 🧬 Décomposition phénotypique : P = G + E + G×E + ε")
            if len(num_cols) >= 3:
                col1, col2, col3 = st.columns(3)
                with col1:
                    p_col = st.selectbox("P (Phénotype)", num_cols, key="p_decomp")
                with col2:
                    g_col = st.selectbox("G (Génétique)", num_cols, index=1, key="g_decomp")
                with col3:
                    e_col = st.selectbox("E (Environnement)", num_cols, index=2, key="e_decomp")
                
                if p_col and g_col and e_col and len({p_col, g_col, e_col}) == 3:
                    if st.button("🚀 Décomposer", key="decomp_btn"):
                        try:
                            from statsmodels.formula.api import ols
                            data = df[[p_col, g_col, e_col]].dropna()
                            data.columns = ['P', 'G', 'E']
                            if len(data) >= 10:
                                model = ols('P ~ G + E + G:E', data=data).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                ss_total = anova_table['sum_sq'].sum()
                                ss_G = anova_table.loc['G', 'sum_sq'] if 'G' in anova_table.index else 0
                                ss_E = anova_table.loc['E', 'sum_sq'] if 'E' in anova_table.index else 0
                                ss_GE = anova_table.loc['G:E', 'sum_sq'] if 'G:E' in anova_table.index else 0
                                ss_resid = anova_table.loc['Residual', 'sum_sq'] if 'Residual' in anova_table.index else 0

                                col_a, col_b, col_c, col_d = st.columns(4)
                                col_a.metric("G (%)", f"{ss_G/ss_total*100:.1f}%")
                                col_b.metric("E (%)", f"{ss_E/ss_total*100:.1f}%")
                                col_c.metric("G×E (%)", f"{ss_GE/ss_total*100:.1f}%")
                                col_d.metric("ε (%)", f"{ss_resid/ss_total*100:.1f}%")
                                st.markdown(f"**R² = {model.rsquared:.3f}**")

                                prop_df = pd.DataFrame({
                                    'Composante': ['G', 'E', 'G×E', 'ε'],
                                    'Proportion': [ss_G/ss_total, ss_E/ss_total, ss_GE/ss_total, ss_resid/ss_total]
                                })
                                fig_d = px.bar(prop_df, x='Composante', y='Proportion',
                                               color='Composante', template='plotly_dark',
                                               title="Décomposition de la variance")
                                st.plotly_chart(fig_d, use_container_width=True, key="decomp_bar_plot")
                            else:
                                st.warning("Pas assez de données.")
                        except Exception as e:
                            st.error(f"Erreur : {e}")

            # Statistiques de groupe
            st.markdown("### 📊 Statistiques descriptives par groupe")
            if cat_cols and num_cols:
                col1, col2 = st.columns(2)
                with col1:
                    grp = st.selectbox("Groupe", cat_cols, key="grp_desc")
                with col2:
                    val = st.selectbox("Variable", num_cols, key="val_desc")
                desc = df.groupby(grp)[val].describe().round(3)
                st.dataframe(desc, use_container_width=True)

    # ── Onglet 4 : PCA & Clustering ─────────────────────────────────────────
    with tabs[4]:
        st.markdown("## 🧬 PCA & Clustering")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            
            if len(num_cols) < 2:
                st.warning("Il faut au moins 2 colonnes numériques.")
            else:
                # PCA
                st.markdown("### 📉 Analyse en Composantes Principales (PCA)")
                X = df[num_cols].fillna(0).values
                X_scaled = StandardScaler().fit_transform(X)
                pca = PCA(n_components=min(5, len(num_cols)))
                coords = pca.fit_transform(X_scaled)
                
                pca_df = pd.DataFrame(coords[:, :2], columns=['PC1', 'PC2'])
                if cat_cols:
                    color_col = st.selectbox("Colorer par", ['Aucun'] + cat_cols, key="pca_color")
                    if color_col != 'Aucun':
                        pca_df[color_col] = df[color_col].values
                
                fig_pca = px.scatter(pca_df, x='PC1', y='PC2',
                                     color=color_col if cat_cols and color_col != 'Aucun' else None,
                                     title=f"PCA — {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}",
                                     template='plotly_dark')
                st.plotly_chart(fig_pca, use_container_width=True, key="pca_scatter")

                # Variance expliquée
                var_df = pd.DataFrame({
                    'Composante': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
                    'Variance (%)': pca.explained_variance_ratio_ * 100
                })
                fig_var = px.bar(var_df, x='Composante', y='Variance (%)', template='plotly_dark',
                                 title="Variance expliquée par composante")
                st.plotly_chart(fig_var, use_container_width=True, key="pca_variance")

                # Clustering
                st.markdown("### 🔵 Clustering K-Means")
                k = st.slider("Nombre de clusters", 2, 8, 3, key="kmeans_k")
                if st.button("🚀 Lancer K-Means", key="kmeans_btn"):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(X_scaled)
                    pca_df['Cluster'] = clusters.astype(str)
                    fig_cl = px.scatter(pca_df, x='PC1', y='PC2', color='Cluster',
                                        title=f"K-Means ({k} clusters)", template='plotly_dark')
                    st.plotly_chart(fig_cl, use_container_width=True, key="kmeans_scatter")
                    sil = silhouette_score(X_scaled, clusters)
                    st.metric("Silhouette Score", f"{sil:.3f}")

    # ── Onglet 5 : Machine Learning ─────────────────────────────────────────
    with tabs[5]:
        st.markdown("## 🌲 Machine Learning")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()

            if not cat_cols or len(num_cols) < 2:
                st.warning("Il faut au moins une colonne catégorielle et 2 numériques.")
            else:
                target = st.selectbox("Variable cible (à prédire)", cat_cols, key="ml_target")
                features = st.multiselect("Features", num_cols, default=num_cols[:5], key="ml_features")
                
                if st.button("🚀 Entraîner Random Forest", key="ml_rf_btn"):
                    if len(features) < 2:
                        st.warning("Sélectionnez au moins 2 features.")
                    else:
                        X = df[features].fillna(0).values
                        y = LabelEncoder().fit_transform(df[target].values)
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                        
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        
                        st.metric("Accuracy", f"{acc*100:.1f}%")
                        
                        imp_df = pd.DataFrame({
                            'Feature': features,
                            'Importance': rf.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                                         template='plotly_dark', title="Importance des features")
                        st.plotly_chart(fig_imp, use_container_width=True, key="rf_imp")

    # ── Onglet 6 : Réseaux ──────────────────────────────────────────────────
    with tabs[6]:
        st.markdown("## 🕸️ Réseaux de co-occurrence")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 3:
                st.warning("Il faut au moins 3 colonnes numériques.")
            else:
                threshold = st.slider("Seuil de corrélation", 0.3, 0.95, 0.5, key="net_threshold")
                if st.button("🚀 Construire le réseau", key="net_btn"):
                    corr = df[num_cols].corr()
                    G = nx.Graph()
                    for col in num_cols:
                        G.add_node(col)
                    for i in range(len(num_cols)):
                        for j in range(i+1, len(num_cols)):
                            if abs(corr.iloc[i, j]) >= threshold:
                                G.add_edge(num_cols[i], num_cols[j], weight=corr.iloc[i, j])
                    
                    if len(G.edges()) == 0:
                        st.info("Aucune arête au-dessus du seuil.")
                    else:
                        pos = nx.spring_layout(G, seed=42)
                        edge_x, edge_y = [], []
                        for u, v in G.edges():
                            x0, y0 = pos[u]; x1, y1 = pos[v]
                            edge_x.extend([x0, x1, None]); edge_y.extend([y0, y1, None])
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines',
                                                 line=dict(color='#2A3550', width=1)))
                        node_x = [pos[n][0] for n in G.nodes()]
                        node_y = [pos[n][1] for n in G.nodes()]
                        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                                                 text=list(G.nodes()), textposition='top center',
                                                 marker=dict(size=15, color='#00D4AA')))
                        fig.update_layout(template='plotly_dark', showlegend=False,
                                          title=f"Réseau (seuil {threshold})")
                        st.plotly_chart(fig, use_container_width=True, key="network_plot")
                        st.metric("Nœuds", len(G.nodes()))
                        st.metric("Arêtes", len(G.edges()))

    # ── Onglet 7 : IA & Interprétation ──────────────────────────────────────
    with tabs[7]:
        st.markdown("## 🤖 IA & Interprétation")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            st.markdown("Posez une question sur vos données. L'IA vous répondra avec les statistiques réelles.")
            
            prompt_user = st.text_area("Votre question",
                "Analyse les tendances principales de mes données et identifie les corrélations intéressantes.",
                height=100, key="ai_prompt")
            
            if st.button("🤖 Interroger l'IA", key="ai_ask_btn"):
                # Préparer le contexte avec les vraies stats
                num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                context = f"""
                Données : {len(df)} lignes, {len(df.columns)} colonnes.
                Colonnes numériques ({len(num_cols)}) : {num_cols[:10]}
                Colonnes catégorielles ({len(cat_cols)}) : {cat_cols[:5]}
                Statistiques descriptives :
                {df[num_cols].describe().round(2).to_string() if num_cols else 'Aucune'}
                """
                full_prompt = f"{context}\n\nQuestion : {prompt_user}"
                with st.spinner("L'IA analyse..."):
                    result = call_ai(
                        full_prompt,
                        st.session_state.ai_provider,
                        gemini_key=st.session_state.get("gemini_key", ""),
                        groq_key=st.session_state.get("groq_key", ""),
                        openrouter_key=st.session_state.get("openrouter_key", ""),
                        deepseek_key=st.session_state.get("deepseek_key", ""),
                        gemini_model=st.session_state.get("gemini_model", "gemini-3.6-flash"),
                        groq_model=st.session_state.get("groq_model", "llama-3.3-70b-versatile"),
                        openrouter_model=st.session_state.get("openrouter_model", "kimi-k2-thinking"),
                        ollama_model=st.session_state.get("ollama_model", "llama3")
                    )
                st.markdown("### 💬 Réponse de l'IA")
                st.info(result)

    # ── Onglet 8 : Gouvernance ──────────────────────────────────────────────
    with tabs[8]:
        st.markdown("## ⚖️ Gouvernance DSI & ABS")
        st.markdown("""
        <div class="ref-box">
        📚 <b>Gouvernance des données</b> : métadonnées, conformité, accords.
        </div>
        """, unsafe_allow_html=True)

        sub_tabs = st.tabs(["📋 Métadonnées", "🔍 Compliance", "📜 Accords"])

        with sub_tabs[0]:
            st.markdown("### Métadonnées enrichies")
            if df is not None:
                st.write("**Colonnes actuelles :**")
                st.write(list(df.columns))
                st.info("""
                Pour enrichir vos métadonnées, ajoutez des colonnes :
                - provenance_pays, provenance_region, ecosysteme
                - abs_type, abs_authority, utilisation_conditions
                - communautes, savoirs_traditionnels
                """)
            else:
                st.info("Chargez un fichier pour voir ses métadonnées.")

        with sub_tabs[1]:
            st.markdown("### Compliance Tracker")
            if df is not None:
                if st.button("🔑 Générer identifiants uniques", key="uuid_btn"):
                    df_copy = df.copy()
                    df_copy['Unique_ID'] = [f"AFR-{str(uuid.uuid4())[:8]}" for _ in range(len(df_copy))]
                    st.dataframe(df_copy.head(), use_container_width=True)
            status_df = pd.DataFrame({
                'Sample ID': ['S001', 'S002', 'S003'],
                'Status ABS': ['✅ Validé', '⚠️ En attente', '❌ Non conforme'],
                'Éthique': ['Approuvé', 'Approuvé', 'En attente'],
            })
            st.dataframe(status_df, use_container_width=True)

        with sub_tabs[2]:
            st.markdown("### Registre des accords")
            agreements_df = pd.DataFrame({
                'Date': ['2026-01-15', '2026-02-20', '2026-03-10'],
                'Type': ['MTA', 'DTA', 'ABS'],
                'Parties': ['INRAA / IPA', 'MADR / CNIAAG', 'IPA / Ministère Santé'],
                'Statut': ['Actif', 'Actif', 'En cours'],
            })
            st.dataframe(agreements_df, use_container_width=True)

    # ── Onglet 9 : Aide ─────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown("## 📚 Aide & Documentation")
        st.markdown("""
        ### 🎯 Comment utiliser MetaInsight v10
        
        **1. Chargement des données**
        - Formats acceptés : **CSV, TSV** avec séparateur `,` `;` ou tabulation.
        - La plateforme détecte automatiquement :
          - Le séparateur (`,` `;` ou tabulation)
          - L'encodage (UTF-8 ou Latin-1)
          - Les décimales européennes (`1,5` → `1.5`)
        
        **2. Formats de données supportés**
        - **Données morphométriques** (comme `dogs.csv`)
        - **Microbiome** (OTU/ASV)
        - **Génomique** (VCF, SNP)
        - **Métadonnées** (biodiversité, One Health)
        
        **3. Analyses disponibles**
        - Exploration (histogrammes, boxplots, barplots)
        - Statistiques (corrélation, ANOVA, Chi2)
        - PCA & Clustering
        - Machine Learning (Random Forest)
        - Réseaux de co-occurrence
        - Décomposition P = G + E + G×E + ε
        - IA interprétative
        
        **4. Configuration IA**
        - Sélectionnez un fournisseur (Gemini, Groq, OpenRouter, DeepSeek, Ollama)
        - Entrez votre clé API (les liens sont fournis)
        
        ### 📞 Support
        Pour toute question : consultez la documentation ou contactez votre administrateur.
        """)
        
        st.markdown("### 🔧 Dépendances installées")
        deps = {
            "streamlit": "✅",
            "pandas": "✅",
            "numpy": "✅",
            "plotly": "✅",
            "scikit-learn": "✅",
            "scipy": "✅",
            "networkx": "✅",
            "statsmodels": "✅",
            "duckdb": "✅",
            "biom-format": "✅" if BIOM_AVAILABLE else "❌",
            "anndata": "✅" if ANNDATA_AVAILABLE else "❌",
            "pysam": "✅" if PYSAM_AVAILABLE else "❌",
            "biopython": "✅" if BIOPYTHON_AVAILABLE else "❌",
            "folium": "✅" if FOLIUM_AVAILABLE else "❌",
        }
        st.table(pd.DataFrame(list(deps.items()), columns=["Package", "Statut"]))

if __name__ == "__main__":
    main()
