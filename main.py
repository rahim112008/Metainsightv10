# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v10 — Édition ZOOTECHNIE & MORPHOMÉTRIE
# Application autonome pour l'analyse morphométrique animale
# Version corrigée : support complet pyarrow + format européen
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, silhouette_score
from scipy.stats import (spearmanr, kruskal, mannwhitneyu, f_oneway,
                         pearsonr, shapiro, ttest_ind)
from scipy.cluster.hierarchy import linkage, dendrogram
import networkx as nx
import requests
import os
import warnings
warnings.filterwarnings('ignore')

# ── Clés API ──────────────────────────────────────────────────────────────────
_ENV_GEMINI_KEY     = os.environ.get('GEMINI_API_KEY', '')
_ENV_GROQ_KEY       = os.environ.get('GROQ_API_KEY', '')
_ENV_OPENROUTER_KEY = os.environ.get('OPENROUTER_API_KEY', '')
_ENV_DEEPSEEK_KEY   = os.environ.get('DEEPSEEK_API_KEY', '')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION STREAMLIT
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MetaInsight — Morphométrie Animale",
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
.ref-box {
    background:#0F1525; border-left:3px solid #00D4AA; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.85rem; color:#A8B5C8; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  LECTURE CSV ROBUSTE (CORRIGÉE pour pyarrow)
# ══════════════════════════════════════════════════════════════════════════════
def read_any_csv(uploaded_file):
    """
    Lit un CSV avec détection auto du séparateur, encodage et décimales.
    CORRECTION : force le backend numpy au lieu de pyarrow.
    """
    try:
        content = uploaded_file.read()
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

        # CORRECTION 1 : forcer le backend numpy (pas Arrow)
        df = pd.read_csv(io.StringIO(text), sep=sep, dtype_backend='numpy_nullable')

        # Conversion des décimales européennes
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    df[col] = df[col].astype(str).str.replace(',', '.').astype(float)
                except (ValueError, TypeError):
                    pass

        # CORRECTION 2 CRITIQUE : forcer les colonnes numériques en float64 numpy
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = df[col].astype('float64')
                except (ValueError, TypeError):
                    pass

        df.columns = df.columns.str.strip()
        df = df.dropna(axis=1, how='all')
        return df
    except Exception as e:
        st.error(f"❌ Erreur de lecture : {e}")
        return None

# ══════════════════════════════════════════════════════════════════════════════
#  IA
# ══════════════════════════════════════════════════════════════════════════════
def call_ai(prompt, provider, gemini_key=None, groq_key=None,
            openrouter_key=None, deepseek_key=None,
            gemini_model="gemini-3.6-flash",
            groq_model="llama-3.3-70b-versatile",
            openrouter_model="kimi-k2-thinking",
            ollama_model="llama3"):
    try:
        if provider == "Gemini (GRATUIT)":
            if not gemini_key: return "🔑 Clé Gemini manquante."
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(url, json=payload, timeout=30)
            return r.json()["candidates"][0]["content"]["parts"][0]["text"] if r.status_code == 200 else f"⚠️ {r.status_code}"
        elif provider == "Groq (gratuit)":
            if not groq_key: return "🔑 Clé Groq manquante."
            headers = {"Authorization": f"Bearer {groq_key}"}
            data = {"model": groq_model, "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"⚠️ {r.status_code}"
        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            if not openrouter_key: return "🔑 Clé OpenRouter manquante."
            headers = {"Authorization": f"Bearer {openrouter_key}"}
            data = {"model": openrouter_model, "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"⚠️ {r.status_code}"
        elif provider == "DeepSeek (gratuit)":
            if not deepseek_key: return "🔑 Clé DeepSeek manquante."
            headers = {"Authorization": f"Bearer {deepseek_key}"}
            data = {"model": "deepseek-chat", "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.deepseek.com/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            return r.json()["choices"][0]["message"]["content"] if r.status_code == 200 else f"⚠️ {r.status_code}"
        elif provider == "Ollama (local)":
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
#  NOMS DES MESURES MORPHOMÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════════
MORPHO_LABELS = {
    'HW': 'Hauteur au garrot (cm)',
    'HR': 'Hauteur à la croupe (cm)',
    'BL': 'Longueur du corps (cm)',
    'HL': 'Longueur de la tête (cm)',
    'HEW': 'Largeur de la tête (cm)',
    'ML': 'Longueur du museau (cm)',
    'HG': 'Tour de tête (cm)',
    'EL': 'Longueur de l\'oreille (cm)',
    'CG': 'Tour de poitrine (cm)',
    'WG': 'Tour de taille (cm)',
    'AG': 'Tour d\'abdomen (cm)',
    'NL': 'Longueur du cou (cm)',
    'LW': 'Largeur du poitrail (cm)',
}

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS D'ANALYSE
# ══════════════════════════════════════════════════════════════════════════════
def compute_morphometric_indices(df):
    """Calcule les indices morphométriques classiques."""
    df = df.copy()
    if 'BL' in df.columns and 'HW' in df.columns:
        df['Indice_format'] = (df['BL'] / df['HW'] * 100).round(2)
    if 'CG' in df.columns and 'HW' in df.columns:
        df['Indice_poitrine'] = (df['CG'] / df['HW'] * 100).round(2)
    if 'HL' in df.columns and 'HEW' in df.columns:
        df['Indice_cephalique'] = (df['HL'] / df['HEW'] * 100).round(2)
    if 'CG' in df.columns and 'BL' in df.columns:
        df['Indice_masse'] = (df['CG'] / df['BL'] * 100).round(2)
    if 'ML' in df.columns and 'HL' in df.columns:
        df['Ratio_museau'] = (df['ML'] / df['HL'] * 100).round(2)
    if 'NL' in df.columns and 'BL' in df.columns:
        df['Ratio_cou'] = (df['NL'] / df['BL'] * 100).round(2)
    return df

def analyze_sexual_dimorphism(df, morpho_cols, sex_col='SEX'):
    """Compare mâles et femelles pour chaque mesure."""
    results = []
    for col in morpho_cols:
        males = df[df[sex_col] == 'M'][col].dropna()
        females = df[df[sex_col] == 'F'][col].dropna()
        if len(males) >= 2 and len(females) >= 2:
            try:
                stat, p = mannwhitneyu(males, females, alternative='two-sided')
                diff_pct = (males.mean() - females.mean()) / females.mean() * 100
                results.append({
                    'Mesure': col,
                    'Mâles (moy)': round(males.mean(), 2),
                    'Femelles (moy)': round(females.mean(), 2),
                    'Différence (%)': round(diff_pct, 2),
                    'p-value': round(p, 4),
                    'Significatif': '✅' if p < 0.05 else '❌'
                })
            except:
                pass
    return pd.DataFrame(results)

def analyze_breed_differences(df, morpho_cols, breed_col='BREED'):
    """Compare les races par Kruskal-Wallis."""
    results = []
    for col in morpho_cols:
        groups = [df[df[breed_col] == b][col].dropna().values for b in df[breed_col].unique()]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                results.append({
                    'Mesure': col,
                    'H (Kruskal)': round(stat, 2),
                    'p-value': round(p, 4),
                    'Significatif': '✅' if p < 0.05 else '❌'
                })
            except:
                pass
    return pd.DataFrame(results)

def compute_allometric_regression(df, x_col, y_col):
    """Régression allométrique log(y) = a * log(x) + b."""
    data = df[[x_col, y_col]].dropna()
    data = data[(data[x_col] > 0) & (data[y_col] > 0)]
    if len(data) < 5:
        return None
    log_x = np.log(data[x_col].values)
    log_y = np.log(data[y_col].values)
    coeffs = np.polyfit(log_x, log_y, 1)
    r = np.corrcoef(log_x, log_y)[0, 1]
    return {'slope': coeffs[0], 'intercept': coeffs[1], 'r': r, 'n': len(data)}

# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    defaults = {
        "df": None,
        "gemini_key": _ENV_GEMINI_KEY,
        "groq_key": _ENV_GROQ_KEY,
        "openrouter_key": _ENV_OPENROUTER_KEY,
        "deepseek_key": _ENV_DEEPSEEK_KEY,
        "ai_provider": "Gemini (GRATUIT)",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🐕 MetaInsight Zootechnie")
        st.markdown('<span style="font-size:0.75rem;color:#7A8BA8;">Analyse morphométrique</span>',
                    unsafe_allow_html=True)
        st.markdown("---")

        st.markdown("### 📂 Charger vos données")
        st.caption("Uploadez votre fichier `dogs.csv` ici :")

        uploaded_file = st.file_uploader(
            "Fichier CSV morphométrique",
            type=["csv", "tsv", "txt"],
            key="morpho_upload"
        )

        if uploaded_file is not None:
            df = read_any_csv(uploaded_file)
            if df is not None:
                st.session_state.df = df
                st.success(f"✅ {len(df)} chiens × {len(df.columns)} colonnes")

        st.markdown("---")
        st.markdown("### 🤖 IA")
        provider = st.selectbox("Fournisseur",
            ["Gemini (GRATUIT)", "Groq (gratuit)", "OpenRouter — Kimi K2 (gratuit)",
             "DeepSeek (gratuit)", "Ollama (local)"], key="ai_prov")
        st.session_state.ai_provider = provider

        if provider == "Gemini (GRATUIT)":
            st.session_state.gemini_key = st.text_input("Clé Gemini", type="password",
                value=st.session_state.get("gemini_key", ""), key="gk")
        elif provider == "Groq (gratuit)":
            st.session_state.groq_key = st.text_input("Clé Groq", type="password",
                value=st.session_state.get("groq_key", ""), key="gqk")
        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            st.session_state.openrouter_key = st.text_input("Clé OpenRouter", type="password",
                value=st.session_state.get("openrouter_key", ""), key="ork")
        elif provider == "DeepSeek (gratuit)":
            st.session_state.deepseek_key = st.text_input("Clé DeepSeek", type="password",
                value=st.session_state.get("deepseek_key", ""), key="dsk")

    df = st.session_state.df

    # ── ONGLETS ─────────────────────────────────────────────────────────────
    tabs = st.tabs([
        "🏠 Accueil",
        "📊 Statistiques",
        "♂♀ Dimorphisme sexuel",
        "🐕 Différences raciales",
        "📐 Indices morphométriques",
        "🧬 PCA & LDA",
        "📈 Allométrie",
        "🔗 Corrélations",
        "⚖️ P = G + E + G×E",
        "🤖 IA",
        "📚 Aide"
    ])

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 0 : Accueil
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        st.markdown("## 🏠 Accueil — Analyse morphométrique")
        if df is None:
            st.info("👈 Uploadez votre fichier `dogs.csv` dans la barre latérale.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total chiens", len(df))
            col2.metric("Races", df['BREED'].nunique() if 'BREED' in df.columns else 0)
            col3.metric("Mesures", len(morpho_cols))
            col4.metric("Sexes", df['SEX'].nunique() if 'SEX' in df.columns else 0)

            st.markdown("---")
            st.markdown("### 📋 Aperçu des données")
            st.dataframe(df.head(20), use_container_width=True)

            if 'BREED' in df.columns:
                st.markdown("### 📊 Répartition des races")
                breed_counts = df['BREED'].value_counts().reset_index()
                breed_counts.columns = ['Race', 'Nombre']
                fig = px.bar(breed_counts, x='Race', y='Nombre', color='Race',
                             template='plotly_dark', title="Nombre de chiens par race")
                st.plotly_chart(fig, use_container_width=True, key="accueil_breed")

            if 'BREED' in df.columns and 'SEX' in df.columns:
                st.markdown("### 📊 Sexe × Race")
                cross = pd.crosstab(df['BREED'], df['SEX'])
                fig2 = px.bar(cross, barmode='group', template='plotly_dark',
                              title="Sexe × Race")
                st.plotly_chart(fig2, use_container_width=True, key="accueil_sex_breed")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 1 : Statistiques
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("## 📊 Statistiques descriptives")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            st.markdown("### 📈 Statistiques globales")
            st.dataframe(df[morpho_cols].describe().round(2), use_container_width=True)

            if 'BREED' in df.columns:
                st.markdown("### 🐕 Statistiques par race")
                breed_stats = df.groupby('BREED')[morpho_cols].agg(['mean', 'std']).round(2)
                st.dataframe(breed_stats, use_container_width=True)

            if 'SEX' in df.columns:
                st.markdown("### ♂♀ Statistiques par sexe")
                sex_stats = df.groupby('SEX')[morpho_cols].agg(['mean', 'std']).round(2)
                st.dataframe(sex_stats, use_container_width=True)

            st.markdown("### 📦 Distribution")
            col1, col2 = st.columns(2)
            with col1:
                sel = st.selectbox("Mesure", morpho_cols,
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="stat_sel")
            with col2:
                grp = st.selectbox("Grouper par",
                    ['Aucun', 'BREED', 'SEX'] if 'BREED' in df.columns else ['Aucun'],
                    key="stat_grp")

            if grp == 'Aucun':
                fig = px.histogram(df, x=sel, nbins=30, template='plotly_dark',
                                   title=f"Distribution de {MORPHO_LABELS.get(sel, sel)}")
            else:
                fig = px.box(df, x=grp, y=sel, color=grp, template='plotly_dark',
                             title=f"{MORPHO_LABELS.get(sel, sel)} par {grp}")
            st.plotly_chart(fig, use_container_width=True, key=f"stat_dist_{sel}_{grp}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 2 : Dimorphisme sexuel
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## ♂♀ Dimorphisme sexuel")
        st.markdown('<div class="ref-box">📚 Comparaison Mâles vs Femelles (Mann-Whitney)</div>',
                    unsafe_allow_html=True)
        if df is None or 'SEX' not in df.columns:
            st.info("Chargez un fichier avec colonne SEX.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            dimor = analyze_sexual_dimorphism(df, morpho_cols)
            if not dimor.empty:
                st.dataframe(dimor.style.background_gradient(cmap='RdBu_r', subset=['Différence (%)']),
                             use_container_width=True)

                melted = df.melt(id_vars=['SEX'], value_vars=morpho_cols,
                                 var_name='Mesure', value_name='Valeur')
                melted['Mesure_label'] = melted['Mesure'].map(MORPHO_LABELS)
                fig = px.box(melted, x='Mesure_label', y='Valeur', color='SEX',
                             template='plotly_dark', title="Dimorphisme sexuel")
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True, key="dimorph_box")

                n_sig = (dimor['Significatif'] == '✅').sum()
                st.metric("Mesures significatives", f"{n_sig} / {len(dimor)}")

                if st.button("🤖 Interpréter", key="dimorph_ai"):
                    summary = dimor[dimor['Significatif'] == '✅'].to_string()
                    prompt = f"""Expert zootechnie, analyse ce dimorphisme sexuel canin :
{summary}
Interprète les différences et leur importance pour la sélection."""
                    st.info(call_ai(prompt, st.session_state.ai_provider,
                                    gemini_key=st.session_state.get("gemini_key", ""),
                                    groq_key=st.session_state.get("groq_key", ""),
                                    openrouter_key=st.session_state.get("openrouter_key", ""),
                                    deepseek_key=st.session_state.get("deepseek_key", "")))

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 3 : Différences raciales
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## 🐕 Différences raciales")
        st.markdown('<div class="ref-box">📚 Comparaison entre races (Kruskal-Wallis)</div>',
                    unsafe_allow_html=True)
        if df is None or 'BREED' not in df.columns:
            st.info("Chargez un fichier avec colonne BREED.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            breed_diff = analyze_breed_differences(df, morpho_cols)
            if not breed_diff.empty:
                st.dataframe(breed_diff.style.background_gradient(cmap='YlOrRd', subset=['H (Kruskal)']),
                             use_container_width=True)

                st.markdown("### 📊 Profil morphologique par race (Radar)")
                breed_means = df.groupby('BREED')[morpho_cols].mean()
                breed_norm = (breed_means - breed_means.min()) / (breed_means.max() - breed_means.min())

                fig_radar = go.Figure()
                for breed in breed_norm.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=breed_norm.loc[breed].values,
                        theta=[MORPHO_LABELS.get(c, c) for c in morpho_cols],
                        fill='toself', name=breed
                    ))
                fig_radar.update_layout(template='plotly_dark',
                    title="Profil morphologique moyen par race",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                st.plotly_chart(fig_radar, use_container_width=True, key="breed_radar")

                sel_measure = st.selectbox("Mesure", morpho_cols,
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="breed_measure")
                fig_box = px.box(df, x='BREED', y=sel_measure, color='BREED',
                                 template='plotly_dark',
                                 title=f"{MORPHO_LABELS.get(sel_measure, sel_measure)} par race")
                st.plotly_chart(fig_box, use_container_width=True, key=f"breed_box_{sel_measure}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 4 : Indices morphométriques
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## 📐 Indices morphométriques")
        st.markdown('<div class="ref-box">📚 Indices : format, poitrine, céphalique, masse</div>',
                    unsafe_allow_html=True)
        if df is None:
            st.info("Chargez un fichier.")
        else:
            df_indices = compute_morphometric_indices(df)
            index_cols = [c for c in df_indices.columns if c.startswith('Indice') or c.startswith('Ratio')]

            if index_cols:
                st.markdown("### 📊 Indices calculés")
                st.dataframe(df_indices[index_cols].describe().round(2), use_container_width=True)

                if 'BREED' in df.columns:
                    st.markdown("### 📊 Distribution par race")
                    idx_melted = df_indices.melt(id_vars=['BREED'], value_vars=index_cols,
                                                  var_name='Indice', value_name='Valeur')
                    fig = px.box(idx_melted, x='Indice', y='Valeur', color='BREED',
                                 template='plotly_dark')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True, key="indices_box")

                st.markdown("### 🔗 Corrélations entre indices")
                corr = df_indices[index_cols].corr()
                fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                     text_auto=True, template='plotly_dark')
                st.plotly_chart(fig_corr, use_container_width=True, key="indices_corr")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 5 : PCA & LDA
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[5]:
        st.markdown("## 🧬 PCA & LDA")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            if len(morpho_cols) < 2:
                st.warning("Il faut au moins 2 mesures.")
            else:
                X = df[morpho_cols].fillna(df[morpho_cols].mean()).values.astype('float64')
                X_scaled = StandardScaler().fit_transform(X)

                st.markdown("### 📉 PCA")
                pca = PCA(n_components=min(5, len(morpho_cols)))
                coords = pca.fit_transform(X_scaled)
                pca_df = pd.DataFrame(coords[:, :2], columns=['PC1', 'PC2'])
                if 'BREED' in df.columns:
                    pca_df['BREED'] = df['BREED'].values
                if 'SEX' in df.columns:
                    pca_df['SEX'] = df['SEX'].values

                fig_pca = px.scatter(pca_df, x='PC1', y='PC2',
                                     color='BREED' if 'BREED' in pca_df.columns else None,
                                     symbol='SEX' if 'SEX' in pca_df.columns else None,
                                     template='plotly_dark',
                                     title=f"PCA — {pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}")
                st.plotly_chart(fig_pca, use_container_width=True, key="pca_morpho")

                st.markdown("### 🎯 Contribution des variables (loadings)")
                loadings = pd.DataFrame(pca.components_[:2].T,
                                        columns=['PC1', 'PC2'],
                                        index=[MORPHO_LABELS.get(c, c) for c in morpho_cols])
                st.dataframe(loadings.round(3), use_container_width=True)

                if 'BREED' in df.columns and df['BREED'].nunique() >= 2:
                    st.markdown("### 🎯 LDA (séparation des races)")
                    le = LabelEncoder()
                    y = le.fit_transform(df['BREED'].values)
                    try:
                        lda = LDA(n_components=min(2, df['BREED'].nunique()-1))
                        coords_lda = lda.fit_transform(X_scaled, y)
                        n_comp = min(2, coords_lda.shape[1])
                        lda_df = pd.DataFrame(coords_lda[:, :n_comp],
                                              columns=['LD1', 'LD2'][:n_comp])
                        lda_df['BREED'] = df['BREED'].values
                        fig_lda = px.scatter(lda_df, x='LD1',
                                             y='LD2' if 'LD2' in lda_df.columns else None,
                                             color='BREED', template='plotly_dark',
                                             title="LDA — Séparation des races")
                        st.plotly_chart(fig_lda, use_container_width=True, key="lda_breed")
                    except Exception as e:
                        st.warning(f"LDA impossible : {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 6 : Allométrie
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("## 📈 Analyse allométrique")
        st.markdown('<div class="ref-box">📚 Régression log-log : log(y) = a × log(x) + b</div>',
                    unsafe_allow_html=True)
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            if len(morpho_cols) < 2:
                st.warning("Il faut au moins 2 mesures.")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("Mesure X", morpho_cols,
                        format_func=lambda x: MORPHO_LABELS.get(x, x), key="allo_x")
                with col2:
                    y_col = st.selectbox("Mesure Y", morpho_cols, index=min(1, len(morpho_cols)-1),
                        format_func=lambda x: MORPHO_LABELS.get(x, x), key="allo_y")

                if x_col != y_col:
                    res = compute_allometric_regression(df, x_col, y_col)
                    if res:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Pente (a)", f"{res['slope']:.3f}")
                        col2.metric("Corrélation (r)", f"{res['r']:.3f}")
                        col3.metric("n", res['n'])

                        if res['slope'] > 1:
                            st.info("🔼 Allométrie positive")
                        elif res['slope'] < 1:
                            st.info("🔽 Allométrie négative")
                        else:
                            st.success("➡️ Isométrie")

                        fig = px.scatter(df, x=x_col, y=y_col,
                                         color='BREED' if 'BREED' in df.columns else None,
                                         trendline='ols', template='plotly_dark',
                                         log_x=True, log_y=True,
                                         title=f"{MORPHO_LABELS.get(y_col,y_col)} vs {MORPHO_LABELS.get(x_col,x_col)}")
                        st.plotly_chart(fig, use_container_width=True, key=f"allo_{x_col}_{y_col}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 7 : Corrélations
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("## 🔗 Corrélations")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            st.markdown("### 🔥 Matrice de corrélation")
            corr = df[morpho_cols].corr()
            fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                            text_auto='.2f', template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True, key="corr_morpho")

            st.markdown("### 🌲 Clustering hiérarchique")
            Z = linkage(corr.values, method='average')
            fig_dend, ax = plt.subplots(figsize=(10, 5))
            dendrogram(Z, labels=[MORPHO_LABELS.get(c, c) for c in morpho_cols], ax=ax)
            plt.title("Dendrogramme des mesures")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig_dend)
            plt.close()

            st.markdown("### 🎯 Corrélation ciblée")
            col1, col2, col3 = st.columns(3)
            with col1:
                c1 = st.selectbox("Mesure 1", morpho_cols,
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="corr1")
            with col2:
                c2 = st.selectbox("Mesure 2", morpho_cols, index=1,
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="corr2")
            with col3:
                method = st.selectbox("Méthode", ['pearson', 'spearman'], key="corr_meth")

            if c1 != c2:
                if method == 'pearson':
                    r, p = pearsonr(df[c1].dropna(), df[c2].dropna())
                else:
                    r, p = spearmanr(df[c1].dropna(), df[c2].dropna())
                st.metric(f"Corrélation {method}", f"r = {r:.3f}", delta=f"p = {p:.4f}")
                fig_sc = px.scatter(df, x=c1, y=c2,
                                    color='BREED' if 'BREED' in df.columns else None,
                                    trendline='ols', template='plotly_dark')
                st.plotly_chart(fig_sc, use_container_width=True, key=f"scatter_{c1}_{c2}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 8 : P = G + E + G×E + ε
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("## ⚖️ Décomposition P = G + E + G×E + ε")
        st.markdown('<div class="ref-box">📚 Modèle linéaire avec interaction</div>',
                    unsafe_allow_html=True)
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            st.markdown("""
            **Interprétation :**
            - **P** : Mesure à expliquer (ex : CG tour de poitrine)
            - **G** : Effet génétique (ex : BREED encodée, HW hauteur)
            - **E** : Effet environnemental (ex : indice corporel)
            - **G×E** : Interaction
            """)

            col1, col2, col3 = st.columns(3)
            with col1:
                p_col = st.selectbox("P (Phénotype)", morpho_cols,
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="p_dec")
            with col2:
                g_col = st.selectbox("G (Génétique)", morpho_cols, index=1,
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="g_dec")
            with col3:
                e_col = st.selectbox("E (Environnement)", morpho_cols, index=min(2, len(morpho_cols)-1),
                    format_func=lambda x: MORPHO_LABELS.get(x, x), key="e_dec")

            if len({p_col, g_col, e_col}) == 3:
                if st.button("🚀 Décomposer", key="decomp_btn"):
                    try:
                        from statsmodels.formula.api import ols
                        import statsmodels.api as sm
                        data = df[[p_col, g_col, e_col]].dropna().astype('float64')
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
                                'Proportion (%)': [ss_G/ss_total*100, ss_E/ss_total*100,
                                                   ss_GE/ss_total*100, ss_resid/ss_total*100]
                            })
                            fig = px.bar(prop_df, x='Composante', y='Proportion (%)',
                                         color='Composante', template='plotly_dark')
                            st.plotly_chart(fig, use_container_width=True, key="decomp_bar_morpho")
                    except Exception as e:
                        st.error(f"Erreur : {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 9 : IA
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[9]:
        st.markdown("## 🤖 Interprétation IA")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            prompt_user = st.text_area("Votre question",
                "Analyse les différences morphologiques entre races et propose des recommandations pour la sélection.",
                height=120, key="ai_prompt_morpho")

            if st.button("🤖 Interroger l'IA", key="ai_btn_morpho"):
                context = f"""
                Données morphométriques canines : {len(df)} chiens
                Races : {df['BREED'].value_counts().to_dict() if 'BREED' in df.columns else 'N/A'}
                Sexes : {df['SEX'].value_counts().to_dict() if 'SEX' in df.columns else 'N/A'}
                Moyennes par race :
                {df.groupby('BREED')[morpho_cols].mean().round(2).to_string() if 'BREED' in df.columns else 'N/A'}
                """
                full_prompt = f"{context}\n\nQuestion : {prompt_user}"
                with st.spinner("Analyse IA..."):
                    res = call_ai(full_prompt, st.session_state.ai_provider,
                                  gemini_key=st.session_state.get("gemini_key", ""),
                                  groq_key=st.session_state.get("groq_key", ""),
                                  openrouter_key=st.session_state.get("openrouter_key", ""),
                                  deepseek_key=st.session_state.get("deepseek_key", ""))
                st.info(res)

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 10 : Aide
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[10]:
        st.markdown("## 📚 Aide & Documentation")
        st.markdown("""
        ### 🐕 Mesures morphométriques reconnues

        | Code | Signification |
        |------|---------------|
        | **HW** | Hauteur au garrot (cm) |
        | **HR** | Hauteur à la croupe (cm) |
        | **BL** | Longueur du corps (cm) |
        | **HL** | Longueur de la tête (cm) |
        | **HEW** | Largeur de la tête (cm) |
        | **ML** | Longueur du museau (cm) |
        | **HG** | Tour de tête (cm) |
        | **EL** | Longueur de l'oreille (cm) |
        | **CG** | Tour de poitrine (cm) |
        | **WG** | Tour de taille (cm) |
        | **AG** | Tour d'abdomen (cm) |
        | **NL** | Longueur du cou (cm) |
        | **LW** | Largeur du poitrail (cm) |

        ### 📊 Analyses disponibles
        - Statistiques descriptives
        - Dimorphisme sexuel (Mann-Whitney)
        - Différences raciales (Kruskal-Wallis + Radar)
        - Indices morphométriques (format, poitrine, céphalique, masse)
        - PCA & LDA
        - Allométrie (régression log-log)
        - Corrélations + clustering
        - Décomposition P = G + E + G×E + ε
        - IA interprétative

        ### 💡 Utilisation
        1. Uploadez votre fichier `dogs.csv` dans la barre latérale
        2. Naviguez dans les onglets
        3. Utilisez l'IA pour interpréter les résultats
        """)

if __name__ == "__main__":
    main()
