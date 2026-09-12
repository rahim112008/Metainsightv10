# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v10.2 — Édition ZOOTECHNIE & MORPHOMÉTRIE
# Version ULTRA-CORRIGÉE : 100% NumPy, aucune dépendance pyarrow
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import io
import os
import requests
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import spearmanr, kruskal, mannwhitneyu, f_oneway, pearsonr
from scipy.cluster.hierarchy import linkage, dendrogram

# ✅ FIX 1 : statsmodels importé au niveau global avec garde-fou
try:
    import statsmodels.api as sm
    from statsmodels.formula.api import ols
    HAS_STATS = True
except ImportError:
    HAS_STATS = False

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
#  LECTURE CSV — 100% NumPy purs (jamais pyarrow)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def read_any_csv_bytes(content: bytes, filename: str = ""):
    """Lit un CSV en forçant les types NumPy purs (jamais pyarrow)."""
    try:
        try:
            text = content.decode('utf-8-sig')
        except UnicodeDecodeError:
            text = content.decode('latin-1')

        first_line = text.split('\n')[0]
        if ';' in first_line:
            sep = ';'
        elif '\t' in first_line:
            sep = '\t'
        else:
            sep = ','

        df = pd.read_csv(
            io.StringIO(text),
            sep=sep,
            na_values=['', 'NA', 'N/A', 'n/a', '?', '-', 'null', 'NULL'],
            keep_default_na=True
        )
        df.columns = [str(c).strip() for c in df.columns]

        new_data = {}
        for col in df.columns:
            series = df[col]
            try:
                converted = pd.to_numeric(
                    series.astype(str).str.replace(',', '.', regex=False),
                    errors='raise'
                )
                new_data[col] = np.array(converted, dtype=np.float64)
            except (ValueError, TypeError):
                new_data[col] = np.array(series.astype(str), dtype=object)

        df_clean = pd.DataFrame(new_data, index=df.index)
        df_clean = df_clean.dropna(axis=1, how='all')

        if 'SEX' in df_clean.columns:
            def norm_sex(v):
                s = str(v).strip().upper()
                if s in ('', 'NAN', 'NONE'):
                    return np.nan
                if s.startswith('M'):
                    return 'M'
                if s.startswith('F'):
                    return 'F'
                return s
            df_clean['SEX'] = df_clean['SEX'].apply(norm_sex)

        if 'BREED' in df_clean.columns:
            df_clean['BREED'] = df_clean['BREED'].astype(str).str.strip()

        return df_clean
    except Exception as e:
        st.error(f"❌ Erreur de lecture : {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  IA
# ══════════════════════════════════════════════════════════════════════════════
def call_ai(prompt, provider, gemini_key=None, groq_key=None,
            openrouter_key=None, deepseek_key=None,
            gemini_model="gemini-2.0-flash",
            groq_model="llama-3.3-70b-versatile",
            openrouter_model="openrouter/auto",
            ollama_model="llama3"):
    try:
        if provider == "Gemini (GRATUIT)":
            if not gemini_key:
                return "🔑 Clé Gemini manquante."
            url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
                   f"{gemini_model}:generateContent?key={gemini_key}")
            payload = {"contents": [{"parts": [{"text": prompt}]}]}
            r = requests.post(url, json=payload, timeout=30)
            if r.status_code == 200:
                try:
                    return r.json()["candidates"][0]["content"]["parts"][0]["text"]
                except (KeyError, IndexError):
                    return f"⚠️ Réponse inattendue : {r.text[:200]}"
            return f"⚠️ Erreur {r.status_code} : {r.text[:200]}"

        elif provider == "Groq (gratuit)":
            if not groq_key:
                return "🔑 Clé Groq manquante."
            headers = {"Authorization": f"Bearer {groq_key}"}
            data = {"model": groq_model,
                    "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur {r.status_code} : {r.text[:200]}"

        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            if not openrouter_key:
                return "🔑 Clé OpenRouter manquante."
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "HTTP-Referer": "http://localhost",
                "X-Title": "MetaInsight"
            }
            data = {"model": openrouter_model,
                    "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur {r.status_code} : {r.text[:200]}"

        elif provider == "DeepSeek (gratuit)":
            if not deepseek_key:
                return "🔑 Clé DeepSeek manquante."
            headers = {"Authorization": f"Bearer {deepseek_key}"}
            data = {"model": "deepseek-chat",
                    "messages": [{"role": "user", "content": prompt}]}
            r = requests.post("https://api.deepseek.com/v1/chat/completions",
                              json=data, headers=headers, timeout=30)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
            return f"⚠️ Erreur {r.status_code} : {r.text[:200]}"

        elif provider == "Ollama (local)":
            try:
                r = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": ollama_model, "prompt": prompt, "stream": False},
                    timeout=60
                )
                if r.status_code == 200:
                    return r.json().get("response", "Réponse vide")
                return f"⚠️ Erreur {r.status_code}"
            except Exception as e:
                return f"❌ Ollama non joignable : {e}"

        return "⚠️ Fournisseur non reconnu."
    except requests.Timeout:
        return "⏱️ Timeout — le serveur IA n'a pas répondu à temps."
    except Exception as e:
        return f"❌ Erreur : {str(e)}"


# ══════════════════════════════════════════════════════════════════════════════
#  LABELS DES MESURES
# ══════════════════════════════════════════════════════════════════════════════
MORPHO_LABELS = {
    'HW': 'Hauteur au garrot (cm)',
    'HR': 'Hauteur à la croupe (cm)',
    'BL': 'Longueur du corps (cm)',
    'HL': 'Longueur de la tête (cm)',
    'HEW': 'Largeur de la tête (cm)',
    'ML': 'Longueur du museau (cm)',
    'HG': 'Tour de tête (cm)',
    'EL': "Longueur de l'oreille (cm)",
    'CG': 'Tour de poitrine (cm)',
    'WG': 'Tour de taille (cm)',
    'AG': "Tour d'abdomen (cm)",
    'NL': 'Longueur du cou (cm)',
    'LW': 'Largeur du poitrail (cm)',
}


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS D'ANALYSE (100% NumPy)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def compute_morphometric_indices(df):
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
    results = []
    if sex_col not in df.columns:
        return pd.DataFrame(results)

    for col in morpho_cols:
        males = df[df[sex_col] == 'M'][col].dropna().values.astype(np.float64)
        females = df[df[sex_col] == 'F'][col].dropna().values.astype(np.float64)
        if len(males) >= 2 and len(females) >= 2:
            try:
                stat, p = mannwhitneyu(males, females, alternative='two-sided')
                mean_f = float(np.mean(females))
                if mean_f == 0:
                    diff_pct = np.nan
                else:
                    diff_pct = (float(np.mean(males)) - mean_f) / mean_f * 100
                results.append({
                    'Mesure': col,
                    'Mâles (moy)': round(float(np.mean(males)), 2),
                    'Femelles (moy)': round(mean_f, 2),
                    'Différence (%)': round(diff_pct, 2) if np.isfinite(diff_pct) else np.nan,
                    'p-value': round(float(p), 4),
                    'Significatif': '✅' if p < 0.05 else '❌'
                })
            except Exception:
                continue
    return pd.DataFrame(results)


def analyze_breed_differences(df, morpho_cols, breed_col='BREED'):
    results = []
    if breed_col not in df.columns:
        return pd.DataFrame(results)

    for col in morpho_cols:
        groups = [
            df[df[breed_col] == b][col].dropna().values.astype(np.float64)
            for b in df[breed_col].unique()
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                results.append({
                    'Mesure': col,
                    'H (Kruskal)': round(float(stat), 2),
                    'p-value': round(float(p), 4),
                    'Significatif': '✅' if p < 0.05 else '❌'
                })
            except Exception:
                continue
    return pd.DataFrame(results)


def compute_allometric_regression(df, x_col, y_col):
    data = df[[x_col, y_col]].dropna()
    x_vals = data[x_col].values.astype(np.float64)
    y_vals = data[y_col].values.astype(np.float64)
    mask = (x_vals > 0) & (y_vals > 0) & np.isfinite(x_vals) & np.isfinite(y_vals)
    x_vals = x_vals[mask]
    y_vals = y_vals[mask]
    if len(x_vals) < 5:
        return None
    log_x = np.log(x_vals)
    log_y = np.log(y_vals)
    if np.std(log_x) == 0:
        return None
    coeffs = np.polyfit(log_x, log_y, 1)
    r = np.corrcoef(log_x, log_y)[0, 1]
    return {'slope': float(coeffs[0]), 'intercept': float(coeffs[1]),
            'r': float(r), 'n': int(len(x_vals))}


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
        st.markdown(
            '<span style="font-size:0.75rem;color:#7A8BA8;">Analyse morphométrique</span>',
            unsafe_allow_html=True
        )
        st.markdown("---")

        st.markdown("### 📂 Charger vos données")
        st.caption("Uploadez votre fichier `dogs.csv` ici :")

        uploaded_file = st.file_uploader(
            "Fichier CSV morphométrique",
            type=["csv", "tsv", "txt"],
            key="morpho_upload"
        )

        if uploaded_file is not None:
            df = read_any_csv_bytes(uploaded_file.getvalue(), uploaded_file.name)
            if df is not None:
                st.session_state.df = df
                st.success(f"✅ {len(df)} chiens × {len(df.columns)} colonnes")

        st.markdown("---")
        st.markdown("### 🤖 IA")
        provider = st.selectbox(
            "Fournisseur",
            ["Gemini (GRATUIT)", "Groq (gratuit)",
             "OpenRouter — Kimi K2 (gratuit)", "DeepSeek (gratuit)",
             "Ollama (local)"],
            key="ai_prov"
        )
        st.session_state.ai_provider = provider

        if provider == "Gemini (GRATUIT)":
            st.session_state.gemini_key = st.text_input(
                "Clé Gemini", type="password",
                value=st.session_state.get("gemini_key", ""), key="gk")
        elif provider == "Groq (gratuit)":
            st.session_state.groq_key = st.text_input(
                "Clé Groq", type="password",
                value=st.session_state.get("groq_key", ""), key="gqk")
        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            st.session_state.openrouter_key = st.text_input(
                "Clé OpenRouter", type="password",
                value=st.session_state.get("openrouter_key", ""), key="ork")
        elif provider == "DeepSeek (gratuit)":
            st.session_state.deepseek_key = st.text_input(
                "Clé DeepSeek", type="password",
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
                breeds_unique, counts = np.unique(
                    df['BREED'].values.astype(str), return_counts=True)
                breed_counts = pd.DataFrame({'Race': breeds_unique, 'Nombre': counts})
                breed_counts = breed_counts.sort_values('Nombre', ascending=False)
                fig = px.bar(breed_counts, x='Race', y='Nombre', color='Race',
                             template='plotly_dark',
                             title="Nombre de chiens par race")
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
            if not morpho_cols:
                st.warning("Aucune colonne morphométrique reconnue (HW, BL, CG…).")
            else:
                st.markdown("### 📈 Statistiques globales")
                st.dataframe(df[morpho_cols].describe().round(2),
                             use_container_width=True)

                if 'BREED' in df.columns:
                    st.markdown("### 🐕 Statistiques par race")
                    rows = []
                    for breed in df['BREED'].unique():
                        sub = df[df['BREED'] == breed]
                        row = {'Race': breed}
                        for col in morpho_cols:
                            vals = sub[col].dropna().values.astype(np.float64)
                            row[f'{col}_mean'] = (round(float(np.mean(vals)), 2)
                                                  if len(vals) > 0 else np.nan)
                            row[f'{col}_std'] = (round(float(np.std(vals)), 2)
                                                 if len(vals) > 0 else np.nan)
                        rows.append(row)
                    breed_stats = pd.DataFrame(rows).set_index('Race')
                    st.dataframe(breed_stats, use_container_width=True)

                if 'SEX' in df.columns:
                    st.markdown("### ♂♀ Statistiques par sexe")
                    rows = []
                    for sex in df['SEX'].dropna().unique():
                        sub = df[df['SEX'] == sex]
                        row = {'Sexe': sex}
                        for col in morpho_cols:
                            vals = sub[col].dropna().values.astype(np.float64)
                            row[f'{col}_mean'] = (round(float(np.mean(vals)), 2)
                                                  if len(vals) > 0 else np.nan)
                            row[f'{col}_std'] = (round(float(np.std(vals)), 2)
                                                 if len(vals) > 0 else np.nan)
                        rows.append(row)
                    sex_stats = pd.DataFrame(rows).set_index('Sexe')
                    st.dataframe(sex_stats, use_container_width=True)

                st.markdown("### 📦 Distribution")
                col1, col2 = st.columns(2)
                with col1:
                    sel = st.selectbox(
                        "Mesure", morpho_cols,
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="stat_sel")
                with col2:
                    grp_options = ['Aucun']
                    if 'BREED' in df.columns:
                        grp_options.append('BREED')
                    if 'SEX' in df.columns:
                        grp_options.append('SEX')
                    grp = st.selectbox("Grouper par", grp_options, key="stat_grp")

                if grp == 'Aucun':
                    fig = px.histogram(df, x=sel, nbins=30, template='plotly_dark',
                                       title=f"Distribution de {MORPHO_LABELS.get(sel, sel)}")
                else:
                    fig = px.box(df, x=grp, y=sel, color=grp, template='plotly_dark',
                                 title=f"{MORPHO_LABELS.get(sel, sel)} par {grp}")
                st.plotly_chart(fig, use_container_width=True,
                                key=f"stat_dist_{sel}_{grp}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 2 : Dimorphisme sexuel
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("## ♂♀ Dimorphisme sexuel")
        st.markdown(
            '<div class="ref-box">📚 Comparaison Mâles vs Femelles (Mann-Whitney)</div>',
            unsafe_allow_html=True)
        if df is None or 'SEX' not in df.columns:
            st.info("Chargez un fichier avec colonne SEX.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            dimor = analyze_sexual_dimorphism(df, morpho_cols)
            if dimor.empty:
                st.warning("Pas assez de données M/F pour comparer.")
            else:
                st.dataframe(
                    dimor.style.background_gradient(cmap='RdBu_r',
                                                    subset=['Différence (%)']),
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
                    prompt = (
                        "Expert zootechnie, analyse ce dimorphisme sexuel canin :\n"
                        f"{summary}\n"
                        "Interprète les différences et leur importance pour la sélection."
                    )
                    st.info(call_ai(
                        prompt, st.session_state.ai_provider,
                        gemini_key=st.session_state.get("gemini_key", ""),
                        groq_key=st.session_state.get("groq_key", ""),
                        openrouter_key=st.session_state.get("openrouter_key", ""),
                        deepseek_key=st.session_state.get("deepseek_key", "")))

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 3 : Différences raciales
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("## 🐕 Différences raciales")
        st.markdown(
            '<div class="ref-box">📚 Comparaison entre races (Kruskal-Wallis)</div>',
            unsafe_allow_html=True)
        if df is None or 'BREED' not in df.columns:
            st.info("Chargez un fichier avec colonne BREED.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            breed_diff = analyze_breed_differences(df, morpho_cols)
            if breed_diff.empty:
                st.warning("Pas assez de races/mesures pour comparer.")
            else:
                st.dataframe(
                    breed_diff.style.background_gradient(cmap='YlOrRd',
                                                         subset=['H (Kruskal)']),
                    use_container_width=True)

                st.markdown("### 📊 Profil morphologique par race (Radar)")
                breeds = list(df['BREED'].unique())
                means_dict = {}
                for breed in breeds:
                    sub = df[df['BREED'] == breed]
                    means_dict[breed] = [
                        float(np.mean(sub[c].dropna().values.astype(np.float64)))
                        if len(sub[c].dropna()) > 0 else 0.0
                        for c in morpho_cols
                    ]
                breed_means = pd.DataFrame(means_dict, index=morpho_cols).T

                mins = breed_means.min()
                maxs = breed_means.max()
                ranges = (maxs - mins).replace(0, 1)
                breed_norm = (breed_means - mins) / ranges

                fig_radar = go.Figure()
                for breed in breed_norm.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=breed_norm.loc[breed].values.astype(np.float64),
                        theta=[MORPHO_LABELS.get(c, c) for c in morpho_cols],
                        fill='toself', name=breed
                    ))
                fig_radar.update_layout(
                    template='plotly_dark',
                    title="Profil morphologique moyen par race",
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
                st.plotly_chart(fig_radar, use_container_width=True,
                                key="breed_radar")

                sel_measure = st.selectbox(
                    "Mesure", morpho_cols,
                    format_func=lambda x: MORPHO_LABELS.get(x, x),
                    key="breed_measure")
                fig_box = px.box(df, x='BREED', y=sel_measure, color='BREED',
                                 template='plotly_dark',
                                 title=f"{MORPHO_LABELS.get(sel_measure, sel_measure)} par race")
                st.plotly_chart(fig_box, use_container_width=True,
                                key=f"breed_box_{sel_measure}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 4 : Indices morphométriques
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[4]:
        st.markdown("## 📐 Indices morphométriques")
        st.markdown(
            '<div class="ref-box">📚 Indices : format, poitrine, céphalique, masse</div>',
            unsafe_allow_html=True)
        if df is None:
            st.info("Chargez un fichier.")
        else:
            df_indices = compute_morphometric_indices(df)
            index_cols = [c for c in df_indices.columns
                          if c.startswith('Indice') or c.startswith('Ratio')]

            if not index_cols:
                st.warning("Aucun indice calculable (vérifiez HW, BL, CG, HL, HEW…).")
            else:
                st.markdown("### 📊 Indices calculés")
                st.dataframe(df_indices[index_cols].describe().round(2),
                             use_container_width=True)

                if 'BREED' in df.columns:
                    st.markdown("### 📊 Distribution par race")
                    idx_melted = df_indices.melt(
                        id_vars=['BREED'], value_vars=index_cols,
                        var_name='Indice', value_name='Valeur')
                    fig = px.box(idx_melted, x='Indice', y='Valeur', color='BREED',
                                 template='plotly_dark')
                    fig.update_layout(xaxis_tickangle=-45)
                    st.plotly_chart(fig, use_container_width=True, key="indices_box")

                st.markdown("### 🔗 Corrélations entre indices")
                corr = df_indices[index_cols].corr()
                fig_corr = px.imshow(corr, color_continuous_scale='RdBu_r',
                                     zmin=-1, zmax=1, text_auto=True,
                                     template='plotly_dark')
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
                X = np.array(df[morpho_cols].values, dtype=np.float64)
                col_means = np.nanmean(X, axis=0)
                col_means = np.where(np.isnan(col_means), 0.0, col_means)
                X = np.where(np.isnan(X), col_means, X)

                std = X.std(axis=0)
                valid_cols = std > 0
                if valid_cols.sum() < 2:
                    st.warning("Trop peu de variabilité dans les données.")
                else:
                    X = X[:, valid_cols]
                    cols_used = [c for c, v in zip(morpho_cols, valid_cols) if v]
                    X_scaled = StandardScaler().fit_transform(X)

                    st.markdown("### 📉 PCA")
                    n_pca = min(5, X_scaled.shape[1], X_scaled.shape[0])
                    pca = PCA(n_components=n_pca)
                    coords = pca.fit_transform(X_scaled)
                    pca_df = pd.DataFrame(coords[:, :2], columns=['PC1', 'PC2'])
                    if 'BREED' in df.columns:
                        pca_df['BREED'] = df['BREED'].values
                    if 'SEX' in df.columns:
                        pca_df['SEX'] = df['SEX'].values

                    fig_pca = px.scatter(
                        pca_df, x='PC1', y='PC2',
                        color='BREED' if 'BREED' in pca_df.columns else None,
                        symbol='SEX' if 'SEX' in pca_df.columns else None,
                        template='plotly_dark',
                        title=(f"PCA — {pca.explained_variance_ratio_[0]:.1%}"
                               f" + {pca.explained_variance_ratio_[1]:.1%}"))
                    st.plotly_chart(fig_pca, use_container_width=True, key="pca_morpho")

                    st.markdown("### 🎯 Contribution des variables (loadings)")
                    loadings = pd.DataFrame(
                        pca.components_[:2].T,
                        columns=['PC1', 'PC2'],
                        index=[MORPHO_LABELS.get(c, c) for c in cols_used])
                    st.dataframe(loadings.round(3), use_container_width=True)

                    if 'BREED' in df.columns and df['BREED'].nunique() >= 2:
                        st.markdown("### 🎯 LDA (séparation des races)")
                        le = LabelEncoder()
                        y = le.fit_transform(df['BREED'].values)
                        n_comp = min(2, df['BREED'].nunique() - 1)
                        if n_comp < 1:
                            st.warning("LDA impossible : une seule classe.")
                        else:
                            try:
                                lda = LDA(n_components=n_comp)
                                coords_lda = lda.fit_transform(X_scaled, y)
                                n_plot = min(2, coords_lda.shape[1])
                                lda_df = pd.DataFrame(
                                    coords_lda[:, :n_plot],
                                    columns=['LD1', 'LD2'][:n_plot])
                                lda_df['BREED'] = df['BREED'].values
                                if n_plot == 2:
                                    fig_lda = px.scatter(
                                        lda_df, x='LD1', y='LD2', color='BREED',
                                        template='plotly_dark',
                                        title="LDA — Séparation des races")
                                else:
                                    fig_lda = px.scatter(
                                        lda_df, x='LD1', y=['0'] * len(lda_df),
                                        color='BREED', template='plotly_dark',
                                        title="LDA — Séparation des races (1D)")
                                st.plotly_chart(fig_lda, use_container_width=True,
                                                key="lda_breed")
                            except Exception as e:
                                st.warning(f"LDA impossible : {e}")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 6 : Allométrie
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[6]:
        st.markdown("## 📈 Analyse allométrique")
        st.markdown(
            '<div class="ref-box">📚 Régression log-log : log(y) = a × log(x) + b</div>',
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
                    x_col = st.selectbox(
                        "Mesure X", morpho_cols,
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="allo_x")
                with col2:
                    y_col = st.selectbox(
                        "Mesure Y", morpho_cols,
                        index=min(1, len(morpho_cols) - 1),
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="allo_y")

                if x_col != y_col:
                    res = compute_allometric_regression(df, x_col, y_col)
                    if res:
                        col1, col2, col3 = st.columns(3)
                        col1.metric("Pente (a)", f"{res['slope']:.3f}")
                        col2.metric("Corrélation (r)", f"{res['r']:.3f}")
                        col3.metric("n", res['n'])

                        if res['slope'] > 1.05:
                            st.info("🔼 Allométrie positive (pente > 1)")
                        elif res['slope'] < 0.95:
                            st.info("🔽 Allométrie négative (pente < 1)")
                        else:
                            st.success("➡️ Isométrie (pente ≈ 1)")

                        fig = px.scatter(
                            df, x=x_col, y=y_col,
                            color='BREED' if 'BREED' in df.columns else None,
                            trendline='ols', template='plotly_dark',
                            log_x=True, log_y=True,
                            title=(f"{MORPHO_LABELS.get(y_col, y_col)} vs "
                                   f"{MORPHO_LABELS.get(x_col, x_col)}"))
                        st.plotly_chart(fig, use_container_width=True,
                                        key=f"allo_{x_col}_{y_col}")
                    else:
                        st.warning("Pas assez de points valides (>= 5 requis, valeurs > 0).")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 7 : Corrélations
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[7]:
        st.markdown("## 🔗 Corrélations")
        if df is None:
            st.info("Chargez un fichier.")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            if len(morpho_cols) < 2:
                st.warning("Il faut au moins 2 mesures.")
            else:
                st.markdown("### 🔥 Matrice de corrélation")
                corr = df[morpho_cols].corr()
                fig = px.imshow(corr, color_continuous_scale='RdBu_r',
                                zmin=-1, zmax=1, text_auto='.2f',
                                template='plotly_dark')
                st.plotly_chart(fig, use_container_width=True, key="corr_morpho")

                st.markdown("### 🌲 Clustering hiérarchique")
                corr_vals = corr.values.copy()
                corr_vals = np.nan_to_num(corr_vals, nan=0.0)
                np.fill_diagonal(corr_vals, 1.0)
                dist_matrix = 1 - np.abs(corr_vals)
                np.fill_diagonal(dist_matrix, 0.0)

                if len(morpho_cols) >= 2:
                    Z = linkage(dist_matrix[np.triu_indices_from(dist_matrix, k=1)],
                                method='average')
                    fig_dend, ax = plt.subplots(figsize=(10, 5))
                    fig_dend.patch.set_facecolor('#0A0E1A')
                    ax.set_facecolor('#0A0E1A')
                    dendrogram(
                        Z,
                        labels=[MORPHO_LABELS.get(c, c) for c in morpho_cols],
                        ax=ax, color_threshold=0)
                    ax.set_title("Dendrogramme des mesures (distance = 1 - |r|)",
                                 color='#E8EDF5')
                    ax.tick_params(colors='#A8B5C8')
                    for spine in ax.spines.values():
                        spine.set_color('#2A3550')
                    plt.xticks(rotation=45, ha='right')
                    plt.tight_layout()
                    st.pyplot(fig_dend)
                    plt.close(fig_dend)

                st.markdown("### 🎯 Corrélation ciblée")
                col1, col2, col3 = st.columns(3)
                with col1:
                    c1 = st.selectbox(
                        "Mesure 1", morpho_cols,
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="corr1")
                with col2:
                    c2 = st.selectbox(
                        "Mesure 2", morpho_cols,
                        index=min(1, len(morpho_cols) - 1),
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="corr2")
                with col3:
                    method = st.selectbox("Méthode", ['pearson', 'spearman'],
                                          key="corr_meth")

                if c1 != c2:
                    sub = df[[c1, c2]].dropna()
                    x_vals = sub[c1].values.astype(np.float64)
                    y_vals = sub[c2].values.astype(np.float64)
                    if len(x_vals) >= 3:
                        if method == 'pearson':
                            r, p = pearsonr(x_vals, y_vals)
                        else:
                            r, p = spearmanr(x_vals, y_vals)
                        st.metric(f"Corrélation {method}",
                                  f"r = {r:.3f}", delta=f"p = {p:.4f}")
                        fig_sc = px.scatter(
                            df, x=c1, y=c2,
                            color='BREED' if 'BREED' in df.columns else None,
                            trendline='ols', template='plotly_dark')
                        st.plotly_chart(fig_sc, use_container_width=True,
                                        key=f"scatter_{c1}_{c2}")
                    else:
                        st.warning("Pas assez de données appariées.")

    # ══════════════════════════════════════════════════════════════════════════
    # Onglet 8 : P = G + E + G×E + ε
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[8]:
        st.markdown("## ⚖️ Décomposition P = G + E + G×E + ε")
        st.markdown(
            '<div class="ref-box">📚 Modèle linéaire avec interaction</div>',
            unsafe_allow_html=True)
        if df is None:
            st.info("Chargez un fichier.")
        elif not HAS_STATS:
            st.error("📦 `statsmodels` n'est pas installé. "
                     "Installez-le avec : pip install statsmodels")
        else:
            morpho_cols = [c for c in df.columns if c in MORPHO_LABELS]
            if len(morpho_cols) < 3:
                st.warning("Il faut au moins 3 mesures.")
            else:
                st.markdown("""
**Interprétation :**
- **P** : Mesure à expliquer (ex : CG tour de poitrine)
- **G** : Effet génétique (ex : HW hauteur)
- **E** : Effet environnemental (ex : BL longueur)
- **G×E** : Interaction
""")

                col1, col2, col3 = st.columns(3)
                with col1:
                    p_col = st.selectbox(
                        "P (Phénotype)", morpho_cols,
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="p_dec")
                with col2:
                    g_col = st.selectbox(
                        "G (Génétique)", morpho_cols,
                        index=min(1, len(morpho_cols) - 1),
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="g_dec")
                with col3:
                    e_col = st.selectbox(
                        "E (Environnement)", morpho_cols,
                        index=min(2, len(morpho_cols) - 1),
                        format_func=lambda x: MORPHO_LABELS.get(x, x),
                        key="e_dec")

                if len({p_col, g_col, e_col}) == 3:
                    if st.button("🚀 Décomposer", key="decomp_btn"):
                        try:
                            data = df[[p_col, g_col, e_col]].dropna().astype(np.float64)
                            data.columns = ['P', 'G', 'E']
                            if len(data) < 10:
                                st.warning("Il faut au moins 10 observations complètes.")
                            else:
                                model = ols('P ~ G + E + G:E', data=data).fit()
                                anova_table = sm.stats.anova_lm(model, typ=2)
                                ss_total = anova_table['sum_sq'].sum()
                                ss_G = (anova_table.loc['G', 'sum_sq']
                                        if 'G' in anova_table.index else 0)
                                ss_E = (anova_table.loc['E', 'sum_sq']
                                        if 'E' in anova_table.index else 0)
                                ss_GE = (anova_table.loc['G:E', 'sum_sq']
                                         if 'G:E' in anova_table.index else 0)
                                ss_resid = (anova_table.loc['Residual', 'sum_sq']
                                            if 'Residual' in anova_table.index else 0)

                                col_a, col_b, col_c, col_d = st.columns(4)
                                col_a.metric("G (%)", f"{ss_G/ss_total*100:.1f}%")
                                col_b.metric("E (%)", f"{ss_E/ss_total*100:.1f}%")
                                col_c.metric("G×E (%)", f"{ss_GE/ss_total*100:.1f}%")
                                col_d.metric("ε (%)", f"{ss_resid/ss_total*100:.1f}%")
                                st.markdown(f"**R² = {model.rsquared:.3f}**")

                                prop_df = pd.DataFrame({
                                    'Composante': ['G', 'E', 'G×E', 'ε'],
                                    'Proportion (%)': [
                                        ss_G/ss_total*100, ss_E/ss_total*100,
                                        ss_GE/ss_total*100, ss_resid/ss_total*100]
                                })
                                fig = px.bar(prop_df, x='Composante',
                                             y='Proportion (%)',
                                             color='Composante',
                                             template='plotly_dark')
                                st.plotly_chart(fig, use_container_width=True,
                                                key="decomp_bar_morpho")

                                with st.expander("📄 Table ANOVA complète"):
                                    st.dataframe(anova_table.round(4),
                                                 use_container_width=True)
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
            prompt_user = st.text_area(
                "Votre question",
                "Analyse les différences morphologiques entre races et propose "
                "des recommandations pour la sélection.",
                height=120, key="ai_prompt_morpho")

            if st.button("🤖 Interroger l'IA", key="ai_btn_morpho"):
                race_summary = ""
                if 'BREED' in df.columns:
                    for breed in df['BREED'].unique():
                        sub = df[df['BREED'] == breed]
                        means = []
                        for c in morpho_cols:
                            vals = sub[c].dropna().values.astype(np.float64)
                            if len(vals) > 0:
                                means.append(f"{c}={np.mean(vals):.2f}")
                            else:
                                means.append(f"{c}=N/A")
                        race_summary += f"{breed}: {', '.join(means)}\n"

                breed_dist = ("N/A" if 'BREED' not in df.columns
                              else dict(zip(*np.unique(
                                  df['BREED'].values.astype(str),
                                  return_counts=True))))
                sex_dist = ("N/A" if 'SEX' not in df.columns
                            else dict(zip(*np.unique(
                                df['SEX'].dropna().values.astype(str),
                                return_counts=True))))

                context = f"""
Données morphométriques canines : {len(df)} chiens
Races : {breed_dist}
Sexes : {sex_dist}
Moyennes par race :
{race_summary}
"""
                full_prompt = f"{context}\n\nQuestion : {prompt_user}"
                with st.spinner("Analyse IA..."):
                    res = call_ai(
                        full_prompt, st.session_state.ai_provider,
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

        st.markdown("### 🐕 Mesures morphométriques reconnues")
        mesures_doc = pd.DataFrame({
            'Code': ['HW', 'HR', 'BL', 'HL', 'HEW', 'ML', 'HG', 'EL',
                     'CG', 'WG', 'AG', 'NL', 'LW'],
            'Signification': [
                'Hauteur au garrot (cm)',
                'Hauteur à la croupe (cm)',
                'Longueur du corps (cm)',
                'Longueur de la tête (cm)',
                'Largeur de la tête (cm)',
                'Longueur du museau (cm)',
                'Tour de tête (cm)',
                "Longueur de l'oreille (cm)",
                'Tour de poitrine (cm)',
                'Tour de taille (cm)',
                "Tour d'abdomen (cm)",
                'Longueur du cou (cm)',
                'Largeur du poitrail (cm)'
            ]
        })
        st.dataframe(mesures_doc, use_container_width=True, hide_index=True)

        st.markdown("### 📊 Analyses disponibles")
        st.markdown("""
- Statistiques descriptives
- Dimorphisme sexuel (Mann-Whitney)
- Différences raciales (Kruskal-Wallis + Radar)
- Indices morphométriques
- PCA & LDA
- Allométrie (régression log-log)
- Corrélations + clustering
- Décomposition P = G + E + G×E + ε
- IA interprétative
""")

        st.markdown("### 💡 Utilisation")
        st.markdown("""
1. Uploadez votre fichier `dogs.csv` dans la barre latérale
2. Naviguez dans les onglets
3. Utilisez l'IA pour interpréter les résultats
""")

        st.markdown("### 📦 Dépendances")
        st.code(
            "pip install streamlit pandas numpy plotly matplotlib seaborn "
            "scikit-learn scipy statsmodels requests",
            language="bash"
        )


if __name__ == "__main__":
    main()
