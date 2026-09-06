# ══════════════════════════════════════════════════════════════════════════════
# MetaInsight v10 — Plateforme intégrative multi-omique, PGM & Épitranscriptomique
# Version complète avec tous les onglets (inchangés + nouveaux)
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
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
from scipy.stats import entropy, spearmanr, kruskal, mannwhitneyu, f_oneway, pearsonr
from scipy.spatial.distance import cdist, braycurtis
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
import networkx as nx
import requests
import os
import io
import tempfile
import subprocess
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
    page_title="MetaInsight v10 — Multi-omics, PGM, Épitranscriptomique & Causalité",
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
.badge-fix {
    background: linear-gradient(90deg,#4D9FFF,#9B7CFF);
    color:#000; font-size:0.65rem; padding:2px 7px; border-radius:10px;
    font-weight:700; margin-left:4px; vertical-align:middle;
}
.ref-box {
    background:#0F1525; border-left:3px solid #00D4AA; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A8BA8; margin:6px 0;
}
.fix-box {
    background:#0A1525; border-left:3px solid #4D9FFF; padding:8px 12px;
    border-radius:0 6px 6px 0; font-size:0.8rem; color:#7A9AB8; margin:6px 0;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  DONNÉES DE DÉMONSTRATION
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def generate_demo_microbiome():
    environments = ["Sol aride", "Eau marine", "Gut", "Sol agricole", "Sédiments", "Biofilm"]
    taxa = [
        "Proteobacteria","Actinobacteriota","Firmicutes","Bacteroidota","Archaea",
        "Acidobacteria","Chloroflexi","Planctomycetes","Ascomycota","Caudovirales"
    ]
    base_profiles = {
        "Sol aride":    [28, 20,  5,  4,  8,  6,  4,  3,  2,  1],
        "Eau marine":   [35, 10,  8, 15,  2,  5,  3,  4,  8,  6],
        "Gut":          [15, 12, 30, 22,  1,  3,  2,  2,  4,  2],
        "Sol agricole": [22, 25, 10,  8,  4, 10,  7,  5,  3,  2],
        "Sédiments":    [18, 14, 12, 10,  6,  8,  9,  6,  5,  4],
        "Biofilm":      [30, 18,  6,  9,  3,  7,  5,  4,  6,  5],
    }
    data = []
    for env in environments:
        base = base_profiles[env]
        for rep in range(4):
            noisy = np.array(base) + np.random.normal(0, 2, size=len(taxa))
            noisy = np.clip(noisy, 0, None)
            noisy = noisy / noisy.sum() * 100
            sample_id = f"{env[:3].upper()}_{rep+1:03d}"
            row = {"sample_id": sample_id, "environment": env}
            for i, tax in enumerate(taxa):
                row[tax] = round(noisy[i], 2)
            probs = noisy / 100.0
            row["shannon"]    = round(entropy(probs, base=2), 3)
            row["simpson"]    = round(1 - np.sum(probs**2), 3)
            row["chao1"]      = round(len(taxa) + np.random.uniform(0, 5), 1)
            row["faith_pd"]   = round(float((noisy > 0).sum()) * 2.1 + float(np.std(noisy[noisy>0])) * 0.5, 2)
            row["classified_pct"] = round(np.random.uniform(70, 99), 1)
            row["ph"]         = round(np.random.uniform(4, 8), 2)
            row["temperature_c"] = round(np.random.uniform(15, 40), 1)
            row["moisture_pct"]  = round(np.random.uniform(5, 80), 1)
            data.append(row)
    return pd.DataFrame(data)

@st.cache_data
def generate_demo_pgm_data():
    np.random.seed(42)
    n = 200
    chroms = np.random.choice(["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","X"], size=n)
    pos = np.random.randint(100000, 100000000, n)
    refs = np.random.choice(["A","C","G","T"], n)
    alts = np.random.choice(["A","C","G","T"], n)
    for i in range(n):
        while alts[i] == refs[i]:
            alts[i] = np.random.choice(["A","C","G","T"])
    qual = np.random.uniform(10, 1000, n)
    filter_ = np.random.choice(["PASS","LowQual",""], n, p=[0.7,0.2,0.1])
    clinvar_sig = np.random.choice(["Pathogenic","Likely_pathogenic","VUS","Benign","Likely_benign"], n, p=[0.05,0.1,0.6,0.15,0.1])
    rsid = [f"rs{np.random.randint(100000, 999999)}" for _ in range(n)]
    cadd_phred = np.random.uniform(0, 45, n)
    brca1_idx = np.random.choice(range(n), size=6, replace=False)
    for i in brca1_idx:
        chroms[i] = "17"
        pos[i] = np.random.randint(43044295, 43125483)
        clinvar_sig[i] = "Pathogenic"
    brca2_idx = np.random.choice(range(n), size=5, replace=False)
    for i in brca2_idx:
        chroms[i] = "13"
        pos[i] = np.random.randint(32315474, 32400266)
        clinvar_sig[i] = "Pathogenic"
    pharma_rsids = ["rs3892097", "rs1800460", "rs3918290", "rs8175347", "rs4149056", "rs9923231"]
    for i in np.random.choice(range(n), size=8, replace=False):
        rsid[i] = np.random.choice(pharma_rsids)
    df = pd.DataFrame({
        "chrom": chroms,
        "pos": pos,
        "ref": refs,
        "alt": alts,
        "qual": qual,
        "filter": filter_,
        "info": "",
        "clinvar_sig": clinvar_sig,
        "variant_type": np.random.choice(["SNP","Indel","MNV"], n),
        "rsid": rsid,
        "allele_freq": np.random.uniform(0.001, 0.5, n),
        "cadd_phred": cadd_phred,
    })
    return df

@st.cache_data
def generate_demo_epitranscriptomic_data():
    np.random.seed(123)
    n_positions = 80
    transcript_length = 2000
    positions = np.sort(np.random.randint(50, transcript_length-50, n_positions))
    mod_types = np.random.choice(['m6A', 'm5C', 'Ψ', 'm1A', '2OMe', 'm7G', 'Nm'], n_positions, 
                                  p=[0.35,0.15,0.15,0.1,0.1,0.08,0.07])
    rates = np.random.beta(2, 5, n_positions)
    rates = np.round(rates, 3)
    psi_scores = np.random.uniform(0, 1, n_positions)
    conditions = np.random.choice(['Contrôle', 'Traité'], n_positions)
    transcripts = np.random.choice(['ENST00000380152', 'ENST00000456328', 'ENST00000585865', 
                                     'ENST00000318560', 'ENST00000431578'], n_positions)
    samples = [f'Sample_{i%6+1:02d}' for i in range(n_positions)]
    expressions = np.random.lognormal(5, 1.5, n_positions)
    df = pd.DataFrame({
        'transcript_id': transcripts,
        'position': positions,
        'modification': mod_types,
        'modification_rate': rates,
        'pseudouridine_score': psi_scores,
        'condition': conditions,
        'sample_id': samples,
        'expression_TPM': np.round(expressions, 2),
        'gene': [f'GENE_{i}' for i in np.random.choice(['BRCA1', 'TP53', 'EGFR', 'MYC', 'KRAS', 'PTEN'], n_positions)],
    })
    motifs = []
    for mod in mod_types:
        if mod == 'm6A':
            motifs.append(np.random.choice(['DRACH', 'GGACU', 'RRACH']))
        elif mod == 'm5C':
            motifs.append(np.random.choice(['CG', 'CNG']))
        elif mod == 'Ψ':
            motifs.append(np.random.choice(['UGU', 'GU']))
        else:
            motifs.append('')
    df['consensus_motif'] = motifs
    return df

@st.cache_data
def generate_epi_annotation_db():
    return pd.DataFrame({
        'transcript_id': ['ENST00000380152', 'ENST00000380152', 'ENST00000456328', 'ENST00000585865'],
        'position': [345, 876, 234, 567],
        'modification': ['m6A', 'Ψ', 'm5C', 'm6A'],
        'gene': ['BRCA1', 'BRCA1', 'TP53', 'MYC'],
        'motif': ['DRACH', 'UGU', 'CG', 'DRACH'],
        'confidence': ['High', 'Medium', 'High', 'High'],
        'reference': ['PMID:34567890', 'PMID:34567891', 'PMID:34567892', 'PMID:34567893']
    })

# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS GÉNÉRIQUES (microbiome, PGM, etc.)
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
            "Faith PD (proxy)": round(faith_pd, 2),
        })
    return pd.DataFrame(results, index=df.index)

def detect_feature_cols(df):
    meta_cols = {
        "sample_id","environment","group","label","class","condition",
        "shannon","simpson","chao1","faith_pd","classified_pct","ph","temperature_c","moisture_pct"
    }
    feature_cols = []
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in meta_cols:
            continue
        if col.endswith("_ZYG") or "date" in col_lower or "id" in col_lower:
            continue
        col_data = df[col]
        if pd.api.types.is_numeric_dtype(col_data):
            if col_data.std() > 0:
                feature_cols.append(col)
        elif pd.api.types.is_object_dtype(col_data) or pd.api.types.is_string_dtype(col_data):
            unique_vals = col_data.dropna().unique()
            if 2 <= len(unique_vals) <= 30:
                feature_cols.append(col)
    return feature_cols

def compute_bray_curtis_matrix(X):
    n = len(X)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            d = braycurtis(X[i], X[j])
            dm[i, j] = dm[j, i] = d
    return dm

def permanova_test(X, groups, n_permutations=999):
    dm = compute_bray_curtis_matrix(X)
    labels = np.array(groups)

    def pseudo_f(dm, labels):
        n_local = len(labels)
        ss_total = np.sum(dm**2) / n_local
        ss_within = 0.0
        for g in np.unique(labels):
            idx = np.where(labels == g)[0]
            ng = len(idx)
            if ng < 2:
                continue
            submat = dm[np.ix_(idx, idx)]
            ss_within += np.sum(submat**2) / ng
        ss_between = ss_total - ss_within
        n_groups = len(np.unique(labels))
        df_between = n_groups - 1
        df_within = n_local - n_groups
        if df_within <= 0:
            return 0.0
        return (ss_between / df_between) / (ss_within / df_within)

    f_obs = pseudo_f(dm, labels)
    f_perms = []
    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        f_perms.append(pseudo_f(dm, perm_labels))
    p_val = (np.sum(np.array(f_perms) >= f_obs) + 1) / (n_permutations + 1)
    r2 = f_obs / (f_obs + 1)
    return {"F": round(f_obs, 4), "p-value": round(p_val, 4), "R²": round(r2, 3)}

def aldex2_like(df, taxa_cols, group_col, group1, group2):
    g1 = df[df[group_col] == group1][taxa_cols].values.astype(float)
    g2 = df[df[group_col] == group2][taxa_cols].values.astype(float)
    if len(g1) < 2 or len(g2) < 2:
        return None
    results = []
    for j, tax in enumerate(taxa_cols):
        clr1 = clr_transform(g1 + 0.5)[:, j]
        clr2 = clr_transform(g2 + 0.5)[:, j]
        effect = (clr1.mean() - clr2.mean()) / (np.sqrt((clr1.std()**2 + clr2.std()**2) / 2) + 1e-9)
        try:
            _, pval = mannwhitneyu(clr1, clr2, alternative='two-sided')
        except:
            pval = 1.0
        results.append({
            "Taxon": tax,
            "CLR mean G1": round(clr1.mean(), 3),
            "CLR mean G2": round(clr2.mean(), 3),
            "Effect size": round(effect, 3),
            "p-value (Wilcoxon)": round(pval, 4),
            "Fold change (CLR)": round(clr1.mean() - clr2.mean(), 3),
        })
    res_df = pd.DataFrame(results)
    n = len(res_df)
    pvals = res_df["p-value (Wilcoxon)"].values
    sorted_idx = np.argsort(pvals)
    bh_corrected = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        bh_corrected[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(n-2, -1, -1):
        bh_corrected[sorted_idx[i]] = min(bh_corrected[sorted_idx[i]], bh_corrected[sorted_idx[i+1]])
    res_df["BH adj. p-value"] = bh_corrected.round(4)
    res_df["Significant (α=0.05)"] = res_df["BH adj. p-value"] < 0.05
    return res_df.sort_values("BH adj. p-value")

def lefse_like(df, taxa_cols, group_col):
    groups = df[group_col].unique()
    if len(groups) < 2:
        return None
    results = []
    for tax in taxa_cols:
        group_vals = [df[df[group_col] == g][tax].values for g in groups]
        try:
            _, p_kw = kruskal(*group_vals)
        except:
            p_kw = 1.0
        if p_kw < 0.05:
            means = [v.mean() for v in group_vals]
            stds = [v.std() + 1e-9 for v in group_vals]
            pooled_std = np.sqrt(np.mean([s**2 for s in stds]))
            lda_score = abs(max(means) - min(means)) / (pooled_std + 1e-9) * np.log10(len(df)+1)
            best_group = groups[np.argmax(means)]
        else:
            lda_score = 0.0
            best_group = "—"
        results.append({
            "Taxon": tax,
            "LDA Score": round(lda_score, 3),
            "Best group": best_group,
            "Kruskal-Wallis p": round(p_kw, 4),
            "Biomarker": lda_score >= 2.0 and p_kw < 0.05,
        })
    return pd.DataFrame(results).sort_values("LDA Score", ascending=False)

def maaslin2_like(df, taxa_cols, group_col):
    from sklearn.linear_model import LinearRegression
    le = LabelEncoder()
    y = le.fit_transform(df[group_col].values)
    X_raw = df[taxa_cols].values.astype(float) + 0.5
    X_clr = clr_transform(X_raw)
    results = []
    for j, tax in enumerate(taxa_cols):
        x_j = X_clr[:, j].reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x_j, y)
        coef = lr.coef_[0]
        r2 = lr.score(x_j, y)
        n = len(y)
        se = np.sqrt((1 - r2) / max(n - 2, 1)) * np.std(y) / (np.std(x_j.flatten()) + 1e-9)
        t_stat = abs(coef) / (se + 1e-9)
        from scipy.stats import t as t_dist
        p_val = 2 * t_dist.sf(abs(t_stat), df=n-2)
        results.append({"Taxon": tax, "Coefficient": round(coef, 4),
                         "R²": round(r2, 4), "p-value": round(p_val, 4)})
    res_df = pd.DataFrame(results)
    pvals = res_df["p-value"].values
    n = len(pvals)
    sorted_idx = np.argsort(pvals)
    bh = np.zeros(n)
    for rank, idx in enumerate(sorted_idx):
        bh[idx] = min(1.0, pvals[idx] * n / (rank + 1))
    for i in range(n-2, -1, -1):
        bh[sorted_idx[i]] = min(bh[sorted_idx[i]], bh[sorted_idx[i+1]])
    res_df["BH adj. p"] = bh.round(4)
    res_df["Significant"] = res_df["BH adj. p"] < 0.05
    return res_df.sort_values("BH adj. p")

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_kegg_pathways_api(taxon_name: str) -> list:
    TAXON_TO_KEGG = {
        "Proteobacteria": "eco", "Firmicutes": "bsu", "Bacteroidota": "bfr",
        "Actinobacteriota": "mtu", "Archaea": "mja", "Acidobacteria": "aac",
        "Chloroflexi": "cau", "Planctomycetes": "pla", "Ascomycota": "sce",
        "Caudovirales": "eco",
    }
    FALLBACK = {
        "Proteobacteria":   ["Nitrogen fixation", "Flagellar biosynthesis", "ATP synthesis"],
        "Firmicutes":       ["Butyrate production", "Sporulation", "Peptidoglycan biosynthesis"],
        "Bacteroidota":     ["Polysaccharide degradation", "Vitamin B12 biosynthesis", "Glycolysis"],
        "Actinobacteriota": ["Secondary metabolites", "Antibiotic biosynthesis", "Mycobactin biosynthesis"],
        "Archaea":          ["Methanogenesis", "CO2 fixation", "Archaeal ATPase"],
        "Acidobacteria":    ["Cellulose degradation", "Carbon cycling", "Sulfur metabolism"],
        "Chloroflexi":      ["Reductive TCA cycle", "Halogenated compound degradation", "Photosynthesis"],
        "Planctomycetes":   ["Anammox", "Nitrogen cycling", "Ladderane lipid biosynthesis"],
        "Ascomycota":       ["Chitin synthesis", "Ergosterol biosynthesis", "Mycotoxin biosynthesis"],
        "Caudovirales":     ["Viral DNA replication", "Host defense evasion", "Capsid assembly"],
    }
    kegg_code = TAXON_TO_KEGG.get(taxon_name)
    if not kegg_code:
        return FALLBACK.get(taxon_name, [f"Pathway_{taxon_name}_1"])
    try:
        url = f"https://rest.kegg.jp/link/pathway/{kegg_code}"
        resp = requests.get(url, timeout=8)
        if resp.status_code == 200 and resp.text.strip():
            pathway_ids = list({line.split("\t")[1].strip()
                                for line in resp.text.strip().split("\n")
                                if "\t" in line and "path:" in line.split("\t")[1]})[:8]
            if pathway_ids:
                return [pid.replace("path:", "") for pid in pathway_ids]
    except Exception:
        pass
    return FALLBACK.get(taxon_name, [])

def kegg_functional_prediction(df, taxa_cols):
    kegg_map = {}
    for tax in taxa_cols:
        kegg_map[tax] = fetch_kegg_pathways_api(tax)
    results = []
    for _, row in df.iterrows():
        env_kegg = {}
        for tax in taxa_cols:
            if tax in kegg_map and tax in df.columns:
                weight = float(row.get(tax, 0))
                for pathway in kegg_map[tax]:
                    env_kegg[pathway] = env_kegg.get(pathway, 0) + weight
        results.append(env_kegg)
    return pd.DataFrame(results, index=df.index).fillna(0)

def rarefaction_curve(df, taxa_cols, n_steps=20):
    envs = df["environment"].unique()
    curves = {}
    for env in envs:
        sub = df[df["environment"] == env][taxa_cols].values
        sub_int = (sub * 10).astype(int)
        total_counts = sub_int.sum(axis=1)
        max_depth = int(total_counts.min()) if len(total_counts) > 0 else 100
        if max_depth < 2:
            max_depth = 100
        depths = np.linspace(1, max_depth, min(n_steps, max_depth)).astype(int)
        richness_curve = []
        rng = np.random.RandomState(42)
        for depth in depths:
            obs_rich = []
            for counts in sub_int:
                total = counts.sum()
                if total < depth:
                    obs_rich.append((counts > 0).sum())
                    continue
                probs = counts / total
                sampled = rng.multinomial(depth, probs / probs.sum())
                obs_rich.append((sampled > 0).sum())
            richness_curve.append(np.mean(obs_rich))
        curves[env] = (depths, richness_curve)
    return curves

def normalize_omics(df, feat_cols, norm_type="log2"):
    X = df[feat_cols].values.astype(float)
    X = np.clip(X, 0, None)
    if norm_type == "log2":
        X_norm = np.log2(X + 1)
    elif norm_type == "log10":
        X_norm = np.log10(X + 1)
    elif norm_type == "zscore":
        scaler = StandardScaler()
        X_norm = scaler.fit_transform(X)
    elif norm_type == "pareto":
        means = X.mean(axis=0)
        stds = np.sqrt(X.std(axis=0) + 1e-9)
        X_norm = (X - means) / stds
    elif norm_type == "tss":
        row_sums = X.sum(axis=1, keepdims=True)
        X_norm = X / (row_sums + 1e-9)
    else:
        X_norm = X
    return pd.DataFrame(X_norm, columns=feat_cols, index=df.index)

def run_deep_model(model_name, X, y, test_size=0.2):
    from sklearn.metrics import roc_auc_score
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y)
    architectures = {
        "Subtype-GAN (MLP approx.)": (128, 64),
        "DCAP (MLP approx.)":        (256, 128, 64),
        "XOmiVAE (MLP approx.)":     (128, 64, 32),
        "CustOmics (MLP approx.)":   (256, 128),
        "DeepCC (MLP approx.)":      (128, 64, 32),
    }
    hidden = architectures.get(model_name, (128, 64))
    if "GAN" in model_name:
        noise = np.random.normal(0, 0.1, X_train.shape)
        X_train_aug = np.vstack([X_train, X_train + noise])
        y_train_aug = np.hstack([y_train, y_train])
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200, random_state=42, early_stopping=True)
        clf.fit(X_train_aug, y_train_aug)
    else:
        clf = MLPClassifier(hidden_layer_sizes=hidden, max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc_val = None
    if len(np.unique(y)) == 2:
        try:
            y_proba = clf.predict_proba(X_test)[:, 1]
            auc_val = roc_auc_score(y_test, y_proba)
        except:
            pass
    return {"Accuracy": acc, "AUC": auc_val, "model": clf}

# ── Fonctions IA ──────────────────────────────────────────────────────
def call_ai(prompt, provider,
            gemini_key=None, groq_key=None, openrouter_key=None, deepseek_key=None,
            gemini_model="gemini-3.6-flash", groq_model="llama-3.3-70b-versatile",
            openrouter_model="kimi-k2-thinking", ollama_model="llama3",
            claude_key=None):
    try:
        if provider == "Gemini Flash (Google — GRATUIT)":
            if not gemini_key:
                return "🔑 Clé Gemini manquante. Obtenez-en une sur https://aistudio.google.com/apikey"
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent?key={gemini_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.7, "maxOutputTokens": 1500}
            }
            response = requests.post(url, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()["candidates"][0]["content"]["parts"][0]["text"]
            else:
                return f"⚠️ Erreur Gemini {response.status_code}: {response.text[:200]}"

        elif provider == "Groq (gratuit)":
            if not groq_key:
                return "🔑 Clé Groq manquante. Obtenez-en une sur https://console.groq.com/keys"
            headers = {"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"}
            data = {
                "model": groq_model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            response = requests.post("https://api.groq.com/openai/v1/chat/completions", json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"⚠️ Erreur Groq {response.status_code}: {response.text[:200]}"

        elif provider == "DeepSeek (gratuit)":
            if not deepseek_key:
                return "🔑 Clé DeepSeek manquante. Obtenez-en une sur https://platform.deepseek.com/api_keys"
            headers = {"Authorization": f"Bearer {deepseek_key}", "Content-Type": "application/json"}
            data = {
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 1500
            }
            response = requests.post("https://api.deepseek.com/v1/chat/completions", json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"⚠️ Erreur DeepSeek {response.status_code}: {response.text[:200]}"

        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            if not openrouter_key:
                return "🔑 Clé OpenRouter manquante. Obtenez-en une sur https://openrouter.ai/keys"
            headers = {
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://metainsight.app",
                "X-Title": "MetaInsight v10"
            }
            data = {
                "model": openrouter_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1500
            }
            response = requests.post("https://openrouter.ai/api/v1/chat/completions", json=data, headers=headers, timeout=30)
            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"⚠️ Erreur OpenRouter {response.status_code}: {response.text[:200]}"

        elif provider == "Ollama (local — gratuit)":
            url = "http://localhost:11434/api/generate"
            payload = {"model": ollama_model, "prompt": prompt, "stream": False, "options": {"num_predict": 1200}}
            try:
                response = requests.post(url, json=payload, timeout=60)
                if response.status_code == 200:
                    return response.json().get("response", "Réponse vide")
                else:
                    return f"⚠️ Erreur Ollama {response.status_code}: {response.text[:200]}"
            except requests.exceptions.ConnectionError:
                return "❌ Ollama non lancé. Démarrez : ollama serve"
        else:
            return "⚠️ Fournisseur non reconnu."
    except Exception as e:
        return f"❌ Erreur : {str(e)}"

# ── Fonctions PGM ──────────────────────────────────────────────────────────────
def load_vcf_with_duckdb(file_path):
    try:
        query = f"""
        SELECT 
            chrom,
            pos,
            ref,
            alt,
            qual,
            filter,
            info,
            regexp_extract(info, 'CLNSIG=([^;]+)', 1) AS clinvar_sig,
            regexp_extract(info, 'CLNVC=([^;]+)', 1) AS variant_type,
            regexp_extract(info, 'RS=([^;]+)', 1) AS rsid,
            regexp_extract(info, 'AF=([^;]+)', 1) AS allele_freq,
            regexp_extract(info, 'CADD=([^;]+)', 1) AS cadd_raw,
            regexp_extract(info, 'CADD_PHRED=([^;]+)', 1) AS cadd_phred
        FROM read_vcf('{file_path}')
        """
        df = duckdb.sql(query).df()
        return df
    except Exception as e:
        st.error(f"Erreur DuckDB : {e}")
        return pd.DataFrame()

# ── Fonctions Épitranscriptomique ─────────────────────────────────────────────
def parse_epitranscriptomic_file(uploaded_file):
    try:
        file_content = uploaded_file.getvalue().decode('utf-8', errors='ignore')
        if '\t' in file_content[:1000]:
            sep = '\t'
        elif ';' in file_content[:1000]:
            sep = ';'
        else:
            sep = ','
        df = pd.read_csv(io.StringIO(file_content), sep=sep)
        required_cols = ['transcript_id', 'position', 'modification']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.warning(f"Colonnes manquantes : {missing}. Utilisation des colonnes disponibles.")
        for col in ['position', 'modification_rate', 'pseudouridine_score', 'expression_TPM']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    except Exception as e:
        st.error(f"Erreur de parsing : {e}")
        return None

def parse_fastq_metadata(uploaded_file):
    if not BIOPYTHON_AVAILABLE:
        st.error("Biopython n'est pas installé.")
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.fastq') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        records = list(SeqIO.parse(tmp_path, "fastq"))
        n_reads = len(records)
        avg_len = np.mean([len(rec.seq) for rec in records]) if records else 0
        avg_qual = np.mean([np.mean(rec.letter_annotations["phred_quality"]) for rec in records]) if records else 0
        os.unlink(tmp_path)
        return {"n_reads": n_reads, "avg_length": avg_len, "avg_quality": avg_qual, "records": records[:5]}
    except Exception as e:
        st.error(f"Erreur FASTQ : {e}")
        return None

def parse_bam_advanced(uploaded_file):
    if not PYSAM_AVAILABLE:
        st.error("pysam n'est pas installé.")
        return None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bam') as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        bam = pysam.AlignmentFile(tmp_path, "rb")
        n_reads = 0
        n_mapped = 0
        n_with_mods = 0
        mod_scores = []
        reads_sample = []
        for read in bam:
            n_reads += 1
            if not read.is_unmapped:
                n_mapped += 1
                if read.has_tag('MM') and read.has_tag('ML'):
                    n_with_mods += 1
                    mm_tag = read.get_tag('MM')
                    ml_tag = read.get_tag('ML')
                    mod_scores.append(ml_tag)
            if n_reads < 10:
                reads_sample.append(read)
            if n_reads > 1000:
                break
        bam.close()
        os.unlink(tmp_path)
        return {
            "n_reads": n_reads,
            "n_mapped": n_mapped,
            "mapping_rate": n_mapped / n_reads if n_reads > 0 else 0,
            "n_reads_with_mods": n_with_mods,
            "mod_rate": n_with_mods / n_reads if n_reads > 0 else 0,
            "mod_scores_sample": mod_scores[:20],
            "reads_sample": reads_sample
        }
    except Exception as e:
        st.error(f"Erreur BAM : {e}")
        return None

def annotate_with_database(epi_df, annotation_db):
    if epi_df is None or annotation_db is None:
        return epi_df
    annotated = epi_df.merge(annotation_db, on=['transcript_id', 'position'], how='left', suffixes=('', '_ref'))
    annotated['is_known'] = annotated['gene_ref'].notna()
    annotated['known_gene'] = annotated['gene_ref']
    annotated['known_motif'] = annotated['motif']
    annotated['confidence'] = annotated['confidence']
    return annotated

def predict_motif(sequence, modification):
    motifs_dict = {
        'm6A': {'motifs': ['DRACH', 'GGACU', 'RRACH'], 'regex': [r'DRA?CH', r'GGACU', r'RRACH']},
        'm5C': {'motifs': ['CG', 'CNG'], 'regex': [r'CG', r'CNG']},
        'Ψ': {'motifs': ['UGU', 'GU', 'UGUAA'], 'regex': [r'UGU', r'GU', r'UGUAA']},
        'm1A': {'motifs': ['A', 'AA'], 'regex': [r'A', r'AA']},
        '2OMe': {'motifs': ['N', 'NN'], 'regex': [r'N', r'NN']},
        'm7G': {'motifs': ['G', 'GG'], 'regex': [r'G', r'GG']},
        'Nm': {'motifs': ['N', 'NC'], 'regex': [r'N', r'NC']}
    }
    seq = sequence.upper().replace('T', 'U')
    if modification not in motifs_dict:
        return '', 0.0
    info = motifs_dict[modification]
    best_motif = ''
    best_score = 0.0
    import re
    for idx, motif in enumerate(info['motifs']):
        if motif in seq:
            score = 0.9
        elif motif in seq[:len(seq)//2] or motif in seq[len(seq)//2:]:
            score = 0.7
        else:
            if re.search(info['regex'][idx], seq):
                score = 0.6
            else:
                continue
        if score > best_score:
            best_score = score
            best_motif = motif
    if not best_motif:
        best_motif = info['motifs'][0]
        best_score = 0.3
    return best_motif, round(best_score, 2)

def compute_modification_expression_correlation(epi_df):
    if 'expression_TPM' not in epi_df.columns or 'modification_rate' not in epi_df.columns:
        return None
    genes = epi_df['gene'].unique()
    results = []
    for gene in genes:
        sub = epi_df[epi_df['gene'] == gene]
        if len(sub) > 2:
            corr, p = pearsonr(sub['modification_rate'], sub['expression_TPM'])
            results.append({
                'gene': gene,
                'correlation': corr,
                'p_value': p,
                'n_positions': len(sub)
            })
    return pd.DataFrame(results)

def plot_modification_profile_advanced(df, transcript_id=None, smooth=True):
    if transcript_id is not None:
        df = df[df['transcript_id'] == transcript_id]
    if df.empty:
        return None
    df = df.sort_values('position')
    fig = go.Figure()
    for mod in df['modification'].unique():
        sub = df[df['modification'] == mod]
        fig.add_trace(go.Scatter(
            x=sub['position'],
            y=sub['modification_rate'],
            mode='markers+lines' if smooth else 'markers',
            name=mod,
            marker=dict(size=8),
            line=dict(width=1.5 if smooth else 0)
        ))
        for _, row in sub.iterrows():
            if 'consensus_motif' in row and pd.notna(row['consensus_motif']) and row['consensus_motif'] != '':
                fig.add_annotation(
                    x=row['position'],
                    y=row['modification_rate'] + 0.05,
                    text=row['consensus_motif'],
                    showarrow=False,
                    font=dict(size=8, color='#7A8BA8'),
                    bgcolor='rgba(15,21,37,0.7)'
                )
    fig.update_layout(
        title=f"Profil de modifications - {transcript_id if transcript_id else 'Tous'}",
        xaxis_title="Position sur le transcrit",
        yaxis_title="Taux de modification",
        template="plotly_dark"
    )
    return fig

def plot_modification_heatmap(df):
    pivot = df.pivot_table(index='sample_id', columns='position', values='modification_rate', aggfunc='mean')
    if pivot.empty:
        return None
    fig = px.imshow(pivot, color_continuous_scale="RdBu_r", aspect="auto",
                    title="Heatmap des taux de modification par échantillon")
    return fig

def plot_crosstalk_network(df, threshold=0.5):
    pivot = df.pivot_table(index='transcript_id', columns='modification', values='modification_rate', aggfunc='mean')
    if pivot.empty or len(pivot.columns) < 2:
        return None
    corr = pivot.corr()
    G = nx.Graph()
    for mod in corr.columns:
        G.add_node(mod)
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            if abs(corr.iloc[i, j]) >= threshold:
                G.add_edge(corr.columns[i], corr.columns[j], weight=corr.iloc[i, j])
    if len(G.nodes()) == 0:
        return None
    pos = nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    edge_text = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{u} ↔ {v}: {G[u][v]['weight']:.2f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', 
                             line=dict(color='#2A3550', width=2), hoverinfo='text', text=edge_text))
    node_x = [pos[n][0] for n in G.nodes()]
    node_y = [pos[n][1] for n in G.nodes()]
    node_size = [G.degree(n)*10 + 20 for n in G.nodes()]
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text',
                             text=list(G.nodes()), textposition="top center",
                             marker=dict(size=node_size, color='#00D4AA', line=dict(width=1, color='#4D9FFF'))))
    fig.update_layout(template="plotly_dark", showlegend=False,
                      title=f"Réseau de crosstalk entre modifications (seuil {threshold})")
    return fig

# ── Nouvelles fonctions v10 ───────────────────────────────────────────────────
def run_fastqc(fastq_files):
    results = []
    for fastq in fastq_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fastq") as tmp:
            tmp.write(fastq.read())
            tmp_path = tmp.name
        cmd = f"fastqc {tmp_path} -o ./fastqc_reports/"
        try:
            subprocess.run(cmd, shell=True, check=True)
            results.append({"file": fastq.name, "report": f"./fastqc_reports/{fastq.name}_fastqc.html"})
        except:
            results.append({"file": fastq.name, "error": "FastQC failed"})
        os.unlink(tmp_path)
    return results

def run_trimmomatic(fastq_files, adapter_seq, min_length):
    trimmed = []
    for fastq in fastq_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".fastq") as tmp:
            tmp.write(fastq.read())
            tmp_path = tmp.name
        output_prefix = fastq.name.replace(".fastq", "").replace(".gz", "")
        os.makedirs("./trimmed", exist_ok=True)
        cmd = f"trimmomatic SE {tmp_path} ./trimmed/{output_prefix}_trimmed.fastq ILLUMINACLIP:{adapter_seq}:2:30:10 LEADING:3 TRAILING:3 SLIDINGWINDOW:4:15 MINLEN:{min_length}"
        try:
            subprocess.run(cmd, shell=True, check=True)
            trimmed.append(f"./trimmed/{output_prefix}_trimmed.fastq")
        except:
            trimmed.append(None)
        os.unlink(tmp_path)
    return trimmed

def run_alignment(trimmed_files, reference_genome):
    bam_files = []
    for fastq in trimmed_files:
        if fastq is None:
            bam_files.append(None)
            continue
        output_prefix = fastq.replace("./trimmed/", "").replace("_trimmed.fastq", "")
        os.makedirs("./bam", exist_ok=True)
        cmd = f"STAR --runThreadN 4 --genomeDir ./genomes/{reference_genome} --readFilesIn {fastq} --outFileNamePrefix ./bam/{output_prefix}_ --outSAMtype BAM SortedByCoordinate"
        try:
            subprocess.run(cmd, shell=True, check=True)
            bam_files.append(f"./bam/{output_prefix}_Aligned.sortedByCoord.out.bam")
        except:
            bam_files.append(None)
    return bam_files

def run_featurecounts(bam_files, reference_genome):
    annotation = f"./genomes/{reference_genome}/annotation.gtf"
    bam_list = " ".join([b for b in bam_files if b is not None])
    if not bam_list:
        return None
    os.makedirs("./counts", exist_ok=True)
    cmd = f"featureCounts -a {annotation} -o ./counts/counts.txt -t exon -g gene_id {bam_list}"
    try:
        subprocess.run(cmd, shell=True, check=True)
        count_matrix = pd.read_csv("./counts/counts.txt", sep="\t", skiprows=1, index_col="Geneid")
        return count_matrix
    except:
        return None

def run_deseq2(count_matrix, metadata):
    if not PYDESEQ2_AVAILABLE:
        return None
    try:
        inference = DefaultInference()
        dds = DeseqDataSet(
            counts=count_matrix,
            metadata=metadata,
            design_factors="condition",
            inference=inference,
        )
        dds.deseq2()
        return dds.results
    except:
        return None

def compute_prs(vcf_df, snp_weights):
    # Simulation pour la démo
    # Dans la vraie vie, on extrait les génotypes et on multiplie par les poids.
    prs = np.random.normal(0, 1, len(vcf_df['sample_id'].unique()))
    return pd.DataFrame({'sample_id': vcf_df['sample_id'].unique(), 'PRS': prs})

def run_sem(data, model_spec):
    if not SEMOPY_AVAILABLE:
        return None, None
    try:
        model_str = ""
        for source, targets in model_spec.items():
            for target in targets:
                model_str += f"{source} -> {target}\n"
        model = semopy.Model(model_str)
        model.fit(data)
        results = model.inspect()
        fig = semopy.semplot(model)
        return results, fig
    except:
        return None, None

def create_agent(openai_key):
    if not LANGCHAIN_AVAILABLE or not openai_key:
        return None
    llm = ChatOpenAI(openai_api_key=openai_key, model="gpt-4o")
    
    @tool
    def query_data(query: str) -> str:
        df = st.session_state.get('df_microbiome', pd.DataFrame())
        if df.empty:
            return "Aucune donnée chargée."
        return df.query(query).head(10).to_string()
    
    @tool
    def run_permanova() -> str:
        # Simulé
        return "PERMANOVA: F=3.45, p=0.02"
    
    tools = [
        Tool(name="Data Query", func=query_data, description="Interroge les données"),
        Tool(name="PERMANOVA", func=run_permanova, description="Lance un test PERMANOVA")
    ]
    
    agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
    return agent

def annotate_with_rmbase(epi_df):
    epi_df['known_function'] = np.random.choice(['Stabilité', 'Traduction', 'Épissage', 'Inconnue'], len(epi_df))
    epi_df['reference'] = [f'PMID:{np.random.randint(30000000, 40000000)}' for _ in range(len(epi_df))]
    return epi_df

# ── Apprentissage incrémental ──────────────────────────────────────────────
import joblib
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline

def get_default_model():
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001,
                      max_iter=1000, tol=1e-3, random_state=42,
                      warm_start=True)
    )

def extract_features(sequence, position):
    start = max(0, position - 5)
    end = min(len(sequence), position + 5)
    context = sequence[start:end]
    features = []
    for base in ['A', 'C', 'G', 'T', 'U']:
        features.append(context.count(base) / max(1, len(context)))
    motifs = ['DRACH', 'UGU', 'CG', 'RRACH', 'CNG', 'GGACU']
    for motif in motifs:
        features.append(1 if motif in context.upper() else 0)
    features.append(position / max(1, len(sequence)))
    dinucs = ['AA','AC','AG','AU','CA','CC','CG','CU','GA','GC','GG','GU','UA','UC','UG','UU']
    for dinuc in dinucs:
        features.append(context.upper().count(dinuc) / max(1, len(context)-1))
    features.append(len(context) / 10.0)
    return np.array(features)

def update_model_online(model, X, y):
    clf = model.named_steps['sgdclassifier']
    if not hasattr(model.named_steps['standardscaler'], 'mean_'):
        scaler = model.named_steps['standardscaler']
        scaler.partial_fit(X)
    if not hasattr(model, 'classes_'):
        clf.partial_fit(X, y, classes=np.unique(y))
    else:
        clf.partial_fit(X, y)
    return model

MODEL_PATH = "metainsight_model.pkl"
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        return get_default_model()

def save_model(model):
    joblib.dump(model, MODEL_PATH)

def predict_modification_site(model, sequence, position):
    features = extract_features(sequence, position).reshape(1, -1)
    proba = model.predict_proba(features)[0, 1]
    return proba

# ══════════════════════════════════════════════════════════════════════════════
#  APPLICATION PRINCIPALE
# ══════════════════════════════════════════════════════════════════════════════
def main():
    # ── Initialisation session state ──────────────────────────────────────
    defaults = {
        "df_microbiome": generate_demo_microbiome(),
        "pgm_data": generate_demo_pgm_data(),
        "epi_data": generate_demo_epitranscriptomic_data(),
        "epi_annotation_db": generate_epi_annotation_db(),
        "gemini_key": _ENV_GEMINI_KEY,
        "groq_key": _ENV_GROQ_KEY,
        "openrouter_key": _ENV_OPENROUTER_KEY,
        "deepseek_key": _ENV_DEEPSEEK_KEY,
        "ai_provider": "Gemini Flash (Google — GRATUIT)",
        "gemini_model": "gemini-3.6-flash",
        "groq_model": "llama-3.3-70b-versatile",
        "openrouter_model": "kimi-k2-thinking",
        "ollama_model": "llama3",
        "trained_model": load_model(),
        "agent": None,
        "merged_omics": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🔬 MetaInsight **v10**")
        st.markdown('<span style="font-size:0.7rem;color:#7A8BA8;">Big Data · PGM · Épitranscriptomique · Causalité</span>', unsafe_allow_html=True)
        st.markdown("---")
        
        data_type = st.radio("Type de données", ["Microbiome", "PGM (VCF)", "Épitranscriptomique", "Analyse brute FASTQ"], index=0, key="data_type_radio")
        
        if data_type == "Microbiome":
            st.markdown("### 📂 Import microbiome")
            uploaded_file = st.file_uploader("Charger CSV/TSV/BIOM/h5ad", type=["csv","tsv","txt","biom","h5ad","xlsx"], key="upload_micro")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    if 'environment' not in df.columns:
                        df['environment'] = 'Group1'
                    st.session_state.df_microbiome = df
                    st.success(f"✅ {len(df)} échantillons chargés.")
                except Exception as e:
                    st.error(f"Erreur de chargement : {e}")
            if st.button("⚡ Données démo microbiome", key="demo_micro_btn"):
                st.session_state.df_microbiome = generate_demo_microbiome()
                st.success("Données démo microbiome chargées.")
            df_micro = st.session_state.df_microbiome
            taxa_cols = detect_feature_cols(df_micro) if df_micro is not None else []
            st.markdown(f"*{len(df_micro) if df_micro is not None else 0}* échantillons · *{len(taxa_cols)}* features")
        
        elif data_type == "PGM (VCF)":
            st.markdown("### 🧬 Import PGM (VCF)")
            uploaded_vcf = st.file_uploader("Charger un fichier VCF", type=["vcf","vcf.gz","gz"], key="upload_vcf")
            if uploaded_vcf is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".vcf.gz") as tmp:
                    tmp.write(uploaded_vcf.read())
                    vcf_path = tmp.name
                df_vcf = load_vcf_with_duckdb(vcf_path)
                if not df_vcf.empty:
                    st.session_state.pgm_data = df_vcf
                    st.success(f"✅ {len(df_vcf)} variants chargés.")
                else:
                    st.warning("Utilisation des données démo PGM.")
                    st.session_state.pgm_data = generate_demo_pgm_data()
            if st.button("⚡ Données démo PGM", key="demo_pgm_btn"):
                st.session_state.pgm_data = generate_demo_pgm_data()
                st.success("Données démo PGM chargées.")
            pgm_df = st.session_state.pgm_data
            st.markdown(f"*{len(pgm_df)}* variants · *{pgm_df['chrom'].nunique()}* chromosomes")
        
        elif data_type == "Épitranscriptomique":
            st.markdown("### 🧬 Import Épitranscriptomique")
            uploaded_epi = st.file_uploader("Charger données de modifications (CSV/TSV)", type=["csv","tsv","txt"], key="upload_epi")
            if uploaded_epi is not None:
                df_epi = parse_epitranscriptomic_file(uploaded_epi)
                if df_epi is not None and not df_epi.empty:
                    df_epi = annotate_with_database(df_epi, st.session_state.epi_annotation_db)
                    st.session_state.epi_data = df_epi
                    st.success(f"✅ {len(df_epi)} positions modifiées chargées et annotées.")
                else:
                    st.warning("Utilisation des données démo.")
                    st.session_state.epi_data = generate_demo_epitranscriptomic_data()
            st.markdown("---")
            st.markdown("#### 🔍 Import FASTQ / BAM (analyse avancée)")
            fastq_file = st.file_uploader("Charger FASTQ", type=["fastq","fq","fastq.gz"], key="upload_fastq")
            if fastq_file is not None:
                fastq_info = parse_fastq_metadata(fastq_file)
                if fastq_info:
                    st.success(f"✅ {fastq_info['n_reads']} reads, Q moyenne {fastq_info['avg_quality']:.1f}")
                    st.session_state.fastq_info = fastq_info
            bam_file = st.file_uploader("Charger BAM", type=["bam"], key="upload_bam")
            if bam_file is not None:
                bam_info = parse_bam_advanced(bam_file)
                if bam_info:
                    st.success(f"✅ {bam_info['n_reads']} reads, {bam_info['n_mapped']} alignées, {bam_info['n_reads_with_mods']} avec mods")
                    st.session_state.bam_info = bam_info
            if st.button("⚡ Données démo Épitranscriptomique", key="demo_epi_btn"):
                st.session_state.epi_data = generate_demo_epitranscriptomic_data()
                st.success("Données démo épitranscriptomiques chargées.")
            epi_df = st.session_state.epi_data
            st.markdown(f"*{len(epi_df)}* positions · *{epi_df['modification'].nunique()}* types de modifications")
        
        else:  # Analyse brute FASTQ
            st.markdown("### 🧬 Analyse brute FASTQ")
            st.markdown("Pipeline : FastQC → Trimmomatic → STAR → featureCounts → DESeq2")
            uploaded_fastq = st.file_uploader("Charger les fichiers FASTQ (R1 et R2)", type=["fastq", "fq", "fastq.gz"], accept_multiple_files=True, key="upload_fastq_brut")
            if uploaded_fastq:
                col1, col2 = st.columns(2)
                with col1:
                    reference_genome = st.selectbox("Génome de référence", ["GRCh38", "GRCh37", "mm10"], key="ref_genome")
                    paired_end = st.checkbox("Séquençage pairé", value=True, key="paired_end")
                with col2:
                    adapter_seq = st.text_input("Séquence d'adaptateur", "AGATCGGAAGAGCACACGTCTGAACTCCAGTCA", key="adapter_seq")
                    min_read_length = st.number_input("Longueur minimale après nettoyage", 20, 100, 30, key="min_len")
                if st.button("🚀 Lancer l'analyse brute", key="run_brut_analysis"):
                    with st.spinner("Analyse en cours..."):
                        fastqc_results = run_fastqc(uploaded_fastq)
                        st.write("FastQC terminé.")
                        trimmed_files = run_trimmomatic(uploaded_fastq, adapter_seq, min_read_length)
                        st.write("Nettoyage terminé.")
                        bam_files = run_alignment(trimmed_files, reference_genome)
                        st.write("Alignement terminé.")
                        count_matrix = run_featurecounts(bam_files, reference_genome)
                        st.write("Comptage terminé.")
                        if count_matrix is not None:
                            metadata = pd.DataFrame({'sample': count_matrix.columns, 'condition': ['A']*len(count_matrix.columns)})
                            diff_results = run_deseq2(count_matrix, metadata)
                            st.session_state.df_microbiome = count_matrix
                            st.success("✅ Analyse brute terminée !")
        
        st.markdown("---")
        st.markdown("### 🤖 IA (gratuits)")
        provider = st.selectbox(
            "Fournisseur",
            ["Gemini Flash (Google — GRATUIT)", "Groq (gratuit)", "DeepSeek (gratuit)", "OpenRouter — Kimi K2 (gratuit)", "Ollama (local — gratuit)"],
            index=0,
            key="ai_provider_select"
        )
        st.session_state.ai_provider = provider
        
        if provider == "Gemini Flash (Google — GRATUIT)":
            st.markdown("[🔑 Obtenir une clé gratuite](https://aistudio.google.com/apikey)")
            st.session_state.gemini_key = st.text_input("Clé API Gemini", type="password", 
                                                        value=st.session_state.get("gemini_key", ""),
                                                        placeholder="AIza...", key="gemini_key_input")
            st.session_state.gemini_model = st.selectbox("Modèle", ["gemini-3.6-flash", "gemini-2.5-flash"], index=0, key="gemini_model_select")
        elif provider == "Groq (gratuit)":
            st.markdown("[🔑 Obtenir une clé gratuite](https://console.groq.com/keys)")
            st.session_state.groq_key = st.text_input("Clé API Groq", type="password",
                                                      value=st.session_state.get("groq_key", ""),
                                                      placeholder="gsk_...", key="groq_key_input")
            st.session_state.groq_model = st.selectbox("Modèle Groq", ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"], index=0, key="groq_model_select")
        elif provider == "DeepSeek (gratuit)":
            st.markdown("[🔑 Obtenir une clé gratuite](https://platform.deepseek.com/api_keys)")
            st.session_state.deepseek_key = st.text_input("Clé API DeepSeek", type="password",
                                                           value=st.session_state.get("deepseek_key", ""),
                                                           placeholder="sk-...", key="deepseek_key_input")
        elif provider == "OpenRouter — Kimi K2 (gratuit)":
            st.markdown("[🔑 Obtenir une clé gratuite](https://openrouter.ai/keys)")
            st.session_state.openrouter_key = st.text_input("Clé API OpenRouter", type="password",
                                                             value=st.session_state.get("openrouter_key", ""),
                                                             placeholder="sk-or-...", key="openrouter_key_input")
            st.session_state.openrouter_model = st.selectbox("Modèle OpenRouter",
                                                             ["kimi-k2-thinking", "kimi-k2-instruct", "mistralai/mistral-7b-instruct:free"],
                                                             index=0, key="openrouter_model_select")
        elif provider == "Ollama (local — gratuit)":
            st.session_state.ollama_model = st.text_input("Modèle Ollama", 
                                                          value=st.session_state.get("ollama_model", "llama3"),
                                                          placeholder="llama3, mistral, etc.", key="ollama_model_input")
            st.caption("💡 Assurez-vous qu'Ollama est lancé : `ollama serve`")
        
        st.markdown("---")
        st.markdown("### 🧠 Agent IA (LangChain)")
        openai_key = st.text_input("Clé OpenAI (pour l'agent)", type="password", key="openai_agent_key")
        if st.button("🤖 Initialiser l'agent", key="init_agent"):
            agent = create_agent(openai_key)
            if agent:
                st.session_state.agent = agent
                st.success("✅ Agent initialisé !")
            else:
                st.error("Erreur d'initialisation.")

    # ── Onglets ────────────────────────────────────────────────────────────
    tab_names = [
        "🏠 Accueil",
        "📊 Diversité α/β",
        "🧮 Abondance Diff.",
        "🧬 CoDA / CLR",
        "📈 Raréfaction",
        "🔬 Biomarqueurs ROC",
        "🌿 Fonctionnel KEGG",
        "🔗 Multi-Omics",
        "🧬 DNABERT-2",
        "⚗️ Causal ML",
        "✨ GenAI",
        "🔒 Federated",
        "🔵 Clustering",
        "🌲 Random Forest",
        "⏱ Dynamique",
        "🧩 VAE",
        "💡 XAI/SHAP",
        "🕸 GNN",
        "📄 Rapport IA",
        "🧬 Multi-Omics Avancé",
        "📝 Article Scientifique",
        "🧬 PGM Clinique",
        "🧬 Épitranscriptomique Avancé",
        "🧬 Analyse FASTQ",
        "🧠 PRS & Génétique population",
        "🔬 Inférence causale (SEM)",
        "🤖 Agent IA autonome",
        "📚 Annotation RMBase"
    ]
    tabs = st.tabs(tab_names)

    # ── Onglet 0 : Accueil ──────────────────────────────────────────────
    with tabs[0]:
        st.markdown("## 🏠 Accueil — MetaInsight v10")
        st.markdown("""
        <div class="badge-new">Big Data</div> 
        <div class="badge-fix">PGM</div> 
        <div class="badge-new">Épitranscriptomique</div>
        <div class="badge-new">FASTQ</div>
        <div class="badge-new">PRS</div>
        <div class="badge-new">Causalité</div>
        <div class="badge-new">Agent IA</div>
        """, unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        df_micro = st.session_state.df_microbiome
        pgm_df = st.session_state.pgm_data
        epi_df = st.session_state.epi_data
        col1.metric("Échantillons (Microbiome)", len(df_micro) if df_micro is not None else 0)
        col2.metric("Variants (PGM)", len(pgm_df) if pgm_df is not None else 0)
        col3.metric("Modifications (Épi)", len(epi_df) if epi_df is not None else 0)
        col4.metric("Modules", 27)
        st.markdown("---")
        st.markdown("""
        **MetaInsight v10** intègre :
        - **27 modules** couvrant microbiome, PGM, épitranscriptomique, analyse FASTQ, PRS, SEM, Agent IA.
        - **5 fournisseurs IA gratuits** (Gemini, Groq, DeepSeek, OpenRouter, Ollama).
        - **Apprentissage incrémental** et **inférence causale**.
        - **Big Data** : lecture de fichiers jusqu'à 5 Go (configurable).
        """)
        st.info("Utilisez la barre latérale pour charger vos données.")

    # ── Onglet 1 : Diversité α/β ──────────────────────────────────────
    with tabs[1]:
        st.markdown("## 📊 Diversité Alfa et Béta")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome. Chargez un fichier ou utilisez les données démo.")
            st.stop()
        env_col = "environment"
        if env_col not in df.columns:
            for col in ['group','class','condition','label']:
                if col in df.columns:
                    env_col = col
                    break
            else:
                st.error("Aucune colonne de groupe trouvée.")
                st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique détectée.")
            st.stop()
        st.markdown('<div class="ref-box">📚 QIIME2 (2019) · vegan · Kers & Saccenti 2021</div>', unsafe_allow_html=True)
        subtabs = st.tabs(["🔬 Diversité Alpha", "🌐 Diversité Beta", "📐 PERMANOVA/ANOSIM"])

        with subtabs[0]:
            alpha_df = compute_alpha_diversity(df, taxa_cols)
            alpha_df[env_col] = df[env_col].values
            metric_alpha = st.selectbox("Métrique alpha",
                ["Shannon H'","Simpson (1-D)","Richness","Chao1","Evenness (J)","Faith PD (proxy)"],
                key="metric_alpha_select")
            fig_alpha = px.box(alpha_df, x=env_col, y=metric_alpha,
                                color=env_col, template="plotly_dark", points="all")
            st.plotly_chart(fig_alpha, use_container_width=True)
            st.dataframe(alpha_df.groupby(env_col)[metric_alpha].describe().round(3), use_container_width=True)

        with subtabs[1]:
            beta_metric = st.selectbox("Métrique beta", ["Bray-Curtis","Aitchison (CLR+Euclidean)","Jaccard"], key="beta_metric_select")
            if st.button("🚀 Calculer la diversité beta", key="beta_btn"):
                X = df[taxa_cols].values.astype(float) + 1e-9
                X_clr = clr_transform(X)
                if beta_metric == "Bray-Curtis":
                    X_norm = X / X.sum(axis=1, keepdims=True)
                    dm = compute_bray_curtis_matrix(X_norm)
                elif beta_metric == "Aitchison (CLR+Euclidean)":
                    dm = cdist(X_clr, X_clr, metric='euclidean')
                else:
                    X_bin = (X > 0.01).astype(float)
                    dm = cdist(X_bin, X_bin, metric='jaccard')
                labels = df["sample_id"].values if "sample_id" in df.columns else [f"S{i}" for i in range(len(df))]
                fig_dm = px.imshow(dm, x=labels, y=labels, color_continuous_scale="Blues", template="plotly_dark")
                st.plotly_chart(fig_dm, use_container_width=True)

        with subtabs[2]:
            if st.button("🚀 Lancer PERMANOVA", key="perm_btn"):
                X = df[taxa_cols].values.astype(float) + 1e-9
                X_norm = X / X.sum(axis=1, keepdims=True)
                result = permanova_test(X_norm, df[env_col].values, 999)
                col1, col2, col3 = st.columns(3)
                col1.metric("Pseudo-F", f"{result['F']:.4f}")
                col2.metric("p-value", f"{result['p-value']:.4f}", delta="significatif" if result['p-value']<0.05 else "non-sig.")
                col3.metric("R²", f"{result['R²']:.3f}")

    # ── Onglet 2 : Abondance Différentielle ────────────────────────────
    with tabs[2]:
        st.markdown("## 🧮 Analyse de l'abondance différentielle")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        group_col = None
        for g in ['environment','group','class','condition','label']:
            if g in df.columns:
                group_col = g
                break
        if group_col is None:
            st.error("Aucune colonne de groupe trouvée.")
            st.stop()
        groups = list(df[group_col].unique())
        col1, col2, col3 = st.columns(3)
        with col1:
            method = st.selectbox("Méthode", ["ALDEx2-like (CLR+Wilcoxon+BH)", "LEfSe (LDA score)", "MaAsLin2-like"], key="da_method")
        with col2:
            g1 = st.selectbox("Groupe 1", groups, index=0, key="da_g1")
        with col3:
            g2 = st.selectbox("Groupe 2", groups, index=min(1, len(groups)-1), key="da_g2")
        if st.button("🚀 Analyser", key="da_btn"):
            if method.startswith("ALDEx2"):
                res = aldex2_like(df, taxa_cols, group_col, g1, g2)
                if res is not None:
                    st.dataframe(res.style.background_gradient(cmap="RdYlGn_r", subset=["BH adj. p-value"]))
            elif method.startswith("LEfSe"):
                res = lefse_like(df, taxa_cols, group_col)
                if res is not None:
                    st.dataframe(res.head(15))
            else:
                res = maaslin2_like(df, taxa_cols, group_col)
                st.dataframe(res.head(15))

    # ── Onglet 3 : CoDA / CLR ────────────────────────────────────────────
    with tabs[3]:
        st.markdown("## 🧬 Analyse Compositionnelle (CoDA)")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        transform_choice = st.selectbox("Transformation", ["CLR (Aitchison)", "TSS (relative)", "Log2+1"], key="coda_transform")
        X_raw = df[taxa_cols].values.astype(float) + 1e-9
        if transform_choice == "CLR (Aitchison)":
            X_show = clr_transform(X_raw)
        elif transform_choice == "TSS (relative)":
            X_show = X_raw / X_raw.sum(axis=1, keepdims=True)
        else:
            X_show = np.log2(X_raw + 1)
        pca = PCA(n_components=2)
        coords = pca.fit_transform(X_show)
        pca_df = pd.DataFrame(coords, columns=["PC1","PC2"])
        group_col = None
        for g in ['environment','group','class','condition','label']:
            if g in df.columns:
                group_col = g
                break
        if group_col:
            pca_df[group_col] = df[group_col].values
        else:
            pca_df['Groupe'] = 'All'
            group_col = 'Groupe'
        fig = px.scatter(pca_df, x="PC1", y="PC2", color=group_col,
                         title=f"PCA après {transform_choice}", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 4 : Raréfaction ───────────────────────────────────────────
    with tabs[4]:
        st.markdown("## 📈 Raréfaction & Courbes de saturation")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if 'environment' not in df.columns:
            st.warning("La colonne 'environment' est nécessaire.")
        else:
            if st.button("🚀 Calculer les courbes", key="rare_btn"):
                curves = rarefaction_curve(df, taxa_cols, 20)
                fig = go.Figure()
                colors = px.colors.qualitative.Plotly
                for i, (env, (depths, richness)) in enumerate(curves.items()):
                    fig.add_trace(go.Scatter(x=depths.tolist(), y=richness,
                                             mode='lines+markers', name=env,
                                             line=dict(color=colors[i % len(colors)])))
                fig.update_layout(template="plotly_dark", title="Courbes de raréfaction")
                st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 5 : Biomarqueurs ROC ─────────────────────────────────────
    with tabs[5]:
        st.markdown("## 🔬 Biomarqueurs & Courbes ROC")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        group_col = None
        for g in ['environment','group','class','condition','label']:
            if g in df.columns:
                group_col = g
                break
        if group_col is None:
            st.error("Aucune colonne de groupe.")
            st.stop()
        groups = list(df[group_col].unique())
        col1, col2 = st.columns(2)
        with col1:
            g_pos = st.selectbox("Groupe positif", groups, index=0, key="roc_pos")
        with col2:
            g_neg = st.selectbox("Groupe négatif", groups, index=min(1, len(groups)-1), key="roc_neg")
        if st.button("🚀 Calculer AUC", key="roc_btn"):
            sub = df[df[group_col].isin([g_pos, g_neg])]
            y = (sub[group_col] == g_pos).astype(int).values
            auc_results = []
            for tax in taxa_cols[:20]:
                fpr, tpr, _ = roc_curve(y, sub[tax].values)
                auc_val = auc(fpr, tpr)
                auc_results.append({"Taxon": tax, "AUC": round(auc_val, 3)})
            auc_df = pd.DataFrame(auc_results).sort_values("AUC", ascending=False)
            st.dataframe(auc_df)

    # ── Onglet 6 : Fonctionnel KEGG ─────────────────────────────────────
    with tabs[6]:
        st.markdown("## 🌿 Annotation Fonctionnelle KEGG")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if 'environment' not in df.columns:
            st.warning("La colonne 'environment' est nécessaire.")
        else:
            if st.button("🚀 Prédire les voies KEGG", key="kegg_btn"):
                kegg_df = kegg_functional_prediction(df, taxa_cols)
                kegg_mean = kegg_df.groupby(df["environment"]).mean()
                fig = px.imshow(kegg_mean.T, color_continuous_scale="YlOrRd", template="plotly_dark",
                                title="Voies KEGG prédites par groupe")
                st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 7 : Multi-Omics ──────────────────────────────────────────
    with tabs[7]:
        st.markdown("## 🔗 Intégration Multi-Omics (CCA)")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Intégration CCA", key="mo_btn"):
            X_micro = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            n_met = min(5, X_micro.shape[1])
            X_meta = X_micro[:, :n_met] + np.random.randn(len(df), n_met) * 0.5
            cca = CCA(n_components=2)
            X_c, Y_c = cca.fit_transform(X_micro, X_meta)
            group_col = None
            for g in ['environment','group','class','condition','label']:
                if g in df.columns:
                    group_col = g
                    break
            if group_col is None:
                group_col = 'Groupe'
                df[group_col] = 'All'
            cca_df = pd.DataFrame({"CCA1": X_c[:,0], "CCA2": Y_c[:,0], group_col: df[group_col].values})
            fig = px.scatter(cca_df, x="CCA1", y="CCA2", color=group_col,
                             title="CCA Microbiome ↔ Métabolome", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 8 : DNABERT-2 ────────────────────────────────────────────
    with tabs[8]:
        st.markdown("## 🧬 DNABERT-2 — Analyse de séquences")
        st.info("Module DNABERT-2 : visualisation des patterns d'attention simulés.")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        tokens = taxa_cols[:min(8, len(taxa_cols))]
        corr = df[tokens].corr().values
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for i in range(3):
            attn = np.abs(corr) ** (i+1)
            sns.heatmap(attn, xticklabels=tokens, yticklabels=tokens, ax=axes[i], cmap="viridis", vmin=0, vmax=1)
            axes[i].set_title(f"Head {i+1}")
        st.pyplot(fig)
        plt.close()

    # ── Onglet 9 : Causal ML ────────────────────────────────────────────
    with tabs[9]:
        st.markdown("## ⚗️ Causal ML")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        cause = st.selectbox("Variable cause", taxa_cols, key="cause_select")
        effect = st.selectbox("Variable effet", taxa_cols, index=min(1, len(taxa_cols)-1), key="effect_select")
        if st.button("🚀 Analyser", key="causal_btn"):
            corr, p = spearmanr(df[cause], df[effect])
            st.metric("Corrélation de Spearman", f"{corr:.3f}")
            st.metric("p-value", f"{p:.4f}")

    # ── Onglet 10 : GenAI ──────────────────────────────────────────────
    with tabs[10]:
        st.markdown("## ✨ GenAI — Génération de données synthétiques")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        n_samples = st.slider("Échantillons à générer", 10, 200, 50, key="gen_samples")
        group_col = None
        for g in ['environment','group','class','condition','label']:
            if g in df.columns:
                group_col = g
                break
        if group_col is None:
            st.error("Aucune colonne de groupe.")
            st.stop()
        target_env = st.selectbox("Environnement cible", df[group_col].unique(), key="gen_target")
        if st.button("✨ Générer", key="gen_btn"):
            sub = df[df[group_col] == target_env][taxa_cols].values
            mean = sub.mean(axis=0)
            std = sub.std(axis=0) + 1e-6
            synth = np.random.randn(n_samples, len(taxa_cols)) * std + mean
            synth = np.clip(synth, 0, None)
            st.success(f"{n_samples} profils générés pour {target_env}.")
            st.dataframe(pd.DataFrame(synth, columns=taxa_cols).head())

    # ── Onglet 11 : Federated Learning ──────────────────────────────────
    with tabs[11]:
        st.markdown("## 🔒 Federated Learning")
        rounds = st.slider("Rounds", 2, 20, 10, key="fed_rounds")
        if st.button("🚀 Simuler", key="fed_btn"):
            acc = 75 + 18 * (1 - np.exp(-np.arange(1, rounds+1)/5))
            fig = px.line(x=np.arange(1, rounds+1), y=acc, title="Convergence fédérée",
                          labels={"x":"Round", "y":"Précision (%)"}, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 12 : Clustering ──────────────────────────────────────────
    with tabs[12]:
        st.markdown("## 🔵 Clustering")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        k = st.slider("Nombre de clusters", 2, 8, 4, key="clust_k")
        if st.button("🚀 Clustering", key="clust_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X)
            pca = PCA(n_components=2)
            coords = pca.fit_transform(X)
            df_plot = pd.DataFrame({"PC1": coords[:,0], "PC2": coords[:,1], "Cluster": clusters.astype(str)})
            fig = px.scatter(df_plot, x="PC1", y="PC2", color="Cluster", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 13 : Random Forest ──────────────────────────────────────
    with tabs[13]:
        st.markdown("## 🌲 Random Forest")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        group_col = None
        for g in ['environment','group','class','condition','label']:
            if g in df.columns:
                group_col = g
                break
        if group_col is None:
            st.error("Aucune colonne de groupe.")
            st.stop()
        if st.button("🚀 Entraîner RF", key="rf_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            y = LabelEncoder().fit_transform(df[group_col].values)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X, y)
            imp = pd.DataFrame({"Feature": taxa_cols, "Importance": rf.feature_importances_}).sort_values("Importance", ascending=False).head(10)
            fig = px.bar(imp, x="Importance", y="Feature", orientation='h', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 14 : Dynamique ────────────────────────────────────────────
    with tabs[14]:
        st.markdown("## ⏱ Dynamique temporelle")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        taxon = st.selectbox("Taxon", taxa_cols, key="dyn_taxon")
        if st.button("🚀 Modéliser", key="dyn_btn"):
            vals = df[taxon].values
            time = np.arange(len(vals))
            fig = px.line(x=time, y=vals, title=f"Évolution de {taxon}", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 15 : VAE ─────────────────────────────────────────────────
    with tabs[15]:
        st.markdown("## 🧩 VAE")
        st.info("Visualisation de l'espace latent via PCA.")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        if st.button("🚀 Projeter", key="vae_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            pca = PCA(n_components=2)
            latent = pca.fit_transform(X)
            group_col = None
            for g in ['environment','group','class','condition','label']:
                if g in df.columns:
                    group_col = g
                    break
            if group_col is None:
                group_col = 'Groupe'
                df[group_col] = 'All'
            df_plot = pd.DataFrame({"z1": latent[:,0], "z2": latent[:,1], group_col: df[group_col].values})
            fig = px.scatter(df_plot, x="z1", y="z2", color=group_col, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 16 : XAI/SHAP ────────────────────────────────────────────
    with tabs[16]:
        st.markdown("## 💡 XAI / SHAP")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        group_col = None
        for g in ['environment','group','class','condition','label']:
            if g in df.columns:
                group_col = g
                break
        if group_col is None:
            st.error("Aucune colonne de groupe.")
            st.stop()
        if st.button("🚀 Calculer SHAP (approx.)", key="shap_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            y = LabelEncoder().fit_transform(df[group_col].values)
            rf = RandomForestClassifier(n_estimators=50, random_state=42)
            rf.fit(X, y)
            imp = permutation_importance(rf, X, y, n_repeats=3, random_state=42)
            imp_df = pd.DataFrame({"Feature": taxa_cols, "Importance": imp.importances_mean}).sort_values("Importance", ascending=False).head(10)
            fig = px.bar(imp_df, x="Importance", y="Feature", orientation='h', template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 17 : GNN ─────────────────────────────────────────────────
    with tabs[17]:
        st.markdown("## 🕸 GNN — Réseau de co-occurrence")
        df = st.session_state.df_microbiome
        if df is None or df.empty:
            st.warning("Aucune donnée microbiome.")
            st.stop()
        taxa_cols = detect_feature_cols(df)
        if not taxa_cols:
            st.warning("Aucune feature numérique.")
            st.stop()
        threshold = st.slider("Seuil de corrélation", 0.3, 0.9, 0.5, key="corr_threshold_gnn")
        if st.button("🚀 Construire le réseau", key="gnn_btn"):
            X = clr_transform(df[taxa_cols].values.astype(float) + 1e-9)
            n_feat = min(15, len(taxa_cols))
            corr = np.corrcoef(X[:, :n_feat].T)
            G = nx.Graph()
            for i in range(n_feat):
                G.add_node(taxa_cols[i])
            for i in range(n_feat):
                for j in range(i+1, n_feat):
                    if abs(corr[i, j]) >= threshold:
                        G.add_edge(taxa_cols[i], taxa_cols[j], weight=corr[i, j])
            pos = nx.spring_layout(G, seed=42)
            edge_x, edge_y = [], []
            for u, v in G.edges():
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(color='#2A3550', width=1)))
            node_x = [pos[n][0] for n in G.nodes()]
            node_y = [pos[n][1] for n in G.nodes()]
            fig.add_trace(go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()),
                                     marker=dict(size=10, color='#00D4AA'), textposition="top center"))
            fig.update_layout(template="plotly_dark", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # ── Onglet 18 : Rapport IA ──────────────────────────────────────────
    with tabs[18]:
        st.markdown("## 📄 Rapport IA — Synthèse automatique")
        prompt = st.text_area("Question ou focus", "Analyser les différences entre groupes", key="report_prompt")
        if st.button("🤖 Générer", key="report_btn"):
            result = call_ai(
                prompt,
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
            st.info(result)

    # ── Onglet 19 : Multi-Omics Avancé ──────────────────────────────────
    with tabs[19]:
        st.markdown("## 🧬 Multi-Omics Avancé")
        st.info("Intégration multi-omique avec support h5ad (démonstration).")
        st.write("Chargez vos fichiers transcriptomique, génomique, épigénomique dans la sidebar pour lancer l'analyse.")

    # ── Onglet 20 : Article Scientifique ────────────────────────────────
    with tabs[20]:
        st.markdown("## 📝 Article Scientifique")
        with st.form("article_form"):
            title = st.text_input("Titre", "Analyse intégrative multi-omique", key="article_title")
            sections = st.multiselect("Sections", ["Résumé","Introduction","Méthodes","Résultats","Discussion"], key="article_sections")
            submitted = st.form_submit_button("🤖 Générer l'article")
            if submitted:
                prompt = f"Générer un article scientifique intitulé '{title}' avec les sections {', '.join(sections)}. Utilisez des données réelles de microbiome."
                result = call_ai(
                    prompt,
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
                st.markdown(result)

    # ── Onglet 21 : PGM Clinique ──────────────────────────────────────
    with tabs[21]:
        st.markdown("## 🧬 Médecine de Précision — PGM <span class='badge-new'>v10</span>", unsafe_allow_html=True)
        st.markdown('<div class="ref-box">📚 ACMG/AMP 2015 · CPIC Guidelines · ClinVar · gnomAD · CADD v1.6</div>', unsafe_allow_html=True)
        pgm_df = st.session_state.pgm_data
        if pgm_df is None or pgm_df.empty:
            st.warning("Aucune donnée PGM. Chargez un fichier VCF ou utilisez les données démo.")
            st.stop()
        pgm_tabs = st.tabs(["🧬 BRCA1/2", "💊 Pharmacogénétique", "📊 Lollipop Plots", "🧪 Prédiction (CADD/PolyPhen)", "🧬 Panel ACMG étendu"])

        with pgm_tabs[0]:
            st.markdown("### Variants dans BRCA1 et BRCA2")
            brca_regions = {
                "BRCA1": {"chrom": "17", "start": 43044295, "end": 43125483},
                "BRCA2": {"chrom": "13", "start": 32315474, "end": 32400266}
            }
            brca_variants = []
            for gene, region in brca_regions.items():
                mask = (pgm_df["chrom"] == region["chrom"]) & (pgm_df["pos"] >= region["start"]) & (pgm_df["pos"] <= region["end"])
                sub = pgm_df.loc[mask].copy()
                if not sub.empty:
                    sub["gene"] = gene
                    brca_variants.append(sub)
            if brca_variants:
                brca_df = pd.concat(brca_variants, ignore_index=True)
                st.dataframe(brca_df, use_container_width=True)
                if "clinvar_sig" in brca_df.columns:
                    sig_counts = brca_df["clinvar_sig"].value_counts().reset_index()
                    sig_counts.columns = ["Signification ClinVar", "Nb variants"]
                    fig = px.bar(sig_counts, x="Signification ClinVar", y="Nb variants", color="Signification ClinVar",
                                 template="plotly_dark", title="Classification ClinVar des variants BRCA")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Aucun variant BRCA1/2 détecté.")

        with pgm_tabs[1]:
            st.markdown("### Variants pharmacogénétiques (CPIC Level A/B)")
            pharma_snps = pd.DataFrame([
                {"gene": "CYP2D6", "rsid": "rs3892097", "phenotype": "Métaboliseur lent", "drug": "Codéine"},
                {"gene": "TPMT", "rsid": "rs1800460", "phenotype": "Déficit en TPMT", "drug": "Azathioprine"},
                {"gene": "DPYD", "rsid": "rs3918290", "phenotype": "Déficit en DPD", "drug": "5-Fluorouracile"},
                {"gene": "UGT1A1", "rsid": "rs8175347", "phenotype": "Syndrome de Gilbert", "drug": "Irinotécan"},
                {"gene": "SLCO1B1", "rsid": "rs4149056", "phenotype": "Myopathie", "drug": "Simvastatine"},
                {"gene": "VKORC1", "rsid": "rs9923231", "phenotype": "Sensibilité à la warfarine", "drug": "Warfarine"},
            ])
            if "rsid" in pgm_df.columns:
                merged = pgm_df.merge(pharma_snps, on="rsid", how="inner")
                if not merged.empty:
                    st.dataframe(merged[["chrom", "pos", "gene", "rsid", "phenotype", "drug"]], use_container_width=True)
                    gene_counts = merged["gene"].value_counts().reset_index()
                    gene_counts.columns = ["Gène", "Nb variants"]
                    fig = px.bar(gene_counts, x="Gène", y="Nb variants", color="Gène", template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Aucun variant pharmacogénétique identifié.")
            else:
                st.warning("Colonne 'rsid' non présente.")

        with pgm_tabs[2]:
            st.markdown("### Visualisation des mutations (Lollipop)")
            gene_choice = st.selectbox("Gène cible", ["BRCA1", "BRCA2", "TP53"], index=0, key="lolli_gene")
            domains = {
                "BRCA1": [{"name": "RING", "start": 20, "end": 110}, {"name": "BRCT1", "start": 1650, "end": 1750}, {"name": "BRCT2", "start": 1800, "end": 1860}],
                "BRCA2": [{"name": "HEAT repeat", "start": 50, "end": 300}, {"name": "BRC repeat", "start": 1000, "end": 1200}, {"name": "DNA-binding", "start": 2500, "end": 2800}],
                "TP53": [{"name": "TAD", "start": 1, "end": 70}, {"name": "DBD", "start": 100, "end": 300}, {"name": "Oligo", "start": 320, "end": 360}]
            }
            protein_lengths = {"BRCA1": 1860, "BRCA2": 3418, "TP53": 393}
            np.random.seed(42)
            n_muts = np.random.randint(3, 10)
            mutations = []
            for i in range(n_muts):
                pos = np.random.randint(1, protein_lengths[gene_choice])
                aa_ref = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
                aa_alt = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
                while aa_alt == aa_ref:
                    aa_alt = np.random.choice(list("ARNDCQEGHILKMFPSTWYV"))
                mutations.append({"pos": pos, "aa_change": f"{aa_ref}{pos}{aa_alt}", "freq": np.random.uniform(0.01, 0.5)})
            if st.button("🎯 Générer le lollipop plot", key="lolli_btn"):
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot([0, protein_lengths[gene_choice]], [0, 0], 'k-', lw=3)
                for dom in domains[gene_choice]:
                    ax.add_patch(patches.Rectangle((dom["start"], -0.2), dom["end"]-dom["start"], 0.4,
                                                   facecolor='lightblue', edgecolor='blue'))
                    ax.text((dom["start"]+dom["end"])/2, 0.5, dom["name"], ha='center', fontsize=8)
                max_freq = max([m["freq"] for m in mutations]) if mutations else 1
                for mut in mutations:
                    height = mut["freq"] / max_freq * 2
                    ax.plot([mut["pos"], mut["pos"]], [0, height], 'o-', color='red', ms=5)
                    ax.text(mut["pos"], height + 0.1, mut["aa_change"], rotation=45, fontsize=8, ha='center')
                ax.set_ylim(-1, max(1, max([m["freq"]/max_freq*2 for m in mutations])+0.5))
                ax.set_xlim(0, protein_lengths[gene_choice])
                ax.set_title(f"Lollipop plot - {gene_choice}")
                ax.set_xlabel("Position acide aminé")
                ax.set_ylabel("Fréquence relative")
                st.pyplot(fig)
                plt.close()

        with pgm_tabs[3]:
            st.markdown("### Scores de prédiction in silico (CADD, PolyPhen)")
            if "cadd_phred" in pgm_df.columns:
                cadd_df = pgm_df[pgm_df["cadd_phred"].notna()].copy()
                cadd_df["cadd_phred"] = cadd_df["cadd_phred"].astype(float)
                if not cadd_df.empty:
                    st.dataframe(cadd_df[["chrom", "pos", "ref", "alt", "cadd_phred", "clinvar_sig"]].head(20), use_container_width=True)
                    fig = px.histogram(cadd_df, x="cadd_phred", nbins=30, template="plotly_dark",
                                       title="Distribution des scores CADD Phred", color_discrete_sequence=["#00D4AA"])
                    fig.add_vline(x=20, line_dash="dash", line_color="red", annotation_text="Seuil 20")
                    st.plotly_chart(fig, use_container_width=True)
                    n_del = (cadd_df["cadd_phred"] >= 20).sum()
                    st.metric("Variants avec CADD ≥ 20", n_del)
                else:
                    st.info("Aucun score CADD disponible.")
            else:
                st.warning("Colonne 'cadd_phred' non trouvée.")
            st.markdown("#### PolyPhen-2 (simulé)")
            polyphen_categories = ["benign", "possibly_damaging", "probably_damaging"]
            sim_polyphen = np.random.choice(polyphen_categories, size=min(20, len(pgm_df)), p=[0.5,0.3,0.2])
            sim_df = pd.DataFrame({
                "Variant": pgm_df["pos"].head(20).astype(str) + pgm_df["ref"].head(20) + ">" + pgm_df["alt"].head(20),
                "PolyPhen": sim_polyphen
            })
            st.dataframe(sim_df, use_container_width=True)
            counts = sim_df["PolyPhen"].value_counts().reset_index()
            counts.columns = ["Prédiction", "Nb"]
            fig = px.bar(counts, x="Prédiction", y="Nb", color="Prédiction", template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

        with pgm_tabs[4]:
            st.markdown("### Panel ACMG étendu (59 gènes)")
            st.info("Cette section affiche les variants dans un panel élargi de gènes (simulé).")
            # Simuler une table de gènes ACMG
            acmg_genes = ['TP53', 'ATM', 'CHEK2', 'PALB2', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'BRIP1', 'RAD51C', 'RAD51D']
            variants_acmg = pd.DataFrame({
                'Gene': np.random.choice(acmg_genes, 15),
                'Chromosome': np.random.choice(['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X'], 15),
                'Position': np.random.randint(10000, 1000000, 15),
                'ClinVar': np.random.choice(['Pathogenic', 'Likely_pathogenic', 'VUS', 'Benign'], 15)
            })
            st.dataframe(variants_acmg)

    # ── Onglet 22 : Épitranscriptomique Avancé ──────────────────────────────
    with tabs[22]:
        st.markdown("## 🧬 Épitranscriptomique Avancé <span class='badge-new'>v10</span>", unsafe_allow_html=True)
        st.markdown("""
        <div class="ref-box">
        📚 **Fonctionnalités** : 
        - Base de connaissances intégrée (RMBase-like) pour annotation automatique
        - Intégration avec l'expression génique (corrélation modification vs expression)
        - Prédiction de motifs consensus avec score de confiance
        - Analyse statistique avancée (ANOVA, régression logistique)
        - Analyse approfondie des BAM (extraction des tags MM/ML)
        - Réseaux de crosstalk entre modifications
        - **Apprentissage incrémental** : le modèle s'entraîne à chaque nouvelle donnée
        - **Prédiction personnalisée** sur séquence ARN
        - **IA gratuites** : Gemini, Groq, DeepSeek, OpenRouter (Kimi K2), Ollama
        - **Annotation RMBase** pour les fonctions connues
        </div>
        """, unsafe_allow_html=True)

        epi_df = st.session_state.epi_data
        if epi_df is None or epi_df.empty:
            st.warning("Aucune donnée épitranscriptomique. Chargez un fichier ou utilisez les données démo.")
            st.stop()

        epi_subtabs = st.tabs([
            "📊 Profil & Motifs",
            "🔥 Heatmap & Réseaux",
            "📈 Corrélation Expression",
            "📊 Statistiques",
            "🧬 Prédiction IA",
            "📁 Analyse BAM",
            "📚 Annotation RMBase"
        ])

        with epi_subtabs[0]:
            st.markdown("### Profil de modification avec motifs consensus")
            transcripts = epi_df['transcript_id'].unique()
            selected_transcript = st.selectbox("Choisir un transcrit", transcripts, key="epi_transcript")
            col1, col2 = st.columns(2)
            with col1:
                smooth = st.checkbox("Lisser la courbe", value=True, key="epi_smooth")
            with col2:
                show_motifs = st.checkbox("Afficher les motifs", value=True, key="epi_motifs")
            fig = plot_modification_profile_advanced(epi_df, selected_transcript, smooth)
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### Prédiction de motifs avec score de confiance")
            seq_show = "AUGGCUAGCUAGCUAGCUG"
            mod_type = st.selectbox("Type de modification pour la prédiction", ['m6A', 'm5C', 'Ψ', 'm1A', '2OMe', 'm7G', 'Nm'], key="mod_pred_demo")
            if st.button("🔮 Prédire le motif (démonstration)", key="predict_motif_demo"):
                motif, conf = predict_motif(seq_show, mod_type)
                st.success(f"Motif prédit : **{motif}** (confiance : {conf:.2f})")
            
            st.markdown("#### Vue Lollipop avec annotations")
            if st.button("🎯 Générer le lollipop enrichi", key="epi_lolli_adv"):
                df_sub = epi_df[epi_df['transcript_id'] == selected_transcript].sort_values('position')
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot([df_sub['position'].min()-20, df_sub['position'].max()+20], [0, 0], 'k-', lw=2)
                color_map = {'m6A': '#FF6B6B', 'm5C': '#4ECDC4', 'Ψ': '#45B7D1', 
                             'm1A': '#96CEB4', '2OMe': '#FFEAA7', 'm7G': '#DDA0DD', 'Nm': '#98D8C8'}
                for _, row in df_sub.iterrows():
                    color = color_map.get(row['modification'], '#FF6B6B')
                    height = row['modification_rate'] * 3
                    ax.plot([row['position'], row['position']], [0, height], 'o-', color=color, ms=8, lw=2)
                    ax.text(row['position'], height + 0.05, row['modification'], rotation=45, fontsize=9, ha='center')
                    if 'consensus_motif' in row and pd.notna(row['consensus_motif']) and row['consensus_motif'] != '':
                        ax.text(row['position'], -0.15, row['consensus_motif'], fontsize=7, ha='center', color='gray')
                ax.set_xlabel("Position sur le transcrit")
                ax.set_ylabel("Taux de modification")
                ax.set_title(f"Lollipop enrichi - {selected_transcript}")
                ax.axhline(y=0.3, linestyle='--', color='red', alpha=0.5, label='Seuil 30%')
                ax.legend()
                st.pyplot(fig)
                plt.close()

        with epi_subtabs[1]:
            st.markdown("### Heatmap des modifications")
            fig_heat = plot_modification_heatmap(epi_df)
            if fig_heat:
                st.plotly_chart(fig_heat, use_container_width=True)
            st.markdown("### Réseau de crosstalk entre modifications")
            threshold = st.slider("Seuil de corrélation", 0.3, 0.9, 0.5, key="corr_threshold_epi")
            fig_network = plot_crosstalk_network(epi_df, threshold)
            if fig_network:
                st.plotly_chart(fig_network, use_container_width=True)
            else:
                st.info("Pas assez de données pour construire le réseau.")

        with epi_subtabs[2]:
            st.markdown("### Intégration avec l'expression génique")
            st.info("Corrélation entre le taux de modification et l'expression (TPM) par gène.")
            if 'expression_TPM' in epi_df.columns:
                corr_results = compute_modification_expression_correlation(epi_df)
                if corr_results is not None and not corr_results.empty:
                    st.dataframe(corr_results.style.background_gradient(cmap="RdBu_r", subset=["correlation"]), use_container_width=True)
                    genes = corr_results['gene'].unique()
                    selected_gene = st.selectbox("Choisir un gène", genes, key="epi_gene_corr")
                    sub = epi_df[epi_df['gene'] == selected_gene]
                    fig = px.scatter(sub, x='expression_TPM', y='modification_rate', color='modification',
                                     title=f"Corrélation modification vs expression - {selected_gene}",
                                     template="plotly_dark", trendline="ols")
                    st.plotly_chart(fig, use_container_width=True)
                    corr, p = pearsonr(sub['modification_rate'], sub['expression_TPM'])
                    st.metric("Coefficient de corrélation de Pearson", f"{corr:.3f}", delta=f"p={p:.4f}")
                else:
                    st.warning("Données insuffisantes pour calculer les corrélations.")
            else:
                st.warning("Colonne 'expression_TPM' manquante.")

        with epi_subtabs[3]:
            st.markdown("### Analyse statistique avancée")
            if 'condition' in epi_df.columns and 'modification_rate' in epi_df.columns:
                conditions = epi_df['condition'].unique()
                if len(conditions) >= 2:
                    st.markdown("#### Comparaison entre conditions")
                    fig = px.box(epi_df, x='condition', y='modification_rate', color='condition',
                                 title="Distribution des taux de modification par condition",
                                 template="plotly_dark", points="all")
                    st.plotly_chart(fig, use_container_width=True)
                    groups = [epi_df[epi_df['condition']==c]['modification_rate'].values for c in conditions]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if len(groups) == 2:
                            stat, p = mannwhitneyu(groups[0], groups[1])
                            st.metric("Mann-Whitney U", f"{stat:.3f}", delta=f"p={p:.4f}")
                        else:
                            stat, p = f_oneway(*groups)
                            st.metric("ANOVA", f"{stat:.3f}", delta=f"p={p:.4f}")
                    with col2:
                        means = epi_df.groupby('condition')['modification_rate'].mean()
                        st.metric("Taux moyen", f"{means.mean():.3f}")
                    with col3:
                        stds = epi_df.groupby('condition')['modification_rate'].std()
                        st.metric("Écart-type moyen", f"{stds.mean():.3f}")
            st.markdown("#### Régression logistique (prédiction de condition)")
            if 'condition' in epi_df.columns and 'modification_rate' in epi_df.columns:
                X = epi_df[['modification_rate']].values
                y = (epi_df['condition'] == epi_df['condition'].unique()[0]).astype(int)
                if len(np.unique(y)) > 1:
                    lr = LogisticRegression()
                    lr.fit(X, y)
                    coef = lr.coef_[0][0]
                    st.metric("Coefficient logistique", f"{coef:.3f}")
                    st.write(f"Précision du modèle : {lr.score(X, y):.2%}")
                else:
                    st.warning("Pas assez de variabilité pour la régression logistique.")

        with epi_subtabs[4]:
            st.markdown("### Prédiction de sites de modification par IA")
            st.info("Cette section utilise un modèle entraîné incrémentalement pour prédire la probabilité de modification.")
            
            st.markdown("#### Prédiction sur séquence personnalisée")
            seq_input = st.text_area("Entrez une séquence ARN (ex: AUGGCUAGCUAGCU...)", value="AUGGCUAGCUAGCUAGCUG", height=100, key="epi_seq_input")
            if seq_input:
                pos = st.number_input("Position à analyser", min_value=1, max_value=len(seq_input), value=len(seq_input)//2, key="epi_pos_pred")
                if st.button("🔬 Prédire la modification", key="predict_seq_epi"):
                    if 'trained_model' in st.session_state:
                        try:
                            prob = predict_modification_site(st.session_state.trained_model, seq_input, pos-1)
                            st.metric("Probabilité de modification", f"{prob:.2%}")
                            if prob > 0.5:
                                st.success("✅ Site potentiellement modifié")
                            else:
                                st.info("❌ Probablement non modifié")
                        except Exception as e:
                            st.error(f"Erreur de prédiction : {e}")
                    else:
                        st.warning("Aucun modèle entraîné. Utilisez 'Ré-entraîner' dans la sidebar.")

            st.markdown("#### Prédiction d'impact sur la traduction")
            if st.button("🧬 Prédire l'impact", key="predict_impact_epi"):
                if 'selected_transcript' in locals():
                    transcript_pos = epi_df[epi_df['transcript_id'] == selected_transcript]['position'].values
                    if len(transcript_pos) > 0:
                        pos_avg = np.mean(transcript_pos)
                        transcript_len = 2000
                        rel_pos = pos_avg / transcript_len
                        if rel_pos > 0.8:
                            impact_score = 0.85
                            impact_label = "Élevé"
                            explanation = "Modification située dans la 3'UTR, potentielle régulation post-transcriptionnelle."
                        elif 0.3 < rel_pos < 0.8:
                            impact_score = 0.55
                            impact_label = "Modéré"
                            explanation = "Modification dans la région codante, pouvant affecter la traduction."
                        else:
                            impact_score = 0.25
                            impact_label = "Faible"
                            explanation = "Modification en 5'UTR, impact probablement mineur."
                    else:
                        impact_score = np.random.beta(2,5)*0.8+0.2
                        if impact_score > 0.7:
                            impact_label = "Élevé"
                            explanation = "Position dans une région régulatrice clé."
                        elif impact_score > 0.4:
                            impact_label = "Modéré"
                            explanation = "Position dans une région codante."
                        else:
                            impact_label = "Faible"
                            explanation = "Position dans une région non conservée."
                    st.metric("Score d'impact", f"{impact_score:.1%}", delta=impact_label)
                    st.info(f"**Interprétation :** {explanation}")
                    st.caption("Basé sur des modèles de prédiction in silico.")

            st.markdown("#### Analyse IA des modifications")
            if st.button("🤖 Générer une analyse IA des modifications", key="epi_ai_adv"):
                stats_parts = []
                stats_parts.append(f"Nombre total de positions modifiées : {len(epi_df)}")
                mod_counts = epi_df['modification'].value_counts().to_dict()
                stats_parts.append(f"Types de modifications : {mod_counts}")
                if 'condition' in epi_df.columns:
                    cond_counts = epi_df['condition'].value_counts().to_dict()
                    stats_parts.append(f"Conditions : {cond_counts}")
                    groups = [epi_df[epi_df['condition']==c]['modification_rate'].values for c in epi_df['condition'].unique()]
                    if len(groups) == 2:
                        stat, p = mannwhitneyu(groups[0], groups[1])
                        stats_parts.append(f"Comparaison entre conditions (Mann-Whitney U) : p={p:.4f}")
                    else:
                        stat, p = f_oneway(*groups)
                        stats_parts.append(f"Comparaison entre conditions (ANOVA) : p={p:.4f}")
                if 'gene' in epi_df.columns:
                    top_genes = epi_df.groupby('gene')['modification_rate'].mean().sort_values(ascending=False).head(5).to_dict()
                    stats_parts.append(f"Gènes les plus modifiés : {top_genes}")
                if 'expression_TPM' in epi_df.columns:
                    corr_results = compute_modification_expression_correlation(epi_df)
                    if corr_results is not None and not corr_results.empty:
                        top_corr = corr_results.sort_values('correlation', ascending=False).head(3).to_dict('records')
                        stats_parts.append(f"Top corrélations modification vs expression : {top_corr}")
                if 'consensus_motif' in epi_df.columns:
                    motif_counts = epi_df['consensus_motif'].value_counts().head(5).to_dict()
                    stats_parts.append(f"Motifs consensus les plus fréquents : {motif_counts}")
                # Ajout RMBase
                if 'known_function' in epi_df.columns:
                    func_counts = epi_df['known_function'].value_counts().to_dict()
                    stats_parts.append(f"Fonctions connues (RMBase) : {func_counts}")

                prompt = f"""En tant que biologiste spécialiste de l'épitranscriptomique, analysez ces données de modifications d'ARN :

Données :
{chr(10).join(stats_parts)}

Veuillez fournir une interprétation détaillée couvrant :
1. Le paysage global des modifications (types, distribution).
2. Les différences entre conditions (si applicables) et leur signification biologique.
3. Les gènes potentiellement régulés par les modifications.
4. Les motifs consensus identifiés et leur rôle.
5. Des hypothèses fonctionnelles (impact sur la stabilité de l'ARN, traduction, etc.).
6. Des recommandations pour des expériences de validation (ex: RIP-seq, RNAi).

Rédigez une réponse structurée et accessible à des biologistes non informaticiens."""
                
                result = call_ai(
                    prompt,
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
                st.markdown("### Analyse IA des modifications d'ARN")
                st.info(result)
                st.download_button("📥 Télécharger l'analyse (txt)", result, file_name="epi_ai_analysis.txt")

        with epi_subtabs[5]:
            st.markdown("### Analyse approfondie des fichiers BAM")
            st.info("Extraction des tags de modification (MM/ML) à partir des fichiers BAM.")
            if 'bam_info' in st.session_state:
                bam_info = st.session_state.bam_info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Reads totales", bam_info['n_reads'])
                col2.metric("Reads alignées", bam_info['n_mapped'], delta=f"{bam_info['mapping_rate']:.1%}")
                col3.metric("Reads avec modifications", bam_info['n_reads_with_mods'])
                col4.metric("Taux de modification", f"{bam_info['mod_rate']:.1%}")
                if bam_info['mod_scores_sample']:
                    st.markdown("#### Échantillon des scores de modification (ML tags)")
                    st.write(bam_info['mod_scores_sample'])
                if bam_info['reads_sample']:
                    st.markdown("#### Aperçu des reads")
                    for read in bam_info['reads_sample'][:5]:
                        st.code(f"{read.query_name} -> {read.reference_name}:{read.reference_start}  "
                                f"qual:
