# 🧬 MetaInsight v10 — Plateforme intégrative multi-omique, PGM & Épitranscriptomique

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://metainsight.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**MetaInsight v10** est une plateforme web open-source intégrative pour l'analyse de données multi-omiques, la génomique clinique (PGM), l'épitranscriptomique, et l'inférence causale. Elle permet d'explorer les interactions entre microbiome, variants génétiques et modifications d'ARN dans une interface interactive sans code.

---

## 📋 Table des matières

- [🔬 Fonctionnalités clés](#-fonctionnalités-clés)
- [🏗️ Architecture](#️-architecture)
- [🚀 Installation](#-installation)
- [📁 Structure du projet](#-structure-du-projet)
- [📊 Données de démonstration](#-données-de-démonstration)
- [🤖 IA intégrée](#-ia-intégrée)
- [🧪 Exemple d'utilisation](#-exemple-dutilisation)
- [🤝 Contributions](#-contributions)
- [📄 Licence](#-licence)
- [🙏 Remerciements](#-remerciements)
- [📧 Contact](#-contact)

---

## 🔬 Fonctionnalités clés

### 🧬 Modules intégrés (27)

| Domaine | Modules |
|---------|---------|
| **Microbiome** | Diversité α/β (Shannon, Simpson, Chao1, Faith PD), PERMANOVA, ALDEx2, LEfSe, MaAsLin2, CLR, raréfaction, biomarqueurs ROC |
| **PGM (Génomique clinique)** | BRCA1/2, pharmacogénétique (CPIC), lollipop plots, scores CADD/PolyPhen, panel ACMG étendu (59 gènes) |
| **Épitranscriptomique** | Profils de modification, motifs consensus (DRACH, UGU, CG), réseaux de crosstalk, heatmaps, prédiction d'impact, apprentissage incrémental |
| **Analyse brute FASTQ** | Pipeline FastQC → Trimmomatic → STAR → featureCounts → DESeq2 |
| **PRS & Génétique population** | Scores polygéniques de risque, PCA génétique, fréquences gnomAD |
| **Inférence causale** | Modélisation par équations structurales (SEM) |
| **IA intégrée** | 5 fournisseurs gratuits (Gemini, Groq, DeepSeek, OpenRouter/Kimi K2, Ollama) |
| **Agent IA autonome** | LangChain pour interroger les données en langage naturel |
| **Annotation RMBase** | Fonctions connues, références, motifs |

### 🖥️ Interface
- Interactive, déployable en un clic sur Streamlit Cloud.
- Visualisations Plotly interactives (heatmaps, réseaux, lollipop, PCA).
- Téléchargement des rapports IA et des figures.

### 📦 Big Data
- Lecture directe des fichiers VCF via DuckDB (out-of-core).
- Support de fichiers jusqu’à **5 Go** (configurable).
- Formats acceptés : CSV, TSV, BIOM, H5AD, VCF, VCF.gz, BAM, FASTQ (métadonnées).

---

## 🏗️ Architecture
