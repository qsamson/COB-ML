# COB-ML: Cognitive Offloading Behavioral Machine Learning

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

A multi-source framework for quantifying human cognitive offloading in AI-integrated professional environments through explainable hybrid learning.

---

## Authors

**Calvin Nobles, PhD**  
Dean and Portfolio Vice President  
School of Cybersecurity and Information Technology  
University of Maryland Global Campus (UMGC)  
ORCID: 0000-0003-4002-1108

**Samson Quaye**  
Ph.D. Student  
Center for Cybersecurity and Forensic Education (C²SAFE)  
Illinois Institute of Technology  
ORCID: 0009-0003-1292-3419

---

## Overview

COB-ML introduces a behavioral measurement framework that distinguishes adaptive cognitive support from maladaptive dependency in AI-assisted professional work. The framework combines a six-layer behavioral feature architecture with a three-branch stacking ensemble trained across five real-world datasets.

**Key Contributions:**

- Six-layer behavioral feature architecture operationalizing cognitive offloading from interaction traces
- Three-branch hybrid ensemble unifying semantic, tabular, and sequential modeling
- Convergent physiological validation against WESAD stress data and STEW EEG workload measures
- Evidence that AI trust disposition, not usage frequency alone, predicts maladaptive offloading risk

---

## Datasets

| Dataset | Records / Windows | Type | Source |
|---------|-------------------|------|--------|
| Stack Overflow Developer Survey | 70,673 | Behavioral survey | [Stack Overflow Survey](https://survey.stackoverflow.co/) |
| WildChat-1M | 837,989 | ChatGPT conversations | [Hugging Face](https://huggingface.co/datasets/allenai/WildChat-1M) |
| LMSYS-Chat-1M | 1,000,000 | Multi-model conversations | [Hugging Face](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |
| WESAD | 55,154 | Physiological stress windows | [University of Siegen](https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download) |
| STEW | 14,208 | EEG workload windows | [IEEE DataPort](https://dx.doi.org/10.21227/44r8-ya50) |

Large external datasets are not redistributed in this repository. See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for dataset access instructions.

---

## Framework Architecture

### Six-Layer Feature Engineering

1. **Participant Attributes** — Demographics and professional experience
2. **Work Context** — Organization size, role, employment type, and remote work status
3. **Task Properties** — AI complexity ratings and cognitive task indicators
4. **AI System Characteristics** — Model tiers and AI tool usage patterns
5. **Interaction Traces** — Verification behavior, turn structure, and dependency indicators
6. **Human Factors** — Trust, sentiment, threat perception, and frustration indicators

### Three-Branch Ensemble

```text
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Branch A      │  │   Branch B      │  │   Branch C      │
│   Semantic      │  │   Tabular       │  │   Sequential    │
├─────────────────┤  ├─────────────────┤  ├─────────────────┤
│ RoBERTa         │  │ XGBoost         │  │ LSTM            │
│ Conversation    │  │ + TabNet        │  │ + 1D-CNN        │
│ embeddings      │  │ Survey features │  │ Physiological   │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┴────────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   Meta-Learner    │
                    │   Logistic Reg.   │
                    └─────────┬─────────┘
                              │
                         COI Prediction
```

---

## Installation

### Requirements

- Python 3.9+
- PyTorch 2.0+ recommended
- 16GB+ RAM for full dataset processing
- CUDA-capable GPU recommended for RoBERTa and deep learning experiments

### Setup with Conda

```bash
git clone https://github.com/qsamson/COB-ML.git
cd COB-ML
conda env create -f environment.yml
conda activate cobml
```

### Setup with pip

```bash
git clone https://github.com/qsamson/COB-ML.git
cd COB-ML
pip install -r requirements.txt
```

---

## Usage

### Load Data

```python
from src.data_loading import load_all_datasets

datasets = load_all_datasets(load_so=True)
df = datasets["stack_overflow"]
```

### Engineer Features

```python
from src.feature_engineering import engineer_all_features

df_features = engineer_all_features(df)
print(f"Engineered features: {df_features.shape}")
```

### Train Models

```python
from src.train import main

main()
```

---

## Reproducibility

To reproduce the main experimental workflow:

```bash
conda env create -f environment.yml
conda activate cobml

python src/data_loading.py
python src/feature_engineering.py
python src/train.py
python src/evaluation.py
```

Dataset downloads and local paths should be configured according to [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md). Raw third-party datasets are not included in this repository.

---

## Results Summary

### H1: Classification Performance on Stack Overflow Behavioral Dataset

Mean performance is reported across ten random seeds. RoBERTa was trained on WildChat-1M conversation text.

| Model | MCC | Macro-F1 | Kappa |
|-------|-----|----------|-------|
| Logistic Regression | 0.329 | 0.544 | 0.328 |
| Random Forest | 0.367 | 0.571 | 0.365 |
| XGBoost | 0.373 | 0.564 | 0.371 |
| LightGBM | 0.371 | 0.575 | 0.370 |
| TabNet | 0.355 | 0.553 | 0.354 |
| LSTM | 0.358 | 0.553 | 0.357 |
| RoBERTa | 0.635 | 0.747 | 0.615 |
| COB-ML Ensemble | 0.373 | 0.577 | 0.371 |
| Occupation-only | 0.070 | 0.375 | 0.069 |

### H3: Cross-Dataset Generalization

The COB-ML ensemble achieves Macro-F1 = 0.577 on the Stack Overflow behavioral classification task and Macro-F1 = 0.762 under the cross-dataset generalization protocol.

| Model | IID Macro-F1 | GroupKFold Macro-F1 | Temporal Macro-F1 | Cross-DS Macro-F1 | ΔF1 |
|-------|--------------|---------------------|-------------------|-------------------|-----|
| Logistic Regression | 0.544 | 0.544 | 0.526 | 0.382 | -0.162 |
| Random Forest | 0.571 | 0.571 | 0.535 | 0.446 | -0.125 |
| XGBoost | 0.564 | 0.563 | 0.467 | 0.446 | -0.118 |
| LightGBM | 0.575 | 0.575 | 0.491 | 0.446 | -0.129 |
| TabNet | 0.553 | 0.553 | 0.531 | 0.498 | -0.055 |
| LSTM | 0.553 | 0.553 | 0.437 | 0.402 | -0.151 |
| RoBERTa | 0.747 | — | 0.747 | 0.747 | 0.000 |
| COB-ML Ensemble | 0.577 | — | 0.469 | 0.762 | +0.185 |

### SHAP-Based Construct Interpretation

The SHAP analysis uses the full construct feature set for interpretation. The strongest overall predictor of cognitive offloading is `ai_trust_score`, indicating that trust disposition toward AI outputs is more informative than usage frequency alone.

The strongest observable interaction-trace predictor is `trust_verification_flag`, which captures whether users seek human verification when they distrust AI output.

Top SHAP features:

1. `ai_trust_score`
2. `ai_complex_rating`
3. `ai_usage_freq`
4. `trust_verification_flag`
5. `ai_full_dependency`

### Physiological Construct Validation

| Dataset | Validation Metric | Result |
|---------|-------------------|--------|
| WESAD | ICC(A,1), subject level | 0.611 |
| STEW | ICC(A,1), window level | 0.964 |
| WESAD | Pearson r | 0.604 |
| STEW | Pearson r | 0.971 |

---

## Paper Figures

The `paper/figures/` directory contains the final figures used in the manuscript:

- Figure 1: Normalized confusion matrices
- Figure 2: SHAP feature importance
- Figure 3: SHAP dependence plot for verification behavior
- Figure 4: Cross-domain generalization results
- Figure 5: ICC physiological validation heatmap
- Figure 6: H4 construct validity summary
- Figure 7: Occupational parity analysis

---

## Repository Structure

```text
COB-ML/
├── README.md
├── requirements.txt
├── environment.yml
├── DATA_AVAILABILITY.md
├── LICENSE
├── src/
│   ├── __init__.py
│   ├── data_loading.py          # Dataset loading and path handling
│   ├── feature_engineering.py   # Six-layer feature architecture
│   ├── evaluation.py            # Metrics, validation, and parity analysis
│   └── train.py                 # Training pipeline
└── paper/
    └── figures/                 # Manuscript visualization outputs
```

---

## License

Code and implementation: **MIT License**

See [LICENSE](LICENSE) for details.

---

## Contact

**Calvin Nobles, PhD**  
Dean and Portfolio Vice President  
School of Cybersecurity and Information Technology  
University of Maryland Global Campus (UMGC)  
Email: calvin.nobles@umgc.edu

**Samson Quaye**  
Ph.D. Student  
Center for Cybersecurity and Forensic Education (C²SAFE)  
Illinois Institute of Technology  
Email: squaye@hawk.illinoistech.edu

For questions about the framework or datasets, please open an issue or contact the authors directly.

---

## Acknowledgments

This research builds upon publicly available datasets:

- Stack Overflow Developer Survey
- WildChat-1M, Allen Institute for AI
- LMSYS-Chat-1M
- WESAD, University of Siegen
- STEW, Nanyang Technological University
