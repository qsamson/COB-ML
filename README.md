# COB-ML: Cognitive Offloading Behavioral Machine Learning

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-research-orange.svg)
![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey.svg)
![Code style](https://img.shields.io/badge/code%20style-black-000000.svg)

A multi-source framework for quantifying human cognitive offloading in AI-integrated professional environments through explainable hybrid learning.

**Authors:** Calvin Nobles, Samson Quaye  
**Institution:** University of Maryland Global Campus

---

## Overview

COB-ML introduces a behavioral measurement framework that distinguishes adaptive cognitive support from maladaptive dependency in AI-assisted professional work. The framework combines a six-layer behavioral feature architecture with a three-branch stacking ensemble trained across five real-world datasets.

**Key Contributions:**
- 6-layer behavioral feature architecture operationalizing cognitive offloading from interaction traces
- 3-branch hybrid ensemble unifying semantic, tabular, and sequential modeling
- Physiological validation against WESAD stress data and STEW EEG workload measures
- Evidence that trust disposition, not usage frequency, predicts maladaptive offloading

---

## Datasets

| Dataset | Records | Type | Source |
|---------|---------|------|--------|
| Stack Overflow Survey | 70,673 | Behavioral survey | [stackoverflow.co](https://survey.stackoverflow.co/) |
| WildChat-1M | 837,989 | ChatGPT conversations | [HuggingFace](https://huggingface.co/datasets/allenai/WildChat-1M) |
| LMSYS-Chat-1M | 1,000,000 | Multi-model conversations | [HuggingFace](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) |
| WESAD | 55,154 | Physiological (stress) | [Download](https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download) |
| STEW | 14,208 | EEG (workload) | [IEEE DataPort](https://dx.doi.org/10.21227/44r8-ya50) |

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for detailed access instructions.

---

## Framework Architecture

### 6-Layer Feature Engineering

1. **Participant Attributes** - Demographics, professional experience
2. **Work Context** - Organization size, role, remote work status
3. **Task Properties** - Complexity ratings, avoidance behaviors
4. **AI System Characteristics** - Model tiers, tool usage patterns
5. **Interaction Traces** - Verification behavior, dependency indicators
6. **Human Factors** - Trust, sentiment, threat perception

### 3-Branch Ensemble

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Branch A      │  │   Branch B      │  │   Branch C      │
│   (Semantic)    │  │   (Tabular)     │  │  (Sequential)   │
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
                    │   (Logistic Reg)  │
                    └─────────┬─────────┘
                              │
                         COI Prediction
```

---

## Installation

### Requirements

- Python ≥ 3.9
- PyTorch ≥ 2.0 (CUDA recommended)
- 16GB+ RAM for full dataset processing

### Setup

**Using Conda:**
```bash
git clone https://github.com/qsamson/COB-ML.git
cd COB-ML
conda env create -f environment.yml
conda activate cobml
```

**Using pip:**
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
df = datasets['stack_overflow']
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

# Run complete training pipeline
main()
```

---

## Results Summary

### Model Performance

| Model | MCC | Macro-F1 | Kappa |
|-------|-----|----------|-------|
| Logistic Regression | 0.329 | 0.544 | 0.328 |
| XGBoost | 0.428 | 0.615 | 0.427 |
| LightGBM | 0.431 | 0.617 | 0.430 |
| RoBERTa | 0.512 | 0.747 | 0.511 |
| **COB-ML Ensemble** | **0.577** | **0.762** | **0.576** |

### Top Predictive Features (SHAP)

1. **trust_verification_flag** - Seeks human verification when distrusts AI
2. ai_trust_score - Overall trust in AI accuracy
3. ai_full_dependency - "No longer need humans" indicator
4. ai_usage_freq - Daily/weekly/monthly usage
5. years_code_num - Professional experience

### Physiological Validation

- **WESAD (stress):** ICC(A,1) = 0.611
- **STEW (EEG workload):** ICC(A,1) = 0.964

---

## Repository Structure

```
COB-ML/
├── README.md
├── requirements.txt
├── environment.yml
├── DATA_AVAILABILITY.md
├── LICENSE
├── src/
│   ├── data_loading.py          # Dataset loading
│   ├── feature_engineering.py   # 6-layer architecture
│   ├── evaluation.py            # Metrics
│   └── train.py                 # Training pipeline
└── paper/
    └── figures/                 # Visualization outputs
```

---

## License

Code and implementation: **MIT License**

See [LICENSE](LICENSE) file for details.

---

## Contact

**Calvin Nobles** – cn8972@gmail.com  
**Samson Quaye** – squaye@hawk.illinoistech.edu

For questions about the framework or datasets, please open an issue or contact the authors directly.

---

## Acknowledgments

This research builds upon publicly available datasets:
- Stack Overflow Developer Survey
- WildChat-1M (Allen Institute for AI)
- LMSYS-Chat-1M
- WESAD (University of Siegen)
- STEW (Nanyang Technological University)
