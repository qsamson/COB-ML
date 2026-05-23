# Data Availability Statement

This document provides detailed instructions for accessing all five datasets used in the COB-ML framework.

---

## 📊 Dataset Overview

| Dataset | Size | Type | Access | License |
|---------|------|------|--------|---------|
| Stack Overflow Developer Survey | 70,673 records | Survey (CSV) | Public | CC BY-SA 4.0 |
| WildChat-1M | 837,989 conversations | Text (Parquet) | Public | ODC-By 1.0 |
| LMSYS-Chat-1M | 1,000,000 conversations | Text (Parquet) | Public | CC BY 4.0 |
| WESAD | 55,154 windows | Physiological (PKL) | Public | Attribution 4.0 |
| STEW | 14,208 windows | EEG (TXT) | Public | CC BY 4.0 |

**Total Dataset Size:** ~4.2 GB (compressed)

---

## 1️⃣ Stack Overflow Developer Survey (2024–2025)

### Description
Annual survey of professional developers covering AI tool usage, trust calibration, verification habits, and occupational demographics.

### Access Method

```python
import pandas as pd

# Load 2024 survey
df_2024 = pd.read_csv(
    "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2024/results.csv",
    low_memory=False
)
df_2024["survey_year"] = 2024

# Load 2025 survey
df_2025 = pd.read_csv(
    "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2025/results.csv",
    low_memory=False
)
df_2025["survey_year"] = 2025
```

### Citation
```bibtex
@misc{stackoverflow2024,
  title={Stack Overflow Developer Survey 2024},
  author={{Stack Overflow}},
  year={2024},
  url={https://survey.stackoverflow.co/2024/}
}
```

---

## 2️⃣ WildChat-1M

### Description
Real ChatGPT conversations from 837,989 naturalistic user sessions.

### Access Method

```python
from datasets import load_dataset

# Load dataset (~2.5 GB download)
wildchat = load_dataset("allenai/WildChat-1M", split="train")
```

**Web:** [HuggingFace Dataset](https://huggingface.co/datasets/allenai/WildChat-1M)

### Citation
```bibtex
@inproceedings{zhao2024wildchat,
  title={WildChat: 1M ChatGPT Interaction Logs in the Wild},
  author={Zhao, Wenting and others},
  booktitle={ICLR},
  year={2024}
}
```

---

## 3️⃣ LMSYS-Chat-1M

### Description
One million conversations across 25 language models.

### Access Method

```python
from datasets import load_dataset

# Load dataset (~3.2 GB download)
lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train")
```

**Web:** [HuggingFace Dataset](https://huggingface.co/datasets/lmsys/lmsys-chat-1m)

### Citation
```bibtex
@article{zheng2023lmsys,
  title={LMSYS-Chat-1M: A Large-Scale Real-World LLM Conversation Dataset},
  author={Zheng, Lianmin and others},
  journal={arXiv:2309.11998},
  year={2023}
}
```

---

## 4️⃣ WESAD (Wearable Stress and Affect Detection)

### Description
Multimodal physiological recordings from 15 subjects under validated stress conditions.

### Access Method

**Direct Download:**
```bash
wget https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download -O WESAD.zip
unzip WESAD.zip
```

**Kaggle Alternative:**
```bash
kaggle datasets download -d orvile/wesad-wearable-stress-affect-detection-dataset
```

### Loading in Python

```python
import pickle

def load_wesad_subject(subject_id='S2'):
    with open(f'WESAD/WESAD/{subject_id}/{subject_id}.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

# Structure:
# data['signal']['chest'] → EDA, ECG, EMG, Temp, Resp
# data['signal']['wrist'] → ACC, BVP, EDA, TEMP
# data['label'] → stress labels
```

### Citation
```bibtex
@inproceedings{schmidt2018wesad,
  title={Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection},
  author={Schmidt, Philip and Reiss, Attila and Duerichen, Robert and Marberger, Claus and Van Laerhoven, Kristof},
  booktitle={ICMI},
  pages={400--408},
  year={2018}
}
```

---

## 5️⃣ STEW (Simultaneous Task EEG Workload Dataset)

### Description
EEG recordings from 48 subjects during rest and high-workload conditions.

### Access Method

**IEEE DataPort (requires free account):**
1. Visit: [https://dx.doi.org/10.21227/44r8-ya50](https://dx.doi.org/10.21227/44r8-ya50)
2. Create free IEEE DataPort account
3. Download: `STEW Dataset.zip` (~380 MB)

### Loading in Python

```python
import numpy as np

# Load EEG data (14 channels × N samples)
eeg_data = np.loadtxt('STEW Dataset/sub01_lo.txt')

# Sampling rate: 128 Hz
# Window size: 256 samples (2 seconds)
```

### Citation
```bibtex
@article{lim2018stew,
  title={STEW: Simultaneous Task EEG Workload Data Set},
  author={Lim, Wei Lun and Sourina, Olga and Wang, Lipo},
  journal={IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  volume={26},
  number={11},
  pages={2106--2114},
  year={2018}
}
```

---

## 🔧 Pre-Processing Pipeline

All preprocessing steps are documented in: `notebooks/COB_ML_Experiments.ipynb`

### Processing Summary

**Stack Overflow:**
- 6-layer feature engineering
- COI proxy score computation
- Tertile-based labeling (Low/Medium/High)

**WildChat & LMSYS:**
- Turn-count extraction
- Length ratio: `user_input_length / ai_output_length`
- Single-turn acceptance flagging
- Model tier encoding (GPT-4=3, GPT-3.5=2, Open-source=1)

**WESAD:**
- Sliding windows (700 samples, 50% overlap)
- Statistical features: mean, std, range, skewness
- Subject-wise splitting

**STEW:**
- Sliding windows (256 samples, 50% overlap)  
- 14-channel EEG feature extraction
- Balanced rest vs. high-workload conditions

---

## 📧 Data Issues or Questions?

Contact the authors:
- **Calvin Nobles** – cn8972@gmail.com
- **Samson Quaye** – squaye@hawk.illinoistech.edu
