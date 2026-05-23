"""
Data Loading Module for COB-ML Framework

Loads and combines data from 5 sources:
1. Stack Overflow Developer Survey (2024-2025)
2. WildChat-1M (ChatGPT conversations)
3. LMSYS-Chat-1M (Multi-model conversations)
4. WESAD (Physiological stress data)
5. STEW (EEG workload data)

Author: Calvin Nobles, Samson Quaye
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from datasets import load_dataset


# ============================================================================
# STACK OVERFLOW DEVELOPER SURVEY
# ============================================================================

# Column definitions for COI-relevant features
SHARED_COI_COLUMNS = [
    "ResponseId", "survey_year",
    # Layer 1: Participant attributes
    "Age", "EdLevel", "YearsCode", "WorkExp", "DevType",
    "OrgSize", "ICorPM", "RemoteWork", "Employment",
    "Country", "Industry",
    # Layer 6: AI trust and reliance (core COI proxies)
    "AISelect", "AISent", "AIAcc", "AIComplex", "AIThreat",
    "JobSat", "ConvertedCompYearly",
]

COI_2024_ONLY = [
    "AIBen", "AIEthics", "AIChallenges",
    "AIToolCurrently Using",
    "AINextMuch more integrated", "AINextMore integrated", "AINextNo change",
    "Frequency_1", "Frequency_2", "Frequency_3",
    "Knowledge_1", "Knowledge_2", "Knowledge_3",
    "Knowledge_4", "Knowledge_5", "Knowledge_6",
    "Knowledge_7", "Knowledge_8",
    "TimeSearching", "TimeAnswering",
    "Frustration",
]

COI_2025_ONLY = [
    "AIHuman",  # KEY: Human vs AI balance → direct offloading measure
    "AIOpen", "AIExplain", "AIFrustration",
    "LearnCodeAI", "AILearnHow",
    "ToolCountWork", "ToolCountPersonal",
    "AIToolCurrently mostly AI", "AIToolCurrently partially AI",
    "AIToolPlan to mostly use AI",
    "AIAgents", "AIAgentChange",
    "AIModelsHaveWorkedWith", "AIAgentKnowledge",
    "SOFriction",
]


def safe_extract(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Extract only available columns from dataframe."""
    available = [c for c in cols if c in df.columns]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"  Missing {len(missing)} cols: {missing[:5]}...")
    return df[available].copy()


def load_stack_overflow_surveys() -> pd.DataFrame:
    """
    Load and combine Stack Overflow Developer Survey data from 2024 and 2025.
    
    Returns:
        Combined dataframe with year indicator and COI-relevant columns.
    """
    print("Loading Stack Overflow Developer Surveys...")
    
    # Load 2024
    url_2024 = "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2024/results.csv"
    df_2024 = pd.read_csv(url_2024, low_memory=False)
    df_2024["survey_year"] = 2024
    
    # Load 2025
    url_2025 = "https://github.com/StackExchange/Survey/raw/refs/heads/main/packages/archive/2025/results.csv"
    df_2025 = pd.read_csv(url_2025, low_memory=False)
    df_2025["survey_year"] = 2025
    
    print(f"  2024: {df_2024.shape} | 2025: {df_2025.shape}")
    
    # Extract COI-relevant columns
    print("\n── 2024 COI extraction ──")
    df_2024_coi = safe_extract(df_2024, SHARED_COI_COLUMNS + COI_2024_ONLY)
    
    print("\n── 2025 COI extraction ──")
    df_2025_coi = safe_extract(df_2025, SHARED_COI_COLUMNS + COI_2025_ONLY)
    
    # Combine with outer join (NaN for year-specific columns is expected)
    df_combined = pd.concat([df_2024_coi, df_2025_coi], ignore_index=True)
    
    print(f"\nCombined: {df_combined.shape}")
    print(f"Total respondents: {len(df_combined):,}")
    
    return df_combined


# ============================================================================
# WILDCHAT-1M (ChatGPT Conversations)
# ============================================================================

def extract_wildchat_features(example: dict) -> dict:
    """Extract behavioral features from a WildChat conversation."""
    conv = example['conversation']
    
    user_msgs = [t for t in conv if t['role'] == 'user']
    asst_msgs = [t for t in conv if t['role'] == 'assistant']
    
    if not user_msgs or not asst_msgs:
        return {
            'conversation_id': example.get('conversation_hash', ''),
            'turn_count': 0,
            'user_total_length': 0,
            'assistant_total_length': 0,
            'length_ratio': 0,
            'single_turn_acceptance': 0,
            'model': example.get('model', ''),
        }
    
    user_total_len = sum(len(m['content']) for m in user_msgs)
    asst_total_len = sum(len(m['content']) for m in asst_msgs)
    
    return {
        'conversation_id': example.get('conversation_hash', ''),
        'turn_count': len(user_msgs),
        'user_total_length': user_total_len,
        'assistant_total_length': asst_total_len,
        'length_ratio': user_total_len / asst_total_len if asst_total_len > 0 else 0,
        'single_turn_acceptance': int(len(user_msgs) == 1),  # One question, accept answer
        'model': example.get('model', ''),
    }


def load_wildchat(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load WildChat-1M conversation dataset.
    
    Args:
        max_samples: Limit number of conversations loaded (for testing).
        
    Returns:
        DataFrame with conversation-level behavioral features.
    """
    print("Loading WildChat-1M...")
    wc = load_dataset("allenai/WildChat-1M", split="train")
    
    if max_samples:
        wc = wc.select(range(min(max_samples, len(wc))))
    
    print(f"  Loaded: {len(wc):,} conversations")
    
    # Extract features
    print("  Extracting conversation features...")
    features = [extract_wildchat_features(ex) for ex in wc]
    df = pd.DataFrame(features)
    
    print(f"  Features extracted: {df.shape}")
    return df


# ============================================================================
# LMSYS-CHAT-1M (Multi-model Conversations)
# ============================================================================

def extract_lmsys_features(example: dict) -> dict:
    """Extract behavioral features from LMSYS conversation."""
    conv = example.get('conversation', [])
    
    user_msgs = [t for t in conv if t.get('role') == 'user']
    asst_msgs = [t for t in conv if t.get('role') == 'assistant']
    
    if not user_msgs or not asst_msgs:
        return {
            'conversation_id': example.get('conversation_id', ''),
            'turn_count': 0,
            'model': example.get('model', ''),
            'language': example.get('language', ''),
        }
    
    return {
        'conversation_id': example.get('conversation_id', ''),
        'turn_count': len(user_msgs),
        'model': example.get('model', ''),
        'language': example.get('language', 'en'),
    }


def load_lmsys_chat(max_samples: Optional[int] = None) -> pd.DataFrame:
    """
    Load LMSYS-Chat-1M dataset.
    
    Args:
        max_samples: Limit number of conversations loaded (for testing).
        
    Returns:
        DataFrame with conversation metadata.
    """
    print("Loading LMSYS-Chat-1M...")
    lmsys = load_dataset("lmsys/lmsys-chat-1m", split="train")
    
    if max_samples:
        lmsys = lmsys.select(range(min(max_samples, len(lmsys))))
    
    print(f"  Loaded: {len(lmsys):,} conversations")
    
    # Extract features
    features = [extract_lmsys_features(ex) for ex in lmsys]
    df = pd.DataFrame(features)
    
    print(f"  Features extracted: {df.shape}")
    return df


# ============================================================================
# WESAD (Physiological Stress Data)
# ============================================================================

def load_wesad(data_path: str = "WESAD/") -> pd.DataFrame:
    """
    Load WESAD physiological stress dataset.
    
    Args:
        data_path: Path to extracted WESAD folder.
        
    Returns:
        DataFrame with windowed physiological features.
        
    Note:
        Requires downloading WESAD from:
        https://uni-siegen.sciebo.de/s/pYjSgfOVs6Ntahr/download
    """
    import pickle
    from pathlib import Path
    
    print("Loading WESAD dataset...")
    
    subjects = [f"S{i}" for i in range(2, 18) if i != 12]  # S12 excluded per paper
    all_windows = []
    
    for subject_id in subjects:
        pkl_path = Path(data_path) / "WESAD" / subject_id / f"{subject_id}.pkl"
        
        if not pkl_path.exists():
            print(f"  Skipping {subject_id} (file not found)")
            continue
            
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # Extract chest signals (EDA, ECG, EMG, Temp, Resp)
        chest = data['signal']['chest']
        labels = data['label']
        
        # Sliding window: 700 samples (7 seconds at 100Hz), 50% overlap
        window_size = 700
        step = window_size // 2
        
        for i in range(0, len(labels) - window_size, step):
            window_label = labels[i:i+window_size].mode()[0] if len(labels[i:i+window_size]) > 0 else 0
            
            # Statistical features per window
            window_features = {
                'subject': subject_id,
                'window_idx': i // step,
                'label': window_label,
            }
            
            for signal_name in ['EDA', 'ECG', 'EMG', 'Temp', 'Resp']:
                if signal_name in chest:
                    signal = chest[signal_name][i:i+window_size]
                    window_features[f'{signal_name}_mean'] = np.mean(signal)
                    window_features[f'{signal_name}_std'] = np.std(signal)
                    window_features[f'{signal_name}_range'] = np.ptp(signal)
                    
            all_windows.append(window_features)
    
    df = pd.DataFrame(all_windows)
    print(f"  Loaded: {len(df):,} windows from {len(subjects)} subjects")
    
    return df


# ============================================================================
# STEW (EEG Workload Data)
# ============================================================================

def load_stew(data_path: str = "STEW Dataset/") -> pd.DataFrame:
    """
    Load STEW EEG workload dataset.
    
    Args:
        data_path: Path to STEW Dataset folder.
        
    Returns:
        DataFrame with windowed EEG features.
        
    Note:
        Download from IEEE DataPort: https://dx.doi.org/10.21227/44r8-ya50
    """
    from pathlib import Path
    
    print("Loading STEW dataset...")
    
    all_windows = []
    data_dir = Path(data_path)
    
    # STEW has files like: sub01_lo.txt, sub01_hi.txt (low/high workload)
    for filepath in data_dir.glob("sub*_*.txt"):
        subject_id = filepath.stem.split('_')[0]
        workload = 1 if 'hi' in filepath.stem else 0  # 1=high, 0=low
        
        # Load EEG data (14 channels × N samples, 128 Hz)
        eeg_data = np.loadtxt(filepath)
        
        # Sliding window: 256 samples (2 seconds), 50% overlap
        window_size = 256
        step = window_size // 2
        
        for i in range(0, eeg_data.shape[1] - window_size, step):
            window = eeg_data[:, i:i+window_size]
            
            # Statistical features per channel
            window_features = {
                'subject': subject_id,
                'window_idx': i // step,
                'workload': workload,
            }
            
            for ch in range(14):
                window_features[f'ch{ch+1}_mean'] = np.mean(window[ch])
                window_features[f'ch{ch+1}_std'] = np.std(window[ch])
                window_features[f'ch{ch+1}_range'] = np.ptp(window[ch])
            
            all_windows.append(window_features)
    
    df = pd.DataFrame(all_windows)
    print(f"  Loaded: {len(df):,} windows")
    
    return df


# ============================================================================
# MAIN LOADING FUNCTION
# ============================================================================

def load_all_datasets(
    load_so: bool = True,
    load_wc: bool = False,
    load_lmsys: bool = False,
    load_wesad: bool = False,
    load_stew: bool = False,
    wc_max: Optional[int] = None,
    lmsys_max: Optional[int] = None,
) -> dict:
    """
    Load all COB-ML datasets.
    
    Args:
        load_so: Load Stack Overflow surveys
        load_wc: Load WildChat
        load_lmsys: Load LMSYS-Chat
        load_wesad: Load WESAD (requires local files)
        load_stew: Load STEW (requires local files)
        wc_max: Max WildChat conversations
        lmsys_max: Max LMSYS conversations
        
    Returns:
        Dictionary of DataFrames {dataset_name: df}
    """
    datasets = {}
    
    if load_so:
        datasets['stack_overflow'] = load_stack_overflow_surveys()
    
    if load_wc:
        datasets['wildchat'] = load_wildchat(max_samples=wc_max)
    
    if load_lmsys:
        datasets['lmsys'] = load_lmsys_chat(max_samples=lmsys_max)
    
    if load_wesad:
        try:
            datasets['wesad'] = load_wesad()
        except Exception as e:
            print(f"WESAD loading failed: {e}")
    
    if load_stew:
        try:
            datasets['stew'] = load_stew()
        except Exception as e:
            print(f"STEW loading failed: {e}")
    
    print(f"\n══════════════════════════════════════")
    print(f"Loaded {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"  {name}: {df.shape}")
    print(f"══════════════════════════════════════\n")
    
    return datasets
