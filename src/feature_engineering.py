"""
Feature Engineering Module for COB-ML Framework

Implements the 6-layer behavioral feature architecture:
Layer 1: Participant attributes (demographics, experience)
Layer 2: Work context (organization, role, environment)
Layer 3: Task properties (complexity, frequency)
Layer 4: AI system characteristics (model tier, tools used)
Layer 5: Interaction traces (usage patterns, verification behavior)
Layer 6: Human factors states (trust, sentiment, frustration)

Author: Calvin Nobles, Samson Quaye
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ============================================================================
# LAYER 1: PARTICIPANT ATTRIBUTES
# ============================================================================

def encode_participant_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode demographic and professional experience features.
    
    Features:
    - Age (ordinal)
    - Education level (ordinal)
    - Years coding experience
    - Years professional experience
    - Developer type (multi-select encoding)
    """
    df_feat = df.copy()
    
    # Age → ordinal encoding
    age_map = {
        'Under 18 years old': 0, '18-24 years old': 1, '25-34 years old': 2,
        '35-44 years old': 3, '45-54 years old': 4, '55-64 years old': 5,
        '65 years or older': 6, 'Prefer not to say': np.nan
    }
    df_feat['age_ord'] = df_feat['Age'].map(age_map)
    
    # Education → ordinal
    ed_map = {
        'I never completed any formal education': 0,
        'Primary/elementary school': 1,
        'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)': 2,
        'Some college/university study without earning a degree': 3,
        'Associate degree (A.A., A.S., etc.)': 4,
        "Bachelor's degree (B.A., B.S., B.Eng., etc.)": 5,
        "Master's degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
        'Professional degree (JD, MD, Ph.D, Ed.D, etc.)': 7,
        'Something else': np.nan
    }
    df_feat['ed_level_ord'] = df_feat['EdLevel'].map(ed_map)
    
    # Years coding / work experience → numeric
    def parse_years(val):
        if pd.isna(val): return np.nan
        if 'Less than 1 year' in str(val): return 0.5
        if 'More than 50 years' in str(val): return 55
        try:
            return float(str(val).split()[0])
        except:
            return np.nan
    
    df_feat['years_code_num'] = df_feat['YearsCode'].apply(parse_years)
    df_feat['years_work_num'] = df_feat['WorkExp'].apply(parse_years)
    
    return df_feat


# ============================================================================
# LAYER 2: WORK CONTEXT
# ============================================================================

def encode_work_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode organizational and role context.
    
    Features:
    - Organization size (ordinal)
    - Individual contributor vs manager (binary)
    - Remote work status (ordinal)
    - Employment type (categorical)
    """
    df_feat = df.copy()
    
    # Organization size → ordinal
    org_map = {
        'Just me - I am a freelancer, sole proprietor, etc.': 0,
        '2 to 9 employees': 1, '10 to 19 employees': 2,
        '20 to 99 employees': 3, '100 to 499 employees': 4,
        '500 to 999 employees': 5, '1,000 to 4,999 employees': 6,
        '5,000 to 9,999 employees': 7, '10,000 or more employees': 8,
        "I don't know": np.nan
    }
    df_feat['org_size_ord'] = df_feat['OrgSize'].map(org_map)
    
    # IC vs Manager → binary
    df_feat['is_manager'] = df_feat['ICorPM'].apply(
        lambda x: 1 if 'manager' in str(x).lower() else 0
    )
    
    # Remote work → ordinal (less remote = 0, more remote = higher)
    remote_map = {
        'In-person': 0, 'Hybrid (some remote, some in-person)': 1, 'Remote': 2
    }
    df_feat['remote_ord'] = df_feat['RemoteWork'].map(remote_map)
    
    return df_feat


# ============================================================================
# LAYER 3: TASK PROPERTIES
# ============================================================================

def encode_task_properties(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode task complexity and frequency signals.
    
    Features:
    - AI task complexity rating
    - Complexity avoidance flag
    - Frustration indicators
    """
    df_feat = df.copy()
    
    # Already encoded in preprocessing, but ensure present
    if 'ai_complex_rating' not in df_feat.columns and 'AIComplex' in df_feat.columns:
        complex_map = {
            'Very poor at handling complex tasks': -2,
            'Bad at handling complex tasks': -1,
            'Neither good or bad at handling complex tasks': 0,
            'Good, but not great at handling complex tasks': 1,
            'Very well at handling complex tasks': 2,
            "I don't use AI tools for complex tasks / I don't know": np.nan
        }
        df_feat['ai_complex_rating'] = df_feat['AIComplex'].map(complex_map)
        df_feat['ai_complex_avoidance'] = (
            df_feat['AIComplex'] == "I don't use AI tools for complex tasks / I don't know"
        ).astype(int)
    
    return df_feat


# ============================================================================
# LAYER 4: AI SYSTEM CHARACTERISTICS
# ============================================================================

def encode_ai_system_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode AI tool and model characteristics.
    
    Features:
    - Model tier (capability encoding)
    - Tool count (work vs personal)
    - AI agent usage
    """
    df_feat = df.copy()
    
    # Tool count features (2025 only)
    if 'ToolCountWork' in df_feat.columns:
        def parse_tool_count(val):
            if pd.isna(val): return np.nan
            if 'None' in str(val): return 0
            if '1-2' in str(val): return 1.5
            if '3-5' in str(val): return 4
            if '6-10' in str(val): return 8
            if 'More than 10' in str(val): return 12
            return np.nan
        
        df_feat['tool_count_work_num'] = df_feat['ToolCountWork'].apply(parse_tool_count)
        if 'ToolCountPersonal' in df_feat.columns:
            df_feat['tool_count_personal_num'] = df_feat['ToolCountPersonal'].apply(parse_tool_count)
    
    # AI agent usage flag (2025 only)
    if 'AIAgents' in df_feat.columns:
        df_feat['uses_ai_agents'] = df_feat['AIAgents'].apply(
            lambda x: 1 if 'Yes' in str(x) else 0
        )
    
    return df_feat


# ============================================================================
# LAYER 5: INTERACTION TRACES
# ============================================================================

def encode_interaction_traces(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode behavioral interaction patterns.
    
    Features:
    - AI usage frequency (ordinal)
    - Verification behavior flags
    - Full dependency indicator (maladaptive signal)
    """
    df_feat = df.copy()
    
    # AI usage frequency → ordinal (0=no use, 4=daily)
    if 'ai_usage_freq' not in df_feat.columns and 'AISelect' in df_feat.columns:
        usage_map = {
            'No, and I don\'t plan to': 0,
            'No, but I plan to soon': 1,
            'Yes, I use AI tools monthly or infrequently': 2,
            'Yes, I use AI tools weekly': 3,
            'Yes, I use AI tools daily': 4,
            'Yes': 3,  # 2024 generic "Yes" → active user
        }
        df_feat['ai_usage_freq'] = df_feat['AISelect'].map(usage_map)
    
    # Verification and dependency flags from AIHuman parsing (done in preprocessing)
    # Ensure they exist
    for col in ['trust_verification_flag', 'learning_verification_flag', 'ai_full_dependency']:
        if col not in df_feat.columns:
            df_feat[col] = np.nan
    
    return df_feat


# ============================================================================
# LAYER 6: HUMAN FACTORS STATES
# ============================================================================

def encode_human_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode cognitive and attitudinal states.
    
    Features:
    - AI sentiment (ordinal: -2 to +2)
    - AI trust score (ordinal: -2 to +2)
    - AI threat perception (binary)
    - Frustration count
    """
    df_feat = df.copy()
    
    # Sentiment → ordinal (-2 to +2)
    if 'ai_sentiment' not in df_feat.columns and 'AISent' in df_feat.columns:
        sent_map = {
            'Very unfavorable': -2, 'Unfavorable': -1, 'Indifferent': 0,
            'Unsure': 0, 'Favorable': 1, 'Very favorable': 2
        }
        df_feat['ai_sentiment'] = df_feat['AISent'].map(sent_map)
    
    # Trust score → ordinal (-2 to +2)
    if 'ai_trust_score' not in df_feat.columns and 'AIAcc' in df_feat.columns:
        acc_map = {
            'Highly distrust': -2, 'Somewhat distrust': -1,
            'Neither trust nor distrust': 0,
            'Somewhat trust': 1, 'Highly trust': 2
        }
        df_feat['ai_trust_score'] = df_feat['AIAcc'].map(acc_map)
    
    # Threat perception → binary
    if 'ai_threat_perceived' not in df_feat.columns and 'AIThreat' in df_feat.columns:
        threat_map = {'Yes': 1, 'No': 0, "I'm not sure": 0}
        df_feat['ai_threat_perceived'] = df_feat['AIThreat'].map(threat_map)
    
    return df_feat


# ============================================================================
# AIHuman MULTI-SELECT PARSING (2025 ONLY - KEY OFFLOADING SIGNALS)
# ============================================================================

def parse_aihuman_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse AIHuman multi-select column into structured features.
    
    Critical features extracted:
    - human_help_count: How many conditions still trigger human help
    - ai_full_dependency: "I don't need humans anymore" flag (MALADAPTIVE)
    - trust_verification_flag: Seeks human when distrusts AI (ADAPTIVE)
    - learning_verification_flag: Seeks human to learn (ADAPTIVE)
    """
    def parse_single(val):
        if pd.isna(val):
            return pd.Series({
                'human_help_count': np.nan,
                'ai_full_dependency': np.nan,
                'trust_verification_flag': np.nan,
                'learning_verification_flag': np.nan,
            })
        
        # Normalize curly apostrophes to straight
        normalized = str(val).replace('\u2019', "'").replace('\u2018', "'")
        items = [x.strip() for x in normalized.split(';')]
        
        return pd.Series({
            'human_help_count': len([
                i for i in items
                if "I don't think" not in i and i != 'Other (please specify):'
            ]),
            'ai_full_dependency': int(
                "I don't think I'll need help from people anymore" in items
            ),
            'trust_verification_flag': int(
                "When I don't trust AI's answers" in items
            ),
            'learning_verification_flag': int(
                "When I want to fully understand something" in items
                or "When I want to learn best practices" in items
            ),
        })
    
    if 'AIHuman' in df.columns:
        aihuman_features = df['AIHuman'].apply(parse_single)
        return pd.concat([df, aihuman_features], axis=1)
    
    return df


# ============================================================================
# COI PROXY SCORE COMPUTATION
# ============================================================================

def compute_coi_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Cognitive Offloading Index (COI) proxy score.
    
    Formula (weighted combination):
        COI = 0.30 × ai_usage_freq
            + 0.25 × ai_trust_score
            + 0.20 × ai_complex_rating
            + 2.00 × ai_full_dependency         (strong maladaptive signal)
            + 1.50 × (1 - trust_verification)   (absence of verification)
    
    Higher COI → Greater tendency toward cognitive offloading
    """
    df_feat = df.copy()
    
    df_feat['coi_proxy'] = (
        df_feat['ai_usage_freq'].fillna(0) * 0.30 +
        df_feat['ai_trust_score'].fillna(0) * 0.25 +
        df_feat['ai_complex_rating'].fillna(0) * 0.20 +
        df_feat['ai_full_dependency'].fillna(0) * 2.00 +
        (1 - df_feat['trust_verification_flag'].fillna(1)) * 1.50
    )
    
    return df_feat


def assign_offloading_labels(df: pd.DataFrame, min_features: int = 2) -> pd.DataFrame:
    """
    Assign tertile-based offloading level labels (Low / Medium / High).
    
    Only labels rows with sufficient feature coverage.
    
    Args:
        df: DataFrame with COI features
        min_features: Minimum number of core features required for labeling
        
    Returns:
        DataFrame with 'offloading_level' column
    """
    df_feat = df.copy()
    
    # Core features for COI computation
    core_features = ['ai_usage_freq', 'ai_trust_score', 'ai_complex_rating']
    
    # Count non-null core features
    df_feat['core_coverage'] = df_feat[core_features].notna().sum(axis=1)
    
    # Only label rows with sufficient coverage
    mask = df_feat['core_coverage'] >= min_features
    labeled = df_feat[mask].copy()
    
    # Tertile-based labeling (q33, q67 thresholds)
    labeled['offloading_level'] = pd.qcut(
        labeled['coi_proxy'],
        q=3,
        labels=['Low', 'Medium', 'High']
    )
    
    # Merge back
    df_feat.loc[mask, 'offloading_level'] = labeled['offloading_level']
    
    return df_feat


# ============================================================================
# MASTER FEATURE ENGINEERING PIPELINE
# ============================================================================

def engineer_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply complete 6-layer feature engineering pipeline.
    
    Pipeline:
    1. Parse AIHuman multi-select
    2. Layer 1: Participant attributes
    3. Layer 2: Work context
    4. Layer 3: Task properties
    5. Layer 4: AI system characteristics
    6. Layer 5: Interaction traces
    7. Layer 6: Human factors states
    8. Compute COI proxy score
    9. Assign offloading level labels
    
    Returns:
        DataFrame with all engineered features
    """
    print("Applying 6-layer feature engineering...")
    
    df_eng = df.copy()
    
    # Parse AIHuman first (2025-specific)
    print("  Parsing AIHuman multi-select...")
    df_eng = parse_aihuman_column(df_eng)
    
    # Apply layer encodings
    print("  Layer 1: Participant attributes...")
    df_eng = encode_participant_attributes(df_eng)
    
    print("  Layer 2: Work context...")
    df_eng = encode_work_context(df_eng)
    
    print("  Layer 3: Task properties...")
    df_eng = encode_task_properties(df_eng)
    
    print("  Layer 4: AI system characteristics...")
    df_eng = encode_ai_system_features(df_eng)
    
    print("  Layer 5: Interaction traces...")
    df_eng = encode_interaction_traces(df_eng)
    
    print("  Layer 6: Human factors states...")
    df_eng = encode_human_factors(df_eng)
    
    # Compute COI proxy
    print("  Computing COI proxy score...")
    df_eng = compute_coi_proxy(df_eng)
    
    # Assign labels
    print("  Assigning offloading level labels...")
    df_eng = assign_offloading_labels(df_eng)
    
    print(f"  Feature engineering complete: {df_eng.shape}")
    
    return df_eng


def get_feature_columns() -> dict:
    """
    Return dictionary of feature column names by layer.
    
    Returns:
        Dict mapping layer names to feature column lists
    """
    return {
        'layer_1_participant': [
            'age_ord', 'ed_level_ord', 'years_code_num', 'years_work_num'
        ],
        'layer_2_work_context': [
            'org_size_ord', 'is_manager', 'remote_ord'
        ],
        'layer_3_task': [
            'ai_complex_rating', 'ai_complex_avoidance', 'frustration_count'
        ],
        'layer_4_ai_system': [
            'tool_count_work_num', 'tool_count_personal_num', 'uses_ai_agents'
        ],
        'layer_5_interaction': [
            'ai_usage_freq', 'trust_verification_flag',
            'learning_verification_flag', 'ai_full_dependency', 'human_help_count'
        ],
        'layer_6_human_factors': [
            'ai_sentiment', 'ai_trust_score', 'ai_threat_perceived'
        ],
        'target': [
            'coi_proxy', 'offloading_level'
        ]
    }
