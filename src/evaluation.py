"""
Evaluation Metrics for COB-ML Framework

Implements metrics from the paper:
- MCC (Matthews Correlation Coefficient)
- Macro-F1
- Cohen's Kappa
- Per-class Precision/Recall/F1
- ICC (Intraclass Correlation)
- Cronbach's Alpha

Author: Calvin Nobles, Samson Quaye
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    matthews_corrcoef, f1_score, cohen_kappa_score,
    precision_recall_fscore_support, roc_auc_score,
    confusion_matrix
)
from scipy.stats import pearsonr
import pingouin as pg


def compute_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Compute full set of classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional)
        
    Returns:
        Dictionary of metrics
    """
    # Core metrics
    mcc = matthews_corrcoef(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1, 2]  # Low, Med, High
    )
    
    metrics = {
        'MCC': mcc,
        'Macro-F1': macro_f1,
        'Kappa': kappa,
        'P_Low': precision[0],
        'R_Low': recall[0],
        'F1_Low': f1[0],
        'P_Med': precision[1],
        'R_Med': recall[1],
        'F1_Med': f1[1],
        'P_High': precision[2],
        'R_High': recall[2],
        'F1_High': f1[2],
    }
    
    # AUROC if probabilities provided
    if y_prob is not None:
        try:
            auroc = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
            metrics['AUROC'] = auroc
        except:
            metrics['AUROC'] = np.nan
    
    return metrics


def compute_icc(df, targets, features, icc_type='ICC2'):
    """
    Compute Intraclass Correlation Coefficient.
    
    Used for construct validation against physiological data.
    
    Args:
        df: DataFrame with raters and targets
        targets: Column name for ratings
        features: Column name for rater IDs
        icc_type: Type of ICC ('ICC1', 'ICC2', 'ICC3')
        
    Returns:
        ICC value
    """
    try:
        icc_result = pg.intraclass_corr(
            data=df, targets=targets, raters=features, ratings='ratings'
        )
        icc_row = icc_result[icc_result['Type'] == icc_type]
        return icc_row['ICC'].values[0] if len(icc_row) > 0 else np.nan
    except:
        return np.nan


def compute_cronbach_alpha(df, items):
    """
    Compute Cronbach's Alpha for internal consistency.
    
    Args:
        df: DataFrame
        items: List of column names (items in scale)
        
    Returns:
        Alpha value
    """
    try:
        return pg.cronbach_alpha(data=df[items])[0]
    except:
        return np.nan


def print_results_table(results_dict):
    """
    Print formatted results table.
    
    Args:
        results_dict: Dict of {model_name: metrics_dict}
    """
    df = pd.DataFrame(results_dict).T
    
    cols = ['MCC', 'Macro-F1', 'Kappa', 'AUROC',
            'P_High', 'R_High', 'F1_High',
            'P_Low', 'R_Low', 'F1_Low',
            'P_Med', 'R_Med', 'F1_Med']
    
    available_cols = [c for c in cols if c in df.columns]
    
    print("\n" + "="*80)
    print("RESULTS TABLE")
    print("="*80)
    print(df[available_cols].round(3).to_string())
    print("="*80 + "\n")
