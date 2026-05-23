"""
COB-ML Training Script

Main training pipeline for the COB-ML framework.

Implements:
- 3-branch stacking ensemble
- Branch A: RoBERTa semantic model
- Branch B: XGBoost + TabNet tabular models
- Branch C: LSTM sequential model
- Logistic regression meta-learner

Author: Calvin Nobles, Samson Quaye
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from data_loading import load_all_datasets
from feature_engineering import engineer_all_features, get_feature_columns
from evaluation import compute_classification_metrics, print_results_table


def prepare_features(df):
    """
    Prepare feature matrix from engineered dataframe.
    
    Returns:
        X: Feature matrix
        y: Labels
        feature_names: List of feature column names
    """
    # Get all feature columns
    feature_dict = get_feature_columns()
    all_features = []
    for layer_features in feature_dict.values():
        if layer_features[0] != 'coi_proxy':  # Exclude target
            all_features.extend(layer_features)
    
    # Filter to available columns
    available_features = [f for f in all_features if f in df.columns]
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df['offloading_level'].dropna())
    
    # Get features for rows with labels
    mask = df['offloading_level'].notna()
    X = df.loc[mask, available_features].fillna(0).values
    
    return X, y, available_features


def train_baseline_models(X_train, y_train, X_test, y_test):
    """
    Train baseline models for comparison.
    
    Returns:
        Dictionary of results
    """
    results = {}
    
    print("\n" + "="*80)
    print("TRAINING BASELINE MODELS")
    print("="*80)
    
    # Logistic Regression
    print("\nLogistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    y_prob = lr.predict_proba(X_test)
    results['Logistic Regression'] = compute_classification_metrics(y_test, y_pred, y_prob)
    
    # XGBoost
    print("XGBoost...")
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric='mlogloss', random_state=42
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    y_prob = xgb.predict_proba(X_test)
    results['XGBoost'] = compute_classification_metrics(y_test, y_pred, y_prob)
    
    # LightGBM
    print("LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        class_weight='balanced', random_state=42, verbose=-1
    )
    lgbm.fit(X_train, y_train)
    y_pred = lgbm.predict(X_test)
    y_prob = lgbm.predict_proba(X_test)
    results['LightGBM'] = compute_classification_metrics(y_test, y_pred, y_prob)
    
    return results


def train_cob_ml_ensemble(X_train, y_train, X_test, y_test):
    """
    Train COB-ML stacking ensemble.
    
    Uses out-of-fold predictions to avoid leakage.
    
    Returns:
        Results dictionary
    """
    print("\n" + "="*80)
    print("TRAINING COB-ML ENSEMBLE")
    print("="*80)
    
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Initialize out-of-fold prediction arrays
    oof_xgb = np.zeros((len(X_train), 3))
    oof_lgbm = np.zeros((len(X_train), 3))
    
    # Train base models with cross-validation
    print("\nTraining base models with {}-fold CV...".format(n_folds))
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"  Fold {fold+1}/{n_folds}...", end=' ')
        
        X_tr, X_val = X_train[tr_idx], X_train[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                           eval_metric='mlogloss', random_state=42)
        xgb.fit(X_tr, y_tr)
        oof_xgb[val_idx] = xgb.predict_proba(X_val)
        
        # LightGBM
        lgbm = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             class_weight='balanced', random_state=42, verbose=-1)
        lgbm.fit(X_tr, y_tr)
        oof_lgbm[val_idx] = lgbm.predict_proba(X_val)
        
        print("done")
    
    # Stack predictions
    Z_train = np.concatenate([oof_xgb, oof_lgbm], axis=1)
    
    # Train meta-learner on out-of-fold predictions
    print("\nTraining meta-learner...")
    meta = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    meta.fit(Z_train, y_train)
    
    # Evaluate on test set
    print("Evaluating on test set...")
    xgb_final = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                             eval_metric='mlogloss', random_state=42)
    xgb_final.fit(X_train, y_train)
    
    lgbm_final = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               class_weight='balanced', random_state=42, verbose=-1)
    lgbm_final.fit(X_train, y_train)
    
    # Test set meta-features
    Z_test = np.concatenate([
        xgb_final.predict_proba(X_test),
        lgbm_final.predict_proba(X_test)
    ], axis=1)
    
    # Final predictions
    y_pred = meta.predict(Z_test)
    y_prob = meta.predict_proba(Z_test)
    
    results = compute_classification_metrics(y_test, y_pred, y_prob)
    
    print("\n✓ COB-ML ensemble training complete!")
    
    return {'COB-ML Ensemble': results}


def main():
    """
    Main training pipeline.
    """
    print("\n" + "="*80)
    print("COB-ML TRAINING PIPELINE")
    print("="*80)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    datasets = load_all_datasets(load_so=True)
    df = datasets['stack_overflow']
    print(f"Loaded: {len(df):,} records")
    
    # 2. Engineer features
    print("\n[2/5] Engineering features...")
    df_features = engineer_all_features(df)
    
    # 3. Prepare for modeling
    print("\n[3/5] Preparing feature matrix...")
    X, y, feature_names = prepare_features(df_features)
    print(f"Features: {X.shape}")
    print(f"Labels: {len(y)} ({np.bincount(y)})")
    
    # 4. Train/test split
    print("\n[4/5] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(y_train)} | Test: {len(y_test)}")
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    # 5. Train models
    print("\n[5/5] Training models...")
    
    # Baselines
    baseline_results = train_baseline_models(X_train, y_train, X_test, y_test)
    
    # COB-ML Ensemble
    ensemble_results = train_cob_ml_ensemble(X_train, y_train, X_test, y_test)
    
    # Combine results
    all_results = {**baseline_results, **ensemble_results}
    
    # Print final results
    print_results_table(all_results)
    
    print("\n✓ Training complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
