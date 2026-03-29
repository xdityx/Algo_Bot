import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report, precision_score, recall_score
import warnings, sys, os
warnings.filterwarnings("ignore")

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from configs.config import LGBM_PARAMS, N_SPLITS, TRAIN_PCT, VAL_PCT, CONFIDENCE_THRESHOLD
from features.engineer import FEATURE_COLS


def walk_forward_train(df: pd.DataFrame):
    """
    Walk-forward validation — no future leakage.
    Trains N models on expanding windows, tests on unseen future data.
    Returns trained models + OOS predictions.
    """
    print(f"\n[Model] Walk-forward training | {N_SPLITS} folds")

    n = len(df)
    fold_size = n // (N_SPLITS + 1)
    models = []
    oos_preds = []
    oos_actuals = []

    for fold in range(N_SPLITS):
        train_end  = fold_size * (fold + 1)
        val_end    = fold_size * (fold + 2)

        train_df = df.iloc[:train_end]
        val_df   = df.iloc[train_end:val_end]

        X_train = train_df[FEATURE_COLS]
        y_train = train_df["target"]
        X_val   = val_df[FEATURE_COLS]
        y_val   = val_df["target"]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set   = lgb.Dataset(X_val,   label=y_val, reference=train_set)

        params = LGBM_PARAMS.copy()
        n_estimators = params.pop("n_estimators")
        early_stop   = params.pop("early_stopping_rounds")

        model = lgb.train(
            params,
            train_set,
            num_boost_round=n_estimators,
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(early_stop, verbose=False),
                lgb.log_evaluation(period=-1),
            ]
        )

        preds = model.predict(X_val)
        auc   = roc_auc_score(y_val, preds)

        print(f"  Fold {fold+1}/{N_SPLITS} | Train: {len(train_df):,} | Val: {len(val_df):,} | AUC: {auc:.4f}")

        models.append(model)
        oos_preds.extend(preds)
        oos_actuals.extend(y_val.values)

    # Final model on all data (for live use)
    print("\n[Model] Training final model on full dataset...")
    n_val = int(len(df) * 0.15)
    X_all   = df[FEATURE_COLS]
    y_all   = df["target"]
    X_final_val = df.iloc[-n_val:][FEATURE_COLS]
    y_final_val = df.iloc[-n_val:]["target"]

    train_set_final = lgb.Dataset(X_all.iloc[:-n_val], label=y_all.iloc[:-n_val])
    val_set_final   = lgb.Dataset(X_final_val, label=y_final_val, reference=train_set_final)

    params = LGBM_PARAMS.copy()
    n_estimators = params.pop("n_estimators")
    early_stop   = params.pop("early_stopping_rounds")

    final_model = lgb.train(
        params,
        train_set_final,
        num_boost_round=n_estimators,
        valid_sets=[val_set_final],
        callbacks=[
            lgb.early_stopping(early_stop, verbose=False),
            lgb.log_evaluation(period=-1),
        ]
    )

    # OOS summary
    oos_preds    = np.array(oos_preds)
    oos_actuals  = np.array(oos_actuals)
    oos_binary   = (oos_preds >= CONFIDENCE_THRESHOLD).astype(int)
    oos_auc      = roc_auc_score(oos_actuals, oos_preds)

    print(f"\n[Model] ─── OOS Performance Summary ───")
    print(f"  AUC:       {oos_auc:.4f}")
    print(f"  Precision: {precision_score(oos_actuals, oos_binary, zero_division=0):.4f}")
    print(f"  Recall:    {recall_score(oos_actuals, oos_binary, zero_division=0):.4f}")
    print(f"  Signals:   {oos_binary.sum()} / {len(oos_binary)} ({oos_binary.mean()*100:.1f}%)")

    feature_importance = pd.Series(
        final_model.feature_importance(importance_type="gain"),
        index=FEATURE_COLS
    ).sort_values(ascending=False)

    print(f"\n[Model] Top 10 Features:")
    print(feature_importance.head(10).to_string())

    return final_model, models, oos_preds, feature_importance


def predict(model, df: pd.DataFrame) -> pd.Series:
    """Generate confidence scores for a DataFrame."""
    X = df[FEATURE_COLS]
    return pd.Series(model.predict(X), index=df.index, name="confidence")


def save_model(model, path="/home/claude/crypto_bot/models/lgbm_final.txt"):
    model.save_model(path)
    print(f"[Model] Saved to {path}")


def load_model(path="/home/claude/crypto_bot/models/lgbm_final.txt"):
    model = lgb.Booster(model_file=path)
    return model


if __name__ == "__main__":
    df = pd.read_parquet("/home/claude/crypto_bot/data/features_btc.parquet")
    final_model, fold_models, oos_preds, feat_imp = walk_forward_train(df)
    save_model(final_model)
