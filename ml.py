import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier


FEATURES_DCF = [
    "implied_upside",
    "t_bond_rate",
    "p_alpha_gt_0",
    "e_alpha",
]

FEATURES_MARKET = [
    "beta_ewm_median",
    "volatility_21d_mean",
    "volatility_21d_max",
    "log_volume_mean",
    "market_deviation_mean",
    "market_deviation_std",
    "market_momentum_mean",
    "market_momentum_std",
    "market_3y_return",
]

FEATURES_FULL = FEATURES_DCF + FEATURES_MARKET
TARGET = "alpha_intensity"


def load_data(path: str, features: list[str]) -> tuple:
    df = pd.read_csv(path)
    df = df.dropna(subset=features + [TARGET])

    X = df[features].copy()

    le = LabelEncoder()
    y = pd.Series(
        le.fit_transform(df[TARGET].astype(int)),
        index=df.index,
        name=TARGET,
    )

    print("Target mapping:")
    for old, new in zip(le.classes_, range(len(le.classes_))):
        print(f"  {old} -> {new}")

    return X, y, df


def make_models() -> dict:
    return {
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                C=1.0,
                random_state=42,
            )),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=4,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            min_samples_leaf=10,
            random_state=42,
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=10,
            random_state=42,
            verbosity=0,
            eval_metric="mlogloss",
        ),
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=3,
            learning_rate=0.05,
            min_data_in_leaf=10,
            random_seed=42,
            verbose=0,
        ),
    }


def evaluate_feature_set(
    features: list[str],
    label: str,
    path: str = "Datasets/train_dataset.csv",
) -> pd.DataFrame:
    X, y, df = load_data(path, features)

    if y.nunique() < 2:
        raise ValueError(f"Target has less than 2 classes for feature set: {label}")

    min_class_count = y.value_counts().min()
    n_splits = min(5, int(min_class_count))

    if n_splits < 2:
        raise ValueError(f"Not enough samples per class for CV in feature set: {label}")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    models = make_models()
    f1_weighted = make_scorer(f1_score, average="weighted")

    print(f"\n{'='*60}")
    print(f"Feature set: {label}  |  n={len(df)}  |  features={len(features)}")
    print(f"Target alpha_intensity distribution:")
    print(df[TARGET].astype(int).value_counts().sort_index().to_string())
    print(f"Encoded target distribution:")
    print(y.value_counts().sort_index().to_string())
    print(f"CV splits: {n_splits}")
    print(f"{'='*60}")

    results = []

    for name, model in models.items():
        n_jobs = -1 if name not in ("CatBoost", "GradientBoosting") else 1

        acc_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring="accuracy",
            n_jobs=n_jobs,
            error_score="raise",
        )

        f1_scores = cross_val_score(
            model, X, y,
            cv=cv,
            scoring=f1_weighted,
            n_jobs=n_jobs,
            error_score="raise",
        )

        results.append({
            "feature_set": label,
            "model": name,
            "ACC_mean": round(acc_scores.mean(), 4),
            "ACC_std": round(acc_scores.std(), 4),
            "F1_mean": round(f1_scores.mean(), 4),
            "F1_std": round(f1_scores.std(), 4),
        })

        print(f"{name:20s}  ACC={acc_scores.mean():.4f} ± {acc_scores.std():.4f}  "
              f"F1={f1_scores.mean():.4f} ± {f1_scores.std():.4f}")

    results_df = pd.DataFrame(results).sort_values("ACC_mean", ascending=False)
    best_name = results_df.iloc[0]["model"]

    print(f"\nЛучшая модель по accuracy: {best_name}")

    best_model = models[best_name]
    best_model.fit(X, y)

    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        m = best_model.named_steps["model"]
        importances = np.abs(m.coef_).mean(axis=0) if hasattr(m, "coef_") else None
    else:
        importances = None

    if importances is not None:
        print(f"Feature importance ({best_name}):")
        for feat, imp in sorted(
            zip(features, importances), key=lambda x: x[1], reverse=True
        ):
            print(f"  {feat:30s}  {imp:.4f}")

    return results_df


def compare_feature_sets(path: str = "Datasets/train_dataset.csv") -> pd.DataFrame:
    sets = [
        (FEATURES_DCF, "DCF only"),
        (FEATURES_MARKET, "Market only"),
        (FEATURES_FULL, "Full"),
    ]

    all_results = []
    for features, label in sets:
        res = evaluate_feature_set(features, label, path=path)
        all_results.append(res)

    combined = pd.concat(all_results, ignore_index=True)

    print(f"\n{'='*60}")
    print("Сводная таблица лучших моделей по каждому набору фичей:")
    print(f"{'='*60}")

    summary = (
        combined.sort_values("ACC_mean", ascending=False)
        .groupby("feature_set", sort=False)
        .first()
        .reset_index()
        [["feature_set", "model", "ACC_mean", "ACC_std", "F1_mean", "F1_std"]]
    )

    print(summary.to_string(index=False))

    return combined


if __name__ == "__main__":
    compare_feature_sets()