import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import optuna
from catboost import CatBoostRegressor
from sklearn.model_selection import RepeatedKFold, StratifiedKFold, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, kendalltau


INPUT_PATH = "Datasets/train_dataset.csv"
OUTPUT_PATH = "Datasets/train_dataset_full.csv"

TARGET = "real_alpha"

MARKET_FEATURES = [
    "beta_ewm_median",
    "volatility_21d_mean",
    "volatility_21d_max",
    "log_volume_mean",
    "market_deviation_mean",
    "market_deviation_std",
    "market_momentum_mean",
    "market_momentum_std",
    "market_3y_return",
    "t_bond_rate",
]

#ML-based copula factor estimation (alpha hat)

def load_market_dataset(path: str = INPUT_PATH) -> pd.DataFrame:
    """
    Input: path to train_dataset.csv
    Output: cleaned DataFrame with market features and real_alpha
    """
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=MARKET_FEATURES + [TARGET])
    return df.reset_index(drop=True)


def build_market_model(
        iterations: int = 300,
        depth: int = 3,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
) -> CatBoostRegressor:
    """
    Input: none
    Output: configured CatBoostRegressor for real_alpha prediction
    """
    return CatBoostRegressor(
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="RMSE",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
    )


def check_fold_stability(df: pd.DataFrame, n_splits: int = 5) -> None:
    """
    Input: DataFrame with market features and real_alpha, number of CV folds
    Output: printed per-fold target statistics to diagnose hidden dataset structure
    """
    X = df[MARKET_FEATURES]
    y = df[TARGET]

    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    print("Fold stability check (target distribution per fold):")
    for fold, (_, test_idx) in enumerate(cv.split(X, y_bins), 1):
        fold_y = y.iloc[test_idx]
        print(f"  Fold {fold} | n={len(fold_y)} | mean={fold_y.mean():.4f} | std={fold_y.std():.4f}")


def optimize_hyperparams(
        df: pd.DataFrame,
        n_trials: int = 50,
        n_splits: int = 5,
) -> dict:
    """
    Input: DataFrame with market features and real_alpha, number of Optuna trials, CV folds
    Output: best hyperparameter dict found by Optuna
    """
    X = df[MARKET_FEATURES]
    y = df[TARGET]
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 800),
            "depth": trial.suggest_int("depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []

        for train_idx, test_idx in cv.split(X, y_bins):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.15, random_state=42
            )

            model = CatBoostRegressor(
                **params,
                loss_function="RMSE",
                early_stopping_rounds=50,
                random_seed=42,
                verbose=0,
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            pred = model.predict(X_test)
            scores.append(mean_squared_error(y_test, pred) ** 0.5)

        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    print(f"\nOptuna best RMSE: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    return study.best_params


def add_alpha_hat_oof(
        path: str = INPUT_PATH,
        output_path: str = OUTPUT_PATH,
        n_splits: int = 5,
        n_repeats: int = 10,
        best_params: dict | None = None,
) -> pd.DataFrame:
    """
    Input: train_dataset.csv path, output path, number of CV folds
    Output: DataFrame with out-of-fold alpha_hat column saved to csv
    """
    df = load_market_dataset(path)

    X = df[MARKET_FEATURES]
    y = df[TARGET]
    y_bins = pd.qcut(y, q=5, labels=False, duplicates="drop")

    params = best_params or {}
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    alpha_hat_accum = np.zeros(len(df))
    counts = np.zeros(len(df))

    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_bins), 1):
        model = build_market_model(**{k: params[k] for k in params if k in build_market_model.__code__.co_varnames})

        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=fold
        )

        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
        pred = model.predict(X_test)

        alpha_hat_accum[test_idx] += pred
        counts[test_idx] += 1

        rmse = mean_squared_error(y_test, pred) ** 0.5
        print(f"Fold {fold:>3}: RMSE={rmse:.4f} | best_iter={model.best_iteration_}")

    df["alpha_hat"] = alpha_hat_accum / np.maximum(counts, 1)

    rmse_total = mean_squared_error(df[TARGET], df["alpha_hat"]) ** 0.5
    spearman = spearmanr(df["alpha_hat"], df[TARGET])
    kendall = kendalltau(df["alpha_hat"], df[TARGET])

    print("\nOOF result:")
    print(f"RMSE:     {rmse_total:.4f}")
    print(f"Spearman: {spearman.statistic:.4f}, p={spearman.pvalue:.4f}")
    print(f"Kendall:  {kendall.statistic:.4f}, p={kendall.pvalue:.4f}")

    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return df

#Plots

def plot_error_minimization(
        path: str = INPUT_PATH,
        test_size: float = 0.2,
        max_iterations: int = 800,
        best_params: dict | None = None,
) -> None:
    """
    Input: train_dataset.csv path, validation size, max CatBoost iterations
    Output: plot of train and validation RMSE by iteration
    """
    df = load_market_dataset(path)

    X = df[MARKET_FEATURES]
    y = df[TARGET]

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        shuffle=True,
    )

    params = best_params or {}
    model = CatBoostRegressor(
        iterations=max_iterations,
        depth=params.get("depth", 3),
        learning_rate=params.get("learning_rate", 0.05),
        l2_leaf_reg=params.get("l2_leaf_reg", 3.0),
        loss_function="RMSE",
        eval_metric="RMSE",
        early_stopping_rounds=50,
        random_seed=42,
        verbose=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_valid, y_valid),
        use_best_model=False,
    )

    evals = model.get_evals_result()
    train_rmse = evals["learn"]["RMSE"]
    valid_rmse = evals["validation"]["RMSE"]

    best_iter = int(np.argmin(valid_rmse))
    best_rmse = float(valid_rmse[best_iter])

    print(f"Best iteration: {best_iter + 1}")
    print(f"Best validation RMSE: {best_rmse:.4f}")

    plt.figure(figsize=(8, 4))
    plt.plot(train_rmse, label="train RMSE")
    plt.plot(valid_rmse, label="validation RMSE")
    plt.axvline(best_iter, linestyle="--", label=f"best iter = {best_iter + 1}")
    plt.xlabel("Iteration")
    plt.ylabel("RMSE")
    plt.title("CatBoost error minimization")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_alpha_hat_dependency(df: pd.DataFrame) -> None:
    """
    Input: DataFrame with real_alpha and alpha_hat
    Output: scatter plot and conditional probability plot
    """
    data = df.dropna(subset=["alpha_hat", TARGET]).copy()

    plt.figure(figsize=(6, 4))
    plt.scatter(data["alpha_hat"], data[TARGET], alpha=0.3)
    plt.xlabel("alpha_hat")
    plt.ylabel("real_alpha")
    plt.title("alpha_hat vs real_alpha")
    plt.grid(True)
    plt.show()

    data["alpha_hat_bin"] = pd.qcut(data["alpha_hat"], 10, duplicates="drop")
    prob = data.groupby("alpha_hat_bin", observed=True)[TARGET].apply(lambda x: (x > 0).mean())

    print("\nP(alpha > 0 | alpha_hat):")
    print(prob)

    plt.figure(figsize=(8, 4))
    prob.plot(kind="bar")
    plt.ylabel("P(alpha > 0)")
    plt.title("P(alpha > 0 | alpha_hat)")
    plt.grid(True)
    plt.show()

#Final model (for production)

def train_final_model(df: pd.DataFrame, best_params: dict | None = None) -> CatBoostRegressor:
    """
    Input: DataFrame with market features and real_alpha
    Output: trained CatBoost model on full dataset
    """
    X = df[MARKET_FEATURES]
    y = df[TARGET]

    params = best_params or {}
    model = build_market_model(**{k: params[k] for k in params if k in build_market_model.__code__.co_varnames})
    model.fit(X, y)

    return model


def save_model(model: CatBoostRegressor, path: str = "Models/market_model.cbm") -> None:
    """
    Input: trained CatBoost model, file path
    Output: saved model to disk
    """
    model.save_model(path)
    print(f"Model saved to: {path}")


def load_model(path: str = "Models/market_model.cbm") -> CatBoostRegressor:
    """
    Input: path to saved model
    Output: loaded CatBoost model
    """
    model = CatBoostRegressor()
    model.load_model(path)
    return model

#Notes

'''
To evaluate the predictive capacity of market information, we trained a regression model using only
aggregated market features (volatility, momentum, beta, and related statistics) to estimate realized
alpha. The results show that while the model cannot precisely predict alpha in magnitude (high RMSE),
it captures a statistically significant monotonic relationship with the true outcomes (positive Spearman
and Kendall correlations). This indicates that market data primarily encodes regime-level information, rather
than idiosyncratic mispricing.
In parallel, the DCF-derived signal (implied upside) demonstrates only weak direct correlation with realized
alpha and exhibits unstable conditional behavior when used in isolation. However, importantly, it remains
largely uncorrelated with the market-based prediction (alpha_hat), suggesting that it carries orthogonal information.
This orthogonality implies that DCF is not redundant with market features, but rather represents an independent
dimension of the valuation signal.
As a result, the project does not treat DCF as a direct predictor within the ML model. Instead, it is incorporated
at a later stage via a copula-based framework, where both market-driven expectations and fundamental signals are
combined to estimate the conditional distribution of alpha.
'''


if __name__ == "__main__":
    try:
        import os
        os.remove("Datasets/train_dataset_full.csv")
        os.remove("Models/market_model.cbm")
    except FileNotFoundError:
        pass

    df = load_market_dataset()

    check_fold_stability(df)

    best_params = optimize_hyperparams(df, n_trials=50)

    df = add_alpha_hat_oof(best_params=best_params)

    # train final model and save
    final_model = train_final_model(df, best_params=best_params)
    save_model(final_model)

    plot_alpha_hat_dependency(df) #Just some visualization for better project presentation
    plot_error_minimization(best_params=best_params) #Found the extreme value of error coefficient -> min