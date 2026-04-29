import logging
import os
import pickle
from pathlib import Path
from typing import Dict, Union

import mlflow
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from matplotlib import pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    make_scorer,
    recall_score,
    roc_auc_score,
    f1_score,
    precision_score,
    log_loss,
)
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, train_test_split

from noshow.config import setup_root_logger

logger = logging.getLogger(__name__)



def pr_auc_metric(
    X_val, y_val, estimator, labels,
    X_train, y_train,
    weight_val=None, weight_train=None,
    *args, **kwargs,
):
    """FLAML custom metric: optimize on PR-AUC, also report ROC-AUC, sensitivity,
    specificity, MCC, F1, precision at threshold 0.5."""
    proba = estimator.predict_proba(X_val)[:, 1]
    pred = (proba >= 0.5).astype(int)

    pr_auc = average_precision_score(y_val, proba)

    metrics_to_log = {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc_score(y_val, proba),
        "sensitivity": recall_score(y_val, pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_val, pred, pos_label=0, zero_division=0),
        "precision": precision_score(y_val, pred, zero_division=0),
        "f1": f1_score(y_val, pred, zero_division=0),
        "mcc": matthews_corrcoef(y_val, pred),
        "log_loss": log_loss(y_val, proba, labels=labels),
    }

    # FLAML minimises the first return value -> 1 - PR-AUC
    return 1 - pr_auc, metrics_to_log

def train_cv_model(
    featuretable: pd.DataFrame,
    output_path: Union[Path, str],
    classifier: BaseEstimator,
    param_grid: Dict,
    save_exp: bool = True,
    use_automl: bool = False,
    automl_time_budget: int = 600,
) -> None:
    """Use Cross validation to train a model and save results and parameters to mlflow.

    Parameters
    ----------
    featuretable : pd.DataFrame
        The featuretable
    output_path : Union[Path, str]
        Path to the output folder where to store the model pickle
    classifier : BaseEstimator
        The classifier to use (only used in the GridSearchCV branch)
    param_grid : Dict
        The parameter grid to search for the best model (GridSearchCV branch)
    save_exp : bool
        If we want to save the experiment to MLFlow, by default True
    use_automl : bool
        If True, use FLAML AutoML instead of GridSearchCV, by default False
    automl_time_budget : int
        Time budget in seconds for FLAML AutoML, by default 600
    """

    if save_exp:
        mlflow.set_experiment("HPO")
        if use_automl:
            mlflow.autolog(disable=True)
        else:
            mlflow.autolog(log_models=False)

        if os.getenv("MLFLOW_TRACKING_URI") is None:
            logger.warning(
                "MLFLOW_TRACKING_URI is not set, will default to mlruns directory."
            )

        run_id = mlflow.start_run().info.run_id

    featuretable["no_show"] = (
        featuretable["no_show"].replace({"no_show": "1", "show": "0"}).astype(int)
    )

    X, y = featuretable.drop(columns="no_show"), featuretable["no_show"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, shuffle=False
    )

    # train_groups = X_train.index.get_level_values("pseudo_id")
    train_groups = np.asarray(X_train.index.get_level_values("pseudo_id"))

    if use_automl:
        from flaml import AutoML

        logger.info(
            "Running FLAML AutoML (time budget=%ds)...", automl_time_budget
        )

        # Assumed class imbalance: 10% no_show (class 1) / 90% show (class 0).
        CLASS_WEIGHTS = {0: 1.0, 1: 9.0}
        sample_weight_train = np.where(y_train == 1, CLASS_WEIGHTS[1], CLASS_WEIGHTS[0])
        automl = AutoML()
        automl.fit(
            X_train=X_train,
            y_train=y_train,
            task="classification",
            # TODO: This doesn't show all metrics in mlflow properly
            metric=pr_auc_metric,
            # metric="ap",
            time_budget=automl_time_budget,
            estimator_list=[
                "lgbm",
                "xgboost",
                "rf",
                # "extra_tree",
                # "histgb",
                "lrl1",
            ],
            eval_method="cv",
            n_splits=5,
            split_type="group",
            groups=train_groups,
            mlflow_logging=True,
            mlflow_exp_name="HPO",
            sample_weight=sample_weight_train,
            seed=0,
            verbose=2,
            log_file_name=str(Path(output_path) / "flaml.log"),
        )
        best_estimator = automl.model.estimator

        if save_exp and mlflow.active_run():
            mlflow.log_params(
                {
                    "automl_best_estimator": automl.best_estimator,
                    "automl_best_config": automl.best_config,
                    "automl_time_budget": automl_time_budget,
                }
            )
    else:
        cv = StratifiedGroupKFold()
        # Specificity = recall of the negative class
        specificity_scorer = make_scorer(recall_score, pos_label=0)

        scoring = {
            "pr_auc": "average_precision",  # PR-AUC, robust to imbalance
            "precision": "precision",  # PPV at threshold 0.5
            "recall": "recall",  # sensitivity at 0.5
            "specificity": specificity_scorer,
            "f1": "f1",
            "mcc": "matthews_corrcoef",
        }
        grid = GridSearchCV(
            classifier,
            param_grid=param_grid,
            cv=cv,
            scoring=scoring,
            verbose=2,
            refit="pr_auc",
            n_jobs=10,
        )
        grid.fit(X_train, y_train, groups=train_groups)
        best_estimator = grid.best_estimator_

    # --- Test-set evaluation
    y_pred_proba = best_estimator.predict_proba(X_test)[:, 1]  # type: ignore
    y_pred_class = best_estimator.predict(X_test)  # type: ignore

    test_roc_auc = roc_auc_score(y_test, y_pred_proba)
    test_pr_auc = average_precision_score(y_test, y_pred_proba)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_class).ravel()
    test_sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    test_specificity = tn / (tn + fp) if (tn + fp) else 0.0
    test_mcc = matthews_corrcoef(y_test, y_pred_class)

    # Create and log calibration curve
    fig, ax = plt.subplots(figsize=(10, 6))
    CalibrationDisplay.from_predictions(y_test, y_pred_proba, n_bins=10, ax=ax)
    ax.set_title("Calibration Curve")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    plt.tight_layout()

    if save_exp and mlflow.active_run():
        mlflow.log_metric("test_roc_auc", float(test_roc_auc))
        mlflow.log_metric("test_pr_auc", float(test_pr_auc))
        mlflow.log_metric("test_sensitivity", float(test_sensitivity))
        mlflow.log_metric("test_specificity", float(test_specificity))
        mlflow.log_metric("test_mcc", float(test_mcc))
        mlflow.log_figure(fig, "calibration_curve.png")
    elif save_exp:
        logger.info("Mlflow run not active, re-activating run to log custom metrics.")
        with mlflow.start_run(run_id=run_id):
            mlflow.log_metric("test_roc_auc", float(test_roc_auc))
            mlflow.log_metric("test_pr_auc", float(test_pr_auc))
            mlflow.log_metric("test_sensitivity", float(test_sensitivity))
            mlflow.log_metric("test_specificity", float(test_specificity))
            mlflow.log_metric("test_mcc", float(test_mcc))
            mlflow.log_figure(fig, "calibration_curve.png")

    # Save the trained model (same convention as before)
    Path(output_path).mkdir(parents=True, exist_ok=True)
    with open(Path(output_path) / "no_show_model_cv.pickle", "wb") as f:
        pickle.dump(best_estimator, f)


if __name__ == "__main__":
    load_dotenv(override=True)
    setup_root_logger()
    project_folder = Path(__file__).parents[3]

    featuretable = pd.read_parquet(
        project_folder / "data" / "processed" / "featuretable.parquet"
    )

    model = HistGradientBoostingClassifier(categorical_features=["hour", "weekday"])

    train_cv_model(
        featuretable=featuretable,
        output_path=project_folder / "output" / "models",
        classifier=model,
        param_grid={
            "max_iter": [200, 300, 500],
            "learning_rate": [0.01, 0.05, 0.1],
        },
        use_automl=False,
    )