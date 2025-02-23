import dataclasses
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
)
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from case_study.model.run_metrics import FitMetrics, RunFitMetrics


@dataclasses.dataclass
class XValParams:
    n_splits: int
    n_repeats: int

    @property
    def total_splits(self) -> int:
        return self.n_splits * self.n_repeats


@dataclasses.dataclass
class ModelParams:
    objective: str
    eval_metric: str
    max_depth: int
    n_estimators: int


class XGBoostCrossValidator:
    def __init__(self, xval_pars: XValParams, model_pars: ModelParams) -> None:
        self.xval_pars = xval_pars
        self.model_pars = model_pars
        self.train_ids = []

    def train(
        self,
        x: pd.DataFrame,
        y: pd.DataFrame,
        export_path: Path,
    ) -> tuple[xgb.XGBClassifier, FitMetrics]:
        """
        Train an XGBoost model on 75% of the dataset and save it as an ONNX model.
        Also, store the training dataset IDs for later validation.

        :param x: Feature matrix.
        :param y: Target vector.
        :param export_path: Export path.
        :return: FitMetrics for the training data.
        """
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            shuffle=True,
        )
        train_ids = x_train.index.tolist()

        model = xgb.XGBClassifier(
            objective=self.model_pars.objective,
            eval_metric=self.model_pars.eval_metric,
            max_depth=self.model_pars.max_depth,
            n_estimators=self.model_pars.n_estimators,
        )
        model.fit(x_train, y_train)
        self.save(train_ids, model, export_path)

        y_pred_proba = model.predict_proba(x_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        metrics = self._compute_metrics(
            model=model,
            y_test=y_test,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
            fold=1,
        )
        return model, metrics

    def crossval(self, x: pd.DataFrame, y: pd.DataFrame) -> RunFitMetrics:
        """
        Perform cross-validation with XGBoost and compute evaluation metrics.

        :param x: Feature matrix.
        :param y: Target vector.
        :return: List of evaluation metrics for each fold.
        """
        results = RunFitMetrics(x.columns)
        kf = RepeatedKFold(
            n_splits=self.xval_pars.n_splits,
            n_repeats=self.xval_pars.n_repeats,
        )

        for fold, (train_idx, test_idx) in enumerate(kf.split(x, y)):
            logger.info(f"processing fold {fold + 1}/{self.xval_pars.total_splits}")

            x_train, x_test = x.iloc[train_idx], x.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = xgb.XGBClassifier(
                objective=self.model_pars.objective,
                eval_metric=self.model_pars.eval_metric,
            )
            model.fit(x_train, y_train)

            y_pred_proba = model.predict_proba(x_test)[:, 1]
            y_pred = (y_pred_proba >= 0.5).astype(int)

            metrics = self._compute_metrics(
                model=model,
                y_test=y_test,
                y_pred=y_pred,
                y_pred_proba=y_pred_proba,
                fold=fold,
            )
            results.append(metrics)

        return results

    @staticmethod
    def _compute_metrics(
        model: xgb.XGBClassifier,
        y_test: pd.DataFrame,
        y_pred: pd.DataFrame,
        y_pred_proba: pd.DataFrame,
        fold: int,
    ) -> FitMetrics:
        """
        Computes and logs evaluation metrics.

        :param model: Trained XGBoost model.
        :param y_test: True labels for the test set.
        :param y_pred: Predicted class labels.
        :param y_pred_proba: Predicted probabilities.
        :param fold: Fold index.
        :return: Dictionary containing computed metrics.
        """
        return FitMetrics(
            fold=fold,
            log_loss=log_loss(y_test, y_pred_proba),
            accuracy=accuracy_score(y_test, y_pred),
            precision=precision_score(y_test, y_pred, average="binary", zero_division=0),
            recall=recall_score(y_test, y_pred, average="binary", zero_division=0),
            f1=f1_score(y_test, y_pred, zero_division=0),
            auc_score=roc_auc_score(y_test, y_pred_proba),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            feature_importances=model.feature_importances_.tolist(),
        )

    @staticmethod
    def save(train_ids: list[int], model: xgb.XGBClassifier, path: Path) -> None:
        # save train ids
        with open(path / "train_ids.json", "w") as f:
            json.dump(train_ids, f)

        # save model
        model.save_model(path / "model.json")
