import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd


@dataclasses.dataclass(frozen=True)
class FitMetrics:
    fold: int
    log_loss: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: list[float]
    feature_importances: list[float]
    auc_score: float


@dataclasses.dataclass
class RunFitMetrics:
    fit_metrics: list[FitMetrics | None]

    def __init__(self, feature_names: list[str]) -> None:
        self.fit_metrics = []
        self.feature_names = feature_names

    def append(self, x: FitMetrics) -> None:
        self.fit_metrics.append(x)

    @property
    def conf_mtx(self) -> npt.NDArray[float] | None:
        if not self.fit_metrics:
            return None
        return np.mean([fm.confusion_matrix for fm in self.fit_metrics], axis=0)

    @property
    def avg_feature_importance(self) -> pd.Series:
        importance_list = [fm.feature_importances for fm in self.fit_metrics if fm is not None]
        avg_importance = np.mean(importance_list, axis=0)
        return pd.Series(avg_importance, index=self.feature_names)

    @property
    def avg_shap_values(self) -> npt.NDArray[float]:
        shap_values_stacked = np.concatenate(self.shap_values, axis=0)
        return shap_values_stacked

    def get_metric(self, metric_name: str) -> list[Any | None]:
        return [getattr(fm, metric_name) for fm in self.fit_metrics]

    def __getattr__(self, name: str) -> list[Any] | None:
        if len(self.fit_metrics) and hasattr(self.fit_metrics[0], name):
            return self.get_metric(name)
        raise AttributeError(f"'RunFitMetrics' object has no attribute '{name}'")

    def save(self, path: Path) -> None:
        out = {
            "features": list(self.feature_names),
            "fms": [dataclasses.asdict(fm) for fm in self.fit_metrics],
        }
        with open(path, "w") as outf:
            json.dump(out, outf)

    @classmethod
    def load(cls, path: Path) -> "RunFitMetrics":
        with open(path, "r") as inf:
            raw = json.load(inf)
        toc = cls(raw["features"])
        toc.fit_metrics = [FitMetrics(**fm) for fm in raw["fms"]]
        return toc
