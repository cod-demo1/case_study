from pathlib import Path

import numpy as np
import numpy.typing as npt
from sklearn.metrics import roc_curve

ROOT_DIR = Path(__file__).resolve().parents[2]


def compute_cost(
    y_test: npt.NDArray[float],
    y_pred: npt.NDArray[float],
    cost_fp: float,
    cost_fn: float,
    cost_check: float,
) -> float:
    fp = np.sum((y_pred == 1) & (y_test == 0))
    fn = np.sum((y_pred == 0) & (y_test == 1))
    total_checked = np.sum(y_pred == 1)  # Total predicted True (TP + FP)

    return (cost_fp * fp) + (cost_fn * fn) + (cost_check * total_checked)


def compute_optimal_threshold(
    y_test: npt.NDArray[float],
    y_pred_proba: npt.NDArray[float],
    cost_fp: float,
    cost_fn: float,
    cost_check: float,
) -> tuple[float, ...]:
    """
    Compute the optimal classification threshold by minimizing total cost, including checking cost.
    """
    _, _, thresholds = roc_curve(y_test, y_pred_proba)

    total_costs = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cur_cost = compute_cost(y_test, y_pred, cost_fp, cost_fn, cost_check)
        total_costs.append(cur_cost)

    optimal_idx = np.argmin(total_costs)
    return thresholds[optimal_idx], total_costs[optimal_idx]


def compare_threshold_vs_random(
    y_test: npt.NDArray[float],
    y_pred_proba: npt.NDArray[float],
    cost_fp: float,
    cost_fn: float,
    cost_check: float,
) -> tuple[float, ...]:
    """
    Compare costs using the computed optimal threshold vs. random selection.
    """
    _, _, thresholds = roc_curve(y_test, y_pred_proba)

    total_costs = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        total_costs.append(compute_cost(y_test, y_pred, cost_fp, cost_fn, cost_check))

    # Optimal threshold cost
    optimal_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[optimal_idx]
    y_pred_opt = (y_pred_proba >= optimal_threshold).astype(int)
    cost_opt = compute_cost(y_test, y_pred_opt, cost_fp, cost_fn, cost_check)

    # Random selection (class mean) cost
    rand_probs = np.random.rand(len(y_test))
    rand_thr = np.mean(y_test)
    y_pred_rand = (rand_probs >= rand_thr).astype(int)
    cost_rand = compute_cost(y_test, y_pred_rand, cost_fp, cost_fn, cost_check)

    return optimal_threshold, cost_opt, cost_rand
