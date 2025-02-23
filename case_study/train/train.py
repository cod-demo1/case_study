import pandas as pd

from case_study.model.model import XValParams, ModelParams, XGBoostCrossValidator
from case_study.utils.data import get_data
from case_study.utils.utils import ROOT_DIR

prod_df = pd.read_csv(ROOT_DIR / "data/Production_Line_Data.csv")
audit_df = pd.read_csv(ROOT_DIR / "data/Audit_Data.csv")
x, y = get_data(prod_df, audit_df)

xval_pars = XValParams(
    n_splits=5,
    n_repeats=10,
)
model_pars = ModelParams(
    objective="binary:logistic",
    eval_metric="logloss",
    max_depth=3,
    n_estimators=100,
)

xgb_xval = XGBoostCrossValidator(xval_pars, model_pars)

bi_metrics = xgb_xval.crossval(x, y.y_has_defect)
bi_metrics.save(ROOT_DIR / "out/run_fit_metrics.json")

xgb_xval.train(x, y.y_has_defect, ROOT_DIR / "out")
