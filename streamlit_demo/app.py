import json

import pandas as pd
import streamlit as st
import xgboost as xgb

from case_study.utils.data import get_data
from case_study.utils.utils import ROOT_DIR
from case_study.model.run_metrics import RunFitMetrics

st.set_page_config(page_title="CoD demo", page_icon="Home", layout="wide")


c1, c2 = st.columns([2, 3], gap="large")
with c1:
    st.markdown(
        """
        # Case Study streamlit demo

        #### :exclamation: upload datasets ->
        ###
        
        ##### When status says :white_check_mark:, proceed to either part through menu on the left.
        - **Part1**: Automating Software Quality Control with AI.
        - **Part2**: Predictive Risk Scoring for End-of-Line Product Audit.
        
        """
    )

with c2:
    prod_f = st.file_uploader("Production line data", type="csv")
    audit_f = st.file_uploader("Audit data", type="csv")

    if prod_f is not None and audit_f is not None:
        prod_data = pd.read_csv(prod_f)
        audit_data = pd.read_csv(audit_f)

        merged_df = get_data(prod_data, audit_data, return_raw=True, debug=True)
        st.session_state["merged_df"] = merged_df

        st.session_state["run_fit_metrics"] = RunFitMetrics.load(ROOT_DIR / "out/run_fit_metrics.json")

        with open(ROOT_DIR / "out/train_ids.json", "r") as inf:
            st.session_state["train_ids"] = json.load(inf)

        model = xgb.XGBClassifier()
        model.load_model(ROOT_DIR / "out/model.json")
        st.session_state["model"] = model

        (
            st.session_state["x_train"],
            st.session_state["y_train"],
            st.session_state["x_test"],
            st.session_state["y_test"],
        ) = get_data(prod_data, audit_data, train_ids=st.session_state["train_ids"])

        st.session_state["x"] = pd.concat(
            [st.session_state["x_train"], st.session_state["x_test"]], axis=0
        ).reset_index(drop=True)
        st.session_state["y"] = pd.concat(
            [st.session_state["y_train"], st.session_state["y_test"]], axis=0
        ).reset_index(drop=True)

        st.session_state["y_test_pred_proba"] = model.predict_proba(st.session_state["x_test"])[:, 1]

    if "y_test_pred_proba" not in st.session_state:
        st.write("### Status: :x:")
    else:
        st.write("### Status: :white_check_mark:")
