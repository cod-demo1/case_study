import numpy as np
import streamlit as st
from PIL import Image

from case_study.utils.plot import (
    cost_comp_plot,
    metrics_comp_plot,
    decision_threshold_plot,
    defects_found_plot,
    defect_code_plot,
    defects_count_per_line_plot,
    times_product_repaired_per_line_plot,
    defect_per_product_type_plot,
    repair_hours_per_product_type_plot,
    corr_mtx_plot,
    confusion_matrix_plot,
    crossval_fit_metrics_plot,
    feature_importance_plot,
)
from case_study.utils.utils import compute_optimal_threshold, compare_threshold_vs_random

st.markdown(
    f"""
    #### :exclamation: Case Study Part II.
    ### Predictive Risk Scoring for End-of-Line Product Audit.
    ####
    
    ##### ***Problem Summary***
    - Current product selection for end-of-line audit is random, therefore not optimised for fault capture.
    - This results in inefficient use of resources and potential delays in detecting quality issues.
    - The goal is to develop a model that would enable targeted product selection for inspection.
    
    ####
    #### :exclamation: Exploratory analysis:
    
    """
)
np.random.seed(42)

merged_df = st.session_state["merged_df"]
x, y = st.session_state["x"], st.session_state["y"]
x_train, x_test, y_train, y_test = (
    st.session_state["x_train"],
    st.session_state["x_test"],
    st.session_state["y_train"],
    st.session_state["y_test"],
)
model = st.session_state["model"]
run_fit_metrics = st.session_state["run_fit_metrics"]
y_test_pred_proba = st.session_state["y_test_pred_proba"]


c1, c2 = st.columns([2, 3], gap="large")
with c1:
    fig = defects_found_plot(y.y_has_defect)
    st.plotly_chart(fig)
with c2:
    fig = defect_code_plot(y)
    st.plotly_chart(fig)


st.write("####")
c1, c2 = st.columns([1, 1], gap="large")
with c1:
    fig = defects_count_per_line_plot(merged_df)
    st.plotly_chart(fig)
with c2:
    fig = times_product_repaired_per_line_plot(merged_df)
    st.plotly_chart(fig)


st.write("####")
c1, c2 = st.columns([1, 3], gap="large")
with c1:
    st.markdown(
        """
        ####
        ##### Defect Occurrence by Product Type
        
        - strongly unbalanced dataset in terms of defect product type
        - strongly unbalanced dataset in terms of defect type
        - `SNOWSPEEDER 2.0` overall most defect prone
        - `SNOWSPEEDER 2.0` overall most repair intensive
        
        ***-> will affect model performance***
        
        ***-> need to reflect in model selection***
        """
    )
with c2:
    fig = defect_per_product_type_plot(merged_df)
    st.plotly_chart(fig)


st.write("####")
c1, c2 = st.columns([1, 3], gap="large")
with c1:
    st.markdown(
        """
        ####
        ##### Repair Hours and Defect Occurrence
        
        - Outliers in repair hours.
        - Long repair times could indicate complex defects / faulty data.
        
        
        """
    )
with c2:
    fig = repair_hours_per_product_type_plot(merged_df)
    st.plotly_chart(fig)


st.write("####")
c1, c2 = st.columns([1, 3], gap="large")
with c1:
    st.markdown(
        """
        ####
        ##### Correlation Heatmap of Production Features

        - mostly 0.1 to 0.3 -> Weak positive correlation.
        - Variables are somewhat related, but the relationship is not strong.
        - Cannot predict one variable based solely on the other with much accuracy.
        
        ***-> Features are not highly redundant with one another.***
        
        ***-> Likely beneficial to keep them as separate features to capture nuanced relationships.***
        
        """
    )
with c2:
    fig = corr_mtx_plot(x)
    st.plotly_chart(fig)


st.markdown(
    """
    ####
    #### :exclamation:  Model validation
    """
)

# xval metrics
c1, c2, c3 = st.columns([1, 1, 1], gap="large")
with c1:
    st.markdown(
        """
        #####
        ##### Repeated k-fold cross-validation
        
        - Splits the dataset into K equally sized folds (or subsets).
        - Model is trained on K-1 folds and tested on the remaining 1 fold.
        - This process is repeated K times, with each fold being used as the test set exactly once.
        - The performance (accuracy, precision, etc.) is averaged over the K rounds to give a more reliable estimate of the model’s performance.
        
        #####
        
        -> ***Binary classification [target=has_defect]***
        
        """
    )
with c2:
    fig = confusion_matrix_plot(run_fit_metrics)
    st.plotly_chart(fig)
with c3:
    fig = crossval_fit_metrics_plot(run_fit_metrics)
    st.plotly_chart(fig)


# feature importances
c1, c2 = st.columns([1.5, 3], gap="large")
with c1:
    st.markdown(
        """
        #####
        ##### Feature importances
        #####
        
        - Score that indicates how useful or valuable each feature is for the model.
        - Averaged across all of the the decision trees within the model.
        - metric = `Gain`
        
        """
    )
with c2:
    fig = feature_importance_plot(run_fit_metrics)
    st.plotly_chart(fig)

c1, c2 = st.columns([2, 3], gap="large")
with c1:
    st.markdown(
        """
        ##### SHAP analysis
        #####
        
        - Explains individual predictions of machine learning models wrt the impact of each feature.
        - SHAP value reflects how much it contributed to moving the prediction away from the mean prediction.
        - Color represents ***feature*** value, and the horizontal position shows SHAP value.
        - **Positive SHAP value**: Indicates pushing the model's prediction ***higher***.
        - **Negative SHAP value**: Indicates pushing the model's prediction ***lower***.
        
        - Computationally expensive.
        
        *(Might look very different for product type specific model).*
        
        *(Might look very different for defect type specific model (multiclass)).*
        
        """
    )
with c2:
    image = Image.open("/home/user/PycharmProjects/case_study/out/shap.jpg")
    st.image(image)

st.markdown(
    """
    ####
    #### :exclamation: Added value
    """
)

# decision threshold
c1, c2, c3 = st.columns([2, 1, 3], gap="large")
with c1:
    st.markdown(
        """
        ####
        #####  Choosing decision threshold
        
        1. Class Imbalance.
        2. Cost of Misclassification.
            - False Negatives (FN) → Missed defective products.
            - False Positives (FP) → Unnecessary inspections.
        
        - -> Influences Evaluation Metrics.
            - Accuracy
            - Precision
            - Recall
        
        ######
        ***-> Compute the cost for different thresholds.***
        """
    )
with c2:
    st.markdown("##")
    cost_fn = st.slider("False negative cost", 0, 100, 50)
    cost_fp = st.slider("False positive cost", 0, 100, 10)
    cost_check = st.slider("Fixed cost per inspection", 0, 100, 1)
with c3:
    opt = compute_optimal_threshold(
        y_test=y_test.y_has_defect.values,
        y_pred_proba=y_test_pred_proba,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        cost_check=cost_check,
    )[0]
    thrs = [0.5, opt]
    fig = decision_threshold_plot(
        x_test,
        y_test.y_has_defect,
        y_test_pred_proba,
        thrs,
    )
    st.plotly_chart(fig)


c1, c2, c3 = st.columns([2, 2, 1.5], gap="large")
with c1:
    st.markdown(
        """
        #####
        ##### Precision-Recall Trade-off
        ######
        
        - **Accuracy** - Good to avoid for imbalanced datasets.
        - **Precision** (Positive Predictive Value): True Positives / All Positives (TP + FP)
        - **Recall** (True Positive Rate): True Positives / All Actual Positives (TP + FN)

        ######
        
        - Optimizing for one metric often causes a decrease in the other.
        - e.g.: to increase precision, only classify most obvious positive cases -> many false negatives & lower recall
        """
    )
with c2:
    opt_threshold, cost_opt, cost_rand = compare_threshold_vs_random(
        y_test=y_test.y_has_defect.values,
        y_pred_proba=y_test_pred_proba,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
        cost_check=cost_check,
    )
    fig = metrics_comp_plot(opt_threshold, y_test_pred_proba, y_test.y_has_defect)
    st.plotly_chart(fig)
with c3:
    fig = cost_comp_plot(cost_opt, cost_rand)
    st.plotly_chart(fig)


st.markdown(
    """
    ####
    #### :exclamation: Next steps / possible improvements
    
    1. Extend training dataset
        - Current dataset: ~2k records
        - ~20k -> improve model robustness and accuracy.
    2. Multiclass Classification
        - Categorize products by defect type.
        - Estimate expected cost wrt defect type.
        - Allows prioritization of critical defects.
    3. Feature Selection.
        - Use tree-based feature selection methods to identify the most important features.
        - Identify and remove weak predictors.
        - Integrate domain knowledge (engineers) to create additional meaningful features.
    4. Advanced Model Architectures.
        - Train more complex model (NNs, per product type, multiclass) only on relevant features.
        - Ensemble methods for increased accuracy.
        - Train separate models per product type.
    5. Real-time model.
        - Integrate into production line monitoring and flag high-risk products in real time.
        - API or embedded model.
        - Alerts for high-risk products, triggering an automatic inspection request.
    6. Continuously retrain the model.
        - Detect model drift.
        - Use new defect data to maintain accuracy.
    
    """
)
