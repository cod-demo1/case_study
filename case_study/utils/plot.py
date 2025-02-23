import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)
import plotly.figure_factory as ff


def defects_found_plot(y):
    defect_counts = y.value_counts().reset_index()
    defect_counts.columns = ["Defect Found", "Count"]
    defect_counts = defect_counts.sort_values("Defect Found", ascending=True)
    fig = px.bar(
        defect_counts,
        x="Defect Found",
        y="Count",
        labels={
            "Defect Found": "Defect Found",
            "Count": "Count",
        },
        title="Distribution of Products with Defects Found in Audit",
        text=defect_counts["Count"].astype(str),
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis=dict(tickmode="array", tickvals=[0, 1]))
    return fig


def defect_code_plot(y):
    y_err = y.loc[y.y_has_defect == 1]
    defect_counts = y_err["y_defect_type"].value_counts().reset_index()
    defect_counts.columns = ["Defect Type", "Count"]
    defect_lookup = {int(d): k for k, d in enumerate(sorted(y.y_defect_type.unique()))}
    defect_counts["Encoded Label"] = defect_counts["Defect Type"].map(defect_lookup)

    fig = px.bar(
        defect_counts,
        x="Encoded Label",
        y="Count",
        labels={
            "Encoded Label": "Defect Code (Encoded)",
            "Count": "Frequency",
        },
        color="Defect Type",
        title="Distribution of Defects Found in Audit",
        text=defect_counts["Count"].astype(str),  # Show count labels
        hover_data={"Encoded Label": False, "Defect Type": True},
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=defect_counts["Encoded Label"],
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def defects_count_per_line_plot(df):
    fig = px.histogram(
        df,
        x="type",
        y="y_defect_type",
        color="production_line",
        title="Defect Counts by Type & Production Line",
        labels={
            "production_line": "Production Line",
            "defects_count_from_production": "Defects Found",
        },
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def times_product_repaired_per_line_plot(df):
    fig = px.histogram(
        df,
        x="type",
        y="times_product_repaired",
        color="production_line",
        title="Times Product Repaired by Type & Production Line",
        labels={
            "production_line": "Production Line",
            "defects_count_from_production": "Defects Found",
        },
    )
    fig.update_xaxes(tickangle=-45)
    return fig


def defect_per_product_type_plot(df):
    product_type_mapping = {ptype: idx + 1 for idx, ptype in enumerate(sorted(df["type"].unique()))}
    df["Encoded Product Type"] = df["type"].map(product_type_mapping)
    fig = px.histogram(
        df,
        x="Encoded Product Type",
        color="y_defect_type",
        title="Defect Occurrence by Product Type",
        barmode="group",
        labels={
            "Encoded Product Type": "Product Type (Encoded)",
            "y_defect_type": "Defect Found",
        },
        hover_data={
            "Encoded Product Type": False,
            "type": True,
        },
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(product_type_mapping.values()),
        ticktext=list(product_type_mapping.keys()),
        tickangle=-45,
    )
    return fig


def repair_hours_per_product_type_plot(df):
    defect_type_mapping = {dtype: idx + 1 for idx, dtype in enumerate(df["y_defect_type"].dropna().unique())}
    df["Encoded Defect Type"] = df["y_defect_type"].map(defect_type_mapping)

    fig = px.strip(
        df,
        x="Encoded Defect Type",
        y="hours_product_repaired_total",
        color="y_defect_type",
        title="Repair Hours and Defect Occurrence",
        labels={
            "Encoded Defect Type": "Defect Type (Encoded)",
            "hours_product_repaired_total": "Total Repair Hours",
        },
        hover_data={"Encoded Defect Type": False, "y_defect_type": True},
        stripmode="overlay",
    )
    fig.update_xaxes(
        tickmode="array",
        tickvals=list(defect_type_mapping.values()),
        ticktext=list(defect_type_mapping.keys()),
        tickangle=-45,
    )
    fig.update_layout(template="plotly_dark")
    return fig


def corr_mtx_plot(x):
    corr_matrix = x.iloc[:, :-7].corr().round(2)
    fig = ff.create_annotated_heatmap(
        z=corr_matrix.values,
        x=list(corr_matrix.columns),
        y=list(corr_matrix.index),
        colorscale="Viridis",
        showscale=True,
    )
    fig.update_layout(
        title="Correlation Heatmap of Production Features",
        width=1200,
        height=1200,
    )
    return fig


def confusion_matrix_plot(run_fit_metrics):
    fig = go.Figure(
        data=go.Heatmap(
            z=run_fit_metrics.conf_mtx,
            colorscale="PuBu",
            colorbar=dict(title="Count"),
            zmin=0,
            zmax=np.max(run_fit_metrics.conf_mtx),
        )
    )
    fig.update_layout(
        title="Mean Confusion Matrix Heatmap",
        xaxis_title="Predicted Labels",
        yaxis_title="True Labels",
        template="plotly",
        xaxis=dict(
            tickvals=[0, 1],
            ticktext=[False, True],
            title="Predicted Labels",
        ),
        yaxis=dict(
            tickvals=[0, 1],
            ticktext=[False, True],
            title="True Labels",
        ),
    )
    return fig


def crossval_fit_metrics_plot(run_fit_metrics):
    dt = pd.DataFrame(
        {
            "accuracy": run_fit_metrics.accuracy,
            "precision": run_fit_metrics.precision,
            "recall": run_fit_metrics.recall,
        }
    )
    melted_data = dt.melt(var_name="Metric", value_name="Value")
    fig = px.box(
        melted_data,
        x="Metric",
        y="Value",
        title="Fit metrics",
    )
    return fig


def feature_importance_plot(run_fit_metrics):
    importance_df = run_fit_metrics.avg_feature_importance.sort_values(ascending=True).reset_index()
    importance_df.columns = ["Feature", "Average Importance"]
    fig = px.bar(
        importance_df.iloc[-15:],
        x="Average Importance",
        y="Feature",
        orientation="h",
        title="Average Feature Importance Across Folds",
        labels={"Average Importance": "Feature Importance"},
    )
    fig.update_layout(
        title="Feature importances",
        width=1200,
        height=500,
    )
    return fig


def decision_threshold_plot(x_test, y_test, probs, thresholds):
    fig = go.Figure()
    for i, threshold in enumerate(thresholds):
        colors = np.where(y_test == 1, "crimson", "cornflowerblue")
        fig.add_trace(
            go.Scatter(
                x=np.full(len(x_test), i) + np.random.uniform(-0.2, 0.2, size=len(x_test)),
                y=probs,
                mode="markers",
                marker=dict(color=colors, size=8),
                name=f"Threshold > {int(threshold * 100)}%",
            )
        )
        fig.add_shape(
            dict(
                type="line",
                x0=i - 0.4,
                x1=i + 0.4,
                y0=threshold,
                y1=threshold,
                line=dict(color="red", width=2, dash="dash"),
            )
        )
    fig.update_layout(
        title="Effect of Decision Threshold on Classification",
        xaxis=dict(
            tickmode="array",
            tickvals=[0, 1],
            ticktext=[
                f"> {int(thresholds[0] * 100)}%",
                f"> {int(thresholds[1] * 100)}%",
            ],
        ),
        yaxis=dict(title="Predicted Probability"),
        showlegend=False,
        height=500,
        width=700,
    )
    return fig


def metrics_comp_plot(opt_threshold, probs, y_test):
    y_pred = (probs >= opt_threshold).astype(int)
    rec_mod = recall_score(y_test, y_pred, zero_division=0)
    prec_mod = precision_score(y_test, y_pred, zero_division=0)
    acc_mod = accuracy_score(y_test, y_pred)

    y_pred_rand = (np.random.rand(len(y_test)) > 0.5).astype(int)
    rec_rand = recall_score(y_test, y_pred_rand, zero_division=0)
    prec_rand = precision_score(y_test, y_pred_rand, zero_division=0)
    acc_rand = accuracy_score(y_test, y_pred_rand)

    metrics = ["Accuracy", "Precision", "Recall"]
    final_model_metrics = [acc_mod, prec_mod, rec_mod]
    random_model_metrics = [acc_rand, prec_rand, rec_rand]

    fig = go.Figure(
        data=[
            go.Bar(
                name="Final Model",
                x=metrics,
                y=final_model_metrics,
                text=[f"{v:.2f}" for v in final_model_metrics],
                textposition="auto",
            ),
            go.Bar(
                name="Random Model",
                x=metrics,
                y=random_model_metrics,
                text=[f"{v:.2f}" for v in random_model_metrics],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Metrics",
        yaxis_title="Scores",
        barmode="group",
    )
    return fig


def cost_comp_plot(cost_opt: float, cost_rand: float) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Bar(
                name="Final Model",
                x=["cost"],
                y=[cost_opt],
                text=[cost_opt.round(2)],
                textposition="auto",
            ),
            go.Bar(
                name="Random Model",
                x=["cost"],
                y=[cost_rand],
                text=[cost_rand.round(2)],
                textposition="auto",
            ),
        ]
    )
    fig.update_layout(
        title="Cost comparison",
        xaxis_title="Model",
        yaxis_title="Cost",
        barmode="group",
    )
    return fig
