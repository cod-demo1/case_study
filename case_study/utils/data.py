import numpy as np

import pandas as pd
from loguru import logger


def get_data(
    prod_df: pd.DataFrame,
    audit_df: pd.DataFrame,
    train_ids: list[int] | None = None,
    return_raw: bool = False,
    debug: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    audit_df["y_has_defect"] = audit_df["defect_type"].notna().astype(int)
    audit_df["y_defect_type"] = audit_df["defect_type"].fillna(0).astype(int)
    merged_df = pd.merge(
        prod_df,
        audit_df[["id", "audit_timestamp", "y_has_defect", "y_defect_type"]],
        on="id",
        how="inner",
    )
    merged_df = validate_data(merged_df, debug)
    if return_raw:
        return merged_df

    num_cols = [a for a in merged_df.columns if any(sub in a for sub in ["emergency", "defects", "repaired"])]
    cat_cols = ["type", "production_line"]
    tar_cols = ["y_has_defect", "y_defect_type"]

    df = pd.get_dummies(merged_df[cat_cols]).astype(int)

    if not train_ids:
        return pd.concat([merged_df[num_cols], df], axis=1), merged_df[tar_cols]

    merged_df_train = merged_df[merged_df.id.isin(train_ids)]
    merged_df_test = merged_df[~merged_df.id.isin(train_ids)]

    return (
        pd.concat([merged_df_train[num_cols], df], axis=1),
        merged_df_train[tar_cols],
        pd.concat([merged_df_test[num_cols], df], axis=1),
        merged_df_test[tar_cols],
    )


def validate_data(df: pd.DataFrame, debug: bool = False) -> pd.DataFrame:
    # check duplicate ids
    duplicate_ids = df.id[df.id.duplicated()]
    if debug:
        logger.info(f"removing {len(duplicate_ids)} duplicate ids")
    df = df[~df.id.isin(duplicate_ids)]

    # check nans
    if debug:
        logger.info(f"removing {df.isna().sum().sum()} nans")
    df = df.dropna()

    # check inconsistent data
    repair_columns = [
        col for col in df.columns if "hours_product_repaired_" in col and col != "hours_product_repaired_total"
    ]
    df["repair_hours_sum"] = df[repair_columns].sum(axis=1)
    inconsistent_repair_hours = df[np.abs(df["repair_hours_sum"] - df["hours_product_repaired_total"]) > 0.1]
    if not inconsistent_repair_hours.empty:
        if debug:
            logger.info(f"removing {inconsistent_repair_hours.shape[0]} inconsistent repair hours rows")
        df = df.drop(inconsistent_repair_hours.index)

    # check prod & audit timestamp logic
    if "audit_timestamp" in df.columns:
        invalid_timestamps = df[df["last_production_timestamp"] > df["audit_timestamp"]]
        if not invalid_timestamps.empty:
            if debug:
                logger.info(
                    f"removing {len(invalid_timestamps)} records where last_production_timestamp is after audit_timestamp."
                )
            df = df.drop(invalid_timestamps.index)

    # check outliers
    high_defect_records = df[df["defects_count_from_production"] > df["defects_count_from_production"].quantile(0.99)]
    if not high_defect_records.empty:
        if debug:
            logger.info(f"removing {len(high_defect_records)} high production defects outliers.")
        df = df.drop(high_defect_records.index)

    high_repair_hours = df[df["hours_product_repaired_total"] > df["hours_product_repaired_total"].quantile(0.99)]
    if not high_repair_hours.empty:
        if debug:
            logger.info(f"removing {len(high_repair_hours)} high repair hours outliers.")
        df = df.drop(high_repair_hours.index)

    return df
