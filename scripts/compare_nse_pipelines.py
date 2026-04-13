from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import polars as pl

ROOT = Path("/Users/nicolaslazaro/Desktop/work/hydrocast")
WV_ROOT = Path("/Users/nicolaslazaro/Desktop/thirdparty/wasservorhersage")
TLP_ROOT = Path("/Users/nicolaslazaro/Desktop/work/transfer-learning-publication")

sys.path.insert(0, str(ROOT / "coach" / "src"))

coach_metrics = importlib.import_module("coach.metrics")
coach_evaluate = importlib.import_module("coach.evaluate")


def _register_package(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    sys.modules[name] = module


_register_package(
    "transfer_learning_publication",
    TLP_ROOT / "src" / "transfer_learning_publication",
)
_register_package(
    "transfer_learning_publication.evaluation",
    TLP_ROOT / "src" / "transfer_learning_publication" / "evaluation",
)
_register_package(
    "transfer_learning_publication.evaluation.metrics",
    TLP_ROOT / "src" / "transfer_learning_publication" / "evaluation" / "metrics",
)

_register_package("wasservorhersage", WV_ROOT / "wasservorhersage")
_register_package(
    "wasservorhersage.evaluation",
    WV_ROOT / "wasservorhersage" / "evaluation",
)

tl_nse = importlib.import_module("transfer_learning_publication.evaluation.metrics.nse").NSE()
MetricCalculator = importlib.import_module("transfer_learning_publication.evaluation.calculator").MetricCalculator
wv_metrics = importlib.import_module("wasservorhersage.evaluation.metrics")


def build_clean_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rows = [
        {
            "group_id": "A",
            "issue_date": "1999-12-31",
            "lead_time": 1,
            "prediction_date": "2000-01-01",
            "prediction": 1.0,
            "observation": 1.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "A",
            "issue_date": "2000-01-01",
            "lead_time": 1,
            "prediction_date": "2000-01-02",
            "prediction": 2.0,
            "observation": 2.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "A",
            "issue_date": "2000-01-02",
            "lead_time": 1,
            "prediction_date": "2000-01-03",
            "prediction": 4.0,
            "observation": 3.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "B",
            "issue_date": "1999-12-31",
            "lead_time": 1,
            "prediction_date": "2000-01-01",
            "prediction": 1.0,
            "observation": 2.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "B",
            "issue_date": "2000-01-01",
            "lead_time": 1,
            "prediction_date": "2000-01-02",
            "prediction": 3.0,
            "observation": 3.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "B",
            "issue_date": "2000-01-02",
            "lead_time": 1,
            "prediction_date": "2000-01-03",
            "prediction": 5.0,
            "observation": 4.0,
            "observation_was_filled": 0,
        },
    ]
    coach_df = pl.DataFrame(rows).with_columns(
        pl.col("issue_date").str.to_date(),
        pl.col("prediction_date").str.to_date(),
    )
    tl_df = coach_df.rename({"group_id": "group_identifier"}).with_columns(
        pl.lit("dummy").alias("model_name"),
    )
    wv_predictions = coach_df.rename(
        {"group_id": "gauge_id", "prediction_date": "datetime", "prediction": "streamflow_pred"}
    ).with_columns(
        pl.col("datetime").cast(pl.Date),
        pl.col("issue_date").alias("forecast_issue_dt"),
    )
    wv_original = (
        coach_df.rename({"group_id": "gauge_id", "prediction_date": "datetime", "observation": "streamflow_original"})
        .select(["gauge_id", "datetime", "streamflow_original"])
        .with_columns(pl.col("datetime").cast(pl.Date))
    )
    return coach_df, tl_df, wv_predictions.join(wv_original, on=["gauge_id", "datetime"], how="left")


def build_duplicate_frame() -> pl.DataFrame:
    rows = [
        {
            "group_id": "A",
            "issue_date": "2000-03-01",
            "lead_time": 10,
            "prediction_date": "2000-03-11",
            "prediction": 1.0,
            "observation": 2.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "A",
            "issue_date": "2000-03-02",
            "lead_time": 9,
            "prediction_date": "2000-03-11",
            "prediction": 3.0,
            "observation": 2.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "A",
            "issue_date": "2000-03-02",
            "lead_time": 10,
            "prediction_date": "2000-03-12",
            "prediction": 4.0,
            "observation": 5.0,
            "observation_was_filled": 0,
        },
        {
            "group_id": "A",
            "issue_date": "2000-03-03",
            "lead_time": 9,
            "prediction_date": "2000-03-12",
            "prediction": 6.0,
            "observation": 5.0,
            "observation_was_filled": 0,
        },
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("issue_date").str.to_date(),
        pl.col("prediction_date").str.to_date(),
    )


def compute_wv_per_basin_nse(df: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    grouped = df.group_by(["gauge_id", "forecast_horizon"], maintain_order=True)
    for (gauge_id, horizon), group_df in grouped:
        rows.append(
            {
                "gauge_id": gauge_id,
                "forecast_horizon": horizon,
                "nse": wv_metrics.nse(group_df["streamflow_pred"], group_df["streamflow_original"]),
            }
        )
    return pl.DataFrame(rows)


def main() -> None:
    coach_df, tl_df, wv_joined = build_clean_frames()

    print("== Raw Formula Check ==")
    preds = pl.Series([1.0, 2.0, 4.0])
    obs = pl.Series([1.0, 2.0, 3.0])
    print("coach.metrics.nse:", coach_metrics.nse(preds, obs))
    print("tl NSE metric:", tl_nse.compute(preds, obs))
    print("wv metrics.nse:", wv_metrics.nse(preds, obs))
    print()

    print("== Clean Per-Basin Check ==")
    coach_metrics_df = coach_evaluate._compute_per_basin_metrics(coach_df)
    coach_aggregate = float(coach_metrics_df["nse"].drop_nulls().drop_nans().median())

    tl_calc = MetricCalculator(tl_df.lazy())
    tl_metrics_df = tl_calc.compute_metrics(
        metrics=["nse"],
        exclude_filled=True,
        lead_times=[1],
        group_by=["group_identifier"],
    )
    tl_aggregate = float(tl_metrics_df["NSE"].drop_nulls().drop_nans().median())

    wv_detailed = wv_joined.with_columns(
        ((pl.col("datetime") - pl.col("forecast_issue_dt")).dt.total_days()).alias("forecast_horizon")
    )
    wv_metrics_df = compute_wv_per_basin_nse(wv_detailed)
    wv_aggregate = float(wv_metrics_df["nse"].drop_nulls().drop_nans().median())

    print("coach per basin:")
    print(coach_metrics_df)
    print("coach median_nse:", coach_aggregate)
    print()

    print("transfer-learning-publication per basin:")
    print(tl_metrics_df)
    print("transfer-learning-publication median_nse:", tl_aggregate)
    print()

    print("wasservorhersage per basin+horizon:")
    print(wv_metrics_df)
    print("wasservorhersage horizon-1 median_nse:", wv_aggregate)
    print()

    print("== Overlapping Forecast Events ==")
    duplicate_df = build_duplicate_frame()
    coach_duplicate = coach_evaluate._compute_per_basin_metrics(duplicate_df)
    coach_by_lead = coach_evaluate._compute_metrics_grouped(
        coach_evaluate._prepare_metric_frame(duplicate_df),
        group_by=("group_id", "lead_time"),
    )
    tl_duplicate = MetricCalculator(
        duplicate_df.rename({"group_id": "group_identifier"})
        .with_columns(
            pl.lit("dummy").alias("model_name"),
        )
        .lazy()
    ).compute_metrics(
        metrics=["nse"],
        exclude_filled=True,
        lead_times=[9, 10],
        group_by=["group_identifier", "lead_time"],
    )
    print("coach pooled forecast-event handling:")
    print(coach_duplicate)
    print("coach per basin + lead_time:")
    print(coach_by_lead)
    print("transfer-learning-publication per basin + lead_time:")
    print(tl_duplicate)


if __name__ == "__main__":
    main()
