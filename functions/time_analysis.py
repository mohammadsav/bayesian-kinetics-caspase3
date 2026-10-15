import numpy as np
import polars as pl


def get_C_B(df, C_A0_array, idata):
    df_list = []

    for i in range(len(C_A0_array)):
        offset = (
            idata["posterior"]["b"]
            .sel(experiment=C_A0_array[i])
            .mean()
            .item()
        )

        one_group = (
            df
            .filter(pl.col("Initial_Conc_Substrate_uM") == C_A0_array[i])
            ["Concentration_of_Product"]
            .to_numpy()
        )

        one_group = one_group - offset
        df_list.append(one_group)

    return df_list

def t_5(df, C_B_array, C_B0_array, C_A0_array, K_m, V_max):
    df_list = []

    for i in range(len(C_A0_array)):
        C_B = C_B_array[i]
        C_B0 = C_B0_array[i]
        C_A0 = C_A0_array[i]

        real_times = (
            df
            .filter(pl.col("Initial_Conc_Substrate_uM") == C_A0)
            ["time"]
            .to_numpy()
        )

        with np.errstate(invalid="ignore", divide="ignore"):
            t = (
                C_B - C_B0
                - K_m * np.log((C_A0 - C_B) / (C_A0 - C_B0))
            ) / V_max

        df_group = pl.DataFrame({
            "t": t,
            "real_time": real_times,
            "C_A0": np.repeat(C_A0, len(t)),
            "C_B": C_B,
        })

        df_list.append(df_group)

    return pl.concat(df_list)

def t_6(df, C_B_array, C_B0_array, C_A0_array, K_m, V_max, h):
    df_list = []

    for i, C_A0 in enumerate(C_A0_array):
        C_B = C_B_array[i]
        C_B0 = C_B0_array[i]

        real_times = (
            df
            .filter(pl.col("Initial_Conc_Substrate_uM") == C_A0)
            ["time"]
            .to_numpy()
        )

        C_A = np.where(C_A0 - C_B >= 0, C_A0 - C_B, np.nan)
        C_A_start = C_A0 - C_B0

        with np.errstate(invalid="ignore", divide="ignore"):
            if h == 1:
                t = (
                    C_B - C_B0
                    - K_m * np.log(C_A / C_A_start)
                ) / V_max
            else:
                t = (
                    C_B - C_B0
                    + K_m**h
                    * (C_A**(1 - h) - C_A_start**(1 - h))
                    / (h - 1)
                ) / V_max

        df_list.append(
            pl.DataFrame({
                "t": t,
                "real_time": real_times,
                "C_A0": np.repeat(C_A0, len(C_B)),
                "C_B": C_B
            })
        )

    return pl.concat(df_list)




def prepare_time_data(df, idata, hill=False):
    df = df.sort(["Initial_Conc_Substrate_uM", "time"])
    levels = df["Initial_Conc_Substrate_uM"].unique().sort().to_numpy()
    posterior = idata["posterior"]
    V_max = posterior["V_max"].mean().item()
    K_m = posterior["K_m"].mean().item()
    starts = posterior["C_B_start"].sel(experiment=levels).mean(dim=("chain", "draw")).values
    products = get_C_B(df, levels, idata)
    if hill:
        data = t_6(df, products, starts, levels, K_m, V_max, posterior["h"].mean().item())
    else:
        data = t_5(df, products, starts, levels, K_m, V_max)
    return (
        data.sort(["C_A0", "real_time"])
        .with_columns((pl.col("real_time") - pl.col("t")).alias("residual"))
        .select("t", "real_time", "residual", "C_A0", "C_B")
        .with_columns((100 * pl.col("C_B").diff().over("C_A0")
            / pl.col("C_B").shift(1).over("C_A0")).alias("concentration_change_pct"))
        .with_columns((pl.col("concentration_change_pct")
            / pl.col("real_time").diff().over("C_A0")).alias("concentration_change_pct_per_hour"))
    )


def cut_plateau(data, cutoff=10, consecutive=2):
    groups = []
    for group in data.partition_by("C_A0", maintain_order=True):
        group = group.sort("real_time")
        rate = group["concentration_change_pct_per_hour"]
        within = (rate.is_finite() & (rate.abs() <= cutoff)).fill_null(False)
        ends = np.flatnonzero(
            within.cast(pl.Int64).rolling_sum(window_size=consecutive).fill_null(0).to_numpy()
            == consecutive
        )
        if len(ends):
            group = group.head(int(ends[0] - consecutive + 1))
        groups.append(group)
    return pl.concat(groups)


def time_r_squared(data):
    valid = data.filter(pl.col("t").is_finite() & pl.col("real_time").is_finite())
    grouped = (
        valid.group_by("C_A0")
        .agg(
            pl.len().alias("Rows used"),
            ((pl.col("real_time") - pl.col("t")) ** 2).sum().alias("SSE"),
            ((pl.col("real_time") - pl.col("real_time").mean()) ** 2).sum().alias("SST")
        )
        .with_columns(pl.when(pl.col("SST") > 0)
            .then(1 - pl.col("SSE") / pl.col("SST")).otherwise(float("nan")).alias("R²"))
        .sort("C_A0")
    )
    actual = valid["real_time"].to_numpy()
    predicted = valid["t"].to_numpy()
    total = np.sum((actual - actual.mean()) ** 2) if len(actual) else 0
    overall = 1 - np.sum((actual - predicted) ** 2) / total if total > 0 else float("nan")
    return grouped, float(overall)


def retained_rows(original, cut):
    return (
        original.group_by("C_A0").agg(pl.len().alias("Total"))
        .join(cut.group_by("C_A0").agg(pl.len().alias("Retained rows")), on="C_A0", how="left")
        .with_columns(pl.col("Retained rows").fill_null(0))
        .with_columns((pl.col("Total") - pl.col("Retained rows")).alias("Lost rows"))
        .select("C_A0", "Retained rows", "Lost rows", "Total").sort("C_A0")
    )
def compare_final_C_B(time_data_5, time_data_5_cut, time_data_6, time_data_6_cut):
    frames = []

    for model, full, cut in [
        ("Model 5", time_data_5, time_data_5_cut),
        ("Model 6", time_data_6, time_data_6_cut)
    ]:
        full_last = (
            full.sort("real_time")
            .group_by("C_A0")
            .agg(pl.col("C_B").last().alias("Full C_B"))
        )

        cut_last = (
            cut.sort("real_time")
            .group_by("C_A0")
            .agg(pl.col("C_B").last().alias("Cut C_B"))
        )

        frames.append(
            full_last
            .join(cut_last, on="C_A0", how="left")
            .with_columns(
                pl.lit(model).alias("Model"),
                (pl.col("Full C_B") - pl.col("Cut C_B"))
                .abs()
                .alias("Absolute difference")
            )
            .with_columns(
                (
                    100 * pl.col("Absolute difference")
                    / pl.col("Full C_B").abs()
                ).alias("Difference (%)")
            )
            .select(
                "Model", "C_A0", "Full C_B", "Cut C_B",
                "Absolute difference", "Difference (%)"
            )
        )

    return pl.concat(frames).sort(["Model", "C_A0"])