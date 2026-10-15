import polars as pl


def load_data(path="../data/caspase.csv"):
    df_raw = pl.read_csv(path, separator="\t")
    time_columns = df_raw.select(pl.col("^Time_.*$")).columns
    df_long = (
        df_raw.drop(time_columns[1:])
        .rename({"Time_hr_3.125": "Time_hr"})
        .unpivot(
            index="Time_hr",
            variable_name="Initial_Conc_Substrate_uM",
            value_name="Concentration_of_Product",
        )
        .with_columns(
            pl.col("Initial_Conc_Substrate_uM")
            .str.strip_prefix("Conc_uM_")
            .cast(pl.Float64)
        )
    )
    return df_long


def prepare_model_frame(df_long, model_number):
    frame = df_long
    if model_number <= 4:
        frame = frame.filter(pl.col("Initial_Conc_Substrate_uM") == 3.125)
    if model_number != 1:
        frame = frame.filter(pl.col("Time_hr") > 10 / 60)
    frame = frame.sort(["Initial_Conc_Substrate_uM", "Time_hr"])
    if model_number == 1:
        frame = frame.with_columns(pl.col("Time_hr").alias("time"))
    else:
        frame = frame.with_columns(
            (pl.col("Time_hr") - pl.col("Time_hr").min()
             .over("Initial_Conc_Substrate_uM")).alias("time")
        )
    if model_number == 2:
        frame = frame.slice(1)
    return frame


def prepare_model_frames(df_long):
    return [prepare_model_frame(df_long, number) for number in range(1, 7)]


def frame_arrays(frame):
    return frame["time"].to_numpy(), frame["Concentration_of_Product"].to_numpy()


def joint_frame_arrays(frame):
    grouped = frame.group_by("Initial_Conc_Substrate_uM", maintain_order=True).agg("time")
    levels = grouped["Initial_Conc_Substrate_uM"].to_numpy()
    curve_times = grouped["time"].to_list()
    times, observed = frame_arrays(frame)
    return frame, levels, curve_times, times, observed


def prepare_model1_data(df_long):
    return frame_arrays(prepare_model_frame(df_long, 1))


def prepare_model2_data(df_long):
    retained = prepare_model_frame(df_long, 3)
    C_B_initial = retained["Concentration_of_Product"][0]
    times, observed = frame_arrays(retained.slice(1))
    return times, observed, C_B_initial


def prepare_model3_data(df_long):
    return frame_arrays(prepare_model_frame(df_long, 3))


def prepare_model4_data(df_long):
    return frame_arrays(prepare_model_frame(df_long, 4))


def prepare_model5_data(df_long):
    return joint_frame_arrays(prepare_model_frame(df_long, 5))


def prepare_model6_data(df_long):
    return joint_frame_arrays(prepare_model_frame(df_long, 6))
