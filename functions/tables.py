import numpy as np
import polars as pl
import arviz as az
from IPython.display import HTML, display


PARAMETER_ORDER = ["V_max", "K_m", "h", "C_B_start", "b", "sigma"]


def _flatten(posterior, name):
    return posterior[name].transpose("chain", "draw", ...).values.reshape(-1)


def _parameter_samples(posterior):
    """Yield (label, samples) in display order. Vector parameters expand to one
    entry per experiment, labelled like az.summary (e.g. C_B_start[3.125])."""
    samples = []
    for name in PARAMETER_ORDER:
        if name not in posterior:
            continue
        variable = posterior[name]
        extra_dims = [dim for dim in variable.dims if dim not in ("chain", "draw")]
        if not extra_dims:
            samples.append((name, _flatten(posterior, name)))
        else:
            dim = extra_dims[0]
            for level in variable[dim].values:
                values = variable.sel({dim: level}).transpose("chain", "draw").values.reshape(-1)
                samples.append((f"{name}[{level}]", values))
        if name == "K_m" and "V_max" in posterior:
            samples.append(("V_max / K_m", _flatten(posterior, "V_max") / _flatten(posterior, "K_m")))
    return samples


def make_estimates_table(posterior):
    rows = []
    for label, values in _parameter_samples(posterior):
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1))
        lower, upper = np.quantile(values, [0.025, 0.975])
        rows.append(
            {
                "Parameter": label,
                "Mean": mean,
                "SD": sd,
                "Relative SD (%)": 100 * sd / abs(mean),
                "Lower 95%": float(lower),
                "Upper 95%": float(upper),
            }
        )
    return pl.DataFrame(rows)


def make_diagnostics_table(idata, parameters):
    summary = az.summary(idata, var_names=parameters, round_to="none", ci_prob=0.95)
    return (
        pl.from_pandas(summary[["ess_bulk", "ess_tail", "r_hat"]])
        .with_columns(pl.Series("Parameter", summary.index.to_list()))
        .select(
            "Parameter",
            pl.col("ess_bulk").alias("ESS bulk"),
            pl.col("ess_tail").alias("ESS tail"),
            pl.col("r_hat").alias("R̂"),
        )
    )


def make_fit_table(observed, posterior_mean, initial_substrate):
    frame = pl.DataFrame(
        {
            "Initial substrate (µM)": np.broadcast_to(initial_substrate, (len(observed),)),
            "observed": observed,
            "predicted": posterior_mean,
        }
    )

    def r_squared(group):
        residual = ((pl.col("observed") - pl.col("predicted")) ** 2).sum()
        total = ((pl.col("observed") - pl.col("observed").mean()) ** 2).sum()
        return group.select((1 - residual / total).alias("R²"))

    by_level = (
        frame.group_by("Initial substrate (µM)", maintain_order=True)
        .agg(
            ((pl.col("observed") - pl.col("predicted")) ** 2).sum().alias("residual_sum"),
            ((pl.col("observed") - pl.col("observed").mean()) ** 2).sum().alias("total_sum"),
        )
        .with_columns((1 - pl.col("residual_sum") / pl.col("total_sum")).alias("R²"))
        .select(pl.col("Initial substrate (µM)").cast(pl.String), "R²")
        .sort("Initial substrate (µM)")
    )

    if by_level.height == 1:
        return by_level

    overall = r_squared(frame).with_columns(pl.lit("All").alias("Initial substrate (µM)"))
    return pl.concat([by_level, overall.select("Initial substrate (µM)", "R²")])


def make_model_tables(idata, product_observed, initial_substrate):
    posterior = idata["posterior"]
    observed = np.asarray(product_observed)
    parameters = [name for name in PARAMETER_ORDER if name in posterior]
    posterior_mean = posterior["mu"].mean(dim=("chain", "draw")).values

    return {
        "estimates": make_estimates_table(posterior),
        "diagnostics": make_diagnostics_table(idata, parameters),
        "fit": make_fit_table(observed, posterior_mean, initial_substrate),
        "divergences": int(idata["sample_stats"]["diverging"].sum().item()),
    }


def make_all_tables(runs, substrates):
    tables = {}
    for name, run in runs.items():
        model, idata, prior, times, product_observed = run
        tables[name] = make_model_tables(idata, product_observed, substrates[name])
    return tables


def display_table(table):
    display(
        HTML(
            table.to_pandas().to_html(
                index=False, float_format=lambda value: f"{value:.6f}"
            )
        )
    )


def display_and_save_tables(model_tables, label, name, table_dir=None):
    """Display the three tables and the divergence count; write CSVs if table_dir is given."""
    for key in ["estimates", "diagnostics", "fit"]:
        print(f"{label}: {key}")
        display_table(model_tables[key])
        if table_dir is not None:
            model_tables[key].write_csv(table_dir / f"{name}_{key}.csv")
    print(f"{label}: divergences = {model_tables['divergences']}")
    if table_dir is not None:
        pl.DataFrame({"Divergences": [model_tables["divergences"]]}).write_csv(
            table_dir / f"{name}_divergences.csv"
        )