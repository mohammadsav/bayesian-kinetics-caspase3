from io import BytesIO
from itertools import combinations
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import numpy as np
import polars as pl
import pymc as pm
from PIL import Image
from scipy.stats import gaussian_kde
from plotnine import *



RANDOM_SEED = 67


def _observations(times, product_observed, initial_substrate):
    return pl.DataFrame(
        {
            "time": np.asarray(times),
            "product": np.asarray(product_observed),
            "initial_substrate": np.broadcast_to(initial_substrate, (len(times),)),
        }
    )


def prepare_prior_data(prior, times, product_observed, initial_substrate):
    observed_data = _observations(times, product_observed, initial_substrate)
    frames = []
    for group, variable, panel in [
        ("prior", "mu", "Prior Expected-Response Curves"),
        ("prior_predictive", "obs", "Prior Predictive Measurements"),
    ]:
        values = prior[group][variable].values
        for chain in range(values.shape[0]):
            for draw in range(values.shape[1]):
                frames.append(
                    observed_data.with_columns(
                        pl.Series("product", values[chain, draw, :]),
                        pl.lit(chain).alias("chain"),
                        pl.lit(draw).alias("draw"),
                        pl.lit(f"{chain}_{draw}").alias("sample"),
                        pl.lit(panel).alias("panel"),
                    )
                )
    return [pl.concat(frames), observed_data]


def plot_prior_check(data, model_label=""):
    prior_data, observed_data = data
    n = observed_data["initial_substrate"].n_unique()
    return (
        ggplot(prior_data, aes(x="time", y="product"))
        + geom_line(aes(group="sample"), color="gray", alpha=0.3, size=0.4)
        + geom_point(data=observed_data, color="black", size=1.5)
        + facet_grid(
            "initial_substrate ~ panel", scales="free_y", labeller="label_both"
        )
        + labs(
            title=f"{model_label}: Prior Checks".lstrip(": "),
            subtitle="Initial substrate in µM; black points are observations",
            x="Time (hours)",
            y="Measured product concentration (µM)",
        )
        + theme_gray()
        + theme(figure_size=(11, 3 * n))
    )


def _r_squared(curve):
    observed = curve["product"].to_numpy()
    total = np.sum((observed - observed.mean()) ** 2)
    if total == 0:
        return "undefined"
    residual = observed - curve["posterior_mean"].to_numpy()
    return f"{1 - np.sum(residual ** 2) / total:.6f}"


def plot_model_dags(runs, ncols=2, model_labels=None):
    runs = list(runs)
    labels = (
        list(model_labels)
        if model_labels is not None
        else [f"Model {i + 1}" for i in range(len(runs))]
    )
    frames = []
    for label, run in zip(labels, runs):
        dag = pm.model_to_graphviz(run[0])
        dag.attr(bgcolor="white", dpi="100")
        with Image.open(BytesIO(dag.pipe(format="png"))) as image:
            pixels = np.asarray(image.convert("L"))
        height, width = pixels.shape
        frames.append(
            pl.DataFrame(
                {
                    "x": np.tile(np.arange(width), height),
                    "y": np.repeat(np.arange(height), width),
                    "shade": pixels.ravel(),
                }
            ).with_columns(pl.lit(label).alias("Model"))
        )
    data = pl.concat(frames).with_columns(pl.col("Model").cast(pl.Enum(labels)))
    ncols = min(ncols, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    return (
        ggplot(data, aes(x="x", y="y", fill="shade"))
        + geom_raster()
        + scale_fill_gradient(low="black", high="white", limits=(0, 255), guide=None)
        + scale_y_reverse()
        + coord_fixed()
        + facet_wrap("~Model", ncol=ncols)
        + theme_void()
        + theme(
            figure_size=(5 * ncols, 5 * nrows),
            strip_text=element_text(size=12, weight="bold"),
        )
    )

def save_model_dags(dag_images, svg_dir, png_dir, dpi=300):
    svg_folder = Path(svg_dir) / "dags"
    png_folder = Path(png_dir) / "dags"

    svg_folder.mkdir(parents=True, exist_ok=True)
    png_folder.mkdir(parents=True, exist_ok=True)

    saved = {}
    figure, axes = plt.subplots(3, 2, figsize=(12, 15))

    for number, (axis, image_bytes) in enumerate(
        zip(axes.flat, dag_images), start=1
    ):
        with Image.open(BytesIO(image_bytes)) as image:
            pixels = np.asarray(image.convert("RGB"))

        axis.imshow(pixels)
        axis.set_title(f"Model {number}")
        axis.axis("off")
        axis.add_patch(
            Rectangle(
                (0, 0), 1, 1,
                transform=axis.transAxes,
                fill=False,
                edgecolor="black",
                linewidth=1,
                clip_on=False,
            )
        )

        name = f"model{number}_dag"
        png_path = png_folder / f"{name}.png"
        svg_path = svg_folder / f"{name}.svg"

        png_path.write_bytes(image_bytes)

        individual, individual_axis = plt.subplots(figsize=(6, 7.5))
        individual_axis.imshow(pixels)
        individual_axis.set_title(f"Model {number}")
        individual_axis.axis("off")
        individual.tight_layout()
        individual.savefig(svg_path, format="svg", bbox_inches="tight")
        plt.close(individual)

        saved[name] = {"svg": svg_path, "png": png_path}

    figure.tight_layout()

    svg_path = svg_folder / "model_dags.svg"
    png_path = png_folder / "model_dags.png"

    figure.savefig(svg_path, format="svg")
    figure.savefig(png_path, format="png", dpi=dpi)
    plt.close(figure)

    saved["model_dags"] = {"svg": svg_path, "png": png_path}
    return saved



def plot_posterior_predictive_zoom(
    model,
    idata,
    times,
    product_observed,
    initial_substrate,
    random_seed=RANDOM_SEED,
    zoom_start=0.4,
    zoom_end=0.6,
    model_label="",
    zoom=True,
):
    data = _observations(times, product_observed, initial_substrate)
    mean = idata["posterior"]["mu"].values.mean(axis=(0, 1))
    with model:
        predictive = pm.sample_posterior_predictive(
            idata, var_names=["obs"], random_seed=random_seed
        )
    predictions = predictive["posterior_predictive"]["obs"].values
    data = data.with_columns(
        pl.Series("posterior_mean", mean),
        pl.Series("lower", np.quantile(predictions, 0.025, axis=(0, 1))),
        pl.Series("upper", np.quantile(predictions, 0.975, axis=(0, 1))),
    )
    curves = data.sort("initial_substrate").partition_by(
        "initial_substrate", maintain_order=True
    )
    reference = curves[0].filter(pl.col("time").is_between(zoom_start, zoom_end))
    lo = min(reference["lower"].min(), reference["product"].min())
    hi = max(reference["upper"].max(), reference["product"].max())
    zoom_height = max(1.2 * (hi - lo), 1e-06)
    rows = []
    for curve in curves:
        curve = curve.sort("time")
        base = (
            ggplot(curve, aes(x="time"))
            + geom_ribbon(aes(ymin="lower", ymax="upper"), fill="steelblue", alpha=0.2)
            + geom_line(aes(y="lower"), color="steelblue", linetype="dashed", size=0.4)
            + geom_line(aes(y="upper"), color="steelblue", linetype="dashed", size=0.4)
            + geom_line(aes(y="posterior_mean"), color="darkblue", size=0.8)
            + geom_point(aes(y="product"), color="black", size=1.3)
            + labs(x="Time (hours)", y="Measured product (µM)")
            + theme_gray()
        )
        title = f"Initial substrate: {curve['initial_substrate'][0]:g} µM"
        full = base + labs(
            title=f"{model_label}: {title}".lstrip(": "), subtitle=f"Full curve | R² = {_r_squared(curve)}"
        )
        if not zoom:
            rows.append(full)
            continue
        center = np.interp(
            (zoom_start + zoom_end) / 2,
            curve["time"].to_numpy(),
            curve["posterior_mean"].to_numpy(),
        )
        zoom_panel = (
            base
            + coord_cartesian(
                xlim=(zoom_start, zoom_end),
                ylim=(center - zoom_height / 2, center + zoom_height / 2),
                expand=False,
            )
            + labs(
                title=f"{model_label}: Zoom {zoom_start}–{zoom_end} hours".lstrip(": "),
                subtitle=f"Shared vertical span: {zoom_height:.3f} µM",
            )
        )
        rows.append(full | zoom_panel)
    output = rows[0]
    for row in rows[1:]:
        output = output / row
    width = 14 if zoom else 8
    return output + theme(figure_size=(width, 3 * len(rows)))


def plot_residual_comparison(
    idata, times, product_observed, initial_substrate, model_label=""
):
    data = _observations(times, product_observed, initial_substrate)
    mean = idata["posterior"]["mu"].values.mean(axis=(0, 1))

    data = data.with_columns(
        pl.Series("posterior_mean", mean)
    ).with_columns(
        (pl.col("product") - pl.col("posterior_mean")).alias("residual")
    )

    n = data["initial_substrate"].n_unique()

    panels = [
        ("Residual (µM)", pl.col("residual"))
    ]

    if n > 1:
        panels.append(
            (
                "Residual (% of initial substrate)",
                100 * pl.col("residual") / pl.col("initial_substrate"),
            )
        )

    plots = []

    for title, expression in panels:
        panel_data = (
            data
            .with_columns(expression.alias("value"))
            .sort(["initial_substrate", "time"])
        )

        plot = (
            ggplot(panel_data, aes(x="time", y="value"))
            + geom_point(color="black", size=1.5, alpha=0.7)
            + geom_hline(
                yintercept=0,
                linetype="dashed",
                color="steelblue",
            )
            + facet_wrap(
                "~initial_substrate",
                ncol=1,
                scales="fixed",
                labeller="label_both",
            )
            + labs(
                title=f"{model_label}: {title}".lstrip(": "),
                subtitle="Initial substrate concentration in µM",
                x="Time (hours)",
                y=title,
            )
            + theme_gray()
        )

        plots.append(plot)

    if n == 1:
        return plots[0] + theme(figure_size=(8.5, 4.5))

    return (plots[0] | plots[1]) + theme(
        figure_size=(13, 2.5 * n)
    )


def plot_joint_posterior(idata, model_label=""):
    posterior = idata["posterior"]
    names = ["V_max", "K_m"] + (["h"] if "h" in posterior else [])
    labels = {"V_max": "V_max (µM/hour)", "K_m": "K_m (µM)", "h": "h (dimensionless)"}
    if "h" in posterior:
        labels["K_m"] = "K₀.₅ (µM)"
    values = {name: posterior[name].values.flatten() for name in names}
    plots = []
    for x, y in combinations(names, 2):
        plots.append(
            ggplot(pl.DataFrame({x: values[x], y: values[y]}), aes(x=x, y=y))
            + geom_point(color="darkblue", alpha=0.2, size=1.5)
            + labs(title=f"{model_label}: Joint Posterior: {x} and {y}".lstrip(": "), x=labels[x], y=labels[y])
            + theme_gray()
        )
    output = plots[0]
    for panel in plots[1:]:
        output = output | panel
    return output + theme(figure_size=(7, 5) if len(plots) == 1 else (16, 5))


def plot_prior_posterior_densities(prior, idata, model_label=""):
    prior_group, posterior = (prior["prior"], idata["posterior"])
    names = ["V_max", "K_m"] + (["h"] if "h" in posterior else [])
    pairs = [
        (name, prior_group[name].values.flatten(), posterior[name].values.flatten())
        for name in names
    ]
    pv, pk = (pairs[0][1], pairs[1][1])
    qv, qk = (pairs[0][2], pairs[1][2])
    pairs.append(("V_max / K_m", pv / pk, qv / qk))
    frames, intervals = ([], [])
    for parameter, prior_values, posterior_values in pairs:
        x_max = (
            max(np.quantile(prior_values, 0.995), np.quantile(posterior_values, 0.995))
            * 1.1
        )
        x = np.geomspace(x_max / 100000, x_max, 2000)
        for distribution, samples in [
            ("Prior", prior_values),
            ("Posterior", posterior_values),
        ]:
            kde = gaussian_kde(np.log(samples))
            frames.append(
                pl.DataFrame({"Value": x, "Density": kde(np.log(x)) / x}).with_columns(
                    pl.lit(parameter).alias("Parameter"),
                    pl.lit(distribution).alias("Distribution"),
                )
            )
        for bound in np.quantile(posterior_values, [0.025, 0.975]):
            intervals.append({"Parameter": parameter, "Bound": bound})
    order = [p[0] for p in pairs]
    data = pl.concat(frames).with_columns(pl.col("Parameter").cast(pl.Enum(order)))
    interval_data = pl.DataFrame(intervals).with_columns(
        pl.col("Parameter").cast(pl.Enum(order))
    )
    subtitle = "Dashed lines: posterior 95% credible intervals"
    if "h" in posterior:
        subtitle += "; V_max/K_m is a descriptive ratio for the Hill model"
    return (
        ggplot(data, aes(x="Value", y="Density", color="Distribution"))
        + geom_line(size=0.8)
        + geom_vline(
            data=interval_data,
            mapping=aes(xintercept="Bound"),
            inherit_aes=False,
            linetype="dashed",
            color="black",
            size=0.5,
        )
        + facet_wrap("~Parameter", nrow=1, scales="free")
        + scale_color_manual(values={"Prior": "darkorange", "Posterior": "darkblue"})
        + labs(
            title=f"{model_label}: Prior vs Posterior Parameter Densities".lstrip(": "),
            subtitle=subtitle,
            x="Parameter value",
            y="Density",
            color="",
        )
        + theme_gray()
        + theme(figure_size=(14 if "h" in posterior else 11, 4.5))
    )


def make_all_plots(runs, substrates, random_seed=RANDOM_SEED):
    plots = {
        "dags": plot_model_dags(list(runs.values()), model_labels=[name.replace("model", "Model ") for name in runs])
    }
    for name, run in runs.items():
        model, idata, prior, times, product_observed = run
        model_label = name.replace("model", "Model ")
        substrate = substrates[name]
        prior_data = prepare_prior_data(
            prior, times, product_observed, initial_substrate=substrate
        )
        plots[name] = {
            "prior": plot_prior_check(prior_data, model_label=model_label),
            "posterior_zoom": plot_posterior_predictive_zoom(
                model,
                idata,
                times,
                product_observed,
                initial_substrate=substrate,
                random_seed=random_seed,
                model_label=model_label,
            ),
            "residuals": plot_residual_comparison(
                idata, times, product_observed, initial_substrate=substrate, model_label=model_label
            ),
            "joint_posterior": plot_joint_posterior(idata, model_label=model_label),
            "parameter_densities": plot_prior_posterior_densities(prior, idata, model_label=model_label),
        }
    return plots
