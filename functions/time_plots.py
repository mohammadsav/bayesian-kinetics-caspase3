import numpy as np
import polars as pl
from plotnine import (ggplot, aes, geom_point, geom_line, geom_text, coord_cartesian,
    geom_vline, geom_segment,
    labs, theme_gray, theme, scale_x_continuous)


def plot_time_overall(data, title="Concentration vs Time",
                      subtitle="Points: measured times | Lines: predicted times"):
    highest_points = (
        data
        .sort("real_time")
        .group_by("C_A0", maintain_order=True)
        .last()
        .with_columns(
            pl.col("C_B")
            .map_elements(
                lambda value: f"{value:.3f} µM",
                return_dtype=pl.String
            )
            .alias("label")
        )
    )

    output = (
        ggplot(
            data,
            aes(y="C_B", color="factor(C_A0)")
        )
        + geom_point(aes(x="real_time"), size=1.5)
        + geom_line(
            data=data.filter(pl.col("t").is_finite()),
            mapping=aes(x="t", group="C_A0"),
            size=0.8
        )
        + geom_text(
            data=highest_points,
            mapping=aes(x="real_time", label="label"),
            nudge_x=0.05,
            nudge_y=0.8,
            ha="left",
            size=9,
            show_legend=False
        )
        + scale_x_continuous(expand=(0.05, 0, 0.25, 0))
        + labs(
            title=title,
            subtitle=subtitle,
            x="Time (hours)",
            y="Offset-corrected product (µM)",
            color="Initial substrate (µM)"
        )
        + theme_gray()
        + theme(figure_size=(9, 6))
    )

    return output


def plot_tldr(data, model_label="Model 5"):
    """The after-cutoff time plot with wording for a reader who has not seen
    the rest of the document."""
    return plot_time_overall(
        data,
        title="Product concentration (µM) vs time (hours)",
        subtitle=(f"Measured vs predicted time to reach each product concentration\n"
                  f"Points: what was measured | Lines: what {model_label} predicts | "
                  f"One color per starting substrate concentration"),
    )

def plot_time_zoom(data, zoom_start, zoom_end, zoom_percent=10, model_label=""):
    concentrations = sorted(
        data["C_A0"].unique().to_list(),
        reverse=True
    )

    plots = []

    for concentration in concentrations:
        curve = (
            data
            .filter(pl.col("C_A0") == concentration)
            .sort("real_time")
        )

        predicted = (
            curve
            .filter(pl.col("t").is_finite())
            .sort("t")
        )

        zoom_center = np.interp(
            (zoom_start + zoom_end) / 2,
            curve["real_time"].to_numpy(),
            curve["C_B"].to_numpy()
        )

        zoom_height = concentration * zoom_percent / 100

        plot = (
            ggplot(curve, aes(y="C_B"))
            + geom_point(
                aes(x="real_time"),
                color="black",
                size=1.5
            )
            + geom_line(
                data=predicted,
                mapping=aes(x="t"),
                color="steelblue",
                size=0.8
            )
            + coord_cartesian(
                xlim=(zoom_start, zoom_end),
                ylim=(
                    zoom_center - zoom_height / 2,
                    zoom_center + zoom_height / 2
                ),
                expand=False
            )
            + labs(
                title=f"{model_label}: C_A0 = {concentration:g} µM".lstrip(": "),
                subtitle=(
                    f"{zoom_start:g}–{zoom_end:g} hours | "
                    f"Y span: {zoom_height:g} µM "
                    f"({zoom_percent:g}% of C_A0)"
                ),
                x="Time (hours)",
                y="Offset-corrected product (µM)"
            )
            + theme_gray()
        )

        plots.append(plot)

    output = plots[0]

    for plot in plots[1:]:
        output = output / plot

    return output + theme(
        figure_size=(8, 3 * len(concentrations))
    )



def make_zoom_plots(data, interval=0.5, zoom_percent=10, model_label=""):
    end = data["real_time"].max()
    return [plot_time_zoom(data, start, min(start + interval, end), zoom_percent, model_label)
            for start in np.arange(0, end, interval)]


def plot_time_sensitivity(V_max, K_m, C_A0, C_B0, sigma, model_label="",
                          probe_fractions=(0.6, 0.9),
                          lower_color="#2E8B57", upper_color="#B03A2E"):
    """Integrated Michaelis-Menten time t(C_B) for one experiment. At two
    product levels, a green segment runs from t(C_B - sigma) up to t(C_B) and a
    red segment from t(C_B) up to t(C_B + sigma), to show how a +/- sigma
    measurement error maps to predicted time and how that grows near the plateau."""
    remaining = C_A0 - C_B0

    def t_of(c):
        return (c - C_B0 - K_m * np.log((C_A0 - c) / remaining)) / V_max

    probes = []
    for fraction in probe_fractions:
        center = C_B0 + fraction * remaining
        low, high = center - sigma, min(center + sigma, C_A0 - 1e-6 * remaining)
        probes.append({
            "C_B": center, "t": t_of(center),
            "t_low": t_of(low), "t_high": t_of(high),
            "label_low": f"-{t_of(center) - t_of(low):.2f} h",
            "label_high": f"+{t_of(high) - t_of(center):.2f} h",
        })
    probe_data = pl.DataFrame(probes)

    y_max = 1.15 * probe_data["t_high"].max()
    C_B = np.linspace(C_B0, C_A0 - 1e-4 * remaining, 2000)
    curve = pl.DataFrame({"C_B": C_B, "t": t_of(C_B)})

    return (
        ggplot(curve, aes(x="C_B", y="t"))
        + geom_line(color="darkblue", size=0.9)
        + geom_vline(xintercept=C_A0, linetype="dashed", color="black")
        + geom_segment(
            data=probe_data,
            mapping=aes(x="C_B", xend="C_B", y="t_low", yend="t"),
            inherit_aes=False, color=lower_color, size=1.4,
        )
        + geom_segment(
            data=probe_data,
            mapping=aes(x="C_B", xend="C_B", y="t", yend="t_high"),
            inherit_aes=False, color=upper_color, size=1.4,
        )
        + geom_point(data=probe_data, mapping=aes(x="C_B", y="t"),
                     inherit_aes=False, color="black", size=2.5)
        + geom_text(
            data=probe_data,
            mapping=aes(x="C_B", y="t_low", label="label_low"),
            inherit_aes=False, ha="right", va="top", size=10, color=lower_color,
            nudge_x=-0.015 * remaining, nudge_y=-0.01 * y_max,
        )
        + geom_text(
            data=probe_data,
            mapping=aes(x="C_B", y="t_high", label="label_high"),
            inherit_aes=False, ha="right", va="bottom", size=10, color=upper_color,
            nudge_x=-0.015 * remaining, nudge_y=0.01 * y_max,
        )
        + coord_cartesian(ylim=(0, y_max))
        + labs(
            title=f"{model_label}: Predicted time versus product concentration".lstrip(": "),
            subtitle=(f"C_A0 = {C_A0:g} µM (dashed line). σ = {sigma:.3f} µM. "
                      f"Green: C_B reads σ low. Red: C_B reads σ high"),
            x="Corrected product C_B (µM)",
            y="Predicted time t(C_B) (hours)",
        )
        + theme_gray()
        + theme(figure_size=(9, 6))
    )
