import inspect

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from pymc.ode import DifferentialEquation


def build_model1(times, product_observed, C_A0=3.125):
    def product_rate(y, t, p):
        C_B = y[0]
        V_max = p[0]
        K_m = p[1]
        C_A = C_A0 - C_B
        return [V_max * C_A / (K_m + C_A)]

    ode_model = DifferentialEquation(
        func=product_rate,
        times=times,
        n_states=1,
        n_theta=2,
        t0=0,
    )

    with pm.Model() as kinetics_model1:
        V_max = pm.LogNormal("V_max", mu=0, sigma=1)
        K_m = pm.LogNormal("K_m", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        product_predicted = pm.Deterministic(
            "product_predicted",
            ode_model(y0=[0], theta=[V_max, K_m])[:, 0],
        )
        mu = pm.Deterministic("mu", product_predicted)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=product_observed)

    return kinetics_model1


def build_model2(times, product_observed, C_B_initial, C_A0=3.125):
    def product_rate(y, t, p):
        C_B = y[0]
        V_max = p[0]
        K_m = p[1]
        C_A = C_A0 - C_B
        return [V_max * C_A / (K_m + C_A)]

    ode_model = DifferentialEquation(
        func=product_rate,
        times=times,
        n_states=1,
        n_theta=2,
        t0=0,
    )

    with pm.Model() as kinetics_model2:
        V_max = pm.LogNormal("V_max", mu=0, sigma=1)
        K_m = pm.LogNormal("K_m", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        product_predicted = pm.Deterministic(
            "product_predicted",
            ode_model(y0=[C_B_initial], theta=[V_max, K_m])[:, 0],
        )
        mu = pm.Deterministic("mu", product_predicted)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=product_observed)

    return kinetics_model2


def build_model3(times, product_observed, C_A0=3.125):
    C_B_initial = float(product_observed[0])

    def product_rate(y, t, p):
        C_B = y[0]
        V_max = p[0]
        K_m = p[1]
        C_A = C_A0 - C_B
        return [V_max * C_A / (K_m + C_A)]

    ode_model = DifferentialEquation(
        func=product_rate,
        times=times,
        n_states=1,
        n_theta=2,
        t0=0,
    )

    with pm.Model() as kinetics_model3:
        V_max = pm.LogNormal("V_max", mu=0, sigma=1)
        K_m = pm.LogNormal("K_m", mu=0, sigma=1)

        b = pm.Normal("b", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        product_predicted = pm.Deterministic(
            "product_predicted",
            ode_model(y0=[C_B_initial], theta=[V_max, K_m])[:, 0],
        )
        mu = pm.Deterministic("mu", product_predicted + b)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=product_observed)

    return kinetics_model3


def build_model4(times, product_observed, C_A0=3.125):
    def product_rate(y, t, p):
        C_B = y[0]
        V_max = p[0]
        K_m = p[1]
        C_A = C_A0 - C_B
        return [V_max * C_A / (K_m + C_A)]

    ode_model = DifferentialEquation(
        func=product_rate,
        times=times,
        n_states=1,
        n_theta=2,
        t0=0,
    )

    with pm.Model() as kinetics_model4:
        V_max = pm.LogNormal("V_max", mu=0, sigma=1)
        K_m = pm.LogNormal("K_m", mu=0, sigma=1)

        C_B_start = pm.Uniform("C_B_start", lower=0, upper=C_A0)
        b = pm.Normal("b", mu=0, sigma=1)
        sigma = pm.HalfNormal("sigma", sigma=1)

        product_predicted = pm.Deterministic(
            "product_predicted",
            ode_model(y0=[C_B_start], theta=[V_max, K_m])[:, 0],
        )
        mu = pm.Deterministic("mu", product_predicted + b)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=product_observed)

    return kinetics_model4


def build_model5(substrate_levels, times_by_concentration, product_observed):
    def product_rate(y, t, p):
        C_B = y[0]
        V_max = p[0]
        K_m = p[1]
        initial_substrate = p[2]
        C_A = initial_substrate - C_B
        return [V_max * C_A / (K_m + C_A)]

    substrate_levels = np.asarray(substrate_levels)
    ode_models = [
        DifferentialEquation(
            func=product_rate,
            times=curve_times,
            n_states=1,
            n_theta=3,
            t0=0,
        )
        for curve_times in times_by_concentration
    ]

    with pm.Model(coords={"experiment": substrate_levels}) as kinetics_model5:
        V_max = pm.LogNormal("V_max", mu=0, sigma=1)
        K_m = pm.LogNormal("K_m", mu=0, sigma=1)

        C_B_start = pm.Uniform(
            "C_B_start", lower=0, upper=substrate_levels, dims="experiment"
        )
        b = pm.Normal("b", mu=0, sigma=1, dims="experiment")
        sigma = pm.HalfNormal("sigma", sigma=1)

        product_curves = []
        offset_curves = []

        for i, substrate in enumerate(substrate_levels):
            product_curve = ode_models[i](
                y0=[C_B_start[i]],
                theta=[V_max, K_m, float(substrate)],
            )[:, 0]
            product_curves.append(product_curve)
            offset_curves.append(np.ones(len(times_by_concentration[i])) * b[i])

        product_predicted = pm.Deterministic(
            "product_predicted", pm.math.concatenate(product_curves)
        )
        mu = pm.Deterministic(
            "mu", product_predicted + pm.math.concatenate(offset_curves)
        )
        pm.Normal("obs", mu=mu, sigma=sigma, observed=product_observed)

    return kinetics_model5


def build_model6(substrate_levels, times_by_concentration, product_observed):
    def product_rate(y, t, p):
        C_B = y[0]
        V_max = p[0]
        K_m = p[1]
        h = p[2]
        initial_substrate = p[3]
        C_A = pt.maximum(initial_substrate - C_B, 1e-12)
        return [V_max * pm.math.sigmoid(h * (pt.log(C_A) - pt.log(K_m)))]

    substrate_levels = np.asarray(substrate_levels)
    ode_models = [
        DifferentialEquation(
            func=product_rate,
            times=curve_times,
            n_states=1,
            n_theta=4,
            t0=0,
        )
        for curve_times in times_by_concentration
    ]

    with pm.Model(coords={"experiment": substrate_levels}) as kinetics_model6:
        V_max = pm.LogNormal("V_max", mu=0, sigma=1)
        K_m = pm.LogNormal("K_m", mu=0, sigma=1)
        h = pm.LogNormal("h", mu=0, sigma=0.5)

        C_B_start = pm.Uniform(
            "C_B_start", lower=0, upper=substrate_levels, dims="experiment"
        )
        b = pm.Normal("b", mu=0, sigma=1, dims="experiment")
        sigma = pm.HalfNormal("sigma", sigma=1)

        product_curves = []
        offset_curves = []

        for i, substrate in enumerate(substrate_levels):
            product_curve = ode_models[i](
                y0=[C_B_start[i]],
                theta=[V_max, K_m, h, float(substrate)],
            )[:, 0]
            product_curves.append(product_curve)
            offset_curves.append(np.ones(len(times_by_concentration[i])) * b[i])

        product_predicted = pm.Deterministic(
            "product_predicted", pm.math.concatenate(product_curves)
        )
        mu = pm.Deterministic(
            "mu", product_predicted + pm.math.concatenate(offset_curves)
        )
        pm.Normal("obs", mu=mu, sigma=sigma, observed=product_observed)

    return kinetics_model6


def print_model_code(number):
    builders = [
        build_model1,
        build_model2,
        build_model3,
        build_model4,
        build_model5,
        build_model6,
    ]
    print(inspect.getsource(builders[number - 1]))
