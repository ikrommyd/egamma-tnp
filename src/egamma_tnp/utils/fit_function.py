# Developed by: Sebastian Arturo Hortua, University of Kansas

from __future__ import annotations

import numpy as np
from iminuit import Minuit, cost
from matplotlib import pyplot as plt

from egamma_tnp.utils.fitter_plot_model import (
    BINS_INFO,
    FIT_CONFIGS,
    CombinedCost,
    PassFailPlotter,
    calculate_custom_chi2,
    create_combined_model,
    logging,
)
from egamma_tnp.utils.fitter_shapes import x_max, x_min
from egamma_tnp.utils.logger_utils_fit import print_fit_summary_rich

fit_prog = []
fit_summary = []
fit_text_sum = []


def fit_function(
    fit_type,
    hist_pass,
    hist_fail,
    fixed_params=None,
    use_cdf=False,
    x_min=x_min,
    x_max=x_max,
    interactive=False,
    args_bin=None,
    args_data=None,
    sigmoid_eff=False,
    args_mass=None,
    data_name=None,
    mc_name=None,
):
    fixed_params = fixed_params or {}

    if fit_type not in FIT_CONFIGS:
        raise ValueError(f"Unknown fit type: {fit_type}")

    config = FIT_CONFIGS[fit_type]

    if use_cdf and (config["signal_cdf"] is None or config["background_cdf"] is None):
        logging.warning(f"[Warning] Model '{fit_type}' missing CDF(s). Disabling CDF mode.")
        use_cdf = False
    param_names = config["param_names"]

    # Get data for pass
    centers_pass = (hist_pass["edges"][:-1] + hist_pass["edges"][1:]) / 2
    edges_pass = hist_pass["edges"]
    values_pass = hist_pass["values"]
    errors_pass = hist_pass["errors"]

    # Get data for fail
    centers_fail = (hist_fail["edges"][:-1] + hist_fail["edges"][1:]) / 2
    edges_fail = hist_fail["edges"]
    values_fail = hist_fail["values"]
    errors_fail = hist_fail["errors"]

    # Crop region to the x range
    mask_pass = (centers_pass >= x_min) & (centers_pass <= x_max)
    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)

    edge_mask = np.zeros(len(edges_pass), dtype=bool)
    edge_mask[:-1] |= mask_pass
    edge_mask[1:] |= mask_pass

    centers_pass = centers_pass[mask_pass]
    edges_pass = edges_pass[edge_mask]
    values_pass = values_pass[mask_pass]
    errors_pass = errors_pass[mask_pass]

    mask_fail = (centers_fail >= x_min) & (centers_fail <= x_max)
    edge_mask = np.zeros(len(edges_fail), dtype=bool)
    edge_mask[:-1] |= mask_fail
    edge_mask[1:] |= mask_fail

    centers_fail = centers_fail[mask_fail]
    edges_fail = edges_fail[edge_mask]
    values_fail = values_fail[mask_fail]
    errors_fail = errors_fail[mask_fail]

    if args_mass == "Z" or args_mass == "Z_muon":
        # Calculate data-based initial guesses
        N_p0 = np.sum(values_pass) + np.sum(values_fail)
        B_p_p0 = max(1, np.median(values_pass[-10:]) * len(values_pass))
        B_f_p0 = max(1, np.median(values_fail[-10:]) * len(values_fail))

        # Scale fixed parameters if present
        for name in ["N", "epsilon", "B_p", "B_f"]:
            if name in fixed_params:
                fixed_params[name]

        # Update bounds with data-based values
        bounds = config["bounds"].copy()
        bounds.update(
            {
                "N": (B_p_p0 + B_f_p0, N_p0 * 10, np.inf),
                "B_p": (0, B_p_p0 / 4, np.inf),
                "B_f": (0, B_f_p0, np.inf),
            }
        )

    elif args_mass == "JPsi" or args_mass == "JPsi_muon":
        # Calculate data-based initial guesses
        N_p0 = (np.sum(values_pass) + np.sum(values_fail)) * 0.1
        sideband_mask = (centers_pass < 2.8) | (centers_pass > 3.4)
        sideband_mask_fail = (centers_fail < 2.8) | (centers_fail > 3.4)
        sideband_values_pass = values_pass[sideband_mask]
        sideband_values_fail = values_fail[sideband_mask_fail]

        if len(sideband_values_pass) == 0:
            B_p_p0 = 1  # fallback guess
        else:
            B_p_p0 = max(1, np.median(sideband_values_pass) * len(values_pass))

        if len(sideband_values_fail) == 0:
            B_f_p0 = 1  # fallback guess
        else:
            B_f_p0 = max(1, np.median(sideband_values_fail) * len(values_fail))

        # Scale fixed parameters if present
        for name in ["N", "epsilon", "B_p", "B_f"]:
            if name in fixed_params:
                fixed_params[name]

        # Update bounds with data-based values
        bounds = config["bounds"].copy()
        bounds.update(
            {
                "N": (0, N_p0 * 10, np.inf),
                "B_p": (0, B_p_p0, np.inf),
                "B_f": (0, B_f_p0, np.inf),
            }
        )

    # Set epsilon bounds and initial guess depending on sigmoid
    if sigmoid_eff:
        bounds["epsilon"] = (-10, 0, 10)  # raw parameter, maps to (0.002, 0.998)
    else:
        bounds["epsilon"] = (0, 0.95, 1)

    # Prepare initial parameter guesses
    p0 = []
    bounds_low = []
    bounds_high = []
    initial_guesses = {}  # All initial guesses are stored here

    for name in param_names:
        if name in fixed_params:
            initial_guesses[name] = fixed_params[name]
            continue
        else:
            b = bounds[name]
            if name == "epsilon" and sigmoid_eff:
                # Set initial guess for raw parameter so sigmoid(epsilon) ~ 0.9
                initial_guesses[name] = 2.2  # sigmoid(2.2) ≈ 0.9
            else:
                initial_guesses[name] = b[1]
        # Set bounds and add to p0 for minimization
        b = bounds[name]
        p0.append(initial_guesses[name])
        bounds_low.append(b[0])
        bounds_high.append(b[2])

    def model_approx_pass(edges, *params):
        result_pass, _ = create_combined_model(fit_type, edges_pass, edges_fail, *params, sigmoid_eff=sigmoid_eff)
        return result_pass

    def model_approx_fail(edges, *params):
        _, result_fail = create_combined_model(fit_type, edges_pass, edges_fail, *params, sigmoid_eff=sigmoid_eff)
        return result_fail

    def model_cdf_pass(edges, *params):
        result_pass, _ = create_combined_model(fit_type, edges_pass, edges_fail, *params, use_cdf=True, sigmoid_eff=sigmoid_eff)
        return result_pass

    def model_cdf_fail(edges, *params):
        _, result_fail = create_combined_model(fit_type, edges_pass, edges_fail, *params, use_cdf=True, sigmoid_eff=sigmoid_eff)
        return result_fail

    bin_widths_pass = np.diff(edges_pass)
    bin_widths_fail = np.diff(edges_fail)

    if use_cdf:
        model_pass = model_cdf_pass
        model_fail = model_cdf_fail
    else:
        model_pass = model_approx_pass
        model_fail = model_approx_fail

    # Cost functions depending on if using CDF or PDF
    if use_cdf:
        c_pass = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_pass)
        c_pass.errdef = Minuit.LIKELIHOOD
        c_fail = cost.ExtendedBinnedNLL(values_fail, edges_fail, model_fail)
        c_fail.errdef = Minuit.LIKELIHOOD
    else:
        c_pass = cost.ExtendedBinnedNLL(values_pass, edges_pass, model_pass, use_pdf="approximate")
        c_pass.errdef = Minuit.LIKELIHOOD
        c_fail = cost.ExtendedBinnedNLL(values_fail, edges_fail, model_fail, use_pdf="approximate")
        c_fail.errdef = Minuit.LIKELIHOOD

    # Create CombinedCost object
    c = CombinedCost(c_pass, c_fail)
    # Set error definition to NLL
    c.errdef = Minuit.LIKELIHOOD

    init = initial_guesses

    # Create Minuit object
    m = Minuit(c, *[init[name] for name in param_names], name=param_names)

    for name in param_names:
        if name in fixed_params:
            init[name] = fixed_params[name]
            m.fixed[name] = True
        elif name in bounds:
            m.limits[name] = (bounds[name][0], bounds[name][2])

    m.hesse()

    # Interactive Fitter
    plotter = PassFailPlotter(c_pass, errors_pass, c_fail, errors_fail, len(edges_pass), edges_pass, edges_fail, fit_type, sigmoid_eff=sigmoid_eff)

    if interactive:
        m.strategy = 1
        m.interactive(plotter)
    else:
        # Set default strategy
        m.strategy = 1

        # Try a full Migrad first
        m.migrad(iterate=50)
        m.hesse()

        if not m.valid:
            fall_back_params = []

            m.strategy = 2
            m.migrad(iterate=1)
            # First fix all parameters
            for p in m.parameters:
                m.fixed[p] = True
            for p in m.parameters:
                m.fixed[p] = False
                m.strategy = 2
                m.migrad()
                if not m.fmin.is_valid:
                    fall_back_params.append(p)
                    m.fixed[p] = True
            m.fixed[p] = True
            for p in fall_back_params:
                m.fixed[p] = True

            # Final Migrad + Hesse
            m.strategy = 1
            m.migrad(iterate=50)
            m.hesse()

    plt.close("all")

    for param in m.parameters:
        try:
            m.minos(param)
        except Exception as e:
            logging.debug(f"MINOS failed: {e!s}")

    # Print results
    m.print_level = 0

    # 2. Extract results
    fcv = m.fval
    popt = m.values.to_dict()
    perr = m.errors.to_dict()

    # NLL chi2and ndof
    chi2 = fcv
    dof = m.ndof
    reduced_chi2 = chi2 / dof

    # compute model predictions at bin centers or edges as appropriate
    model_pass_vals = model_approx_pass(edges_pass, *m.values)
    model_fail_vals = model_approx_fail(edges_fail, *m.values)

    # Scale model predictions by bin widths
    model_pass_vals = model_pass_vals * bin_widths_pass
    model_fail_vals = model_fail_vals * bin_widths_fail

    # Calculate Pearson and Poisson Chi2 and ndof
    n_params = len([name for name in m.parameters if not m.fixed[name]])
    Pearson_chi2_pass, Poisson_chi2_pass, ndof_pass = calculate_custom_chi2(values_pass, errors_pass, model_pass_vals, n_params)
    Pearson_chi2_fail, Poisson_chi2_fail, ndof_fail = calculate_custom_chi2(values_fail, errors_fail, model_fail_vals, n_params)
    Pearson_chi2 = Pearson_chi2_pass + Pearson_chi2_fail
    Poisson_chi2 = Poisson_chi2_pass + Poisson_chi2_fail
    total_ndof = (len(values_pass) + len(values_fail)) - n_params
    Pearson_tot_red_chi2 = Pearson_chi2 / total_ndof
    Poisson_tot_red_chi2 = Poisson_chi2 / total_ndof

    # Output fit summary to terminal
    summary_text = print_fit_summary_rich(
        m,
        popt,
        perr,
        edges_pass,
        args_bin,
        BINS_INFO,
        Pearson_chi2,
        Poisson_chi2,
        total_ndof,
        args_data,
        fit_type,
        sigmoid_eff=sigmoid_eff,
        DATA_NAME=data_name,
        MC_NAME=mc_name,
    )

    fit_text_sum.append(summary_text)

    # Check convergence
    if m.fmin.is_valid:
        converged = "Fit converged: minimum is valid."
    else:
        issues = []
        if m.fmin.has_parameters_at_limit:
            issues.append("\nparameters at limit")
        if m.fmin.edm > m.tol:
            issues.append(f"\nEDM too high (EDM = {m.fmin.edm:.2g}, tol = {m.tol:.2g})")
        if not m.fmin.has_valid_parameters:
            issues.append("\ninvalid Hesse (covariance matrix)")
        if m.fmin.has_reached_call_limit:
            issues.append(f"\ncall limit reached (Nfcn = {m.fmin.nfcn})")

        if not issues:
            issues.append("unknown issue")

        converged = "Fit did not converge: " + "; ".join(issues)

    minos_errors = {}
    for param in m.parameters:
        if param in m.merrors:
            minos_errors[param] = {"lower": m.merrors[param].lower, "upper": m.merrors[param].upper}
        else:
            minos_errors[param] = {
                "lower": -m.errors[param],  # Fall back to symmetric errors
                "upper": m.errors[param],
            }

    # Return results
    results = {
        "minos_errors": minos_errors,
        "m": m,
        "type": fit_type,
        "popt": popt,
        "perr": perr,
        "cov": m.covariance,
        "chi_squared": chi2,
        "reduced_chi_squared": reduced_chi2,
        "dof": dof,
        "Pearson_chi2": Pearson_chi2,
        "Poisson_chi2": Poisson_chi2,
        "Pearson_tot_red_chi2": Pearson_tot_red_chi2,
        "Poisson_tot_red_chi2": Poisson_tot_red_chi2,
        "total_ndof": total_ndof,
        "success": m.valid,
        "message": m.fmin,
        "param_names": param_names,
        "centers_pass": centers_pass,
        "values_pass": values_pass,
        "errors_pass": errors_pass,
        "centers_fail": centers_fail,
        "values_fail": values_fail,
        "errors_fail": errors_fail,
        "x_min": x_min,
        "x_max": x_max,
        "converged": converged,
        "bin_widths_pass": bin_widths_pass,
        "bin_widths_fail": bin_widths_fail,
        "sum_text": summary_text,
    }

    results["bin"] = args_bin
    return results
