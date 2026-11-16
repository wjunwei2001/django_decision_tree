import numpy as np
from scipy import stats
from .utils import load_file_data, format_num


def run_samplesize_est(request):
    """
    Computes minimum sample size required to achieve a target statistical power.
    """
    try:
        delta = float(request.POST.get('delta', 5.0))
        power_needed = float(request.POST.get('power_needed', 0.8))
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        delta, power_needed, conf_int = 5.0, 0.8, 95.0

    samp = load_file_data(request, 'data_file', 'samplesize.dat')
    if len(samp) == 0:
        return {"result_text": "Error: No pilot sample dataset found for sample size estimation."}

    if samp.ndim > 1:
        samp = samp[:, 0]

    sigma = np.std(samp, ddof=1)
    n_samp = len(samp)
    alpha = (100.0 - conf_int) / 100.0

    n_new = 1
    power = 0.0
    while power < power_needed and n_new < 10000:
        n_new += 1
        d = delta / (sigma * np.sqrt(1.0 / n_samp + 1.0 / n_new))
        dof = n_samp + n_new - 2
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, dof)
        t_star = t_crit - d
        power = 1.0 - stats.t.cdf(t_star, dof)

    return {
        "result_text": (
            f"<b>Sample Size Determination:</b><br>"
            f"• Expected Effect Size (δ) = {delta}<br>"
            f"• Estimated Standard Deviation (σ) = {format_num(sigma)}<br>"
            f"• Target Power (1 - β) = {power_needed * 100:.1f}% (Significance α = {alpha:.2f})<br>"
            f"• <b>Minimum Required Sample Size (N_new): {n_new}</b> (Achieved Power: {format_num(power, 3)})"
        )
    }


def run_power_est(request):
    """
    Calculates statistical power achieved given two sample groups and an effect size.
    """
    try:
        delta = float(request.POST.get('delta', 5.0))
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        delta, conf_int = 5.0, 95.0

    dat = load_file_data(request, 'data_file', 'power.dat')
    if len(dat) == 0 or dat.shape[1] < 2:
        return {"result_text": "Error: Power estimation requires two sample columns."}

    samp1, samp2 = dat[:, 0], dat[:, 1]
    n1, n2 = len(samp1), len(samp2)
    sd1, sd2 = np.std(samp1, ddof=1), np.std(samp2, ddof=1)

    sigma = np.sqrt(((n1 - 1) * sd1**2 + (n2 - 1) * sd2**2) / (n1 + n2 - 2))
    d = delta / (sigma * np.sqrt(1.0 / n1 + 1.0 / n2))
    alpha = (100.0 - conf_int) / 100.0
    dof = n1 + n2 - 2
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, dof)
    t_star = t_crit - d
    power = 1.0 - stats.t.cdf(t_star, dof)

    return {
        "result_text": (
            f"<b>Statistical Power Analysis:</b><br>"
            f"• Difference to Detect (δ) = {delta}<br>"
            f"• Pooled Standard Deviation = {format_num(sigma)}<br>"
            f"• Sample Sizes: N1 = {n1}, N2 = {n2}<br>"
            f"• <b>Calculated Statistical Power = {power * 100:.2f}%</b>"
        )
    }
