import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from .utils import load_file_data, STATIC_PLOT_DIR, format_num


def run_paired_test(request):
    """
    Runs a Paired Samples t-test (pre/post measurements).
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    paired_data = load_file_data(request, 'data_file', 'paired.dat')
    if len(paired_data) == 0 or paired_data.shape[1] < 2:
        return {"result_text": "Error: Paired test requires 2 paired numeric columns."}

    before = paired_data[:, 0]
    after = paired_data[:, 1]
    n = len(paired_data)

    diff_array = before - after
    mean_diff = np.mean(diff_array)
    sd_diff = np.std(diff_array, ddof=1)

    t_stat = mean_diff / (sd_diff / np.sqrt(n))
    alpha = (100.0 - conf_int) / 100.0
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), n - 1))

    ci_low = mean_diff - t_crit * sd_diff / np.sqrt(n)
    ci_high = mean_diff + t_crit * sd_diff / np.sqrt(n)

    if np.abs(t_stat) > t_crit:
        conclusion = "<b>Conclusion:</b> Reject Null Hypothesis — Statistically significant difference between pre- and post-treatments."
    else:
        conclusion = "<b>Conclusion:</b> Fail to Reject Null Hypothesis — Insufficient evidence to show a significant treatment difference."

    plot_filename = "paired_diff_boxplot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6, 4))
    plt.boxplot([before, after, diff_array], labels=['Before', 'After', 'Difference (B - A)'])
    plt.title('Paired Comparison & Differences')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Paired Samples t-Test Results:</b><br>"
            f"• Sample Size (Pairs) = {n}<br>"
            f"• Mean Difference (d̄) = {format_num(mean_diff)}<br>"
            f"• SD of Differences (s_d) = {format_num(sd_diff)}<br>"
            f"• t-statistic = {format_num(t_stat)} (t-critical = {format_num(t_crit)})<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br>"
            f"• {conf_int:.1f}% CI for Mean Difference: [{format_num(ci_low)}, {format_num(ci_high)}]<br><br>"
            f"{conclusion}"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }


def run_1samp_hyp_testing(request):
    """
    Runs a 1-Sample t-test (two-sided, upper-tailed, or lower-tailed).
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
        mu_0 = float(request.POST.get('mu', 0.0))
        alt_hyp = request.POST.get('alt_hyp', 'two-sided')
    except (ValueError, TypeError, AttributeError):
        conf_int, mu_0, alt_hyp = 95.0, 0.0, 'two-sided'

    dat = load_file_data(request, 'data_file', 'hyp.dat')
    if len(dat) == 0:
        return {"result_text": "Error: No dataset found for 1-sample test."}

    if dat.ndim > 1:
        dat = dat[:, 0]

    n = len(dat)
    mean = np.mean(dat)
    sd = np.std(dat, ddof=1)
    se = sd / np.sqrt(n)
    t_stat = (mean - mu_0) / se
    alpha = (100.0 - conf_int) / 100.0

    if alt_hyp == 'greater':
        t_crit = stats.t.ppf(1.0 - alpha, n - 1)
        p_value = 1.0 - stats.t.cdf(t_stat, n - 1)
        is_sig = t_stat > t_crit
        h0_text = f"μ ≤ {mu_0}"
        h1_text = f"μ > {mu_0} (Upper-Tailed)"
        if is_sig:
            conclusion = f"<b>Conclusion:</b> Reject Null Hypothesis (p < {alpha:.3f}) — Sample mean (x̄ = {format_num(mean)}) is statistically significantly greater than {mu_0}."
        else:
            conclusion = f"<b>Conclusion:</b> Fail to Reject Null Hypothesis — Insufficient evidence to show the sample mean is significantly greater than {mu_0}."
    elif alt_hyp == 'less':
        t_crit = -stats.t.ppf(1.0 - alpha, n - 1)
        p_value = stats.t.cdf(t_stat, n - 1)
        is_sig = t_stat < t_crit
        h0_text = f"μ ≥ {mu_0}"
        h1_text = f"μ < {mu_0} (Lower-Tailed)"
        if is_sig:
            conclusion = f"<b>Conclusion:</b> Reject Null Hypothesis (p < {alpha:.3f}) — Sample mean (x̄ = {format_num(mean)}) is statistically significantly less than {mu_0}."
        else:
            conclusion = f"<b>Conclusion:</b> Fail to Reject Null Hypothesis — Insufficient evidence to show the sample mean is significantly less than {mu_0}."
    else:
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
        p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), n - 1))
        is_sig = np.abs(t_stat) > t_crit
        h0_text = f"μ = {mu_0}"
        h1_text = f"μ ≠ {mu_0} (Two-Sided)"
        if is_sig:
            conclusion = f"<b>Conclusion:</b> Reject Null Hypothesis (p < {alpha:.3f}) — Sample mean (x̄ = {format_num(mean)}) is statistically significantly different from {mu_0}."
        else:
            conclusion = f"<b>Conclusion:</b> Fail to Reject Null Hypothesis — Insufficient evidence to show a statistically significant difference from {mu_0}."

    t_crit_ci = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
    low_ci = mean - t_crit_ci * se
    upp_ci = mean + t_crit_ci * se

    plot_filename = "1hyp_testing_boxplot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(5.5, 4))
    plt.boxplot(dat, labels=['Sample Data'])
    plt.axhline(mu_0, color='r', linestyle='--', label=f'Hypothesized μ₀ ({mu_0})')
    plt.title('1-Sample Data Boxplot')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Null Hypothesis (H<sub>0</sub>):</b> {h0_text}<br>"
            f"<b>Alternative Hypothesis (H<sub>1</sub>):</b> {h1_text}<br><br>"
            f"<b>One-Sample t-Test Results:</b><br>"
            f"• Hypothesized Mean (μ₀) = {mu_0}<br>"
            f"• Sample Mean (x̄) = {format_num(mean)}, Sample SD (s) = {format_num(sd)}, N = {n}<br>"
            f"• Standard Error (SE) = {format_num(se)}<br>"
            f"• t-statistic = {format_num(t_stat)} (Critical t* = {format_num(t_crit)})<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br>"
            f"• {conf_int:.1f}% Confidence Interval for μ: [{format_num(low_ci)}, {format_num(upp_ci)}]<br><br>"
            f"{conclusion}"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }


def run_2samp_hyp_testing(request):
    """
    Runs a 2-Sample Welch's t-test with Satterthwaite approximation for unequal variances.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    dat1 = load_file_data(request, 'data_file', 'data1.dat')
    dat2 = load_file_data(request, 'data_file_2', 'data2.dat')

    if len(dat1) == 0 or len(dat2) == 0:
        return {"result_text": "Error: Missing data for two-sample test."}

    if dat1.ndim > 1: dat1 = dat1[:, 0]
    if dat2.ndim > 1: dat2 = dat2[:, 0]

    n1, n2 = len(dat1), len(dat2)
    mean1, mean2 = np.mean(dat1), np.mean(dat2)
    sd1, sd2 = np.std(dat1, ddof=1), np.std(dat2, ddof=1)

    se_diff = np.sqrt(sd1**2 / n1 + sd2**2 / n2)
    t_stat = (mean1 - mean2) / se_diff

    eta = ((sd1**2 / n1 + sd2**2 / n2)**2) / (
        (sd1**2 / n1)**2 / (n1 - 1) + (sd2**2 / n2)**2 / (n2 - 1)
    )
    alpha = (100.0 - conf_int) / 100.0
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, eta)
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), eta))

    conclusion = (
        "<b>Conclusion:</b> Reject Null Hypothesis — Significant difference between the two sample means."
        if np.abs(t_stat) > t_crit else
        "<b>Conclusion:</b> Fail to Reject Null Hypothesis — No significant difference detected between sample means."
    )

    plot_filename = "2hyp_testing_boxplot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6, 4))
    plt.boxplot([dat1, dat2], labels=['Sample 1', 'Sample 2'])
    plt.title('Two-Sample Boxplot Comparison')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Two-Sample Welch's t-Test Results:</b><br>"
            f"• Group 1: Mean = {format_num(mean1)}, SD = {format_num(sd1)}, N = {n1}<br>"
            f"• Group 2: Mean = {format_num(mean2)}, SD = {format_num(sd2)}, N = {n2}<br>"
            f"• Welch's Degrees of Freedom (η) = {format_num(eta, 2)}<br>"
            f"• t-statistic = {format_num(t_stat)} (t-critical = {format_num(t_crit)})<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br><br>"
            f"{conclusion}"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }
