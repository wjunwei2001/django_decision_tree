import os
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from .utils import load_file_data, STATIC_PLOT_DIR, format_num


def run_anova(request):
    """
    Runs One-Way Analysis of Variance (ANOVA) with Bonferroni post-hoc pairwise comparisons.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    all_data = load_file_data(request, 'data_file', 'anova.dat')
    if len(all_data) == 0 or all_data.ndim < 2:
        return {"result_text": "Error: ANOVA data must contain multiple columns (groups)."}

    n_groups = all_data.shape[1]
    n_sub = all_data.shape[0]

    groups = {f"Group {i+1}": all_data[:, i] for i in range(n_groups)}
    means = {name: np.mean(values) for name, values in groups.items()}
    sds = {name: np.std(values, ddof=1) for name, values in groups.items()}

    mean_mean = np.mean(list(means.values()))
    sx2 = sum((m - mean_mean)**2 for m in means.values()) / (n_groups - 1)
    s2wit = sum(sd**2 for sd in sds.values()) / n_groups
    s2bet = n_sub * sx2
    f_ratio = s2bet / s2wit

    dfn = n_groups - 1
    dfd = n_groups * (n_sub - 1)
    f_crit = stats.f.ppf(conf_int / 100.0, dfn, dfd)
    p_value = 1.0 - stats.f.cdf(f_ratio, dfn, dfd)

    posthoc_text = ""
    if f_ratio > f_crit:
        conclusion = "<b>Conclusion:</b> Reject Null Hypothesis — Statistically significant difference among groups.<br>"
        group_pairs = list(combinations(groups.keys(), 2))
        alpha = (100.0 - conf_int) / 100.0
        t_crit_bonf = stats.t.ppf(1.0 - (alpha / (2.0 * len(group_pairs))), dfd)
        posthoc_lines = []
        for (gA, gB) in group_pairs:
            t_pair = (means[gA] - means[gB]) / np.sqrt(2.0 * s2wit / n_sub)
            p_pair = 2.0 * (1.0 - stats.t.cdf(np.abs(t_pair), dfd))
            sig_label = "<b>Significant</b>" if np.abs(t_pair) > t_crit_bonf else "Not Significant"
            posthoc_lines.append(f"• {gA} vs {gB}: t = {format_num(t_pair, 3)}, p = {format_num(p_pair)} ({sig_label})")
        posthoc_text = "<br><b>Bonferroni Post-Hoc Pairwise Comparisons:</b><br>" + "<br>".join(posthoc_lines)
    else:
        conclusion = "<b>Conclusion:</b> Fail to Reject Null Hypothesis. No significant variance detected among group means."

    plot_filename = "anova_boxplot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6.5, 4.5))
    plt.boxplot([groups[g] for g in groups], labels=list(groups.keys()))
    plt.title('One-Way ANOVA Group Comparisons')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>One-Way ANOVA Results:</b><br>"
            f"• Number of Groups = {n_groups}, Subjects per group = {n_sub}<br>"
            f"• F-Ratio = {format_num(f_ratio)} (F-Critical = {format_num(f_crit)})<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br><br>"
            f"{conclusion}"
            f"{posthoc_text}<br><br>"
            f"<small>Note: Results are only valid if data points in each group are independent of one another.</small>"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }
