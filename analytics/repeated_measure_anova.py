import os
from itertools import combinations
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from .utils import load_file_data, STATIC_PLOT_DIR, format_num


def run_repeated_anova(request):
    """
    Runs One-Way Repeated Measures Analysis of Variance (RM-ANOVA)
    with Bonferroni post-hoc pairwise comparisons.
    Tests for differences across repeated measurements / conditions on the same subjects.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    all_data = load_file_data(request, 'data_file', 'repeated_anova.dat')
    if len(all_data) == 0 or all_data.ndim < 2:
        return {"result_text": "Error: Repeated measures ANOVA data must contain multiple columns (measurements)."}

    n_sub = all_data.shape[0]        # number of subjects (rows)
    n_measures = all_data.shape[1]   # number of measurements per subject (columns)
    n_total = n_sub * n_measures     # total number of observations

    # Grand total and correction factor (C)
    grand_total = np.sum(all_data)
    correction = (grand_total ** 2) / n_total

    # Sum of Squares (SS)
    treatment_totals = np.sum(all_data, axis=0)  # column sums (treatments/conditions)
    subject_totals = np.sum(all_data, axis=1)    # row sums (subjects)

    ss_total = np.sum(all_data ** 2) - correction
    ss_betw = (np.sum(treatment_totals ** 2) / n_sub) - correction
    ss_subjects = (np.sum(subject_totals ** 2) / n_measures) - correction
    ss_error = ss_total - ss_betw - ss_subjects

    # Degrees of Freedom (df)
    df_betw = n_measures - 1
    df_subjects = n_sub - 1
    df_error = df_betw * df_subjects
    df_total = n_total - 1

    # Mean Squares (MS)
    ms_betw = ss_betw / df_betw
    ms_subjects = ss_subjects / df_subjects
    ms_error = ss_error / df_error

    # F-Ratio and P-Value
    f_ratio = ms_betw / ms_error
    f_crit = stats.f.ppf(conf_int / 100.0, df_betw, df_error)
    p_value = 1.0 - stats.f.cdf(f_ratio, df_betw, df_error)

    # Condition means
    group_names = [f"Measure {j+1}" for j in range(n_measures)]
    means = {f"Measure {j+1}": np.mean(all_data[:, j]) for j in range(n_measures)}
    means_summary = ", ".join([f"{name}: {format_num(val, 3)}" for name, val in means.items()])

    # Conclusion and Bonferroni Post-Hoc Comparisons
    posthoc_text = ""
    if f_ratio > f_crit:
        conclusion = "<b>Conclusion:</b> Reject Null Hypothesis. Statistically significant difference across repeated measurements.<br>"
        group_pairs = list(combinations(group_names, 2))
        n_comp = len(group_pairs)
        alpha = (100.0 - conf_int) / 100.0
        alpha_bonf = alpha / n_comp
        t_crit_bonf = stats.t.ppf(1.0 - (alpha_bonf / 2.0), df_error)

        se_pair = np.sqrt(2.0 * ms_error / n_sub)
        posthoc_lines = []
        for (gA, gB) in group_pairs:
            t_pair = (means[gA] - means[gB]) / se_pair
            p_pair = 2.0 * (1.0 - stats.t.cdf(np.abs(t_pair), df_error))
            sig_label = "<b>Significant</b>" if np.abs(t_pair) > t_crit_bonf else "Not Significant"
            p_str = format_num(p_pair)
            posthoc_lines.append(f"• {gA} vs {gB}: t = {format_num(t_pair, 3)}, p = {p_str} ({sig_label})")

        posthoc_text = (
            f"<br><b>Bonferroni Post-Hoc Pairwise Comparisons:</b><br>"
            f"• Number of Comparisons (m) = {n_comp}<br>"
            f"• Bonferroni-corrected Critical t* (alpha' = {format_num(alpha_bonf)}, df = {df_error}) = {format_num(t_crit_bonf)}<br>"
            + "<br>".join(posthoc_lines)
        )
    else:
        conclusion = "<b>Conclusion:</b> Fail to Reject Null Hypothesis. No significant variance detected across repeated measurements."

    # Generate Profile Plot / Boxplot
    plot_filename = "repeated_anova_plot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)

    plt.figure(figsize=(7, 4.5))
    # Boxplot of each measurement condition
    plt.boxplot([all_data[:, j] for j in range(n_measures)], labels=group_names, patch_artist=True,
                boxprops=dict(facecolor='#e3f2fd', color='#1976d2'),
                medianprops=dict(color='#d32f2f', linewidth=2))

    # Overlay subject trajectories
    for i in range(n_sub):
        plt.plot(range(1, n_measures + 1), all_data[i, :], marker='o', linestyle='--', alpha=0.5, label=f'Subject {i+1}' if n_sub <= 10 else None)

    plt.title('Repeated Measures ANOVA: Group Distribution & Subject Trajectories')
    plt.xlabel('Measurement Point / Condition')
    plt.ylabel('Observed Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    if n_sub <= 10:
        plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Null Hypothesis (H<sub>0</sub>):</b> Mean values across all repeated measurements are equal.<br>"
            f"<b>Alternative Hypothesis (H<sub>1</sub>):</b> At least one measurement condition differs significantly.<br><br>"
            f"<b>Repeated Measures ANOVA Results:</b><br>"
            f"• Number of Subjects (N) = {n_sub}, Repeated Measures per Subject (k) = {n_measures}<br>"
            f"• Condition Means: {means_summary}<br>"
            f"• SS_between = {format_num(ss_betw)} (df = {df_betw}, MS = {format_num(ms_betw)})<br>"
            f"• SS_subjects = {format_num(ss_subjects)} (df = {df_subjects}, MS = {format_num(ms_subjects)})<br>"
            f"• SS_error = {format_num(ss_error)} (df = {df_error}, MS = {format_num(ms_error)})<br>"
            f"• SS_total = {format_num(ss_total)} (df = {df_total})<br><br>"
            f"• <b>F-Ratio = {format_num(f_ratio)}</b> (Critical F* = {format_num(f_crit)})<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br><br>"
            f"{conclusion}"
            f"{posthoc_text}"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }