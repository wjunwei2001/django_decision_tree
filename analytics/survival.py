import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from .utils import load_file_data, STATIC_PLOT_DIR, format_num


def compute_survival(unsorted_time_of_events, unsorted_type_of_events):
    """
    Kaplan-Meier survival curve computation.
    """
    indices = np.argsort(unsorted_time_of_events)
    time_of_events = np.zeros(len(indices))
    type_of_events = np.zeros(len(indices))
    for i in range(len(indices)):
        index = indices[i]
        time_of_events[i] = unsorted_time_of_events[index]
        if unsorted_type_of_events[index] == 1:
            type_of_events[i] = 1

    N = len(time_of_events)
    total_surviving = N
    n_i = [N]
    times_of_death = [0.0]
    times_of_death_plot = [0.0]
    S_hat = [1.0]
    S_hat_plot = [1.0]
    d_i = [0]
    lost_i = [0]
    i = 0
    all_times = [0.0]

    while i < N:
        time_of_interest = time_of_events[i]
        n_events = np.count_nonzero(time_of_events == time_of_interest)
        deaths_at_i = 0
        lost_at_i = 0
        for j in range(i, i + n_events):
            if type_of_events[j] == 1:
                deaths_at_i += 1
            else:
                lost_at_i += 1

        if deaths_at_i > 0:
            S_hat_plot.append(S_hat[-1])
            times_of_death_plot.append(time_of_interest)
            new_surv_frac = float(n_i[-1] - deaths_at_i) / n_i[-1]
            new_value = S_hat[-1] * new_surv_frac
            S_hat.append(new_value)
            S_hat_plot.append(new_value)
            times_of_death.append(time_of_interest)
            times_of_death_plot.append(time_of_interest)

        all_times.append(time_of_interest)
        i = i + deaths_at_i + lost_at_i
        total_surviving = total_surviving - deaths_at_i - lost_at_i
        n_i.append(total_surviving)
        d_i.append(deaths_at_i)
        lost_i.append(lost_at_i)

    return [times_of_death, S_hat, all_times, n_i, d_i, lost_i, times_of_death_plot, S_hat_plot]


def compare_survivals(survival_1, survival_2):
    """
    Log-rank test calculation between two survival curves.
    """
    to_be_added = []
    for i in range(len(survival_2[2])):
        position = np.where(np.isclose(survival_1[2], survival_2[2][i], 1e-4))
        if len(position[0]) == 0:
            to_be_added.append(survival_2[2][i])

    all_times = survival_1[2] + to_be_added
    all_times.sort()

    u_l = 0.0
    s_2_l = 0.0
    n_1_i = survival_1[3][0]
    n_2_i = survival_2[3][0]
    d_1_i = 0
    d_2_i = 0

    for i in range(1, len(all_times)):
        pos_2 = np.where(np.isclose(survival_2[2], all_times[i], 1e-4))
        pos_1 = np.where(np.isclose(survival_1[2], all_times[i], 1e-4))
        if len(pos_2[0]) > 0:
            d_2_i = survival_2[4][pos_2[0][0]] if isinstance(survival_2[4][pos_2[0][0]], (int, float)) else 1
            n_2_i = survival_2[3][pos_2[0][0] - 1]
        else:
            n_2_i = max(0, n_2_i - d_2_i)
            d_2_i = 0

        if len(pos_1[0]) > 0:
            d_1_i = survival_1[4][pos_1[0][0]] if isinstance(survival_1[4][pos_1[0][0]], (int, float)) else 1
            n_1_i = survival_1[3][pos_1[0][0] - 1]
        else:
            n_1_i = max(0, n_1_i - d_1_i)
            d_1_i = 0

        if d_2_i > 0 or d_1_i > 0:
            d_total_i = d_2_i + d_1_i
            n_total_i = n_2_i + n_1_i
            if n_total_i > 0:
                f_i = float(d_total_i) / n_total_i
                e_i = n_2_i * f_i
                u_l += (d_2_i - e_i)
                if n_total_i > 1:
                    s_2_l += (float(n_1_i) * n_2_i * d_total_i * (n_total_i - d_total_i)) / (n_total_i * n_total_i * (n_total_i - 1))

    return [u_l, s_2_l]


def run_survival_analysis(request):
    """
    Analyzes survival datasets for two groups, computes Kaplan-Meier curves,
    median survival times, and runs a Log-Rank comparison test.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    control = load_file_data(request, 'data_file', 'existing.dat')
    treatment = load_file_data(request, 'data_file_2', 'new_drug.dat')

    if len(control) == 0 or len(treatment) == 0:
        return {"result_text": "Error: Missing survival data for control or new group."}

    times_control = control[:, 2] - control[:, 1]
    types_control = control[:, 3]
    survival_control = compute_survival(times_control, types_control)

    times_new = treatment[:, 2] - treatment[:, 1]
    types_new = treatment[:, 3]
    survival_new = compute_survival(times_new, types_new)

    t_death_control = survival_control[0]
    s_hat_control = survival_control[1]
    median_survival_control = next((t_death_control[i] for i in range(len(t_death_control)) if s_hat_control[i] < 0.5), t_death_control[-1])

    t_death_new = survival_new[0]
    s_hat_new = survival_new[1]
    median_survival_new = next((t_death_new[i] for i in range(len(t_death_new)) if s_hat_new[i] < 0.5), t_death_new[-1])

    [uL, s2L] = compare_survivals(survival_control, survival_new)
    z_stat = (uL / np.sqrt(s2L)) if s2L > 0 else 0.0
    alpha = (100.0 - conf_int) / 100.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)
    p_value = 2.0 * (1.0 - stats.norm.cdf(np.abs(z_stat)))

    if np.abs(z_stat) > z_crit:
        conclusion = "<b>Conclusion:</b> Reject Null Hypothesis — Statistically significant difference in survival between the two groups."
    else:
        conclusion = "<b>Conclusion:</b> Fail to Reject Null Hypothesis — Insufficient evidence to show a statistically significant difference in survival outcome."

    plot_filename = "survival_plot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(7, 4.5))
    plt.plot(survival_control[6], survival_control[7], 'r-', label='Control Group', linewidth=2)
    plt.plot(survival_new[6], survival_new[7], 'b-', label='New Treatment Group', linewidth=2)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Estimated Survival Probability S(t)')
    plt.title('Kaplan-Meier Survival Curves')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Null Hypothesis:</b> There is no difference in survival between the two groups.<br>"
            f"<b>Alternative Hypothesis:</b> There is a difference in survival between the two groups.<br><br>"
            f"<b>Kaplan-Meier Survival & Log-Rank Test Results:</b><br>"
            f"• Median Survival (Control Group): {median_survival_control:.2f} units<br>"
            f"• Median Survival (New Group): {median_survival_new:.2f} units<br>"
            f"• Log-Rank Z-statistic = {format_num(z_stat)} (Z-critical = {format_num(z_crit)})<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br><br>"
            f"{conclusion}"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }
