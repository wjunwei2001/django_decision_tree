import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from .utils import load_file_data, STATIC_PLOT_DIR, format_num


def run_estimation(request):
    """
    Computes sample mean, standard deviation, and confidence interval for 1 continuous parameter.
    Generates a diagnostic boxplot.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    df = load_file_data(request, 'data_file', 'est.dat')
    if len(df) == 0:
        return {"result_text": "Error: No valid data found for estimation."}

    if df.ndim > 1:
        df = df[:, 0]

    # Generate Boxplot
    plot_filename = "est_boxplot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6, 4))
    plt.boxplot(df, labels=['Sample'])
    plt.title('Boxplot of Sample Data')
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    # Calculate statistics
    n = len(df)
    x_bar = np.mean(df)
    sample_sd = np.std(df, ddof=1)
    population_sd = np.std(df)

    alpha = (100.0 - conf_int) / 100.0
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
    margin_error = t_crit * sample_sd / np.sqrt(n)

    lowerbound = x_bar - margin_error
    upperbound = x_bar + margin_error

    return {
        "result_text": (
            f"<b>Parameter Estimation Results:</b><br>"
            f"• Sample Size (N) = {n}<br>"
            f"• Sample Mean (x̄) = {format_num(x_bar)}<br>"
            f"• Sample Standard Deviation (s) = {format_num(sample_sd)}<br>"
            f"• Population Standard Deviation (σ) = {format_num(population_sd)}<br>"
            f"• Critical t-value (t*) = {format_num(t_crit)}<br>"
            f"• <b>{conf_int:.1f}% Confidence Interval:</b> [{format_num(lowerbound)}, {format_num(upperbound)}]"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }
