import itertools
import numpy as np
from scipy import stats
from .utils import load_file_data, format_num


def compute_chi_sq_2_by_2(table):
    """
    Computes Chi-Square statistic with Yates' continuity correction for a 2x2 contingency table.
    Matches the exact reference implementation from nominal-copy.py:
    chi_sq = sum((|observed - expected| - 0.5)^2 / expected)
    """
    tot_row_0 = sum(table[0, :])
    tot_row_1 = sum(table[1, :])
    tot_col_0 = sum(table[:, 0])
    tot_col_1 = sum(table[:, 1])
    total = tot_row_0 + tot_row_1
    if total == 0:
        return 0.0

    e00 = tot_row_0 * tot_col_0 / total
    e10 = tot_row_1 * tot_col_0 / total
    e01 = tot_row_0 * tot_col_1 / total
    e11 = tot_row_1 * tot_col_1 / total

    chi_sq_stat = 0.0
    for obs, exp in [(table[0, 0], e00), (table[1, 0], e10), (table[0, 1], e01), (table[1, 1], e11)]:
        if exp > 0:
            chi_sq_stat += (np.abs(obs - exp) - 0.5)**2 / exp
    return chi_sq_stat


def parse_contingency_data(raw_data):
    """
    Robustly parses a contingency table from string array or numeric array.
    Supports:
    1. Purely numeric matrices (e.g. pancreatic_cancer_study.dat)
    2. Tables with header row and group label columns (e.g. nominal_data.dat)
    3. Tables with only group labels in column 0
    """
    if raw_data.ndim == 1:
        raw_data = raw_data.reshape(1, -1)

    # Case 1: Pure numeric matrix without text headers
    try:
        data_table = raw_data.astype(float)
        group_names = [f"Group {i+1}" for i in range(data_table.shape[0])]
        return group_names, data_table
    except (ValueError, TypeError):
        pass

    # Case 2: Row 0 is column headers, Column 0 is group names
    try:
        data_table = raw_data[1:, 1:].astype(float)
        group_names = list(raw_data[1:, 0])
        return group_names, data_table
    except (ValueError, TypeError):
        pass

    # Case 3: Column 0 is group names, no column header row
    try:
        data_table = raw_data[:, 1:].astype(float)
        group_names = list(raw_data[:, 0])
        return group_names, data_table
    except (ValueError, TypeError):
        pass

    raise ValueError("Unable to parse numeric contingency table from the uploaded file.")


def run_nominal_analysis(request):
    """
    Runs Chi-Square test of independence on nominal data with Bonferroni-corrected pairwise comparisons.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    raw_data = load_file_data(request, 'data_file', 'nominal_data.dat', dtype=str)
    if len(raw_data) == 0:
        return {"result_text": "Error: No nominal dataset found."}

    try:
        group_names, data_table = parse_contingency_data(raw_data)
    except Exception as e:
        return {"result_text": f"Error parsing contingency table: {str(e)}"}

    n_rows, n_cols = data_table.shape
    if n_rows < 2 or n_cols < 2:
        return {"result_text": "Error: Contingency table must have at least 2 rows and 2 columns."}

    # Calculate row totals, column totals, and grand total
    row_totals = data_table.sum(axis=1)
    col_totals = data_table.sum(axis=0)
    total = data_table.sum()

    # Calculate expected frequencies matrix
    expected = np.outer(row_totals, col_totals) / total

    # Calculate overall Chi-Square test statistic
    chi_sq_stat = (((data_table - expected)**2) / expected).sum()
    dof = (n_rows - 1) * (n_cols - 1)
    alpha = (100.0 - conf_int) / 100.0
    chi_crit = stats.chi2.ppf(1.0 - alpha, dof)
    p_value = 1.0 - stats.chi2.cdf(chi_sq_stat, dof)

    # Hypothesis testing decision
    is_significant = chi_sq_stat > chi_crit
    if is_significant:
        conclusion_text = (
            f"<b>Conclusion:</b> Reject Null Hypothesis (p < {alpha:.3f}) — "
            f"Data supports a statistically significant difference / association among the {n_rows} groups.<br>"
        )
    else:
        conclusion_text = (
            f"<b>Conclusion:</b> Fail to Reject Null Hypothesis — "
            f"Insufficient evidence to support any significant difference among the {n_rows} groups.<br>"
        )

    # Pairwise 2x2 comparisons with Bonferroni correction
    pairwise_html = ""
    if is_significant:
        pairwise_results = []
        for r1, r2 in itertools.combinations(range(n_rows), 2):
            sub_table = data_table[[r1, r2], :]
            if n_cols == 2:
                chi_2x2 = compute_chi_sq_2_by_2(sub_table)
                pairwise_results.append((group_names[r1], group_names[r2], chi_2x2))

        n_comp = len(pairwise_results)
        if n_comp > 0:
            chi_crit_bonf = stats.chi2.ppf(1.0 - (alpha / n_comp), 1)
            sig_pairs = [
                f"{g1} vs {g2} (χ² = {c:.3f})"
                for (g1, g2, c) in pairwise_results if c > chi_crit_bonf
            ]

            pairwise_html = (
                f"<br><b>Pairwise Post-Hoc Comparisons (2x2 Yates' Corrected):</b><br>"
                f"• Number of Comparisons (m) = {n_comp}<br>"
                f"• Bonferroni-corrected Critical χ² (α' = {format_num(alpha/n_comp)}, df = 1) = {format_num(chi_crit_bonf)}<br>"
            )

            if sig_pairs:
                pairwise_html += f"• <b>Statistically Significant Pairs (χ²<sub>pairwise</sub> > {format_num(chi_crit_bonf)}):</b><br>" + "<br>".join([f"&nbsp;&nbsp;→ {p}" for p in sig_pairs]) + "<br>"
            else:
                pairwise_html += "• <i>No individual pairs remained statistically significant after Bonferroni correction.</i><br>"

    return {
        "result_text": (
            f"<b>Null Hypothesis (H<sub>0</sub>):</b> Disease incidence / proportions are equal across all groups.<br>"
            f"<b>Alternative Hypothesis (H<sub>1</sub>):</b> Disease incidence / proportions differ across groups.<br><br>"
            f"<b>Chi-Square Contingency Table Analysis:</b><br>"
            f"• Groups Analyzed ({n_rows}): {', '.join(group_names)}<br>"
            f"• Degrees of Freedom (df) = {dof}<br>"
            f"• Chi-Square Statistic (χ²) = {format_num(chi_sq_stat)}<br>"
            f"• Critical χ² ({conf_int:.1f}% Confidence) = {format_num(chi_crit)}<br>"
            f"• <b>P-value = {format_num(p_value)}</b><br><br>"
            f"{conclusion_text}"
            f"{pairwise_html}"
        )
    }
