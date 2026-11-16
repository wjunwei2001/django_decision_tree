import html
import io
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from django.conf import settings
from .utils import STATIC_PLOT_DIR


def parse_logistic_data(request, file_key, default_filename):
    """
    Robustly loads and parses a dataset for logistic regression.
    Supports .dat, .csv, .txt files with comma, tab, or whitespace separation.
    Automatically handles headers and identifies predictor (X) and outcome (y) variables.
    """
    content = None
    if request and hasattr(request, 'FILES') and request.FILES and file_key in request.FILES:
        try:
            uploaded = request.FILES[file_key]
            content = uploaded.read().decode('utf-8', errors='ignore')
        except Exception:
            content = None

    if not content:
        # Check sample_data/ then root
        sample_path = os.path.join(settings.BASE_DIR, 'sample_data', default_filename)
        if os.path.exists(sample_path):
            with open(sample_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        else:
            root_path = os.path.join(settings.BASE_DIR, default_filename)
            if os.path.exists(root_path):
                with open(root_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

    if not content or not content.strip():
        raise ValueError(f"No dataset found. Please upload a file or verify '{default_filename}'.")

    # Detect header: check if the first row contains non-numeric tokens
    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]
    first_tokens = lines[0].replace(',', ' ').split()
    is_numeric_header = True
    for tok in first_tokens:
        try:
            float(tok)
        except ValueError:
            is_numeric_header = False
            break

    header_opt = None if is_numeric_header else 0

    df = None
    for sep in [',', r'\s+', r'\t']:
        try:
            temp_df = pd.read_csv(io.StringIO(content), sep=sep, header=header_opt, engine='python')
            if temp_df.shape[1] >= 2 and len(temp_df) >= 2:
                df = temp_df
                break
        except Exception:
            continue

    if df is None or df.shape[1] < 2:
        raise ValueError("Invalid dataset structure: Logistic regression requires at least one predictor column and one outcome column.")

    df = df.dropna()
    if len(df) < 5:
        raise ValueError("Insufficient sample size: At least 5 observations are required to fit a logistic model.")

    col_names = [str(c).strip() for c in df.columns]
    df.columns = col_names

    # Identify target column (y)
    target_col = None
    common_target_names = {
        'y', 'target', 'outcome', 'response', 'label', 'class', 'status',
        'passed', 'severity', 'choice', 'remission', 'condition', 'stage',
        'grade', 'treatment_outcome', 'disease_stage', 'adverse_event'
    }
    for col in col_names:
        if col.lower() in common_target_names:
            target_col = col
            break

    if target_col is None:
        # Check unique count: usually outcome has few unique values compared to predictors
        nunique = df.nunique()
        # Prefer last column if it is discrete
        if nunique.iloc[-1] <= 15 and nunique.iloc[-1] < len(df):
            target_col = col_names[-1]
        elif nunique.iloc[0] <= 15 and nunique.iloc[0] < len(df):
            target_col = col_names[0]
        else:
            target_col = col_names[-1]

    feature_cols = [c for c in col_names if c != target_col]
    X_df = df[feature_cols].copy()
    y_raw = df[target_col].copy()

    # Ensure predictors are numeric
    for col in feature_cols:
        X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
    valid_idx = ~X_df.isna().any(axis=1)
    X_df = X_df[valid_idx]
    y_raw = y_raw[valid_idx]

    return X_df, y_raw, feature_cols, target_col


def _format_p_value(p):
    """Clean scientific or decimal formatting for p-values."""
    if p < 0.0001:
        return f"{p:.4e}"
    return f"{p:.4f}"


def _render_tooltip(text, align="center"):
    """
    Renders an accessible interactive (?) button with an interactive tooltip.
    The text is safely stored in data-tooltip and native title attributes so that
    it NEVER renders inline into the document flow under any circumstance,
    preventing messy layouts and eliminating text leaks.
    """
    clean_text = text.strip()
    safe_attr = html.escape(clean_text, quote=True)
    plain_text = re.sub(r'<[^>]+>', '', clean_text).replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&').replace('&quot;', '"')
    safe_title = html.escape(plain_text, quote=True)
    return (
        f'<button type="button" class="info-help-btn" '
        f'data-tooltip="{safe_attr}" '
        f'title="{safe_title}" '
        f'aria-label="Explanation" tabindex="0">?</button>'
    )


# ==============================================================================
# 1. BINOMIAL LOGISTIC REGRESSION
# ==============================================================================
def run_binomial_lr(request):
    """
    Fits Binomial Logistic Regression: ln(p / (1 - p)) = beta0 + beta1*x1 + ...
    Reports coefficients, Standard Errors, z-stats, p-values, Odds Ratios (OR),
    Confidence Intervals for OR, McFadden's Pseudo R^2, Likelihood Ratio Chi-Square test,
    classification metrics, and diagnostic probability/ROC visualization.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    alpha = (100.0 - conf_int) / 100.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    try:
        X_df, y_raw, feature_cols, target_col = parse_logistic_data(request, 'data_file', 'binomial_lr.dat')
    except Exception as e:
        return {"result_text": f"<div class='alert alert-danger'>Data Loading Error: {str(e)}</div>"}

    unique_classes = pd.Series(y_raw).unique()
    if len(unique_classes) != 2:
        return {
            "result_text": (
                f"<div class='alert alert-danger'><b>Binomial Regression Error:</b> "
                f"Dependent variable '{target_col}' must contain exactly 2 unique categories. "
                f"Found {len(unique_classes)} unique classes: {list(unique_classes)}. "
                f"For >2 categories, please select Ordinal or Multinomial Logistic Regression.</div>"
            )
        }

    # Map classes to 0 and 1
    sorted_classes = sorted(unique_classes, key=lambda x: (str(x).isdigit(), str(x)))
    class_map = {sorted_classes[0]: 0, sorted_classes[1]: 1}
    y = y_raw.map(class_map).astype(int).values
    n = len(y)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))

    X_const = sm.add_constant(X_df.values, has_constant='add')
    var_names = ['Constant (Intercept)'] + list(feature_cols)

    try:
        model = sm.Logit(y, X_const)
        res = model.fit(disp=False, maxiter=100)
    except Exception as e:
        # Fallback with BFGS solver if standard Newton-Raphson has convergence issues
        try:
            model = sm.Logit(y, X_const)
            res = model.fit(method='bfgs', disp=False, maxiter=200)
        except Exception as e2:
            return {"result_text": f"<div class='alert alert-danger'>Model Convergence Error: {str(e2)}</div>"}

    # Extract statistics
    params = res.params
    bse = res.bse
    z_stats = res.tvalues
    p_values = res.pvalues
    ll = res.llf
    ll_null = res.llnull
    llr_chi2 = 2.0 * (ll - ll_null)
    df_model = len(feature_cols)
    llr_pvalue = 1.0 - stats.chi2.cdf(llr_chi2, df_model) if df_model > 0 else 1.0
    llr_sig_class = "stat-sig" if llr_pvalue < alpha else ""
    aic = res.aic
    bic = res.bic

    # Predicted probabilities and classification performance (at threshold 0.5)
    pred_probs = res.predict(X_const)
    pred_labels = (pred_probs >= 0.5).astype(int)
    tp = int(np.sum((pred_labels == 1) & (y == 1)))
    tn = int(np.sum((pred_labels == 0) & (y == 0)))
    fp = int(np.sum((pred_labels == 1) & (y == 0)))
    fn = int(np.sum((pred_labels == 0) & (y == 1)))
    accuracy = (tp + tn) / n * 100.0

    # Diagnostic performance metrics
    sensitivity = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    specificity = (tn / (tn + fp) * 100.0) if (tn + fp) > 0 else 0.0
    ppv_str = f"{(tp / (tp + fp) * 100.0):.1f}%" if (tp + fp) > 0 else "N/A"
    npv_str = f"{(tn / (tn + fn) * 100.0):.1f}%" if (tn + fn) > 0 else "N/A"

    preds_label = ", ".join(feature_cols)
    ref_label = f"{sorted_classes[0]}"
    event_label = f"{sorted_classes[1]}"

    # Build Coefficient Table HTML with Tooltips
    table_rows = []
    for i, name in enumerate(var_names):
        coef = params[i]
        se = bse[i]
        z_val = z_stats[i]
        p_val = p_values[i]
        or_val = np.exp(coef)
        ci_low = np.exp(coef - z_crit * se)
        ci_upp = np.exp(coef + z_crit * se)

        sig_star = ""
        if p_val < 0.001:
            sig_star = " ***"
        elif p_val < 0.01:
            sig_star = " **"
        elif p_val < 0.05:
            sig_star = " *"

        sig_class = "stat-sig" if p_val < alpha else ""
        or_str = f"{or_val:.4f}" if i > 0 else "—"
        ci_str = f"[{ci_low:.4f}, {ci_upp:.4f}]" if i > 0 else "—"

        table_rows.append(
            f"<tr>"
            f"<td><b>{name}</b></td>"
            f"<td>{coef:.4f}</td>"
            f"<td>{se:.4f}</td>"
            f"<td>{z_val:.3f}</td>"
            f"<td class='{sig_class}'>{_format_p_value(p_val)}{sig_star}</td>"
            f"<td><b>{or_str}</b></td>"
            f"<td><small>{ci_str}</small></td>"
            f"</tr>"
        )

    # Model Significance Interpretation
    if llr_pvalue < alpha:
        overall_eval = (
            f"<b>Model Evaluation:</b> Reject Null Hypothesis (H₀). The regression model is statistically significant "
            f"(LLR χ² = {llr_chi2:.3f}, df = {df_model}, p = {_format_p_value(llr_pvalue)}), indicating that the predictors "
            f"collectively explain significant variance in the log-odds of {event_label}."
        )
    else:
        overall_eval = (
            f"<b>Model Evaluation:</b> Fail to Reject Null Hypothesis (H₀). The model does not achieve statistical significance "
            f"(LLR χ² = {llr_chi2:.3f}, df = {df_model}, p = {_format_p_value(llr_pvalue)})."
        )

    # Significant Predictors Summary
    sig_preds = [feature_cols[i-1] for i in range(1, len(var_names)) if p_values[i] < alpha]
    if sig_preds:
        pred_summary = "• <b>Statistically Significant Predictors (p < " + f"{alpha:.2f}):</b> " + ", ".join(sig_preds) + "<br>"
    else:
        pred_summary = "• <i>No individual predictors reached statistical significance at the selected confidence level.</i><br>"

    # Diagnostic Visualizations
    plot_filename = "binomial_lr_plot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)

    plt.figure(figsize=(7.5, 4.8))
    if len(feature_cols) == 1:
        # Single Predictor: Sigmoid Curve
        x_col = feature_cols[0]
        x_vals = X_df[x_col].values
        x_grid = np.linspace(np.min(x_vals), np.max(x_vals), 300)
        X_grid_const = sm.add_constant(x_grid, has_constant='add')
        p_grid = res.predict(X_grid_const)

        # Standard Error band on log-odds
        cov = res.cov_params()
        linear_grid = X_grid_const @ params
        se_linear = np.sqrt(np.diag(X_grid_const @ cov @ X_grid_const.T))
        ci_lower_p = 1.0 / (1.0 + np.exp(-(linear_grid - z_crit * se_linear)))
        ci_upper_p = 1.0 / (1.0 + np.exp(-(linear_grid + z_crit * se_linear)))

        jitter = np.random.uniform(-0.03, 0.03, size=n)
        plt.scatter(x_vals, y + jitter, color='#0f172a', alpha=0.6, s=35, label='Observations (Jittered)')
        plt.plot(x_grid, p_grid, color='#059669', linewidth=2.5, label='Fitted Logistic Curve')
        plt.fill_between(x_grid, ci_lower_p, ci_upper_p, color='#10b981', alpha=0.18, label=f'{conf_int:.0f}% CI Band')
        plt.axhline(0.5, color='#dc2626', linestyle='--', linewidth=1.2, label='Decision Threshold (p=0.5)')
        plt.xlabel(f"Predictor: {x_col}", fontsize=11, fontweight='600')
        plt.ylabel(f"P({target_col} = {event_label})", fontsize=11, fontweight='600')
        plt.title('Binomial Logistic Regression Sigmoid Fit', fontsize=12, fontweight='700')
        plt.legend(loc='best', frameon=True)

        plot_guide_html = f"""
        <div class="plot-guide-card">
            <div class="plot-guide-header">
                <span class="plot-guide-icon">📈</span>
                <b>How to Read & Interpret the Sigmoid Logistic Curve:</b>
                {_render_tooltip("Visual representation of modeled event probability across increasing levels of your predictor.")}
            </div>
            <div class="plot-guide-grid">
                <div class="plot-guide-box" style="grid-column: 1 / -1;">
                    <ul class="guide-list">
                        <li><b>S-Shaped Sigmoid Fit (Green Curve):</b> Displays the modeled probability of the event (<b>{target_col} = {event_label}</b>) as the predictor ({x_col}) changes. A steeper slope indicates a more rapid shift in probability per unit increase.</li>
                        <li><b>Decision Threshold (0.50 Red Dashed Line):</b> Represents the clinical tipping point. Observations with predictor values where the curve is above 0.50 are predicted to experience the event.</li>
                        <li><b>Confidence Band ({conf_int:.0f}% Shaded Green Area):</b> Represents estimation precision. A narrow band indicates high confidence in predicted probability; widening bands (especially at extreme low or high predictor values) reflect sparse data.</li>
                        <li><b>Jittered Observed Data Points:</b> Real patient data plotted at 0 (negative) and 1 (positive). A slight vertical jitter is added so overlapping subjects can be seen clearly.</li>
                    </ul>
                </div>
            </div>
        </div>
        """
    else:
        # Multiple Predictors: ROC Curve and Predicted Probability Distribution
        plt.subplot(1, 2, 1)
        thresholds = np.linspace(0, 1, 200)
        tpr_list, fpr_list = [], []
        for thresh in thresholds:
            y_pred_th = (pred_probs >= thresh).astype(int)
            tpr = np.sum((y_pred_th == 1) & (y == 1)) / max(n_pos, 1)
            fpr = np.sum((y_pred_th == 1) & (y == 0)) / max(n_neg, 1)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        # Sort for clean curve
        sorted_indices = np.argsort(fpr_list)
        fpr_sorted = np.array(fpr_list)[sorted_indices]
        tpr_sorted = np.array(tpr_list)[sorted_indices]
        auc = np.trapz(tpr_sorted, fpr_sorted)

        plt.plot(fpr_sorted, tpr_sorted, color='#059669', lw=2.5, label=f'ROC Curve (AUC = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='#94a3b8', linestyle='--', label='Chance Line (AUC = 0.50)')
        plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=10)
        plt.ylabel('True Positive Rate (Sensitivity)', fontsize=10)
        plt.title('Receiver Operating Characteristic (ROC)', fontsize=11, fontweight='700')
        plt.legend(loc='lower right')
        plt.grid(True, linestyle=':', alpha=0.6)

        plt.subplot(1, 2, 2)
        plt.hist(pred_probs[y == 0], bins=10, alpha=0.55, color='#64748b', label=f'{ref_label} (y=0)', density=True)
        plt.hist(pred_probs[y == 1], bins=10, alpha=0.55, color='#059669', label=f'{event_label} (y=1)', density=True)
        plt.axvline(0.5, color='#dc2626', linestyle='--', label='Cutoff 0.5')
        plt.xlabel('Predicted Probability', fontsize=10)
        plt.ylabel('Density', fontsize=10)
        plt.title('Predicted Probability Distribution', fontsize=11, fontweight='700')
        plt.legend(loc='upper center')
        plt.grid(True, linestyle=':', alpha=0.6)

        plot_guide_html = f"""
        <div class="plot-guide-card">
            <div class="plot-guide-header">
                <span class="plot-guide-icon">📊</span>
                <b>How to Read & Interpret These Visualizations (What to Look Out For):</b>
                {_render_tooltip("Visual diagnostic guides for evaluating model discrimination and classification confidence.")}
            </div>
            <div class="plot-guide-grid">
                <div class="plot-guide-box">
                    <h4 class="guide-box-title">
                        1. ROC Curve (Left Panel)
                        {_render_tooltip("Receiver Operating Characteristic: plots True Positive Rate vs False Positive Rate across all decision thresholds.")}
                    </h4>
                    <ul class="guide-list">
                        <li><b>What it shows:</b> The model's ability to discriminate between subjects who experienced the outcome event ({event_label}) versus those who did not ({ref_label}) across all possible probability cutoffs.</li>
                        <li><b>What to look out for:</b>
                            <ul>
                                <li><b>Upward Arch:</b> The higher the curve arches towards the upper-left corner (0, 1), the better the model balances high sensitivity with low false alarm rates.</li>
                                <li><b>Dashed Grey Line:</b> Represents an AUC of 0.50 (no better than guessing or flipping a coin).</li>
                                <li><b>AUC (Area Under the Curve) Benchmark:</b>
                                    <ul>
                                        <li><b>AUC = {auc:.3f}</b> in this model.</li>
                                        <li><b>0.50:</b> No discriminative power.</li>
                                        <li><b>0.70 – 0.80:</b> Acceptable clinical discrimination.</li>
                                        <li><b>0.80 – 0.90:</b> Excellent clinical discrimination.</li>
                                        <li><b>&gt; 0.90:</b> Outstanding diagnostic accuracy.</li>
                                    </ul>
                                </li>
                            </ul>
                        </li>
                    </ul>
                </div>
                <div class="plot-guide-box">
                    <h4 class="guide-box-title">
                        2. Probability Distribution (Right Panel)
                        {_render_tooltip("Histogram showing how confident the model was when predicting probabilities for actual negative vs positive patients.")}
                    </h4>
                    <ul class="guide-list">
                        <li><b>What it shows:</b> The spread of model-calculated probabilities for actual negative cases (grey bars) versus actual positive cases (green bars).</li>
                        <li><b>What to look out for:</b>
                            <ul>
                                <li><b>Separation of Colors:</b> An effective model shows distinct separation: grey bars cluster near 0.0 (low predicted risk) and green bars cluster near 1.0 (high predicted risk).</li>
                                <li><b>Red Cutoff Line (0.50):</b> Observations to the right of 0.50 are predicted positive; observations to the left are predicted negative.</li>
                                <li><b>Ambiguity Overlap:</b> Overlapping bars around the 0.50 cutoff line indicate borderline subjects where the model has higher prediction uncertainty.</li>
                            </ul>
                        </li>
                    </ul>
                </div>
            </div>
        </div>
        """

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    result_html = f"""
    <div class="analysis-section">
        <h3 style="color: #065f46; margin-bottom: 0.5rem;">Binomial Logistic Regression Results</h3>
        <p style="color: #64748b; margin-top: 0; font-size: 0.9rem;">
            Dependent Variable: <b>{target_col}</b> (Reference: {ref_label} = 0, Event: {event_label} = 1) | Total N = {n} ({n_neg} vs {n_pos})
        </p>

        <!-- 1. Formal Statistical Hypotheses -->
        <div class="hypotheses-box">
            <div class="hypotheses-header">
                <span class="hypotheses-badge">Statistical Hypotheses</span>
                {_render_tooltip("Formal hypotheses evaluated by the overall Likelihood Ratio (Omnibus) test.")}
            </div>
            <div class="hypotheses-content">
                <div class="hypothesis-item">
                    <span class="hyp-label">Null Hypothesis (H₀):</span>
                    <span class="hyp-text">The predictor variables (<b>{preds_label}</b>) have no effect on the likelihood of <b>{target_col} = {event_label}</b> (all slope parameters β₁ = β₂ = ... = 0). The baseline intercept-only model is sufficient.</span>
                    {_render_tooltip("H₀ states that there is no true statistical relationship between the predictors and the outcome.")}
                </div>
                <div class="hypothesis-item">
                    <span class="hyp-label">Alternative Hypothesis (H₁):</span>
                    <span class="hyp-text">At least one predictor variable significantly influences the likelihood of <b>{target_col} = {event_label}</b> (at least one slope parameter β ≠ 0). The model with predictors explains outcome variance significantly better than chance.</span>
                    {_render_tooltip("H₁ states that the predictors collectively improve prediction of the outcome beyond chance.")}
                </div>
            </div>
        </div>

        <!-- 2. Model Fit & Diagnostics Summary -->
        <div class="model-fit-card">
            <div class="model-fit-title">
                <b>Model Fit & Diagnostic Summary:</b>
                {_render_tooltip("Global goodness-of-fit metrics assessing whether the model fits the dataset well.")}
            </div>
            <div class="model-fit-grid">
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Likelihood Ratio χ²:
                        {_render_tooltip("Omnibus test statistic comparing the full model against an intercept-only model. Higher values indicate stronger collective predictive power.")}
                    </span>
                    <span class="fit-metric-value"><b>{llr_chi2:.4f}</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">Degrees of Freedom (df):</span>
                    <span class="fit-metric-value"><b>{df_model}</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Model p-value:
                        {_render_tooltip("Probability of obtaining this improvement in fit by random chance. p &lt; 0.05 indicates the overall model is statistically significant.")}
                    </span>
                    <span class="fit-metric-value"><b class="{llr_sig_class}">{_format_p_value(llr_pvalue)}</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Log-Likelihood:
                        {_render_tooltip("Natural log of the model likelihood. Values closer to 0 (less negative) indicate better fit.")}
                    </span>
                    <span class="fit-metric-value">{ll:.3f}</span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Null Log-Likelihood:
                        {_render_tooltip("Log-likelihood of the baseline model with no predictors (intercept only).")}
                    </span>
                    <span class="fit-metric-value">{ll_null:.3f}</span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        AIC:
                        {_render_tooltip("Akaike Information Criterion: penalizes model complexity. Lower AIC values indicate a better, more parsimonious model.")}
                    </span>
                    <span class="fit-metric-value">{aic:.2f}</span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        BIC:
                        {_render_tooltip("Bayesian Information Criterion: imposes a stronger penalty for additional parameters. Lower BIC indicates a preferred model.")}
                    </span>
                    <span class="fit-metric-value">{bic:.2f}</span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Overall Classification Accuracy:
                        {_render_tooltip("Percentage of all cases correctly predicted (TN + TP) / N at probability threshold 0.50.")}
                    </span>
                    <span class="fit-metric-value"><b>{accuracy:.1f}%</b></span>
                </div>
            </div>
        </div>

        <!-- 3. Classification Confusion Matrix (Below Accuracy) -->
        <div class="confusion-matrix-card">
            <div class="matrix-header">
                <span class="matrix-title">
                    2×2 Classification Confusion Matrix
                    {_render_tooltip("Contingency table cross-classifying observed patient outcomes versus model predictions at the 0.50 decision threshold.")}
                </span>
                <span class="matrix-cutoff-badge">Decision Threshold = 0.50</span>
            </div>
            <div style="overflow-x: auto;">
                <table class="confusion-matrix-table">
                    <thead>
                        <tr>
                            <th rowspan="2" class="matrix-corner">
                                Actual Condition
                                {_render_tooltip("The true recorded status of the subjects in the dataset.")}
                            </th>
                            <th colspan="2" class="matrix-pred-header">Model Predicted Status</th>
                            <th rowspan="2" class="matrix-total-header">Total Actual</th>
                        </tr>
                        <tr>
                            <th class="matrix-sub-header">
                                Predicted Negative ({ref_label})
                                {_render_tooltip("Model predicted probability P(Y=1) &lt; 0.50.")}
                            </th>
                            <th class="matrix-sub-header">
                                Predicted Positive ({event_label})
                                {_render_tooltip("Model predicted probability P(Y=1) ≥ 0.50.")}
                            </th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td class="actual-label-cell">
                                <b>Actual {ref_label}</b> (0)
                            </td>
                            <td class="matrix-data-cell cell-tn">
                                <div class="cell-number">{tn}</div>
                                <div class="cell-badge badge-tn">True Neg (TN)</div>
                            </td>
                            <td class="matrix-data-cell cell-fp">
                                <div class="cell-number">{fp}</div>
                                <div class="cell-badge badge-fp">False Pos (FP)</div>
                            </td>
                            <td class="matrix-data-cell cell-subtotal">
                                <b>{n_neg}</b>
                            </td>
                        </tr>
                        <tr>
                            <td class="actual-label-cell">
                                <b>Actual {event_label}</b> (1)
                            </td>
                            <td class="matrix-data-cell cell-fn">
                                <div class="cell-number">{fn}</div>
                                <div class="cell-badge badge-fn">False Neg (FN)</div>
                            </td>
                            <td class="matrix-data-cell cell-tp">
                                <div class="cell-number">{tp}</div>
                                <div class="cell-badge badge-tp">True Pos (TP)</div>
                            </td>
                            <td class="matrix-data-cell cell-subtotal">
                                <b>{n_pos}</b>
                            </td>
                        </tr>
                        <tr class="matrix-totals-row">
                            <td class="actual-label-cell"><b>Total Predicted</b></td>
                            <td class="matrix-data-cell cell-subtotal"><b>{tn + fn}</b></td>
                            <td class="matrix-data-cell cell-subtotal"><b>{fp + tp}</b></td>
                            <td class="matrix-data-cell cell-grandtotal"><b>N = {n}</b></td>
                        </tr>
                    </tbody>
                </table>
            </div>
            <div class="matrix-stats-pills">
                <div class="stat-pill">
                    <span class="stat-pill-label">Sensitivity (Recall):</span>
                    <span class="stat-pill-val">{sensitivity:.1f}%</span>
                    {_render_tooltip("TP / (TP + FN): Proportion of actual positive cases that were correctly identified.")}
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-label">Specificity:</span>
                    <span class="stat-pill-val">{specificity:.1f}%</span>
                    {_render_tooltip("TN / (TN + FP): Proportion of actual negative cases that were correctly identified.")}
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-label">Positive Predictive Value (PPV):</span>
                    <span class="stat-pill-val">{ppv_str}</span>
                    {_render_tooltip("TP / (TP + FP): Probability that a subject predicted positive actually has the outcome.")}
                </div>
                <div class="stat-pill">
                    <span class="stat-pill-label">Negative Predictive Value (NPV):</span>
                    <span class="stat-pill-val">{npv_str}</span>
                    {_render_tooltip("TN / (TN + FN): Probability that a subject predicted negative actually does not have the outcome.")}
                </div>
            </div>
        </div>

        <!-- 4. Model Coefficients & Odds Ratios Table with Tooltips -->
        <b>Model Coefficients & Odds Ratios ({conf_int:.0f}% Confidence Level):</b>
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Coefficient (β) {_render_tooltip("Estimated change in log-odds of the outcome per 1-unit increase in predictor, holding other variables constant.")}</th>
                    <th>Std. Error {_render_tooltip("Standard error of the coefficient, measuring estimation uncertainty.")}</th>
                    <th>z-statistic {_render_tooltip("Wald z = β / SE. Measures how many standard errors the coefficient is from zero.")}</th>
                    <th>p-value {_render_tooltip("Wald test two-tailed p-value. p &lt; 0.05 indicates the predictor has a statistically significant independent association.")}</th>
                    <th>Odds Ratio (eᵝ) {_render_tooltip("Multiplicative factor by which the odds of the outcome event change for a 1-unit increase in the predictor. OR &gt; 1 indicates increased likelihood; OR &lt; 1 indicates decreased likelihood.", align="right")}</th>
                    <th>{conf_int:.0f}% CI for OR</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        <small style="color: #64748b;">Significance codes: *** p &lt; 0.001, ** p &lt; 0.01, * p &lt; 0.05</small><br><br>

        <!-- 5. Conclusion & Clinical Interpretation Callout -->
        <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 0.85rem 1rem; border-radius: 4px;">
            {overall_eval}<br><br>
            {pred_summary}
            <small><b>Odds Ratio Interpretation:</b> The Odds Ratio (OR) indicates the multiplicative change in the odds of '{event_label}' for each 1-unit increase in the predictor, holding other variables constant. An OR &gt; 1 indicates increased likelihood; an OR &lt; 1 indicates decreased likelihood.</small>
        </div>
    </div>
    """

    return {
        "result_text": result_html,
        "boxplot_url": f"questions/{plot_filename}",
        "plot_guide": plot_guide_html,
    }


# ==============================================================================
# 2. ORDINAL LOGISTIC REGRESSION
# ==============================================================================
def _brant_test(y, X_df_values, main_betas, main_cov, n_preds, num_classes):
    """
    Brant (1990) test for the proportional odds assumption.

    For each of the K-1 cumulative splits, a separate binary logistic model is
    fitted. Each predictor's binary-model coefficient is then compared to the
    coefficient from the proportional odds model.  Under H0 (proportional odds
    holds) the differences should not be significantly larger than zero.

    Returns (chi2, df, p_value, per_predictor_details) or (None, None, None, msg)
    if the test cannot be computed.
    """
    X_c = sm.add_constant(X_df_values, has_constant='add')
    binary_results = []
    for k in range(1, num_classes):
        y_bin = (y >= k).astype(int)
        # Skip if outcome is perfectly separated
        if len(np.unique(y_bin)) < 2:
            return None, None, None, f"Cannot test: cumulative split y >= {k} has no variance."
        try:
            br = sm.Logit(y_bin, X_c).fit(disp=False, maxiter=150)
            binary_results.append(br)
        except Exception as e:
            return None, None, None, f"Binary model fit failed for split {k}: {e}"

    if len(binary_results) < 1:
        return None, None, None, "No binary models could be fitted."

    total_chi2 = 0.0
    total_df = 0
    per_pred = []
    for j in range(n_preds):
        pred_chi2 = 0.0
        pred_df = 0
        for br in binary_results:
            diff = br.params[j + 1] - main_betas[j]
            # Var(diff) = Var(beta_binary) + Var(beta_main)  [models independent]
            var_diff = br.bse[j + 1] ** 2 + main_cov[j, j]
            if var_diff > 0:
                pred_chi2 += diff ** 2 / var_diff
                pred_df += 1
        pval_j = 1.0 - stats.chi2.cdf(pred_chi2, pred_df) if pred_df > 0 else 1.0
        per_pred.append((pred_chi2, pred_df, pval_j))
        total_chi2 += pred_chi2
        total_df += pred_df

    if total_df == 0:
        return None, None, None, "Brant test degrees of freedom are zero."

    total_pval = 1.0 - stats.chi2.cdf(total_chi2, total_df)
    return total_chi2, total_df, total_pval, per_pred


def run_ordinal_lr(request):
    """
    Fits Ordinal Logistic Regression (Proportional Odds Cumulative Logit Model):
    ln(P(Y <= k) / P(Y > k)) = alpha_k - beta^T X
    Reports LLR test, coefficients with plain-language OR interpretations,
    Brant proportional odds test, confusion matrix, macro F1, expandable
    advanced sections, and improved cumulative probability visualisation.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    alpha = (100.0 - conf_int) / 100.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    try:
        X_df, y_raw, feature_cols, target_col = parse_logistic_data(request, 'data_file', 'ordinal_lr.dat')
    except Exception as e:
        return {"result_text": f"<div class='alert alert-danger'>Data Loading Error: {str(e)}</div>"}

    unique_classes = pd.Series(y_raw).unique()
    if len(unique_classes) < 2:
        return {
            "result_text": f"<div class='alert alert-danger'>Ordinal Regression requires at least 2 ordered categories. Found {len(unique_classes)}.</div>"
        }

    # Order classes naturally
    sorted_classes = sorted(unique_classes, key=lambda x: (str(x).isdigit(), str(x)))
    class_map = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    y = y_raw.map(class_map).astype(int).values
    n = len(y)
    num_classes = len(sorted_classes)

    try:
        mod = OrderedModel(y, X_df.values, distr='logit')
        res = mod.fit(method='bfgs', disp=False, maxiter=300)
    except Exception as e:
        try:
            mod = OrderedModel(y, X_df.values, distr='logit')
            res = mod.fit(method='nm', disp=False, maxiter=500)
        except Exception as e2:
            return {"result_text": f"<div class='alert alert-danger'>Ordinal Model Fitting Error: {str(e2)}</div>"}

    params   = res.params
    bse      = res.bse
    p_values = res.pvalues
    z_stats  = res.tvalues
    ll       = res.llf
    ll_null  = res.llnull
    aic      = res.aic
    bic      = res.bic
    llr_chi2 = res.llr
    llr_pval = res.llr_pvalue
    n_preds  = len(feature_cols)

    pred_betas   = params[:n_preds]
    pred_bse     = bse[:n_preds]
    pred_pvals   = p_values[:n_preds]
    pred_z       = z_stats[:n_preds]
    cutoff_params = params[n_preds:]
    cutoff_bse    = bse[n_preds:]
    cutoff_pvals  = p_values[n_preds:]
    main_cov      = res.cov_params()[:n_preds, :n_preds]

    # ── Predictions & Classification ──────────────────────────────────────────
    pred_probs  = res.predict(X_df.values)
    pred_labels = np.argmax(pred_probs, axis=1)
    acc = np.mean(pred_labels == y) * 100.0

    # Macro-averaged precision, recall, F1 (zero_division=0)
    from collections import defaultdict
    tp_per = defaultdict(int); fp_per = defaultdict(int); fn_per = defaultdict(int)
    for true, pred in zip(y, pred_labels):
        if true == pred:
            tp_per[true] += 1
        else:
            fp_per[pred] += 1
            fn_per[true] += 1
    precisions, recalls, f1s = [], [], []
    for k in range(num_classes):
        tp = tp_per[k]; fp = fp_per[k]; fn = fn_per[k]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        precisions.append(prec); recalls.append(rec); f1s.append(f1)
    macro_prec = float(np.mean(precisions))
    macro_rec  = float(np.mean(recalls))
    macro_f1   = float(np.mean(f1s))

    # ── Category distribution ─────────────────────────────────────────────────
    class_counts = pd.Series(y).value_counts().sort_index()
    counts_str = ", ".join([f"{sorted_classes[idx]}: {cnt}" for idx, cnt in class_counts.items()])

    # ── Brant proportional odds test ──────────────────────────────────────────
    brant_chi2, brant_df, brant_pval, brant_detail = _brant_test(
        y, X_df.values, pred_betas, main_cov, n_preds, num_classes
    )

    # ── Helper: plain-language OR interpretation ──────────────────────────────
    def _or_interpretation(name, or_val, p_val, alpha_thr):
        pct = abs(or_val - 1.0) * 100.0
        direction = "increase" if or_val >= 1.0 else "decrease"
        sig = "statistically significant" if p_val < alpha_thr else "not statistically significant"
        return (
            f"Each 1-unit increase in <b>{name}</b> is associated with approximately "
            f"a <b>{pct:.0f}% {direction}</b> in the odds of being in a higher "
            f"severity category ({sig} at p {'< ' if p_val < alpha_thr else '= '}{_format_p_value(p_val)}). "
            f"This represents an association, not necessarily a causal relationship."
        )

    # ── Build predictor table ─────────────────────────────────────────────────
    table_rows = []
    interp_blocks = []
    for i, name in enumerate(feature_cols):
        b      = pred_betas[i]
        se     = pred_bse[i]
        z_val  = pred_z[i]
        p_val  = pred_pvals[i]
        or_val = np.exp(b)
        ci_low = np.exp(b - z_crit * se)
        ci_upp = np.exp(b + z_crit * se)
        sig_star  = " ***" if p_val < 0.001 else (" **" if p_val < 0.01 else (" *" if p_val < 0.05 else ""))
        sig_class = "stat-sig" if p_val < alpha else ""
        table_rows.append(
            f"<tr>"
            f"<td><b>{name}</b></td>"
            f"<td>{b:.4f}</td>"
            f"<td>{se:.4f}</td>"
            f"<td>{z_val:.3f}</td>"
            f"<td class='{sig_class}'>{_format_p_value(p_val)}{sig_star}</td>"
            f"<td><b>{or_val:.4f}</b></td>"
            f"<td><small>[{ci_low:.4f}, {ci_upp:.4f}]</small></td>"
            f"</tr>"
        )
        interp_text = _or_interpretation(name, or_val, p_val, alpha)
        interp_blocks.append(
            f"<div style='margin-bottom:0.6rem; padding:0.55rem 0.75rem; "
            f"background:#f0fdf4; border-left:3px solid #10b981; border-radius:4px; font-size:0.88rem;'>"
            f"{interp_text}</div>"
        )

    # ── Build threshold cutpoints table ───────────────────────────────────────
    cutoff_rows = []
    for j in range(len(cutoff_params)):
        c_label = (f"Threshold {sorted_classes[j]} | {sorted_classes[j+1]}"
                   if j + 1 < len(sorted_classes) else f"Cutoff {j+1}")
        cutoff_rows.append(
            f"<tr>"
            f"<td>{c_label}</td>"
            f"<td>{cutoff_params[j]:.4f}</td>"
            f"<td>{cutoff_bse[j]:.4f}</td>"
            f"<td>{_format_p_value(cutoff_pvals[j])}</td>"
            f"</tr>"
        )

    # ── Confusion matrix (multi-class) ────────────────────────────────────────
    cm_size = num_classes
    cm = np.zeros((cm_size, cm_size), dtype=int)
    for true, pred in zip(y, pred_labels):
        cm[true][pred] += 1

    cm_header = "".join(
        f"<th style='background:#e0f2fe;color:#0369a1;'>Pred: {sorted_classes[k]}</th>"
        for k in range(cm_size)
    )
    cm_body = ""
    for i_row in range(cm_size):
        row_total = int(np.sum(cm[i_row, :]))
        cm_body += f"<tr><td style='font-weight:700;background:#f8fafc;'>Actual: {sorted_classes[i_row]}</td>"
        for j_col in range(cm_size):
            cell_val = cm[i_row][j_col]
            if i_row == j_col:
                # Diagonal – correct
                cell_style = "background:#dcfce7;font-weight:700;color:#15803d;"
            else:
                cell_style = "background:#fff;"
            cm_body += f"<td style='{cell_style}text-align:center;'>{cell_val}</td>"
        cm_body += f"<td style='background:#f1f5f9;font-weight:700;text-align:center;'>{row_total}</td></tr>"
    # Totals row
    col_totals = cm.sum(axis=0)
    cm_body += "<tr><td style='font-weight:700;background:#f1f5f9;'>Total Predicted</td>"
    for ct in col_totals:
        cm_body += f"<td style='text-align:center;font-weight:700;background:#f1f5f9;'>{ct}</td>"
    cm_body += f"<td style='text-align:center;font-weight:700;background:#e2e8f0;'>N = {n}</td></tr>"

    # ── Brant test HTML block ─────────────────────────────────────────────────
    if brant_chi2 is not None:
        if brant_pval < 0.05:
            brant_colour = "#dc2626"; brant_bg = "#fef2f2"
            brant_interp = (
                "<b>Evidence of a violation</b> of the proportional odds assumption "
                "(p < 0.05). The effect of at least one predictor may differ across "
                "cumulative outcome splits. Interpret Odds Ratios with caution; "
                "consider a multinomial logistic regression or a partial proportional odds model."
            )
        else:
            brant_colour = "#059669"; brant_bg = "#f0fdf4"
            brant_interp = (
                "<b>No significant evidence</b> of a proportional odds violation "
                "(p ≥ 0.05). The assumption that each predictor shifts the log-odds "
                "equally across all cumulative splits is consistent with the data."
            )
        per_pred_rows = ""
        if isinstance(brant_detail, list):
            for j, (pc, pd_, pp) in enumerate(brant_detail):
                sig = " *" if pp < 0.05 else ""
                per_pred_rows += (
                    f"<tr><td>{feature_cols[j]}</td>"
                    f"<td style='text-align:center;'>{pc:.3f}</td>"
                    f"<td style='text-align:center;'>{pd_}</td>"
                    f"<td style='text-align:center;{'color:#dc2626;font-weight:700;' if pp < 0.05 else ''}'>"
                    f"{_format_p_value(pp)}{sig}</td></tr>"
                )
        brant_html = f"""
        <div style="border:1px solid #e2e8f0; border-radius:6px; padding:1rem; margin-bottom:1.25rem; background:{brant_bg};">
            <div style="font-weight:700; color:{brant_colour}; margin-bottom:0.5rem;">
                Brant Test for Proportional Odds (Omnibus): χ²({brant_df}) = {brant_chi2:.3f}, p = {_format_p_value(brant_pval)}
            </div>
            <p style="margin:0 0 0.75rem; font-size:0.875rem;">{brant_interp}</p>
            <details>
                <summary style="cursor:pointer; font-size:0.85rem; color:#475569; margin-bottom:0.4rem;">Per-predictor Brant results</summary>
                <table class="stats-table" style="max-width:480px; margin-top:0.5rem;">
                    <thead><tr><th>Predictor</th><th>χ²</th><th>df</th><th>p-value</th></tr></thead>
                    <tbody>{per_pred_rows}</tbody>
                </table>
                <small style="color:#64748b; display:block; margin-top:0.35rem;">
                    Note: the Brant test is based on comparing binary logistic models for each cumulative split
                    (y ≥ k) to the pooled proportional odds model. Small per-split samples may reduce reliability.
                </small>
            </details>
        </div>"""
    else:
        brant_html = f"""
        <div style="border:1px solid #fde68a; border-radius:6px; padding:0.85rem 1rem; margin-bottom:1.25rem; background:#fffbeb;">
            <b style="color:#b45309;">Proportional Odds Assumption:</b>
            The Brant test could not be completed. Reason: {brant_detail}.
            Consider visually inspecting whether the predictor effects are consistent
            across cumulative splits or use a multinomial model as an alternative.
        </div>"""

    # ── LLR significance banner ───────────────────────────────────────────────
    if llr_pval < alpha:
        llr_banner = (
            f"<b>Model Evaluation:</b> Reject H₀. The fitted model is statistically "
            f"significant (LLR χ²({n_preds}) = {llr_chi2:.3f}, p = {_format_p_value(llr_pval)}). "
            f"The predictors collectively explain a meaningful portion of variance in {target_col}."
        )
        llr_bg = "#f0fdf4"; llr_border = "#10b981"
    else:
        llr_banner = (
            f"<b>Model Evaluation:</b> Fail to reject H₀. The model does not reach "
            f"statistical significance (LLR χ²({n_preds}) = {llr_chi2:.3f}, p = {_format_p_value(llr_pval)})."
        )
        llr_bg = "#f8fafc"; llr_border = "#94a3b8"

    # ── Diagnostic plot ───────────────────────────────────────────────────────
    plot_filename = "ordinal_lr_plot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)

    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    first_pred = feature_cols[0]
    x_vals = X_df[first_pred].values

    if len(feature_cols) == 1:
        # Single predictor: smooth predicted probability curves
        x_grid = np.linspace(np.min(x_vals), np.max(x_vals), 300)
        X_grid = x_grid.reshape(-1, 1)
        p_grid = res.predict(X_grid)

        plt.figure(figsize=(7.5, 4.8))
        for k in range(num_classes):
            col = colors[k % len(colors)]
            plt.plot(x_grid, p_grid[:, k],
                     label=f'P({target_col} = {sorted_classes[k]})',
                     color=col, lw=2.2)
        plt.axhline(1/num_classes, color='#94a3b8', linestyle=':', linewidth=1,
                    label='Equal probability reference')
        plt.xlabel(f"Predictor: {first_pred}", fontsize=11, fontweight='600')
        plt.ylabel("Predicted Category Probability", fontsize=11, fontweight='600')
        plt.title(f"Predicted Probabilities by {first_pred}", fontsize=12, fontweight='700')
        plt.legend(frameon=True, loc='best')
        plt.grid(True, linestyle='--', alpha=0.45)
        plt.ylim(-0.02, 1.02)
    else:
        # Multiple predictors: 2-panel – category probability curves + bar chart of mean predicted
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8))

        # Panel 1: Predicted probability curves for first predictor (others at mean)
        x_grid = np.linspace(np.min(x_vals), np.max(x_vals), 200)
        other_means = X_df.drop(columns=[first_pred]).mean().values
        X_grid_full = np.column_stack([
            x_grid,
            np.tile(other_means, (len(x_grid), 1))
        ] if len(feature_cols) > 1 else [x_grid])

        # Match column order to original X_df
        col_order = list(feature_cols)
        first_idx = col_order.index(first_pred)
        X_grid_arr = np.zeros((len(x_grid), len(col_order)))
        for ci, cname in enumerate(col_order):
            if cname == first_pred:
                X_grid_arr[:, ci] = x_grid
            else:
                X_grid_arr[:, ci] = X_df[cname].mean()
        p_grid = res.predict(X_grid_arr)

        for k in range(num_classes):
            axes[0].plot(x_grid, p_grid[:, k],
                         label=f'{sorted_classes[k]}',
                         color=colors[k % len(colors)], lw=2.2)
        axes[0].set_xlabel(f"{first_pred}\n(other predictors at mean)", fontsize=10)
        axes[0].set_ylabel("Predicted Category Probability", fontsize=10)
        axes[0].set_title("Predicted Probabilities", fontsize=11, fontweight='700')
        axes[0].legend(title=target_col, frameon=True, loc='best', fontsize=9)
        axes[0].grid(True, linestyle='--', alpha=0.45)
        axes[0].set_ylim(-0.02, 1.02)

        # Panel 2: Actual vs Predicted category counts
        actual_counts   = [np.sum(y == k) for k in range(num_classes)]
        predicted_counts = [np.sum(pred_labels == k) for k in range(num_classes)]
        x_pos = np.arange(num_classes)
        width = 0.35
        axes[1].bar(x_pos - width/2, actual_counts, width, label='Actual',
                    color='#3b82f6', alpha=0.8)
        axes[1].bar(x_pos + width/2, predicted_counts, width, label='Predicted',
                    color='#10b981', alpha=0.8)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([str(s) for s in sorted_classes], fontsize=10)
        axes[1].set_xlabel(f"{target_col} Category", fontsize=10)
        axes[1].set_ylabel("Count", fontsize=10)
        axes[1].set_title("Actual vs Predicted Counts", fontsize=11, fontweight='700')
        axes[1].legend(fontsize=9)
        axes[1].grid(True, linestyle='--', alpha=0.4, axis='y')

    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    # ── Assemble HTML result ──────────────────────────────────────────────────
    result_html = f"""
    <div class="analysis-section">
        <h3 style="color: #065f46; margin-bottom: 0.5rem;">Ordinal Logistic Regression Results</h3>
        <p style="color: #64748b; margin-top: 0; font-size: 0.9rem;">
            Dependent Variable: <b>{target_col}</b>
            (Ordered Categories: {' &lt; '.join([str(c) for c in sorted_classes])}) | Total N = {n}
            <br><small style="color:#94a3b8;">
                Model: Proportional Odds (Cumulative Logit) · Estimation: MLE (BFGS).
                Odds Ratios reflect associations, not necessarily causal effects.
                Metrics are calculated on the training data, which may overestimate performance.
            </small>
        </p>

        <!-- ── 1. Model Fit Summary ────────────────────────────── -->
        <div class="model-fit-card">
            <div class="model-fit-title"><b>Model Fit Summary</b></div>
            <div class="model-fit-grid">
                <div class="fit-metric-item">
                    <span class="fit-metric-name">Ordered Categories:</span>
                    <span class="fit-metric-value">{counts_str}</span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Likelihood Ratio χ²:
                        {_render_tooltip("Compares the fitted model against an intercept-only (null) model. A large value with small p-value means the predictors collectively improve prediction.")}
                    </span>
                    <span class="fit-metric-value"><b>{llr_chi2:.3f}</b> (df = {n_preds})</span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Model p-value:
                        {_render_tooltip("Probability of this chi-square improvement by chance if the null model were true. p < 0.05 is conventionally significant.")}
                    </span>
                    <span class="fit-metric-value">
                        <b style="color:{'#059669' if llr_pval < 0.05 else '#64748b'};">{_format_p_value(llr_pval)}</b>
                    </span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        McFadden's Pseudo R²:
                        {_render_tooltip("Ranges from 0 to 1. Values of 0.2–0.4 are considered good fit for logistic models. Cannot be interpreted like R² in linear regression.")}
                    </span>
                    <span class="fit-metric-value"><b>{res.prsquared:.4f}</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        AIC:
                        {_render_tooltip("Akaike Information Criterion — lower is better. Useful for comparing models on the same data.")}
                    </span>
                    <span class="fit-metric-value">{aic:.2f}</span>
                </div>
            </div>
            <!-- Advanced: Log-likelihood and BIC -->
            <details style="margin-top:0.85rem;">
                <summary style="cursor:pointer; font-size:0.85rem; color:#475569; padding:0.25rem 0;">
                    ▸ Advanced statistics (Log-likelihood, BIC)
                </summary>
                <div style="margin-top:0.6rem; font-size:0.875rem; color:#475569; line-height:1.8;">
                    • Log-Likelihood (fitted): {ll:.3f}<br>
                    • Log-Likelihood (null, intercepts only): {ll_null:.3f}<br>
                    • BIC: {bic:.2f}
                </div>
            </details>
        </div>

        <!-- ── 2. Statistical Hypotheses ──────────────────────── -->
        <div class="hypotheses-box">
            <div class="hypotheses-header">
                <span class="hypotheses-badge">Statistical Hypotheses</span>
            </div>
            <div class="hypotheses-content">
                <div class="hypothesis-item">
                    <span class="hyp-label">H₀:</span>
                    <span class="hyp-text">None of the predictor variables (<b>{', '.join(feature_cols)}</b>)
                    have an effect on the log-odds of attaining a higher <b>{target_col}</b> category
                    (all β = 0).</span>
                </div>
                <div class="hypothesis-item">
                    <span class="hyp-label">H₁:</span>
                    <span class="hyp-text">At least one predictor significantly shifts the cumulative
                    log-odds of a higher <b>{target_col}</b> outcome (at least one β ≠ 0).</span>
                </div>
            </div>
        </div>

        <!-- ── 3. Predictor Effects ────────────────────────────── -->
        <b>Predictor Effects &amp; Proportional Odds Ratios ({conf_int:.0f}% CI):</b>
        <p style="font-size:0.82rem; color:#64748b; margin: 0.2rem 0 0.6rem;">
            The Odds Ratio (OR) tells you how the odds of being in a <i>higher</i>
            outcome category change for a one-unit increase in the predictor.
            OR &gt; 1 means higher predictor values are associated with a more severe
            outcome; OR &lt; 1 means the opposite.
            <b>Odds are not the same as probabilities</b> — an OR of 2 does not mean
            the probability doubles.
        </p>
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Coefficient (β) {_render_tooltip("The estimated log-odds coefficient. Negative means the predictor is associated with lower outcome categories.")}</th>
                    <th>Std. Error {_render_tooltip("Uncertainty in the coefficient estimate.")}</th>
                    <th>z-statistic {_render_tooltip("Wald test: how many standard errors the coefficient is from zero.")}</th>
                    <th>p-value {_render_tooltip("If p < 0.05, the predictor has a statistically significant association with the outcome.")}</th>
                    <th>Odds Ratio (eᵝ) {_render_tooltip("The multiplicative change in cumulative odds of a higher category per 1-unit increase in the predictor.")}</th>
                    <th>{conf_int:.0f}% CI for OR</th>
                </tr>
            </thead>
            <tbody>{''.join(table_rows)}</tbody>
        </table>
        <small style="color: #64748b;">Significance codes: *** p &lt; 0.001, ** p &lt; 0.01, * p &lt; 0.05</small><br><br>

        <b>Plain-Language Interpretation of Predictor Effects:</b>
        <div style="margin-top:0.5rem;">{''.join(interp_blocks)}</div>

        <!-- ── 4. Proportional Odds Assumption ────────────────── -->
        <b>Proportional Odds Assumption:</b>
        <p style="font-size:0.82rem; color:#64748b; margin: 0.2rem 0 0.6rem;">
            The model assumes that each predictor shifts the cumulative log-odds by the
            same amount regardless of which category boundary is being crossed
            (i.e., the Odds Ratios are the same for every cumulative split).
            The Brant test checks whether this holds.
        </p>
        {brant_html}

        <!-- ── 5. Classification Performance ─────────────────── -->
        <div class="model-fit-card">
            <div class="model-fit-title">
                <b>Model Classification Performance</b>
                {_render_tooltip("All metrics are computed on the training data. In-sample performance may overestimate generalisation to new patients.")}
            </div>
            <div class="model-fit-grid">
                <div class="fit-metric-item">
                    <span class="fit-metric-name">Overall Accuracy:</span>
                    <span class="fit-metric-value"><b>{acc:.1f}%</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Macro Precision:
                        {_render_tooltip("Average proportion of correctly predicted cases among all cases predicted in each category. Macro = unweighted average across categories.")}
                    </span>
                    <span class="fit-metric-value"><b>{macro_prec*100:.1f}%</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Macro Recall:
                        {_render_tooltip("Average proportion of actual cases in each category that were correctly identified. Also called sensitivity.")}
                    </span>
                    <span class="fit-metric-value"><b>{macro_rec*100:.1f}%</b></span>
                </div>
                <div class="fit-metric-item">
                    <span class="fit-metric-name">
                        Macro F1-Score:
                        {_render_tooltip("Harmonic mean of Macro Precision and Macro Recall. A balanced metric; 100% is perfect, 0% is worst.")}
                    </span>
                    <span class="fit-metric-value"><b>{macro_f1*100:.1f}%</b></span>
                </div>
            </div>
        </div>

        <!-- Confusion Matrix -->
        <div class="confusion-matrix-card">
            <div class="matrix-header">
                <span class="matrix-title">Classification Confusion Matrix</span>
                <span class="matrix-cutoff-badge">Predicted = argmax of category probabilities</span>
            </div>
            <div style="overflow-x: auto;">
                <table class="confusion-matrix-table">
                    <thead>
                        <tr>
                            <th class="matrix-corner" style="background:#f8fafc;"></th>
                            {cm_header}
                            <th class="matrix-total-header">Total Actual</th>
                        </tr>
                    </thead>
                    <tbody>{cm_body}</tbody>
                </table>
            </div>
            <small style="color:#64748b;">Diagonal cells (green) = correctly classified observations.</small>
        </div>

        <!-- ── 6. Advanced: Threshold Cutpoints ──────────────── -->
        <details style="margin-bottom:1.25rem;">
            <summary style="cursor:pointer; font-size:0.9rem; font-weight:700; color:#475569; padding:0.5rem 0.75rem; background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px;">
                ▸ Advanced Model Parameters — Threshold Cutpoints (Intercepts)
            </summary>
            <div style="padding:0.85rem 0.5rem 0 0.5rem;">
                <p style="font-size:0.82rem; color:#64748b; margin-bottom:0.6rem;">
                    These parameters define the boundary values on the latent severity scale
                    that separate adjacent outcome categories. They are estimated alongside
                    the predictor coefficients but are typically of secondary interest for
                    interpretation.
                </p>
                <table class="stats-table" style="max-width: 520px;">
                    <thead>
                        <tr>
                            <th>Threshold (Cutpoint)</th>
                            <th>Estimate</th>
                            <th>Std. Error</th>
                            <th>p-value</th>
                        </tr>
                    </thead>
                    <tbody>{''.join(cutoff_rows)}</tbody>
                </table>
            </div>
        </details>

        <!-- ── 7. Overall Conclusion ──────────────────────────── -->
        <div style="background:{llr_bg}; border-left: 4px solid {llr_border}; padding: 0.85rem 1rem; border-radius: 4px;">
            {llr_banner}
        </div>
    </div>
    """

    plot_guide_html = f"""
    <div class="plot-guide-card">
        <div class="plot-guide-header">
            <span class="plot-guide-icon">📊</span>
            <b>How to Read the Visualisation:</b>
        </div>
        <div class="plot-guide-grid">
            <div class="plot-guide-box">
                <h4 class="guide-box-title">Predicted Probability Curves</h4>
                <ul class="guide-list">
                    <li>Each coloured line shows the model's predicted probability that
                        a patient falls into each <b>{target_col}</b> category as <b>{first_pred}</b>
                        changes (other predictors held at their mean).</li>
                    <li>The three curves must <b>sum to 1</b> at every point — if one rises,
                        another must fall.</li>
                    <li>Steeper slopes = stronger predictor effect.</li>
                    <li><b>What to look for:</b> Do high values of {first_pred} clearly shift
                        probability toward more severe (or milder) categories? If curves are
                        nearly flat, the predictor has little discriminating power.</li>
                </ul>
            </div>
            {"" if len(feature_cols) == 1 else '''
            <div class="plot-guide-box">
                <h4 class="guide-box-title">Actual vs Predicted Counts</h4>
                <ul class="guide-list">
                    <li>Blue bars = actual number of patients in each category.</li>
                    <li>Green bars = number of patients the model predicted into each category.</li>
                    <li>Large discrepancies suggest the model struggles with that category
                        (common for minority categories).</li>
                </ul>
            </div>'''}
        </div>
    </div>
    """

    return {
        "result_text": result_html,
        "boxplot_url": f"questions/{plot_filename}",
        "plot_guide": plot_guide_html,
    }

    """
    Fits Ordinal Logistic Regression (Proportional Odds Cumulative Logit Model):
    ln(P(Y <= k) / P(Y > k)) = alpha_k - beta^T X
    Reports coefficients, threshold cutoffs, SEs, p-values, proportional odds ratios,
    AIC/BIC, classification performance, and cumulative category probability plots.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    alpha = (100.0 - conf_int) / 100.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    try:
        X_df, y_raw, feature_cols, target_col = parse_logistic_data(request, 'data_file', 'ordinal_lr.dat')
    except Exception as e:
        return {"result_text": f"<div class='alert alert-danger'>Data Loading Error: {str(e)}</div>"}

    unique_classes = pd.Series(y_raw).unique()
    if len(unique_classes) < 2:
        return {
            "result_text": f"<div class='alert alert-danger'>Ordinal Regression requires at least 2 ordered categories. Found {len(unique_classes)}.</div>"
        }

    # Order classes naturally
    sorted_classes = sorted(unique_classes, key=lambda x: (str(x).isdigit(), str(x)))
    class_map = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    y = y_raw.map(class_map).astype(int).values
    n = len(y)
    num_classes = len(sorted_classes)

    try:
        # OrderedModel uses proportional odds logit formulation
        mod = OrderedModel(y, X_df.values, distr='logit')
        res = mod.fit(method='bfgs', disp=False, maxiter=300)
    except Exception as e:
        try:
            mod = OrderedModel(y, X_df.values, distr='logit')
            res = mod.fit(method='nm', disp=False, maxiter=500)
        except Exception as e2:
            return {"result_text": f"<div class='alert alert-danger'>Ordinal Model Fitting Error: {str(e2)}</div>"}

    params = res.params
    bse = res.bse
    p_values = res.pvalues
    z_stats = res.tvalues
    ll = res.llf
    aic = res.aic
    bic = res.bic

    # Coefficients correspond to predictors, followed by threshold cutoffs
    n_preds = len(feature_cols)
    pred_betas = params[:n_preds]
    pred_bse = bse[:n_preds]
    pred_pvals = p_values[:n_preds]
    pred_z = z_stats[:n_preds]

    cutoff_params = params[n_preds:]
    cutoff_bse = bse[n_preds:]
    cutoff_pvals = p_values[n_preds:]

    # Build Predictor Coefficients Table
    table_rows = []
    for i, name in enumerate(feature_cols):
        b = pred_betas[i]
        se = pred_bse[i]
        z_val = pred_z[i]
        p_val = pred_pvals[i]
        or_val = np.exp(b)
        ci_low = np.exp(b - z_crit * se)
        ci_upp = np.exp(b + z_crit * se)

        sig_star = ""
        if p_val < 0.001:
            sig_star = " ***"
        elif p_val < 0.01:
            sig_star = " **"
        elif p_val < 0.05:
            sig_star = " *"

        sig_class = "stat-sig" if p_val < alpha else ""

        table_rows.append(
            f"<tr>"
            f"<td><b>{name}</b></td>"
            f"<td>{b:.4f}</td>"
            f"<td>{se:.4f}</td>"
            f"<td>{z_val:.3f}</td>"
            f"<td class='{sig_class}'>{_format_p_value(p_val)}{sig_star}</td>"
            f"<td><b>{or_val:.4f}</b></td>"
            f"<td><small>[{ci_low:.4f}, {ci_upp:.4f}]</small></td>"
            f"</tr>"
        )

    # Cutoffs Table
    cutoff_rows = []
    for j in range(len(cutoff_params)):
        c_label = f"Threshold {sorted_classes[j]} | {sorted_classes[j+1]}" if j+1 < len(sorted_classes) else f"Cutoff {j+1}"
        cutoff_rows.append(
            f"<tr>"
            f"<td>{c_label}</td>"
            f"<td>{cutoff_params[j]:.4f}</td>"
            f"<td>{cutoff_bse[j]:.4f}</td>"
            f"<td>{_format_p_value(cutoff_pvals[j])}</td>"
            f"</tr>"
        )

    # Class Counts breakdown
    class_counts = pd.Series(y).value_counts().sort_index()
    counts_str = ", ".join([f"{sorted_classes[idx]}: {cnt}" for idx, cnt in class_counts.items()])

    # Predictions and Accuracy
    pred_probs = res.predict(X_df.values)
    pred_labels = np.argmax(pred_probs, axis=1)
    acc = np.mean(pred_labels == y) * 100.0

    # Diagnostic Plot: Category Probability Profiles
    plot_filename = "ordinal_lr_plot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)

    plt.figure(figsize=(7.5, 4.8))
    first_pred = feature_cols[0]
    sort_idx = np.argsort(X_df[first_pred].values)
    sorted_x = X_df[first_pred].values[sort_idx]
    sorted_probs = pred_probs[sort_idx]

    colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6']
    for k in range(num_classes):
        col = colors[k % len(colors)]
        plt.plot(sorted_x, sorted_probs[:, k], label=f'P({target_col} = {sorted_classes[k]})', color=col, lw=2.2)

    plt.xlabel(f"Predictor: {first_pred}", fontsize=11, fontweight='600')
    plt.ylabel("Predicted Class Probability", fontsize=11, fontweight='600')
    plt.title(f"Ordinal Proportional Odds Predictions across {first_pred}", fontsize=12, fontweight='700')
    plt.legend(frameon=True, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    result_html = f"""
    <div class="analysis-section">
        <h3 style="color: #065f46; margin-bottom: 0.5rem;">Ordinal Logistic Regression Results</h3>
        <p style="color: #64748b; margin-top: 0; font-size: 0.9rem;">
            Dependent Variable: <b>{target_col}</b> (Ordered Categories: {' &lt; '.join([str(c) for c in sorted_classes])}) | Total N = {n}
        </p>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; margin-bottom: 1.25rem;">
            <b>Model Fit Summary:</b><br>
            • Category Distribution: {counts_str}<br>
            • Log-Likelihood: {ll:.3f}<br>
            • AIC: {aic:.2f} | BIC: {bic:.2f}<br>
            • Model Classification Accuracy: <b>{acc:.1f}%</b>
        </div>

        <b>Predictor Effects & Proportional Odds Ratios ({conf_int:.0f}% CI):</b>
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Coefficient (β)</th>
                    <th>Std. Error</th>
                    <th>z-statistic</th>
                    <th>p-value</th>
                    <th>Odds Ratio (eᵝ)</th>
                    <th>{conf_int:.0f}% CI for OR</th>
                </tr>
            </thead>
            <tbody>
                {''.join(table_rows)}
            </tbody>
        </table>
        <small style="color: #64748b;">Significance codes: *** p &lt; 0.001, ** p &lt; 0.01, * p &lt; 0.05</small><br><br>

        <b>Model Threshold Parameters (Cutoffs):</b>
        <table class="stats-table" style="max-width: 500px;">
            <thead>
                <tr>
                    <th>Threshold</th>
                    <th>Intercept / Cutoff</th>
                    <th>Std. Error</th>
                    <th>p-value</th>
                </tr>
            </thead>
            <tbody>
                {''.join(cutoff_rows)}
            </tbody>
        </table>

        <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 0.85rem 1rem; border-radius: 4px;">
            <b>Proportional Odds Interpretation:</b><br>
            Under the proportional odds assumption, each predictor's Odds Ratio reflects the change in odds of attaining a higher outcome category versus all lower categories combined for a 1-unit increase in that predictor.
        </div>
    </div>
    """

    return {
        "result_text": result_html,
        "boxplot_url": f"questions/{plot_filename}",
    }


# ==============================================================================
# 3. MULTINOMIAL LOGISTIC REGRESSION
# ==============================================================================
def run_multinomial_lr(request):
    """
    Fits Multinomial Logistic Regression for nominal, unordered categorical outcomes:
    ln(P(Y = k) / P(Y = ref)) = beta_0k + beta_1k*X_1 + ...
    Reports Relative Risk Ratios (RRR), SEs, p-values, McFadden's Pseudo R^2,
    LLR Chi-Square test, AIC/BIC, classification matrix, and probability visualization.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    alpha = (100.0 - conf_int) / 100.0
    z_crit = stats.norm.ppf(1.0 - alpha / 2.0)

    try:
        X_df, y_raw, feature_cols, target_col = parse_logistic_data(request, 'data_file', 'multinomial_lr.dat')
    except Exception as e:
        return {"result_text": f"<div class='alert alert-danger'>Data Loading Error: {str(e)}</div>"}

    unique_classes = pd.Series(y_raw).unique()
    if len(unique_classes) < 3:
        return {
            "result_text": (
                f"<div class='alert alert-danger'><b>Multinomial Regression Warning:</b> "
                f"Multinomial regression is designed for nominal outcomes with >2 categories. "
                f"Found {len(unique_classes)} categories: {list(unique_classes)}. "
                f"For 2 categories, consider Binomial Logistic Regression.</div>"
            )
        }

    sorted_classes = sorted(unique_classes, key=lambda x: (str(x).isdigit(), str(x)))
    class_map = {cls_name: i for i, cls_name in enumerate(sorted_classes)}
    y = y_raw.map(class_map).astype(int).values
    n = len(y)
    num_classes = len(sorted_classes)
    ref_class = sorted_classes[0]

    X_const = sm.add_constant(X_df.values, has_constant='add')
    var_names = ['Constant'] + list(feature_cols)

    try:
        mod = sm.MNLogit(y, X_const)
        res = mod.fit(disp=False, maxiter=200)
    except Exception as e:
        return {"result_text": f"<div class='alert alert-danger'>Multinomial Model Fitting Error: {str(e)}</div>"}

    pseudo_r2 = res.prsquared
    ll = res.llf
    ll_null = res.llnull
    llr_chi2 = 2.0 * (ll - ll_null)
    df_model = (num_classes - 1) * len(feature_cols)
    llr_pvalue = 1.0 - stats.chi2.cdf(llr_chi2, df_model) if df_model > 0 else 1.0
    aic = res.aic
    bic = res.bic

    pred_probs = res.predict(X_const)
    pred_labels = np.argmax(pred_probs, axis=1)
    acc = np.mean(pred_labels == y) * 100.0

    # Build Comparison Tables for Each Non-Reference Category vs Reference
    tables_html = []
    # res.params is shape (n_vars, n_categories - 1)
    for k in range(num_classes - 1):
        target_category_name = sorted_classes[k + 1]
        k_params = res.params[:, k]
        k_bse = res.bse[:, k]
        k_pvals = res.pvalues[:, k]
        k_z = res.tvalues[:, k]

        rows = []
        for i, var_name in enumerate(var_names):
            coef = k_params[i]
            se = k_bse[i]
            z_val = k_z[i]
            p_val = k_pvals[i]
            rrr = np.exp(coef)
            ci_low = np.exp(coef - z_crit * se)
            ci_upp = np.exp(coef + z_crit * se)

            sig_star = ""
            if p_val < 0.001:
                sig_star = " ***"
            elif p_val < 0.01:
                sig_star = " **"
            elif p_val < 0.05:
                sig_star = " *"

            sig_class = "stat-sig" if p_val < alpha else ""
            rrr_str = f"{rrr:.4f}" if i > 0 else "—"
            ci_str = f"[{ci_low:.4f}, {ci_upp:.4f}]" if i > 0 else "—"

            rows.append(
                f"<tr>"
                f"<td><b>{var_name}</b></td>"
                f"<td>{coef:.4f}</td>"
                f"<td>{se:.4f}</td>"
                f"<td>{z_val:.3f}</td>"
                f"<td class='{sig_class}'>{_format_p_value(p_val)}{sig_star}</td>"
                f"<td><b>{rrr_str}</b></td>"
                f"<td><small>{ci_str}</small></td>"
                f"</tr>"
            )

        tables_html.append(f"""
        <h4 style="color: #1e293b; margin-top: 1.25rem; margin-bottom: 0.4rem;">
            Outcome Category: <span style="color: #059669;">{target_category_name}</span> vs. Reference (<span style="color: #64748b;">{ref_class}</span>)
        </h4>
        <table class="stats-table">
            <thead>
                <tr>
                    <th>Variable</th>
                    <th>Coefficient (β)</th>
                    <th>Std. Error</th>
                    <th>z-statistic</th>
                    <th>p-value</th>
                    <th>Relative Risk Ratio (RRR)</th>
                    <th>{conf_int:.0f}% CI for RRR</th>
                </tr>
            </thead>
            <tbody>
                {''.join(rows)}
            </tbody>
        </table>
        """)

    # Diagnostic Visualization: Predicted Probability Comparison
    plot_filename = "multinomial_lr_plot.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)

    plt.figure(figsize=(7.5, 4.8))
    first_pred = feature_cols[0]
    sort_idx = np.argsort(X_df[first_pred].values)
    sorted_x = X_df[first_pred].values[sort_idx]
    sorted_probs = pred_probs[sort_idx]

    colors = ['#64748b', '#059669', '#3b82f6', '#f59e0b', '#ec4899']
    for k in range(num_classes):
        plt.plot(sorted_x, sorted_probs[:, k], label=f'P({target_col} = {sorted_classes[k]})', color=colors[k % len(colors)], lw=2.2)

    plt.xlabel(f"Predictor: {first_pred}", fontsize=11, fontweight='600')
    plt.ylabel("Predicted Probability", fontsize=11, fontweight='600')
    plt.title(f"Multinomial Response Curves across {first_pred}", fontsize=12, fontweight='700')
    plt.legend(frameon=True, loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    plt.close()

    result_html = f"""
    <div class="analysis-section">
        <h3 style="color: #065f46; margin-bottom: 0.5rem;">Multinomial Logistic Regression Results</h3>
        <p style="color: #64748b; margin-top: 0; font-size: 0.9rem;">
            Dependent Variable: <b>{target_col}</b> | Reference Category: <b>{ref_class}</b> | Total N = {n}
        </p>

        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 6px; padding: 1rem; margin-bottom: 1.25rem;">
            <b>Model Fit Summary:</b><br>
            • McFadden's Pseudo R²: <b>{pseudo_r2:.4f}</b><br>
            • Likelihood Ratio χ²: <b>{llr_chi2:.4f}</b> (df = {df_model}, p = <b>{_format_p_value(llr_pvalue)}</b>)<br>
            • Log-Likelihood: {ll:.3f} (Null Log-Likelihood: {ll_null:.3f})<br>
            • AIC: {aic:.2f} | BIC: {bic:.2f}<br>
            • Overall Classification Accuracy: <b>{acc:.1f}%</b>
        </div>

        {''.join(tables_html)}
        <small style="color: #64748b;">Significance codes: *** p &lt; 0.001, ** p &lt; 0.01, * p &lt; 0.05</small><br><br>

        <div style="background: #f0fdf4; border-left: 4px solid #10b981; padding: 0.85rem 1rem; border-radius: 4px;">
            <b>Relative Risk Ratio (RRR) Interpretation:</b><br>
            An RRR indicates the relative likelihood of a subject belonging to that specific category compared to the baseline reference category (<b>{ref_class}</b>) for each 1-unit increase in the predictor.
        </div>
    </div>
    """

    return {
        "result_text": result_html,
        "boxplot_url": f"questions/{plot_filename}",
    }
