import os
import ast
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit, minimize
from .utils import load_file_data, STATIC_PLOT_DIR, format_num


# ==============================================================================
# SIMPLE LINEAR REGRESSION
# ==============================================================================
def run_simple_linear_regression(request, dat=None):
    """
    Fits y = beta0 + beta1 * x, computes R^2, slope/intercept CIs, trend test,
    and generates regression scatter with confidence bands.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    if dat is None:
        dat = load_file_data(request, 'data_file', 'linear.dat')

    if len(dat) == 0 or dat.shape[1] < 2:
        return {"result_text": "Error: Linear regression requires x and y columns."}

    x, y = dat[:, 0], dat[:, 1]
    n = len(x)
    x_bar, y_bar = np.mean(x), np.mean(y)

    ssx = np.sum((x - x_bar)**2)
    num = np.sum((y - y_bar) * (x - x_bar))
    beta1 = num / ssx if ssx > 0 else 0.0
    beta0 = y_bar - beta1 * x_bar
    regr_line = beta0 + beta1 * x

    sse = np.sum((y - regr_line)**2)
    sst = np.sum((y - y_bar)**2)
    r_square = 1.0 - (sse / sst) if sst > 0 else 0.0

    s_y = np.sqrt(sse / max(n - 2, 1))
    s_beta0 = s_y * np.sqrt(1.0 / n + (x_bar**2 / ssx)) if ssx > 0 else 0.0
    s_beta1 = s_y * np.sqrt(1.0 / ssx) if ssx > 0 else 0.0

    alpha = (100.0 - conf_int) / 100.0
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, max(n - 2, 1))

    low_slope = beta1 - t_crit * s_beta1
    upp_slope = beta1 + t_crit * s_beta1
    low_intercept = beta0 - t_crit * s_beta0
    upp_intercept = beta0 + t_crit * s_beta0

    t_stat = beta1 / s_beta1 if s_beta1 > 0 else 0.0
    p_value = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stat), max(n - 2, 1)))

    trend_msg = (
        f"Significant {'positive' if beta1 > 0 else 'negative'} linear trend detected (p = {format_num(p_value)})."
        if np.abs(t_stat) > t_crit else
        f"No statistically significant trend detected (p = {format_num(p_value)})."
    )

    sorted_idx = np.argsort(x)
    sorted_x = x[sorted_idx]
    s_yh = s_y * np.sqrt(1.0 / n + (sorted_x - x_bar)**2 / max(ssx, 1e-12))
    y_h = beta0 + beta1 * sorted_x
    low_conf = y_h - t_crit * s_yh
    upp_conf = y_h + t_crit * s_yh

    plot_filename = "simple_linear_regres.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(x, y, 'ko', label='Observed Data', markersize=5)
    plt.plot(sorted_x, y_h, 'r-', label=f'Fit: y = {beta0:.2f} + {beta1:.2f}x', linewidth=2)
    plt.plot(sorted_x, upp_conf, 'g--', label=f'{conf_int:.0f}% CI Band')
    plt.plot(sorted_x, low_conf, 'g--')
    plt.fill_between(sorted_x, low_conf, upp_conf, color='green', alpha=0.1)
    plt.title('Simple Linear Regression & Confidence Bands')
    plt.xlabel('Predictor Variable (x)')
    plt.ylabel('Outcome Variable (y)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Simple Linear Regression Model:</b><br>"
            f"• <b>Regression Equation:</b> ŷ = {format_num(beta0)} + {format_num(beta1)}x<br>"
            f"• Coefficient of Determination (R²) = {format_num(r_square)}<br>"
            f"• Slope (β₁): {format_num(beta1)} [{conf_int:.1f}% CI: {format_num(low_slope)}, {format_num(upp_slope)}]<br>"
            f"• Intercept (β₀): {format_num(beta0)} [{conf_int:.1f}% CI: {format_num(low_intercept)}, {format_num(upp_intercept)}]<br>"
            f"• Sum of Squared Errors (SSE): {format_num(sse)}<br>"
            f"• Residual Standard Error (s_y): {format_num(s_y)}<br>"
            f"• <b>Trend Hypothesis Test:</b> {trend_msg}"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }


# ==============================================================================
# MULTIPLE LINEAR REGRESSION
# ==============================================================================
def run_multiple_linear_regression(request, dat=None):
    """
    Fits Multiple Linear Regression on multi-predictor datasets:
    y = beta0 + beta1 * x1 + beta2 * x2 + ... + betak * xk.
    Computes R^2, Adjusted R^2, F-test, individual coefficients CIs & p-values,
    and generates an Observed vs. Predicted diagnostic plot.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    if dat is None:
        dat = load_file_data(request, 'data_file', 'multi_single.dat')

    if request.POST.get('regression_mode') == 'force_nonlinear':
        res = run_custom_nonlinear_regression(request, dat)
        notice = (
            f"<div style='background-color:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; padding:12px 16px; border-radius:10px; margin-bottom:14px;'>"
            f"<b>ℹ Direct Multivariable Non-Linear Modeling Mode:</b> Linear regression bypassed by user selection. "
            f"Fitting specified non-linear model directly to the multivariable dataset.</div>"
        )
        res["result_text"] = notice + res.get("result_text", "")
        return res

    if len(dat) == 0 or dat.shape[1] < 2:
        return {"result_text": "Error: Multiple regression requires outcome y (column 1) and predictor columns (column 2+)."}

    y = dat[:, 0]
    X = dat[:, 1:]
    m, n_x = X.shape

    if m <= n_x + 1:
        return {"result_text": f"Error: Number of observations ({m}) must exceed number of parameters ({n_x + 1})."}

    B = np.column_stack([np.ones(m), X])
    try:
        inv_XtX = np.linalg.inv(B.T @ B)
        betas = inv_XtX @ B.T @ y
    except np.linalg.LinAlgError:
        betas = np.linalg.lstsq(B, y, rcond=None)[0]
        inv_XtX = np.linalg.pinv(B.T @ B)

    y_pred = B @ betas
    sse = np.sum((y - y_pred)**2)
    sst = np.sum((y - np.mean(y))**2)
    r_square = 1.0 - (sse / sst) if sst > 0 else 0.0

    df_model = n_x
    df_resid = max(m - (n_x + 1), 1)
    s_y_sq = sse / df_resid
    s_y = np.sqrt(s_y_sq)

    r_adj = 1.0 - ((sse / df_resid) / (sst / (m - 1))) if (m > 1 and sst > 0) else 0.0
    ms_model = (sst - sse) / df_model if df_model > 0 else 0.0
    ms_resid = s_y_sq
    f_stat = ms_model / ms_resid if ms_resid > 0 else 0.0
    f_pval = 1.0 - stats.f.cdf(f_stat, df_model, df_resid)

    cov_betas = s_y_sq * inv_XtX
    se_betas = np.sqrt(np.maximum(np.diag(cov_betas), 1e-14))

    alpha = (100.0 - conf_int) / 100.0
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df_resid)
    t_stats = betas / se_betas
    p_vals = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df_resid))

    # Construct equation string
    eqn_terms = [format_num(betas[0])]
    for i in range(1, n_x + 1):
        sgn = "+" if betas[i] >= 0 else "-"
        eqn_terms.append(f"{sgn} {format_num(abs(betas[i]))}x_{i}")
    eqn = "ŷ = " + " ".join(eqn_terms)

    # Table of coefficients
    rows_html = []
    # Intercept
    low_b0 = betas[0] - t_crit * se_betas[0]
    upp_b0 = betas[0] + t_crit * se_betas[0]
    rows_html.append(
        f"<tr>"
        f"<td style='padding:6px 10px; font-weight:600;'>Intercept (β₀)</td>"
        f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(betas[0])}</td>"
        f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(se_betas[0])}</td>"
        f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(t_stats[0], 3)}</td>"
        f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(p_vals[0])}</td>"
        f"<td style='padding:6px 10px; font-family:monospace;'>[{format_num(low_b0)}, {format_num(upp_b0)}]</td>"
        f"</tr>"
    )
    # Predictors
    for i in range(1, n_x + 1):
        low_bi = betas[i] - t_crit * se_betas[i]
        upp_bi = betas[i] + t_crit * se_betas[i]
        rows_html.append(
            f"<tr>"
            f"<td style='padding:6px 10px; font-weight:600;'>Predictor x_{i} (β_{i})</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(betas[i])}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(se_betas[i])}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(t_stats[i], 3)}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(p_vals[i])}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>[{format_num(low_bi)}, {format_num(upp_bi)}]</td>"
            f"</tr>"
        )
    table_body = "\n".join(rows_html)

    # Diagnostic Plot: Observed vs Predicted
    plot_filename = "multiple_linear_regres.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6.5, 4.5))
    plt.scatter(y, y_pred, color='#2563eb', alpha=0.75, edgecolors='k', s=45, label='Observations')
    min_val = min(np.min(y), np.min(y_pred))
    max_val = max(np.max(y), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Identity Line (y = ŷ)')
    plt.title(f'Multiple Linear Regression: Observed vs Predicted (R² = {format_num(r_square)})')
    plt.xlabel('Observed Outcome (y)')
    plt.ylabel('Fitted / Predicted Outcome (ŷ)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    result_text = (
        f"<div style='margin-bottom:12px;'>"
        f"<h3 style='margin:0 0 6px 0; font-size:1.15rem; color:#0f172a;'>Multiple Linear Regression Results</h3>"
        f"• <b>Regression Equation:</b> <code style='font-size:0.95rem; font-weight:700;'>{eqn}</code><br>"
        f"• <b>Number of Predictors (k):</b> {n_x} | <b>Observations (n):</b> {m}<br>"
        f"• <b>Coefficient of Determination (R²):</b> {format_num(r_square)} (Adjusted R²: {format_num(r_adj)})<br>"
        f"• <b>Overall Model F-test:</b> F({df_model}, {df_resid}) = {format_num(f_stat, 3)}, p = {format_num(f_pval)}<br>"
        f"• <b>Residual Standard Error (s_y):</b> {format_num(s_y)} (Residual Variance: {format_num(s_y_sq)})<br>"
        f"• <b>Sum of Squared Errors (SSE):</b> {format_num(sse)}"
        f"</div>"
        f"<div style='overflow-x:auto; margin-top:12px; border:1px solid #e2e8f0; border-radius:8px;'>"
        f"<table style='width:100%; border-collapse:collapse; text-align:left; font-size:0.875rem;'>"
        f"<thead style='background-color:#f1f5f9; border-bottom:1px solid #cbd5e1;'>"
        f"<tr>"
        f"<th style='padding:8px 10px;'>Coefficient</th>"
        f"<th style='padding:8px 10px;'>Estimate</th>"
        f"<th style='padding:8px 10px;'>Std. Error</th>"
        f"<th style='padding:8px 10px;'>t-stat</th>"
        f"<th style='padding:8px 10px;'>p-value</th>"
        f"<th style='padding:8px 10px;'>{conf_int:.0f}% Conf. Interval</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>"
        f"{table_body}"
        f"</tbody>"
        f"</table>"
        f"</div>"
    )

    cached_data_json = json.dumps(dat.tolist())

    return {
        "result_text": result_text,
        "boxplot_url": f"questions/{plot_filename}",
    }


# ==============================================================================
# GENERAL LINEAR REGRESSION (POLYNOMIAL & MULTIPLE)
# ==============================================================================
def run_general_linear_regression(request):
    """
    Fits polynomial or multiple linear regression via normal equations.
    Maintains compatibility with existing tests and workflows.
    """
    try:
        regression_type = request.POST.get('regression_type', 'multiple')
        deg = int(request.POST.get('deg', 2))
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        regression_type, deg, conf_int = 'multiple', 2, 95.0

    if regression_type == 'polynomial':
        poly = load_file_data(request, 'data_file', 'poly.dat')
        if len(poly) == 0 or poly.shape[1] < 2:
            return {"result_text": "Error: Polynomial regression requires x and y columns."}

        x, y = poly[:, 0], poly[:, 1]
        n = len(x)
        A = np.zeros((n, deg + 1))
        for i in range(n):
            for j in range(deg + 1):
                A[i, j] = x[i]**j

        K_matrix = np.linalg.inv(A.T @ A) @ A.T
        betas = K_matrix @ y
        sse = np.sum((y - A @ betas)**2)
        s_y_sq = sse / max(n - (deg + 1), 1)

        terms = [f"{betas[0]:.3f}"]
        for i in range(1, deg + 1):
            terms.append(f"{betas[i]:+.3f}x^{i}" if i > 1 else f"{betas[i]:+.3f}x")
        eqn_str = "ŷ = " + " ".join(terms)

        x_plot = np.linspace(min(x), max(x), 200)
        y_fit = sum(betas[i] * (x_plot**i) for i in range(deg + 1))

        plot_filename = "general_linear_regres.png"
        plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(x, y, 'ko', label='Data Points')
        plt.plot(x_plot, y_fit, 'r-', label=f'Degree {deg} Fit', linewidth=2)
        plt.title('Polynomial Regression Fit')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        return {
            "result_text": (
                f"<b>Polynomial Regression (Degree {deg}):</b><br>"
                f"• <b>Fitted Equation:</b> {eqn_str}<br>"
                f"• Residual Variance (s_y²) = {s_y_sq:.4f}<br>"
                f"• Sum of Squared Errors (SSE) = {sse:.4f}"
            ),
            "boxplot_url": f"questions/{plot_filename}",
        }

    else:
        return run_multiple_linear_regression(request)


# ==============================================================================
# AST SAFE EQUATION COMPILER & PARSER (SINGLE & MULTI-PREDICTOR)
# ==============================================================================
ALLOWED_MATH_FUNCS = {
    'exp': np.exp,
    'log': np.log,
    'ln': np.log,
    'log10': np.log10,
    'sqrt': np.sqrt,
    'sin': np.sin,
    'cos': np.cos,
    'tan': np.tan,
    'abs': np.abs,
    'sinh': np.sinh,
    'cosh': np.cosh,
}


def compile_equation(expr_str):
    """
    Safely compiles a mathematical formula into a callable function using AST validation.
    Predictor variables: 'x' (single predictor) or 'x1', 'x2', 'x3', etc. (multiple predictors).
    Parameters: any other identifiers (e.g. 'a', 'b', 'c', 'beta0', 'km', etc.).
    Returns (model_func, param_names, pred_names, clean_formula_str).
    """
    clean = expr_str.strip()
    if '=' in clean:
        clean = clean.split('=', 1)[1].strip()
    clean = clean.replace('^', '**')

    try:
        tree = ast.parse(clean, mode='eval')
    except SyntaxError as e:
        raise ValueError(f"Equation syntax error: {e.msg} at position {e.offset}.")

    allowed_names = set(ALLOWED_MATH_FUNCS.keys()) | {'pi', 'e', 'np'}
    param_nodes = []
    pred_nodes = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            name_lower = node.id.lower()
            if name_lower == 'x' or re.match(r'^x_?[0-9]+$', name_lower):
                pred_nodes.append(node)
            elif node.id in allowed_names:
                pass
            else:
                param_nodes.append(node)
        elif isinstance(node, (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call,
                               ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow,
                               ast.USub, ast.UAdd, ast.Load)):
            pass
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            pass
        else:
            raise ValueError(f"Unsupported syntax construct: '{type(node).__name__}'. Only standard math operations and functions are allowed.")

    if not pred_nodes:
        raise ValueError("The equation must contain at least one predictor variable (e.g., 'x' or 'x1', 'x2').")

    # Order parameters by first appearance in the equation
    param_nodes.sort(key=lambda n: getattr(n, 'col_offset', 0))
    params = list(dict.fromkeys([n.id for n in param_nodes]))

    # Order predictor names
    pred_names = sorted(list(dict.fromkeys([n.id for n in pred_nodes])))

    if not params:
        raise ValueError("The equation must contain at least one parameter to estimate (e.g., 'a', 'b').")

    compiled = compile(tree, '<custom_equation>', 'eval')
    math_env = dict(ALLOWED_MATH_FUNCS)
    math_env.update({'pi': np.pi, 'e': np.e, 'np': np})

    def model_func(x_data, *args):
        env = dict(math_env)
        # Handle 1D array or 2D matrix of predictors
        if isinstance(x_data, (tuple, list)):
            for idx, p_col in enumerate(x_data):
                if idx < len(pred_names):
                    env[pred_names[idx]] = np.asarray(p_col, dtype=float)
        elif getattr(x_data, 'ndim', 1) == 1:
            for p_name in pred_names:
                env[p_name] = np.asarray(x_data, dtype=float)
        else:
            for idx, p_name in enumerate(pred_names):
                col_idx = idx if idx < x_data.shape[1] else 0
                env[p_name] = np.asarray(x_data[:, col_idx], dtype=float)

        for p_name, val in zip(params, args):
            env[p_name] = float(val)
        return eval(compiled, {'__builtins__': {}}, env)

    return model_func, params, pred_names, clean


def parse_initial_guesses(guesses_str, params):
    """
    Parses comma- or semicolon-separated initial guesses (e.g. 'a=1.5, b=0.2' or '1.5, 0.2').
    """
    guesses = {}
    if guesses_str:
        parts = [p.strip() for p in guesses_str.replace(';', ',').split(',') if p.strip()]
        for idx, part in enumerate(parts):
            if '=' in part:
                k, v = part.split('=', 1)
                try:
                    guesses[k.strip()] = float(v.strip())
                except ValueError:
                    pass
            else:
                try:
                    if idx < len(params):
                        guesses[params[idx]] = float(part)
                except ValueError:
                    pass

    return [guesses.get(p, 1.0) for p in params]


def format_substituted_equation(clean_formula, params, popt):
    """
    Substitutes fitted parameter values into formula for clear user display.
    """
    sub_formula = clean_formula
    sorted_pairs = sorted(zip(params, popt), key=lambda x: len(x[0]), reverse=True)
    for p, val in sorted_pairs:
        sub_formula = re.sub(r'\b' + re.escape(p) + r'\b', f"({format_num(val)})", sub_formula)
    return sub_formula


def render_equation_input_form(cached_data_json, current_eq="a * x**2 + b * x + c", current_guesses="", error_msg="", title="📐 Further Assessment / User Selection of Model", description="", is_multivariable=False):
    """
    Generates the interactive HTML model selection and equation input card.
    Supports presets for single predictor (x) or multiple predictors (x1, x2, ...).
    """
    err_banner = ""
    if error_msg:
        err_banner = (
            f"<div style='background-color:#fef2f2; border:1px solid #fecaca; color:#991b1b; padding:10px 14px; border-radius:8px; margin-bottom:12px; font-size:0.9rem;'>"
            f"<b>Fit Error:</b> {error_msg}</div>"
        )

    desc_text = description if description else (
        f"Choose a candidate {'multivariable' if is_multivariable else ''} model from the presets below or enter any custom mathematical equation "
        f"in terms of <code>{'x1, x2, ...' if is_multivariable else 'x'}</code> and parameters (e.g. <code>a</code>, <code>b</code>, <code>c</code>)."
    )

    if is_multivariable:
        preset_options = """
            <option value="custom">Custom Multivariable Equation (Type below)</option>
            <option value="a * (x1**b) * (x2**c)" selected>Cobb-Douglas Power: y = a · x1^b · x2^c</option>
            <option value="a * x1**2 + b * x2**2 + c * x1 * x2 + d">Polynomial with Interaction: y = a·x1² + b·x2² + c·x1·x2 + d</option>
            <option value="a * exp(b * x1) + c * exp(d * x2)">Multi-Exponential: y = a·exp(b·x1) + c·exp(d·x2)</option>
            <option value="(a * x1) / (b + x1) + (c * x2) / (d + x2)">Multi-Rational Saturation: y = a·x1/(b+x1) + c·x2/(d+x2)</option>
        """
        hint_text = "Use <code>x1, x2, ...</code> as predictor variables. Any other letters are parameters to fit."
        default_val = current_eq if current_eq else "a * (x1**b) * (x2**c)"
    else:
        preset_options = """
            <option value="custom">Custom Equation (Type your own below)</option>
            <option value="a * x**2 + b * x + c" selected>Quadratic Polynomial: y = a · x² + b · x + c</option>
            <option value="a * exp(b * x)">Exponential Growth/Decay: y = a · exp(b · x)</option>
            <option value="a * x**b">Power Law: y = a · x^b</option>
            <option value="a + b * log(x)">Logarithmic: y = a + b · ln(x)</option>
            <option value="a * x / (b + x)">Michaelis-Menten / Rational: y = a · x / (b + x)</option>
            <option value="100 / (1 + (beta / x)**2)">Hill Equation (Sigmoidal): y = 100 / (1 + (beta/x)²)</option>
            <option value="a / (1 + exp(-b * (x - c)))">Logistic Growth: y = a / (1 + exp(-b(x - c)))</option>
        """
        hint_text = "Independent predictor variable must be <code>x</code> (or <code>x1</code>). Supported functions: <code>exp</code>, <code>log</code>, <code>sqrt</code>, <code>sin</code>, <code>cos</code>, <code>tan</code>, <code>^</code> or <code>**</code>."
        default_val = current_eq if current_eq else "a * x**2 + b * x + c"

    return f"""
    <div style="margin-top: 1.5rem; background: #ffffff; border: 1.5px solid #cbd5e1; border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);">
        <h3 style="font-size: 1.15rem; font-weight: 700; color: #0f172a; margin: 0 0 0.5rem 0; display: flex; align-items: center; gap: 0.5rem;">
            <span>{title[:2] if title and title[0] in '📐💡' else '📐'}</span> {title[2:].strip() if title and title[0] in '📐💡' else title}
        </h3>
        <p style="font-size: 0.9rem; color: #475569; margin: 0 0 1rem 0; line-height: 1.5;">
            {desc_text}
        </p>

        {err_banner}

        <form method="POST" action="" enctype="multipart/form-data" style="display: flex; flex-direction: column; gap: 1rem;">
            <input type="hidden" name="run_action" value="true">
            <input type="hidden" name="action_name" value="run_custom_nonlinear_regression">
            <input type="hidden" name="cached_data_json" value='{cached_data_json}'>

            <div>
                <label style="display: block; font-size: 0.875rem; font-weight: 600; color: #1e293b; margin-bottom: 0.35rem;">
                    Candidate Model Presets:
                </label>
                <select id="preset_model_select" onchange="var inp=document.getElementById('custom_equation'); if(this.value!=='custom'){{inp.value=this.value;}}"
                        style="width: 100%; padding: 0.65rem 0.75rem; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 0.9rem; background: #f8fafc; color: #0f172a;">
                    {preset_options}
                </select>
            </div>

            <div>
                <label for="custom_equation" style="display: block; font-size: 0.875rem; font-weight: 600; color: #1e293b; margin-bottom: 0.35rem;">
                    Model Equation:
                </label>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span style="font-family: monospace; font-weight: 700; color: #059669; font-size: 1.1rem;">ŷ = </span>
                    <input type="text" id="custom_equation" name="custom_equation" value="{default_val}" required
                           style="flex: 1; padding: 0.65rem 0.75rem; border: 1.5px solid #059669; border-radius: 8px; font-family: monospace; font-size: 0.95rem; background: #ffffff;">
                </div>
                <small style="display: block; font-size: 0.8rem; color: #64748b; margin-top: 0.35rem;">
                    {hint_text}
                </small>
            </div>

            <div>
                <label for="initial_guesses" style="display: block; font-size: 0.875rem; font-weight: 600; color: #1e293b; margin-bottom: 0.35rem;">
                    Initial Parameter Guesses (Optional):
                </label>
                <input type="text" id="initial_guesses" name="initial_guesses" value="{current_guesses}"
                       placeholder="e.g. a=1.0, b=0.5, c=1.0 (or leave empty for automatic starting values)"
                       style="width: 100%; padding: 0.6rem 0.75rem; border: 1px solid #cbd5e1; border-radius: 8px; font-size: 0.9rem; background: #ffffff;">
            </div>

            <div style="display: flex; gap: 0.75rem; align-items: center; margin-top: 0.5rem;">
                <button type="submit" style="background: #059669; color: #ffffff; padding: 0.75rem 1.5rem; border: none; border-radius: 8px; font-weight: 700; font-size: 0.95rem; cursor: pointer; display: inline-flex; align-items: center; gap: 0.5rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05);">
                    <span>⚡ Fit Non-Linear Model</span>
                </button>
                <span style="font-size: 0.85rem; color: #64748b;">Fits model directly to the current dataset</span>
            </div>
        </form>
    </div>
    """


# ==============================================================================
# CUSTOM NON-LINEAR REGRESSION (SINGLE & MULTI-PREDICTOR)
# ==============================================================================
def run_custom_nonlinear_regression(request, dat=None):
    """
    Fits user-specified non-linear equation or candidate model on single- or multi-predictor data.
    Uses AST validation and Levenberg-Marquardt / Nelder-Mead optimization.
    """
    try:
        conf_int = float(request.POST.get('conf_int', 95))
    except (ValueError, TypeError, AttributeError):
        conf_int = 95.0

    # Retrieve dataset: direct param -> uploaded file -> cached JSON -> sample poly.dat
    if dat is None:
        if request.FILES.get('data_file'):
            dat = load_file_data(request, 'data_file', None)
        elif request.POST.get('cached_data_json'):
            try:
                raw_json = request.POST.get('cached_data_json')
                dat = np.array(json.loads(raw_json), dtype=float)
            except Exception:
                dat = load_file_data(request, 'data_file', 'poly.dat')
        else:
            dat = load_file_data(request, 'data_file', 'poly.dat')

    if len(dat) == 0 or dat.shape[1] < 2:
        return {"result_text": "Error: Non-linear regression requires at least 2 columns: outcome y and predictor(s)."}

    n = len(dat)
    cached_data_json = json.dumps(dat.tolist())

    # Detect single vs multi-predictor layout
    # If 2 columns, outcome is col 1 (or col 0), predictor is other
    # By standard convention in this app: col 0 is x, col 1 is y (single predictor) OR col 0 is y, col 1+ is X (multi-predictor)
    eq_str = request.POST.get('custom_equation', '').strip()
    guesses_str = request.POST.get('initial_guesses', '').strip()

    is_multi = (dat.shape[1] > 2)
    if not eq_str:
        eq_str = "a * (x1**b) * (x2**c)" if is_multi else "a * x**2 + b * x + c"

    try:
        model_func, params, pred_names, clean_eq = compile_equation(eq_str)
    except ValueError as e:
        form_html = render_equation_input_form(cached_data_json, eq_str, guesses_str, error_msg=str(e), is_multivariable=is_multi)
        return {
            "result_text": f"<div style='color:#dc2626; font-weight:700; margin-bottom:12px;'>Equation Error: {e}</div>" + form_html
        }

    # Extract x and y based on predictors
    if is_multi:
        y = dat[:, 0]
        X = dat[:, 1:]
    else:
        # Check if 2 columns: col 0 x, col 1 y
        x = dat[:, 0]
        y = dat[:, 1]
        X = x

    p0 = parse_initial_guesses(guesses_str, params)

    # Optimization via curve_fit with fallback
    popt = None
    pcov = None
    fit_error = None

    try:
        if is_multi:
            popt, pcov = curve_fit(lambda X_in, *p: model_func(X_in, *p), X, y, p0=p0, maxfev=10000)
        else:
            popt, pcov = curve_fit(model_func, X, y, p0=p0, maxfev=10000)
    except Exception as e_fit:
        # Fallback to Nelder-Mead on SSE
        def loss(p_vals):
            try:
                preds = model_func(X, *p_vals)
                if np.any(np.isnan(preds)) or np.any(np.isinf(preds)):
                    return 1e12
                return np.sum((y - preds)**2)
            except Exception:
                return 1e12

        res = minimize(loss, p0, method='Nelder-Mead', options={'maxiter': 5000})
        if res.success or res.fun < 1e10:
            popt = res.x
            pcov = None
        else:
            fit_error = str(e_fit)

    if popt is None:
        form_html = render_equation_input_form(cached_data_json, eq_str, guesses_str, error_msg=f"Convergence failed: {fit_error}. Try providing initial parameter guesses.", is_multivariable=is_multi)
        return {
            "result_text": f"<div style='color:#dc2626; font-weight:700; margin-bottom:12px;'>Model fitting did not converge. Please check the equation or supply initial parameter guesses.</div>" + form_html
        }

    y_pred = model_func(X, *popt)
    sse = np.sum((y - y_pred)**2)
    sst = np.sum((y - np.mean(y))**2)
    r_square = 1.0 - (sse / sst) if sst > 0 else 0.0

    p_count = len(params)
    df_resid = max(n - p_count, 1)
    s_y_sq = sse / df_resid
    s_y = np.sqrt(s_y_sq)

    alpha = (100.0 - conf_int) / 100.0
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df_resid)

    # Standard errors and CIs
    se_params = [None] * p_count
    if pcov is not None and not np.any(np.isinf(pcov)) and not np.any(np.isnan(pcov)):
        se_diag = np.sqrt(np.maximum(np.diag(pcov), 0.0))
        se_params = list(se_diag)

    sub_formula = format_substituted_equation(clean_eq, params, popt)

    # Table rows
    table_rows = []
    for idx, p_name in enumerate(params):
        est = popt[idx]
        se = se_params[idx]
        if se is not None and se > 0:
            low_ci = est - t_crit * se
            upp_ci = est + t_crit * se
            se_str = format_num(se)
            ci_str = f"[{format_num(low_ci)}, {format_num(upp_ci)}]"
        else:
            se_str = "N/A"
            ci_str = "N/A"

        table_rows.append(
            f"<tr>"
            f"<td style='padding:6px 10px; font-weight:600; font-family:monospace;'>{p_name}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{format_num(est)}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{se_str}</td>"
            f"<td style='padding:6px 10px; font-family:monospace;'>{ci_str}</td>"
            f"</tr>"
        )
    param_table = "\n".join(table_rows)

    # Plot
    plot_filename = "custom_nonlinear_regres.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6.5, 4.5))

    if not is_multi:
        plt.plot(x, y, 'ko', label='Observed Data', markersize=5)
        x_fine = np.linspace(min(x), max(x), 300)
        try:
            y_fine = model_func(x_fine, *popt)
            plt.plot(x_fine, y_fine, 'r-', label=f'Non-Linear Fit (R² = {format_num(r_square)})', linewidth=2)
        except Exception:
            plt.plot(x[np.argsort(x)], y_pred[np.argsort(x)], 'r-', label=f'Non-Linear Fit (R² = {format_num(r_square)})', linewidth=2)
        plt.xlabel('Predictor Variable (x)')
        plt.ylabel('Outcome Variable (y)')
    else:
        plt.scatter(y, y_pred, color='#059669', edgecolors='k', alpha=0.75, s=45, label='Observations')
        min_v = min(np.min(y), np.min(y_pred))
        max_v = max(np.max(y), np.max(y_pred))
        plt.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=2, label='Identity Line (y = ŷ)')
        plt.xlabel('Observed Outcome (y)')
        plt.ylabel('Model Fitted Outcome (ŷ)')

    plt.title(f"{'Multivariable ' if is_multi else ''}Non-Linear Model Fit (R² = {format_num(r_square)})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    result_header = (
        f"<div style='margin-bottom:14px;'>"
        f"<div style='background-color:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; padding:12px 16px; border-radius:10px; margin-bottom:14px;'>"
        f"<b>✓ Custom {'Multivariable ' if is_multi else ''}Non-Linear Model Successfully Fitted</b></div>"
        f"• <b>Specified Equation:</b> <code style='font-size:0.95rem; font-weight:700;'>ŷ = {clean_eq}</code><br>"
        f"• <b>Fitted Equation with Parameters:</b> <code style='font-size:0.95rem; font-weight:700;'>ŷ = {sub_formula}</code><br>"
        f"• <b>Coefficient of Determination (R²):</b> {format_num(r_square)}<br>"
        f"• <b>Residual Standard Error (s_y):</b> {format_num(s_y)} | <b>Sum of Squared Errors (SSE):</b> {format_num(sse)}<br>"
        f"• <b>Observations (n):</b> {n} | <b>Degrees of Freedom:</b> {df_resid}"
        f"</div>"
        f"<div style='overflow-x:auto; margin-top:12px; margin-bottom:16px; border:1px solid #e2e8f0; border-radius:8px;'>"
        f"<table style='width:100%; border-collapse:collapse; text-align:left; font-size:0.875rem;'>"
        f"<thead style='background-color:#f1f5f9; border-bottom:1px solid #cbd5e1;'>"
        f"<tr>"
        f"<th style='padding:8px 10px;'>Parameter</th>"
        f"<th style='padding:8px 10px;'>Optimized Estimate</th>"
        f"<th style='padding:8px 10px;'>Std. Error</th>"
        f"<th style='padding:8px 10px;'>{conf_int:.0f}% Conf. Interval</th>"
        f"</tr>"
        f"</thead>"
        f"<tbody>"
        f"{param_table}"
        f"</tbody>"
        f"</table>"
        f"</div>"
    )

    refit_form = render_equation_input_form(cached_data_json, eq_str, guesses_str, is_multivariable=is_multi)

    return {
        "result_text": result_header + refit_form,
        "boxplot_url": f"questions/{plot_filename}",
    }


# ==============================================================================
# SINGLE NON-LINEAR REGRESSION (HILL EQUATION COMPATIBILITY)
# ==============================================================================
def run_single_nonlinear_regression(request, dat=None):
    """
    Fits Hill Equation: response = 100 / (1 + (beta / conc)**2)
    using Golden Section Search optimization (from nonlinear.py).
    Preserved for backward compatibility.
    """
    if dat is not None:
        exp_data = dat
    else:
        exp_data = load_file_data(request, 'data_file', 'response_data.dat')

    if len(exp_data) == 0 or exp_data.shape[1] < 2:
        return {"result_text": "Error: Non-linear regression data requires concentration and response columns."}

    conc = exp_data[:, 0]
    response = exp_data[:, 1]

    def obj_fun(beta):
        return np.sum((response - (100.0 / (1.0 + (beta / conc)**2)))**2)

    a = float(np.min(conc) * 0.1)
    b = float(np.max(conc) * 2.0)
    if a <= 0:
        a = 0.001
    r = (np.sqrt(5) - 1.0) / 2.0

    x1 = (1.0 - r) * (b - a) + a
    x2 = r * (b - a) + a
    f_x1 = obj_fun(x1)
    f_x2 = obj_fun(x2)

    tolerance = 1e-10
    for _ in range(10000):
        if f_x1 >= f_x2:
            a = x1
            x1 = x2
            x2 = r * (b - a) + a
            f_x1 = f_x2
            f_x2 = obj_fun(x2)
        else:
            b = x2
            x2 = x1
            x1 = (1.0 - r) * (b - a) + a
            f_x2 = f_x1
            f_x1 = obj_fun(x1)
        if np.abs(b - a) < tolerance:
            break

    opt_beta = (a + b) / 2.0
    min_sse = obj_fun(opt_beta)

    x_plot = np.logspace(np.log10(max(min(conc), 0.01)), np.log10(max(conc)), 200)
    fitted = 100.0 / (1.0 + (opt_beta / x_plot)**2)

    plot_filename = "single_nonlinear_regres.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6.5, 4.5))
    plt.semilogx(conc, response, 'ko', label='Experimental Data', markersize=6)
    plt.semilogx(x_plot, fitted, 'r-', label=f'Hill Model Fit (EC50 β = {opt_beta:.3f})', linewidth=2)
    plt.title('Single Non-Linear Regression (Hill Equation)')
    plt.xlabel('Concentration (log scale)')
    plt.ylabel('Response (%)')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Single Non-Linear Regression (Hill Equation):</b><br>"
            f"• <b>Optimized EC50 Parameter (β):</b> {format_num(opt_beta)}<br>"
            f"• Model Equation: Response = 100 / (1 + ({format_num(opt_beta)} / Concentration)²)<br>"
            f"• Minimized Sum of Squared Errors (SSE): {format_num(min_sse)}<br>"
            f"• Optimization Method: Golden Section Search"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }


# ==============================================================================
# MULTIPLE NON-LINEAR REGRESSION (GAUSSIAN PEAK COMPATIBILITY)
# ==============================================================================
def run_multiple_nonlinear_regression(request):
    """
    Fits multi-parameter Gaussian intensity curve:
    y = (a1 / sqrt(2*pi*sigma^2)) * exp(-(coord - mu)^2 / sigma^2) + a2
    using Nelder-Mead simplex optimization. Preserved for backward compatibility.
    """
    exp_data = load_file_data(request, 'data_file', 'image_analysis_data.dat')
    if len(exp_data) == 0 or exp_data.shape[1] < 2:
        return {"result_text": "Error: Multi-parameter non-linear regression data requires 2 columns."}

    coord = exp_data[:, 0]
    pixel_int = exp_data[:, 1]

    def obj_fun(params):
        a1, a2, mu, sigma = params
        if sigma <= 0:
            return 1e10
        y_model = (a1 / np.sqrt(2.0 * np.pi * sigma**2)) * np.exp(-((coord - mu)**2) / (sigma**2)) + a2
        return np.sum((pixel_int - y_model)**2)

    a2_init = np.min(pixel_int)
    mu_init = coord[np.argmax(pixel_int)]
    sigma_init = np.std(coord) / 2.0
    a1_init = (np.max(pixel_int) - a2_init) * np.sqrt(2.0 * np.pi * sigma_init**2)
    init_guess = [a1_init, a2_init, mu_init, sigma_init]

    res = minimize(obj_fun, init_guess, method='Nelder-Mead', options={'xatol': 1e-8, 'maxiter': 2000})

    a1_opt, a2_opt, mu_opt, sigma_opt = res.x
    sse_opt = res.fun

    coord_fine = np.linspace(min(coord), max(coord), 200)
    fitted_curve = (a1_opt / np.sqrt(2.0 * np.pi * sigma_opt**2)) * np.exp(-((coord_fine - mu_opt)**2) / (sigma_opt**2)) + a2_opt

    plot_filename = "multiple_nonlinear_regres.png"
    plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
    plt.figure(figsize=(6.5, 4.5))
    plt.plot(coord, pixel_int, 'ko', label='Experimental Data', markersize=5)
    plt.plot(coord_fine, fitted_curve, 'r-', label=f'Fitted Gaussian Model', linewidth=2)
    plt.title('Multi-Parameter Non-Linear Model Fit')
    plt.xlabel('Spatial Coordinate')
    plt.ylabel('Signal Intensity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()

    return {
        "result_text": (
            f"<b>Multi-Parameter Non-Linear Regression Results:</b><br>"
            f"• <b>Optimized Model Parameters:</b><br>"
            f"  - Peak Amplitude (a₁): {format_num(a1_opt)}<br>"
            f"  - Baseline Offset (a₂): {format_num(a2_opt)}<br>"
            f"  - Peak Center Location (μ): {format_num(mu_opt)}<br>"
            f"  - Peak Spread (σ): {format_num(sigma_opt)}<br>"
            f"• Minimized SSE: {format_num(sse_opt)}<br>"
            f"• Optimization: Nelder-Mead Simplex Algorithm"
        ),
        "boxplot_url": f"questions/{plot_filename}",
    }


# ==============================================================================
# SINGLE-PREDICTOR LINEARITY & QUADRATIC IMPROVEMENT EVALUATION
# ==============================================================================
def check_singlelinearity(request, dat=None):
    """
    Evaluates whether single-predictor data (x, y) has a linear relationship
    by assessing statistically significant quadratic improvement (nested model F-test).
    - If NO significant quadratic improvement: proceeds with Simple Linear Regression.
    - If SIGNIFICANT quadratic improvement: presents non-linear curvature diagnostics
      and prompts the user for further assessment / model selection / custom equation.
    """
    if dat is None:
        dat = load_file_data(request, 'data_file', 'linear.dat')

    # Check if user explicitly chose direct non-linear regression mode
    if request.POST.get('regression_mode') == 'force_nonlinear':
        res = run_custom_nonlinear_regression(request, dat)
        notice = (
            f"<div style='background-color:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; padding:12px 16px; border-radius:10px; margin-bottom:14px;'>"
            f"<b>ℹ Direct Non-Linear Modeling Mode:</b> Linearity test bypassed by user selection. "
            f"Fitting specified non-linear model directly to the dataset.</div>"
        )
        res["result_text"] = notice + res.get("result_text", "")
        return res

    if len(dat) == 0 or dat.shape[1] < 2:
        return run_simple_linear_regression(request)

    x, y = dat[:, 0], dat[:, 1]
    n = len(x)
    if n < 4:
        return run_simple_linear_regression(request, dat)

    # 1. Linear Fit: y = beta0 + beta1 * x
    p_lin = np.polyfit(x, y, 1)
    y_pred_lin = np.polyval(p_lin, x)
    sse_lin = np.sum((y - y_pred_lin)**2)
    sst = np.sum((y - np.mean(y))**2)
    r2_lin = 1.0 - (sse_lin / sst) if sst > 0 else 0.0

    # 2. Quadratic Fit: y = beta0 + beta1 * x + beta2 * x^2
    p_quad = np.polyfit(x, y, 2)
    y_pred_quad = np.polyval(p_quad, x)
    sse_quad = np.sum((y - y_pred_quad)**2)
    r2_quad = 1.0 - (sse_quad / sst) if sst > 0 else 0.0

    # Nested F-test for quadratic term addition (df1 = 1, df2 = n - 3)
    df_extra = 1
    df_resid_quad = n - 3
    extra_ss = sse_lin - sse_quad
    f_stat = (extra_ss / df_extra) / (sse_quad / df_resid_quad) if sse_quad > 0 else 0.0
    f_crit = stats.f.ppf(0.95, df_extra, df_resid_quad)
    p_val = 1.0 - stats.f.cdf(f_stat, df_extra, df_resid_quad)

    if f_stat <= f_crit:
        # NO SIGNIFICANT QUADRATIC IMPROVEMENT -> Simple Linear Regression
        res = run_simple_linear_regression(request, dat)

        prefix = (
            f"<div style='background-color:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; padding:14px 16px; border-radius:10px; margin-bottom:16px;'>"
            f"<div style='font-weight:700; font-size:1.025rem; margin-bottom:4px;'>"
            f"✓ Linearity Assessment Passed: No Significant Quadratic Improvement"
            f"</div>"
            f"<div style='font-size:0.92rem; line-height:1.5;'>"
            f"Adding a quadratic term does not provide a statistically significant improvement over a linear model "
            f"(F = {format_num(f_stat, 3)} &le; F_crit = {format_num(f_crit, 3)}, p = {format_num(p_val)} &gt; 0.05). "
            f"The relationship between predictor and outcome is adequately linear. Proceeding with <b>Simple Linear Regression</b>."
            f"</div>"
            f"</div>"
        )
        res["result_text"] = prefix + res.get("result_text", "")
        return res
    else:
        # SIGNIFICANT QUADRATIC IMPROVEMENT -> Further assessment / model selection
        cached_data_json = json.dumps(dat.tolist())

        # Generate diagnostic plot comparing linear vs quadratic fits
        plot_filename = "linearity_assessment.png"
        plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
        plt.figure(figsize=(6.5, 4.5))
        plt.plot(x, y, 'ko', label='Observed Data', markersize=5)

        x_sorted = np.linspace(min(x), max(x), 200)
        plt.plot(x_sorted, np.polyval(p_lin, x_sorted), 'r--', label=f'Linear Fit (R² = {format_num(r2_lin, 3)})', linewidth=2)
        plt.plot(x_sorted, np.polyval(p_quad, x_sorted), 'b-', label=f'Quadratic Fit (R² = {format_num(r2_quad, 3)})', linewidth=2)

        plt.title('Relationship Assessment: Linearity vs Quadratic Curvature')
        plt.xlabel('Predictor Variable (x)')
        plt.ylabel('Outcome Variable (y)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        assessment_report = (
            f"<div style='background-color:#fffbeb; border:1px solid #fde68a; color:#92400e; padding:14px 16px; border-radius:10px; margin-bottom:16px;'>"
            f"<div style='font-weight:700; font-size:1.025rem; margin-bottom:4px;'>"
            f"⚠ Non-Linearity Detected: Significant Quadratic Improvement"
            f"</div>"
            f"<div style='font-size:0.92rem; line-height:1.5;'>"
            f"Adding a quadratic term yielded a statistically significant improvement in model fit "
            f"(F = {format_num(f_stat, 3)} &gt; F_crit = {format_num(f_crit, 3)}, p = {format_num(p_val)} &le; 0.05). "
            f"The assumption of linearity is rejected; the relationship exhibits significant curvature. "
            f"Proceeding to <b>Further Assessment & Non-Linear Model Selection</b>."
            f"</div>"
            f"</div>"
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px; margin-bottom:16px;'>"
            f"<h4 style='margin:0 0 8px 0; color:#1e293b; font-size:0.95rem;'>Quadratic Improvement Test Summary:</h4>"
            f"• <b>Linear Model SSE:</b> {format_num(sse_lin)} (R² = {format_num(r2_lin)})<br>"
            f"• <b>Quadratic Model SSE:</b> {format_num(sse_quad)} (R² = {format_num(r2_quad)})<br>"
            f"• <b>Extra Sum of Squares (Reduction in Error):</b> {format_num(extra_ss)}<br>"
            f"• <b>Nested Model F-statistic:</b> F(1, {df_resid_quad}) = {format_num(f_stat, 3)} (p = {format_num(p_val)})"
            f"</div>"
        )

        model_selection_form = render_equation_input_form(cached_data_json, is_multivariable=False)

        return {
            "result_text": assessment_report + model_selection_form,
            "boxplot_url": f"questions/{plot_filename}",
        }


# ==============================================================================
# MULTIVARIABLE LINEARITY & QUADRATIC IMPROVEMENT EVALUATION
# ==============================================================================
def check_multilinearity(request, dat=None):
    """
    Evaluates whether multi-predictor data has a linear relationship across predictors
    via a multivariable nested F-test on quadratic curvature terms (x_1^2, x_2^2, ...).
    - If NO significant quadratic improvement: proceeds with Multiple Linear Regression
      (with optional non-linear override).
    - If SIGNIFICANT quadratic improvement: reports multivariable non-linearity and
      prompts user to select or input a multi-predictor non-linear equation.
    """
    if dat is None:
        dat = load_file_data(request, 'data_file', 'multi_single.dat')

    # Check if user explicitly chose direct non-linear regression mode
    if request.POST.get('regression_mode') == 'force_nonlinear':
        res = run_custom_nonlinear_regression(request, dat)
        notice = (
            f"<div style='background-color:#eff6ff; border:1px solid #bfdbfe; color:#1e40af; padding:12px 16px; border-radius:10px; margin-bottom:14px;'>"
            f"<b>ℹ Direct Multivariable Non-Linear Modeling Mode:</b> Linearity test bypassed by user selection. "
            f"Fitting specified non-linear model directly to the multivariable dataset.</div>"
        )
        res["result_text"] = notice + res.get("result_text", "")
        return res

    if len(dat) == 0 or dat.shape[1] < 2:
        return run_multiple_linear_regression(request, dat)

    y = dat[:, 0]
    X = dat[:, 1:]
    n, k = X.shape

    if n <= 2 * k + 1:
        return run_multiple_linear_regression(request, dat)

    # 1. Linear model: [1, X]
    B_lin = np.column_stack([np.ones(n), X])
    try:
        betas_lin = np.linalg.lstsq(B_lin, y, rcond=None)[0]
        y_pred_lin = B_lin @ betas_lin
        sse_lin = np.sum((y - y_pred_lin)**2)
    except Exception:
        return run_multiple_linear_regression(request, dat)

    # 2. Quadratic curvature model: [1, X, X^2]
    B_quad = np.column_stack([np.ones(n), X, X**2])
    try:
        betas_quad = np.linalg.lstsq(B_quad, y, rcond=None)[0]
        y_pred_quad = B_quad @ betas_quad
        sse_quad = np.sum((y - y_pred_quad)**2)
    except Exception:
        return run_multiple_linear_regression(request, dat)

    sst = np.sum((y - np.mean(y))**2)
    r2_lin = 1.0 - (sse_lin / sst) if sst > 0 else 0.0
    r2_quad = 1.0 - (sse_quad / sst) if sst > 0 else 0.0

    df_extra = k
    df_resid_quad = n - (2 * k + 1)
    extra_ss = sse_lin - sse_quad
    f_stat = (extra_ss / df_extra) / (sse_quad / df_resid_quad) if sse_quad > 0 else 0.0
    f_crit = stats.f.ppf(0.95, df_extra, df_resid_quad)
    p_val = 1.0 - stats.f.cdf(f_stat, df_extra, df_resid_quad)

    if f_stat <= f_crit:
        # Linearity holds
        res = run_multiple_linear_regression(request, dat)
        prefix = (
            f"<div style='background-color:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; padding:14px 16px; border-radius:10px; margin-bottom:16px;'>"
            f"<div style='font-weight:700; font-size:1.025rem; margin-bottom:4px;'>"
            f"✓ Multivariable Linearity Assessment Passed: No Significant Quadratic Improvement"
            f"</div>"
            f"<div style='font-size:0.92rem; line-height:1.5;'>"
            f"Adding quadratic terms across predictors did not yield statistically significant improvement "
            f"(F = {format_num(f_stat, 3)} &le; F_crit = {format_num(f_crit, 3)}, p = {format_num(p_val)} &gt; 0.05). "
            f"Proceeding with <b>Multiple Linear Regression</b>."
            f"</div>"
            f"</div>"
        )
        res["result_text"] = prefix + res.get("result_text", "")
        return res
    else:
        # Significant non-linearity detected across predictors
        cached_data_json = json.dumps(dat.tolist())
        plot_filename = "multilinearity_assessment.png"
        plot_path = os.path.join(STATIC_PLOT_DIR, plot_filename)
        plt.figure(figsize=(6.5, 4.5))
        plt.scatter(y, y_pred_lin, color='#ef4444', alpha=0.6, label=f'Linear Fit (R² = {format_num(r2_lin, 3)})')
        plt.scatter(y, y_pred_quad, color='#2563eb', alpha=0.7, label=f'Quadratic Fit (R² = {format_num(r2_quad, 3)})')
        min_v = min(np.min(y), np.min(y_pred_lin), np.min(y_pred_quad))
        max_v = max(np.max(y), np.max(y_pred_lin), np.max(y_pred_quad))
        plt.plot([min_v, max_v], [min_v, max_v], 'k--', label='Ideal 1:1 Line')
        plt.title('Multivariable Assessment: Linear vs Quadratic Model Fits')
        plt.xlabel('Observed Outcome (y)')
        plt.ylabel('Fitted Outcome (ŷ)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        assessment_report = (
            f"<div style='background-color:#fffbeb; border:1px solid #fde68a; color:#92400e; padding:14px 16px; border-radius:10px; margin-bottom:16px;'>"
            f"<div style='font-weight:700; font-size:1.025rem; margin-bottom:4px;'>"
            f"⚠ Multivariable Non-Linearity Detected: Significant Quadratic Improvement"
            f"</div>"
            f"<div style='font-size:0.92rem; line-height:1.5;'>"
            f"Adding quadratic terms across the {k} predictor variables significantly improved model fit "
            f"(F = {format_num(f_stat, 3)} &gt; F_crit = {format_num(f_crit, 3)}, p = {format_num(p_val)} &le; 0.05). "
            f"A multiple linear model is inadequate; proceed to <b>Further Assessment & Multivariable Non-Linear Model Selection</b>."
            f"</div>"
            f"</div>"
            f"<div style='background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px; padding:14px 16px; margin-bottom:16px;'>"
            f"<h4 style='margin:0 0 8px 0; color:#1e293b; font-size:0.95rem;'>Multivariable Linearity Test Summary:</h4>"
            f"• <b>Multiple Linear SSE:</b> {format_num(sse_lin)} (R² = {format_num(r2_lin)})<br>"
            f"• <b>Quadratic Model SSE:</b> {format_num(sse_quad)} (R² = {format_num(r2_quad)})<br>"
            f"• <b>Extra Sum of Squares (Reduction in Error):</b> {format_num(extra_ss)}<br>"
            f"• <b>Nested Model F-statistic:</b> F({df_extra}, {df_resid_quad}) = {format_num(f_stat, 3)} (p = {format_num(p_val)})"
            f"</div>"
        )
        model_selection_form = render_equation_input_form(
            cached_data_json,
            current_eq="a * (x1**b) * (x2**c)" if k >= 2 else "a * x1**2 + b * x1 + c",
            is_multivariable=(k > 1)
        )
        return {
            "result_text": assessment_report + model_selection_form,
            "boxplot_url": f"questions/{plot_filename}",
        }
