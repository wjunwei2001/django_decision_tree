import io
import os
import re
import numpy as np
from django.conf import settings

# Output directory for diagnostic plots
STATIC_PLOT_DIR = os.path.join(settings.BASE_DIR, 'questions', 'static', 'questions')
os.makedirs(STATIC_PLOT_DIR, exist_ok=True)


def format_num(val, decimals=4):
    """
    Formats a numeric result value:
    - If the first non-zero digit is in the 5th decimal place onwards (i.e. 0 < |val| < 0.0001):
      formats in scientific notation to 4 significant figures (e.g. 1.372e-6).
    - Otherwise (e.g. 0.0153, 0.0001):
      formats as standard decimal with `decimals` decimal places (default 4).
    - If val is 0: formats as 0.0000 (or `decimals` zeros).
    """
    if val is None:
        return ""
    try:
        f = float(val)
    except (ValueError, TypeError):
        return str(val)

    if np.isnan(f) or np.isinf(f):
        return str(val)

    if f == 0.0:
        return f"{0.0:.{decimals}f}"

    abs_f = abs(f)
    if abs_f < 0.0001:
        # Scientific notation to 4 significant figures (1 digit before dot, 3 after)
        s = f"{f:.3e}"
        # Strip leading zeros from exponent (e-06 -> e-6)
        return re.sub(r'e([+-])0+(\d+)', r'e\1\2', s)
    else:
        return f"{f:.{decimals}f}"


def load_file_data(request, file_key, default_filename, dtype=float):
    """
    Loads dataset from request.FILES if uploaded.
    Falls back to sample_data/ or root directory if not provided.
    """
    if request and hasattr(request, 'FILES') and request.FILES and file_key in request.FILES:
        try:
            uploaded = request.FILES[file_key]
            content = uploaded.read().decode('utf-8', errors='ignore')
            return np.genfromtxt(io.StringIO(content), dtype=dtype)
        except Exception:
            pass

    # Search in sample_data/ first, then root directory
    sample_path = os.path.join(settings.BASE_DIR, 'sample_data', default_filename)
    if os.path.exists(sample_path):
        return np.genfromtxt(sample_path, dtype=dtype)

    root_path = os.path.join(settings.BASE_DIR, default_filename)
    if os.path.exists(root_path):
        return np.genfromtxt(root_path, dtype=dtype)

    return np.array([])
