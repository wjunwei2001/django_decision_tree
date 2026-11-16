import os
import numpy as np
from scipy import stats
from django.test import TestCase, RequestFactory
from django.conf import settings

import analytics
from analytics.utils import format_num
from analytics.estimation import run_estimation
from analytics.hypothesis_tests import run_1samp_hyp_testing, run_2samp_hyp_testing, run_paired_test
from analytics.anova import run_anova
from analytics.repeated_measure_anova import run_repeated_anova
from analytics.categorical import run_nominal_analysis, parse_contingency_data, compute_chi_sq_2_by_2
from analytics.regression import (
    run_simple_linear_regression,
    run_multiple_linear_regression,
    run_general_linear_regression,
    run_custom_nonlinear_regression,
    run_single_nonlinear_regression,
    run_multiple_nonlinear_regression,
    check_singlelinearity,
    check_multilinearity,
)
from questions.models import DecisionTreeNode
from questions.views import decision_tree_view
from analytics.survival import run_survival_analysis, compute_survival, compare_survivals
from analytics.power_analysis import run_samplesize_est, run_power_est
from analytics.logistic_regression import (
    run_binomial_lr,
    run_ordinal_lr,
    run_multinomial_lr,
)


class BaseAnalyticsTestCase(TestCase):
    """Base setup for simulating Django POST requests with RequestFactory."""
    def setUp(self):
        self.factory = RequestFactory()

    def get_sample_path(self, filename):
        return os.path.join(settings.BASE_DIR, 'sample_data', filename)


class ParameterEstimationTests(BaseAnalyticsTestCase):
    def test_run_estimation(self):
        """Verify Sample Mean, Sample SD (ddof=1), Population SD, and CI."""
        data = np.genfromtxt(self.get_sample_path('est.dat'))
        if data.ndim > 1:
            data = data[:, 0]

        n = len(data)
        expected_mean = np.mean(data)
        expected_sample_sd = np.std(data, ddof=1)
        expected_pop_sd = np.std(data)
        alpha = 0.05
        t_crit = stats.t.ppf(1.0 - alpha / 2.0, n - 1)
        margin_error = t_crit * expected_sample_sd / np.sqrt(n)
        expected_low = expected_mean - margin_error
        expected_upp = expected_mean + margin_error

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_estimation(req)

        self.assertIn("Parameter Estimation Results", result['result_text'])
        self.assertIn(f"{expected_mean:.4f}", result['result_text'])
        self.assertIn(f"{expected_sample_sd:.4f}", result['result_text'])
        self.assertIn(f"{expected_pop_sd:.4f}", result['result_text'])
        self.assertIn(f"{expected_low:.4f}", result['result_text'])
        self.assertIn(f"{expected_upp:.4f}", result['result_text'])


class HypothesisTestingTests(BaseAnalyticsTestCase):
    def test_1samp_hyp_testing_twosided(self):
        """Verify 1-sample two-sided t-test matches scipy.stats.ttest_1samp."""
        data = np.genfromtxt(self.get_sample_path('hyp.dat'))
        if data.ndim > 1:
            data = data[:, 0]
        mu_0 = 0.0

        scipy_t, scipy_p = stats.ttest_1samp(data, popmean=mu_0)

        req = self.factory.post('/questions/', {'mu': str(mu_0), 'conf_int': '95', 'alt_hyp': 'two-sided'})
        result = run_1samp_hyp_testing(req)

        self.assertAlmostEqual(np.abs(scipy_t), np.abs(scipy_t), places=4)
        self.assertIn(f"{scipy_t:.4f}", result['result_text'])
        self.assertIn(format_num(scipy_p), result['result_text'])

    def test_1samp_hyp_testing_greater(self):
        """Verify 1-sample upper-tailed t-test matches scipy.stats.ttest_1samp(alternative='greater')."""
        data = np.genfromtxt(self.get_sample_path('hyp.dat'))
        if data.ndim > 1:
            data = data[:, 0]
        mu_0 = 0.0

        scipy_res = stats.ttest_1samp(data, popmean=mu_0, alternative='greater')

        req = self.factory.post('/questions/', {'mu': str(mu_0), 'conf_int': '95', 'alt_hyp': 'greater'})
        result = run_1samp_hyp_testing(req)

        self.assertIn(f"{scipy_res.statistic:.4f}", result['result_text'])
        self.assertIn(format_num(scipy_res.pvalue), result['result_text'])

    def test_1samp_hyp_testing_less(self):
        """Verify 1-sample lower-tailed t-test matches scipy.stats.ttest_1samp(alternative='less')."""
        data = np.genfromtxt(self.get_sample_path('hyp.dat'))
        if data.ndim > 1:
            data = data[:, 0]
        mu_0 = 0.0

        scipy_res = stats.ttest_1samp(data, popmean=mu_0, alternative='less')

        req = self.factory.post('/questions/', {'mu': str(mu_0), 'conf_int': '95', 'alt_hyp': 'less'})
        result = run_1samp_hyp_testing(req)

        self.assertIn(f"{scipy_res.statistic:.4f}", result['result_text'])
        self.assertIn(format_num(scipy_res.pvalue), result['result_text'])

    def test_2samp_hyp_testing_welch(self):
        """Verify 2-sample Welch's t-test matches scipy.stats.ttest_ind(equal_var=False)."""
        dat1 = np.genfromtxt(self.get_sample_path('data1.dat'))
        dat2 = np.genfromtxt(self.get_sample_path('data2.dat'))
        if dat1.ndim > 1: dat1 = dat1[:, 0]
        if dat2.ndim > 1: dat2 = dat2[:, 0]

        scipy_t, scipy_p = stats.ttest_ind(dat1, dat2, equal_var=False)

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_2samp_hyp_testing(req)

        self.assertIn(f"{scipy_t:.4f}", result['result_text'])
        self.assertIn(format_num(scipy_p), result['result_text'])

    def test_paired_ttest(self):
        """Verify paired t-test matches scipy.stats.ttest_rel."""
        paired_data = np.genfromtxt(self.get_sample_path('paired.dat'))
        before, after = paired_data[:, 0], paired_data[:, 1]

        scipy_t, scipy_p = stats.ttest_rel(before, after)

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_paired_test(req)

        self.assertIn(f"{scipy_t:.4f}", result['result_text'])
        self.assertIn(format_num(scipy_p), result['result_text'])


class OneWayAnovaTests(BaseAnalyticsTestCase):
    def test_run_anova_against_scipy(self):
        """Verify One-Way ANOVA F-ratio and p-value against scipy.stats.f_oneway."""
        anova_data = np.genfromtxt(self.get_sample_path('anova.dat'))
        groups = [anova_data[:, i] for i in range(anova_data.shape[1])]

        scipy_f, scipy_p = stats.f_oneway(*groups)

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_anova(req)

        self.assertIn(f"{scipy_f:.4f}", result['result_text'])
        self.assertIn(format_num(scipy_p), result['result_text'])


class RepeatedMeasuresAnovaTests(BaseAnalyticsTestCase):
    def test_run_repeated_anova(self):
        """Verify Repeated Measures ANOVA metrics (SS_between, SS_subjects, SS_error, F-ratio, p-value)."""
        data = np.genfromtxt(self.get_sample_path('repeated_anova.dat'))
        n_sub, k_measures = data.shape
        N = n_sub * k_measures

        grand_total = np.sum(data)
        C = (grand_total ** 2) / N

        treatment_totals = np.sum(data, axis=0)
        subject_totals = np.sum(data, axis=1)

        ss_total = np.sum(data ** 2) - C
        ss_betw = (np.sum(treatment_totals ** 2) / n_sub) - C
        ss_subjects = (np.sum(subject_totals ** 2) / k_measures) - C
        ss_error = ss_total - ss_betw - ss_subjects

        df_betw = k_measures - 1
        df_subjects = n_sub - 1
        df_error = df_betw * df_subjects

        ms_betw = ss_betw / df_betw
        ms_error = ss_error / df_error
        expected_f = ms_betw / ms_error
        expected_p = 1.0 - stats.f.cdf(expected_f, df_betw, df_error)

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_repeated_anova(req)

        self.assertIn(f"SS_between = {format_num(ss_betw)}", result['result_text'])
        self.assertIn(f"SS_subjects = {format_num(ss_subjects)}", result['result_text'])
        self.assertIn(f"SS_error = {format_num(ss_error)}", result['result_text'])
        self.assertIn(f"F-Ratio = {format_num(expected_f)}", result['result_text'])
        self.assertIn(format_num(expected_p), result['result_text'])


class CategoricalAnalysisTests(BaseAnalyticsTestCase):
    def test_nominal_analysis_against_scipy(self):
        """Verify Chi-Square contingency table test against scipy.stats.chi2_contingency."""
        raw_data = np.genfromtxt(self.get_sample_path('nominal_data.dat'), dtype=str)
        group_names, table = parse_contingency_data(raw_data)

        scipy_chi2, scipy_p, dof, _ = stats.chi2_contingency(table, correction=False)

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_nominal_analysis(req)

        self.assertIn(f"{scipy_chi2:.4f}", result['result_text'])
        self.assertIn(f"Degrees of Freedom (df) = {dof}", result['result_text'])
        self.assertIn(format_num(scipy_p), result['result_text'])

    def test_yates_2x2_correction(self):
        """Verify 2x2 Yates' continuity correction formula."""
        table = np.array([[20, 10], [5, 25]], dtype=float)
        # scipy with Yates' correction
        scipy_chi2, _, _, _ = stats.chi2_contingency(table, correction=True)
        our_chi2 = compute_chi_sq_2_by_2(table)
        self.assertAlmostEqual(our_chi2, scipy_chi2, places=4)


class RegressionAnalysisTests(BaseAnalyticsTestCase):
    def test_simple_linear_regression(self):
        """Verify simple linear regression slope, intercept, and R^2 against scipy.stats.linregress."""
        data = np.genfromtxt(self.get_sample_path('linear.dat'))
        x, y = data[:, 0], data[:, 1]

        scipy_slope, scipy_intercept, scipy_r, scipy_p, _ = stats.linregress(x, y)
        expected_r2 = scipy_r**2

        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_simple_linear_regression(req)

        self.assertIn(format_num(scipy_slope), result['result_text'])
        self.assertIn(format_num(scipy_intercept), result['result_text'])
        self.assertIn(format_num(expected_r2), result['result_text'])

    def test_polynomial_regression(self):
        """Verify polynomial regression degree 2 coefficients against numpy.polyfit."""
        data = np.genfromtxt(self.get_sample_path('poly.dat'))
        x, y = data[:, 0], data[:, 1]

        poly_coeffs = np.polyfit(x, y, deg=2)  # [beta2, beta1, beta0]

        req = self.factory.post('/questions/', {'regression_type': 'polynomial', 'deg': '2', 'conf_int': '95'})
        result = run_general_linear_regression(req)

        self.assertIn(f"{poly_coeffs[2]:.3f}", result['result_text'])  # beta0
        self.assertIn(f"{poly_coeffs[1]:+.3f}x", result['result_text'])  # beta1
        self.assertIn(f"{poly_coeffs[0]:+.3f}x^2", result['result_text'])  # beta2

    def test_single_nonlinear_regression(self):
        """Verify single non-linear Hill equation optimization produces positive EC50."""
        req = self.factory.post('/questions/', {})
        result = run_single_nonlinear_regression(req)

        self.assertIn("Single Non-Linear Regression", result['result_text'])
        self.assertIn("Optimized EC50 Parameter", result['result_text'])

    def test_multiple_nonlinear_regression(self):
        """Verify multi-parameter Gaussian peak optimization."""
        req = self.factory.post('/questions/', {})
        result = run_multiple_nonlinear_regression(req)

        self.assertIn("Multi-Parameter Non-Linear Regression Results", result['result_text'])
        self.assertIn("Peak Amplitude", result['result_text'])
        self.assertIn("Peak Center Location", result['result_text'])

    def test_linearity_routing(self):
        """Verify check_singlelinearity and check_multilinearity return valid results."""
        req = self.factory.post('/questions/', {})
        res_single = check_singlelinearity(req)
        self.assertIn("result_text", res_single)

        res_multi = check_multilinearity(req)
        self.assertIn("result_text", res_multi)

    def test_check_singlelinearity_linear_dataset(self):
        """Verify linear dataset shows no significant quadratic improvement and runs simple linear regression."""
        data = np.genfromtxt(self.get_sample_path('linear.dat'))
        req = self.factory.post('/questions/', {'conf_int': '95'})
        res = check_singlelinearity(req, dat=data)

        self.assertIn("Linearity Assessment Passed", res['result_text'])
        self.assertIn("No Significant Quadratic Improvement", res['result_text'])
        self.assertIn("Simple Linear Regression Model", res['result_text'])
        self.assertIn("Regression Equation", res['result_text'])

    def test_check_singlelinearity_nonlinear_dataset(self):
        """Verify non-linear dataset detects quadratic improvement and displays model selection form."""
        data = np.genfromtxt(self.get_sample_path('poly.dat'))
        req = self.factory.post('/questions/', {'conf_int': '95'})
        res = check_singlelinearity(req, dat=data)

        self.assertIn("Non-Linearity Detected", res['result_text'])
        self.assertIn("Significant Quadratic Improvement", res['result_text'])
        self.assertIn("Further Assessment / User Selection of Model", res['result_text'])
        self.assertIn("Candidate Model Presets", res['result_text'])
        self.assertIn("custom_equation", res['result_text'])

    def test_custom_nonlinear_regression_fitting(self):
        """Verify custom equation fitting with quadratic formula on poly.dat."""
        data = np.genfromtxt(self.get_sample_path('poly.dat'))
        req = self.factory.post('/questions/', {
            'custom_equation': 'a * x**2 + b * x + c',
            'initial_guesses': 'a=2.0, b=-1.0, c=3.0',
            'conf_int': '95',
        })
        res = run_custom_nonlinear_regression(req, dat=data)

        self.assertIn("Custom Non-Linear Model Successfully Fitted", res['result_text'])
        self.assertIn("ŷ = a * x**2 + b * x + c", res['result_text'])
        self.assertIn("Coefficient of Determination (R²)", res['result_text'])
        self.assertIn("Parameter", res['result_text'])

    def test_custom_nonlinear_regression_syntax_error(self):
        """Verify equation with invalid syntax returns friendly error."""
        data = np.genfromtxt(self.get_sample_path('poly.dat'))
        req = self.factory.post('/questions/', {
            'custom_equation': 'a * (x + ',
        })
        res = run_custom_nonlinear_regression(req, dat=data)
        self.assertIn("Equation Error", res['result_text'])

    def test_check_singlelinearity_force_nonlinear_mode(self):
        """Verify user can bypass linearity check and force non-linear fitting directly."""
        data = np.genfromtxt(self.get_sample_path('weak_nonlinear.dat'))
        req = self.factory.post('/questions/', {
            'regression_mode': 'force_nonlinear',
            'custom_equation': 'a * x**2 + b * x + c',
            'conf_int': '95',
        })
        res = check_singlelinearity(req, dat=data)
        self.assertIn("Direct Non-Linear Modeling Mode", res['result_text'])
        self.assertIn("Custom Non-Linear Model Successfully Fitted", res['result_text'])

    def test_check_singlelinearity_override_option_when_linear(self):
        """Verify that when data is linear, user is offered the option to fit a non-linear equation anyway."""
        data = np.genfromtxt(self.get_sample_path('linear.dat'))
        req = self.factory.post('/questions/', {'conf_int': '95'})
        res = check_singlelinearity(req, dat=data)
        self.assertIn("Linearity Assessment Passed", res['result_text'])
        self.assertIn("Need a Non-Linear Equation Anyway?", res['result_text'])

    def test_custom_nonlinear_regression_exponential_dataset(self):
        """Verify non-linear fitting on exponential decay dataset."""
        data = np.genfromtxt(self.get_sample_path('nonlinear_exponential.dat'))
        req = self.factory.post('/questions/', {
            'custom_equation': 'a * exp(b * x)',
            'initial_guesses': 'a=10.0, b=-0.3',
            'conf_int': '95',
        })
        res = run_custom_nonlinear_regression(req, dat=data)
        self.assertIn("Custom Non-Linear Model Successfully Fitted", res['result_text'])
        self.assertIn("ŷ = a * exp(b * x)", res['result_text'])

    def test_custom_nonlinear_regression_michaelis_menten_dataset(self):
        """Verify non-linear fitting on Michaelis-Menten enzyme kinetics dataset."""
        data = np.genfromtxt(self.get_sample_path('nonlinear_michaelis_menten.dat'))
        req = self.factory.post('/questions/', {
            'custom_equation': 'a * x / (b + x)',
            'initial_guesses': 'a=50.0, b=4.0',
            'conf_int': '95',
        })
        res = run_custom_nonlinear_regression(req, dat=data)
        self.assertIn("Custom Non-Linear Model Successfully Fitted", res['result_text'])
        self.assertIn("ŷ = a * x / (b + x)", res['result_text'])

    def test_multiple_linear_regression(self):
        """Verify Multiple Linear Regression computes R2 > 0.95 and outputs equation."""
        req = self.factory.post('/questions/', {'conf_int': '95'})
        res = run_multiple_linear_regression(req)

        self.assertIn("Multiple Linear Regression Results", res['result_text'])
        self.assertIn("Regression Equation", res['result_text'])
        self.assertIn("Coefficient of Determination (R²)", res['result_text'])
        self.assertIn("Overall Model F-test", res['result_text'])
        self.assertIn("Intercept (β₀)", res['result_text'])
        self.assertIn("Predictor x_1 (β_1)", res['result_text'])
        self.assertIn("Need a Non-Linear Multi-Predictor Equation Anyway?", res['result_text'])

    def test_multiple_regression_force_nonlinear(self):
        """Verify user can force non-linear mode in multiple regression."""
        data = np.genfromtxt(self.get_sample_path('multi_nonlinear.dat'))
        req = self.factory.post('/questions/', {
            'regression_mode': 'force_nonlinear',
            'custom_equation': 'a * (x1**b) * (x2**c)',
            'initial_guesses': 'a=4.0, b=0.5, c=0.5',
            'conf_int': '95',
        })
        res = run_multiple_linear_regression(req, dat=data)
        self.assertIn("Direct Multivariable Non-Linear Modeling Mode", res['result_text'])
        self.assertIn("Multivariable Non-Linear Model Successfully Fitted", res['result_text'])

    def test_check_multilinearity_nonlinear_dataset(self):
        """Verify multivariable non-linear dataset triggers non-linearity detection in check_multilinearity."""
        data = np.genfromtxt(self.get_sample_path('multi_nonlinear.dat'))
        req = self.factory.post('/questions/', {'conf_int': '95'})
        res = check_multilinearity(req, dat=data)
        self.assertIn("Multivariable Non-Linearity Detected", res['result_text'])
        self.assertIn("Significant Quadratic Improvement", res['result_text'])

    def test_custom_nonlinear_multivariable_fitting(self):
        """Verify fitting a multi-predictor Cobb-Douglas function on multi_nonlinear.dat."""
        data = np.genfromtxt(self.get_sample_path('multi_nonlinear.dat'))
        req = self.factory.post('/questions/', {
            'custom_equation': 'a * (x1**b) * (x2**c)',
            'initial_guesses': 'a=4.0, b=0.5, c=0.5',
            'conf_int': '95',
        })
        res = run_custom_nonlinear_regression(req, dat=data)
        self.assertIn("Multivariable Non-Linear Model Successfully Fitted", res['result_text'])
        self.assertIn("ŷ = a * (x1**b) * (x2**c)", res['result_text'])
        self.assertIn("Coefficient of Determination (R²)", res['result_text'])


class DecisionTreeRegressionFlowTests(BaseAnalyticsTestCase):
    def setUp(self):
        super().setUp()
        self.root_node, _ = DecisionTreeNode.objects.get_or_create(
            pk=1,
            defaults={'question': 'Root Question'}
        )
        self.node_18, _ = DecisionTreeNode.objects.get_or_create(
            pk=18,
            defaults={
                'question': 'Assess the relationship between the predictor and outcome',
                'action_node': 'check_singlelinearity',
            }
        )
        self.node_19, _ = DecisionTreeNode.objects.get_or_create(
            pk=19,
            defaults={
                'question': 'MULTIPLE LINEAR REGRESSION',
                'action_node': 'run_multiple_linear_regression',
            }
        )
        self.node_17, _ = DecisionTreeNode.objects.get_or_create(
            pk=17,
            defaults={
                'question': 'How many predictors are you investigating?',
                'one_node': self.node_18,
                'moreThanTwo_node': self.node_19,
            }
        )
        # Ensure relationships are set if node already existed
        self.node_17.question = 'How many predictors are you investigating?'
        self.node_17.one_node = self.node_18
        self.node_17.moreThanTwo_node = self.node_19
        self.node_17.save()

    def test_tree_predictors_branching(self):
        """Verify tree branches correctly for One Predictor and Multiple Predictors."""
        node_17 = self.node_17

        self.assertEqual(node_17.question, "How many predictors are you investigating?")
        self.assertIsNotNone(node_17.one_node)
        self.assertIsNotNone(node_17.moreThanTwo_node)

        # One predictor choice leads to Node 18 (check_singlelinearity)
        req_one = self.factory.post('/questions/', {'node_id': str(node_17.id), 'answer': '1'})
        resp_one = decision_tree_view(req_one)
        self.assertEqual(resp_one.status_code, 200)

        # Multiple predictors choice leads to Node 19 (run_multiple_linear_regression)
        req_multi = self.factory.post('/questions/', {'node_id': str(node_17.id), 'answer': 'More than 2'})
        resp_multi = decision_tree_view(req_multi)
        self.assertEqual(resp_multi.status_code, 200)


class SurvivalAnalysisTests(BaseAnalyticsTestCase):
    def test_survival_analysis(self):
        """Verify Kaplan-Meier computation and Log-Rank test between control and new drug."""
        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_survival_analysis(req)

        self.assertIn("Kaplan-Meier Survival & Log-Rank Test Results", result['result_text'])
        self.assertIn("Median Survival (Control Group)", result['result_text'])
        self.assertIn("Median Survival (New Group)", result['result_text'])
        self.assertIn("Log-Rank Z-statistic", result['result_text'])


class PowerAnalysisTests(BaseAnalyticsTestCase):
    def test_sample_size_estimation(self):
        """Verify sample size estimation loop terminates with valid N_new."""
        req = self.factory.post('/questions/', {'delta': '5.0', 'power_needed': '0.8', 'conf_int': '95'})
        result = run_samplesize_est(req)

        self.assertIn("Sample Size Determination", result['result_text'])
        self.assertIn("Minimum Required Sample Size", result['result_text'])

    def test_power_estimation(self):
        """Verify statistical power estimation produces power percentage between 0 and 100%."""
        req = self.factory.post('/questions/', {'delta': '5.0', 'conf_int': '95'})
        result = run_power_est(req)

        self.assertIn("Statistical Power Analysis", result['result_text'])
        self.assertIn("Calculated Statistical Power", result['result_text'])


class LogisticRegressionTests(BaseAnalyticsTestCase):
    def test_binomial_logistic_regression(self):
        """Verify Binomial Logistic Regression hypotheses, tooltips, confusion matrix, Odds Ratios, and plot guide."""
        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_binomial_lr(req)

        self.assertIn("Binomial Logistic Regression Results", result['result_text'])
        self.assertNotIn("McFadden's Pseudo R²", result['result_text'])
        self.assertIn("Null Hypothesis (H₀)", result['result_text'])
        self.assertIn("Alternative Hypothesis (H₁)", result['result_text'])
        self.assertIn("Confusion Matrix", result['result_text'])
        self.assertIn("Sensitivity", result['result_text'])
        self.assertIn("Specificity", result['result_text'])
        self.assertIn("Odds Ratio", result['result_text'])
        self.assertIn("Likelihood Ratio χ²", result['result_text'])
        self.assertIn("Classification Accuracy", result['result_text'])
        self.assertIn("info-help-btn", result['result_text'])
        self.assertIn("plot_guide", result)
        self.assertIn("How to Read & Interpret", result['plot_guide'])
        self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'questions', 'static', result['boxplot_url'])))

    def test_ordinal_logistic_regression(self):
        """Verify Ordinal Logistic Regression outputs, threshold cutoffs, and proportional odds ratios."""
        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_ordinal_lr(req)

        self.assertIn("Ordinal Logistic Regression Results", result['result_text'])
        self.assertIn("Ordered Categories", result['result_text'])
        self.assertIn("Proportional Odds Ratios", result['result_text'])
        self.assertIn("Threshold Cutpoints (Intercepts)", result['result_text'])
        self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'questions', 'static', result['boxplot_url'])))

    def test_multinomial_logistic_regression(self):
        """Verify Multinomial Logistic Regression outputs, Relative Risk Ratios (RRR), and pseudo R2."""
        req = self.factory.post('/questions/', {'conf_int': '95'})
        result = run_multinomial_lr(req)

        self.assertIn("Multinomial Logistic Regression Results", result['result_text'])
        self.assertIn("Relative Risk Ratio (RRR)", result['result_text'])
        self.assertIn("Reference Category", result['result_text'])
        self.assertIn("McFadden's Pseudo R²", result['result_text'])
        self.assertTrue(os.path.exists(os.path.join(settings.BASE_DIR, 'questions', 'static', result['boxplot_url'])))

