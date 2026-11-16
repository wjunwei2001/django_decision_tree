from django.db import migrations


def update_node_19_multilinearity(apps, schema_editor):
    DecisionTreeNode = apps.get_model('questions', 'DecisionTreeNode')
    try:
        node_19 = DecisionTreeNode.objects.get(pk=19)
        node_19.question = "Assess the relationship between the predictors and outcome"
        node_19.description = (
            "ASSESS MULTIVARIABLE PREDICTOR-OUTCOME RELATIONSHIP. Upload your multivariable dataset "
            "(column 1: outcome y, subsequent columns: predictors x₁, x₂, ... xₖ). We will assess the "
            "relationship across predictors by testing for significant quadratic curvature. If the relationship "
            "is linear (no significant quadratic improvement), Multiple Linear Regression is performed. If "
            "significant curvature is detected, you will be guided to select a multivariable model or specify "
            "a custom non-linear equation."
        )
        node_19.action_node = "check_multilinearity"
        node_19.save()
    except DecisionTreeNode.DoesNotExist:
        pass


def rollback_node_19_multilinearity(apps, schema_editor):
    DecisionTreeNode = apps.get_model('questions', 'DecisionTreeNode')
    try:
        node_19 = DecisionTreeNode.objects.get(pk=19)
        node_19.question = "MULTIPLE LINEAR REGRESSION"
        node_19.action_node = "run_multiple_linear_regression"
        node_19.save()
    except DecisionTreeNode.DoesNotExist:
        pass


class Migration(migrations.Migration):

    dependencies = [
        ('questions', '0011_update_regression_decision_tree_nodes'),
    ]

    operations = [
        migrations.RunPython(update_node_19_multilinearity, rollback_node_19_multilinearity),
    ]

