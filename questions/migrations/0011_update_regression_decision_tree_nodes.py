from django.db import migrations


def update_regression_nodes(apps, schema_editor):
    DecisionTreeNode = apps.get_model('questions', 'DecisionTreeNode')

    try:
        node_18 = DecisionTreeNode.objects.get(pk=18)
        node_18.question = "Assess the relationship between the predictor and outcome"
        node_18.description = (
            "ASSESS PREDICTOR-OUTCOME RELATIONSHIP. Upload your single-predictor data file "
            "(column 1: predictor x, column 2: outcome y) to evaluate linearity. If no significant "
            "quadratic improvement is found, Simple Linear Regression is performed. If significant "
            "quadratic improvement is detected, you will be guided to select a candidate model or specify "
            "a custom non-linear equation."
        )
        node_18.action_node = "check_singlelinearity"
        node_18.save()
    except DecisionTreeNode.DoesNotExist:
        node_18 = None

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
        node_19 = None

    try:
        node_17 = DecisionTreeNode.objects.get(pk=17)
        node_17.question = "How many predictors are you investigating?"
        node_17.description = "Select whether you are investigating a single predictor or multiple predictors."
        node_17.yes_node = None
        node_17.no_node = None
        node_17.one_node = node_18
        node_17.two_node = None
        node_17.moreThanTwo_node = node_19
        node_17.save()
    except DecisionTreeNode.DoesNotExist:
        pass


def rollback_regression_nodes(apps, schema_editor):
    DecisionTreeNode = apps.get_model('questions', 'DecisionTreeNode')
    try:
        node_17 = DecisionTreeNode.objects.get(pk=17)
        node_18 = DecisionTreeNode.objects.get(pk=18)
        node_19 = DecisionTreeNode.objects.get(pk=19)

        node_17.question = "Is there only 1 x (independent) variable?"
        node_17.yes_node = node_18
        node_17.no_node = node_19
        node_17.one_node = None
        node_17.moreThanTwo_node = None
        node_17.save()

        node_18.question = "We will check if the relationship is approximately linear. Upload your data in this format:"
        node_18.action_node = "check_singlelinearity"
        node_18.save()

        node_19.question = "We will check if the relationship is approximately linear. Upload your data in this format:"
        node_19.action_node = "check_multilinearity"
        node_19.save()
    except DecisionTreeNode.DoesNotExist:
        pass


class Migration(migrations.Migration):

    dependencies = [
        ('questions', '0010_rename_action_decisiontreenode_action_node'),
    ]

    operations = [
        migrations.RunPython(update_regression_nodes, rollback_regression_nodes),
    ]

