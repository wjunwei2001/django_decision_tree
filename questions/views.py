from django.shortcuts import render, get_object_or_404, redirect
from .models import DecisionTreeNode
import analytics


def decision_tree_view(request):
    """
    Main interactive view for the statistical decision tree.
    Handles traversal across question nodes and execution of statistical analysis.
    """
    root_node = get_object_or_404(DecisionTreeNode, pk=1)
    all_nodes = list(DecisionTreeNode.objects.all())
    nodes = []

    if request.method == 'POST':
        # Case A: User submitted parameters / file to RUN an analysis
        if 'run_action' in request.POST:
            action_name = request.POST.get('action_name')
            func = getattr(analytics, action_name, None)
            if func:
                result = func(request)
                context = {
                    'result': result,
                    'action_name': action_name,
                    'result_text': result.get('result_text', ''),
                    'boxplot_url': result.get('boxplot_url') or result.get('plot_url'),
                }
                return render(request, "questions/analysis_result.html", context)
            else:
                return render(request, "questions/analysis_result.html", {
                    "result_text": f"<div class='alert alert-danger'>Error: Statistical function '{action_name}' not found.</div>"
                })

        # Case B: User answered a decision tree question
        node_pk = request.POST.get('node_id')
        current_node = get_object_or_404(DecisionTreeNode, pk=node_pk)
        answer = request.POST.get('answer')

        next_node = None
        if answer in ('Yes', 'yes'):
            next_node = current_node.yes_node
        elif answer in ('No', 'no'):
            next_node = current_node.no_node
        elif answer in ('1', 'One Predictor', 'One predictor', 'one'):
            next_node = current_node.one_node
        elif answer in ('2', 'Two Categories', '2 Categories', '2 Groups', '2 Predictors'):
            next_node = current_node.two_node
        elif answer in ('More than 2', 'Multiple Predictors', 'Multiple predictors', 'Multiple', '> 1', '> 2 Categories', '> 2 Groups'):
            next_node = current_node.moreThanTwo_node

        nodes.append(current_node)
        if next_node:
            nodes.append(next_node)

        # Reconstruct path from root to current node
        def retrieve_previous_nodes(node):
            for x in all_nodes:
                if (x.yes_node_id == node.id or x.no_node_id == node.id or
                        x.one_node_id == node.id or x.two_node_id == node.id or
                        x.moreThanTwo_node_id == node.id):
                    nodes.insert(0, x)
                    retrieve_previous_nodes(x)

        retrieve_previous_nodes(current_node)

    else:
        # Initial GET request: Start at root
        nodes = [root_node]

    return render(request, 'questions/form.html', {'nodes': nodes})
