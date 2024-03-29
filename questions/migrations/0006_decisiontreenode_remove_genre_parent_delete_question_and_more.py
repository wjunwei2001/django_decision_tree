# Generated by Django 4.1.3 on 2022-12-13 03:31

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('questions', '0005_genre'),
    ]

    operations = [
        migrations.CreateModel(
            name='DecisionTreeNode',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('question', models.TextField()),
                ('no_node', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='no_nodes', to='questions.decisiontreenode')),
                ('yes_node', models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, related_name='yes_nodes', to='questions.decisiontreenode')),
            ],
        ),
        migrations.RemoveField(
            model_name='genre',
            name='parent',
        ),
        migrations.DeleteModel(
            name='Question',
        ),
        migrations.DeleteModel(
            name='Genre',
        ),
    ]
