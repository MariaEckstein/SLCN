from alien_main import simulate, fit, plot_heatmaps, interactive_game
import glob

# Set parameters
settings = {'run_on_cluster': False,
            'data_set': 'Aliens',
            'fit_par_names': ['alpha', 'beta', 'forget'],
            'learning_style': 'hierarchical',
            'set_specific_parameters': False,
            'use_humans': False}

agent_ids = range(300, 302)

# Do things
# interactive_game(settings)

for agent_id in agent_ids:
    simulate(agent_id=agent_id, sets=settings)

for file_name in glob.glob('C:/Users/maria/MEGAsync/Berkeley/TaskSets/GenRec/' + '*.csv'):
    fit(file_name=file_name, sets=settings)

for agent_id in agent_ids:
    plot_heatmaps(sets=settings, agent_id=agent_id)
