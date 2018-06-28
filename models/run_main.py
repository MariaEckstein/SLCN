from main import *
import glob


# Set and check parameters
sets = {'run_on_cluster': False,
        'data_set': 'PS',  # 'PS' or 'Aliens'
        'fit_par_names': ['alpha', 'beta', 'epsilon'],
        'learning_style': 'simple_flat',
        'set_specific_parameters': False,
        'use_humans': False,
        'n_agents': 2}
check_user_settings(sets)

# Get data paths, plot paths, etc.
paths = get_paths(sets['use_humans'], sets['data_set'], sets['run_on_cluster'])
agent_ids = range(100, 111)

# Interactive game
# interactive_game(sets)

# Simulate random agents based on the model specified above
for agent_id in agent_ids:
    simulate(sets=sets, agent_id=agent_id,
             save_path=paths['agent_data_path'])

# Fit data to files in the given directory (either human data or simulated agents)
for file_name in glob.glob(paths['agent_data_path'] + '*.csv'):
    fit(sets=sets, file_name=file_name,
        fitted_data_path=paths['fitted_data_path'],
        heatmap_data_path=paths['heatmap_data_path'])

# # Plot heatmaps to show how the fitting went
# for agent_id in agent_ids:
#     plot_heatmaps(sets=sets, agent_id=agent_id,
#                   heatmap_data_path=paths['heatmap_data_path'],
#                   heatmap_plot_path=paths['heatmap_plot_path'])

# Simulate data based on fitted parameters
for file_name in glob.glob(paths['fitted_data_path'] + '*.csv'):
    simulate_based_on_data(sets=sets, file_name=file_name,
                           simulation_data_path=paths['simulation_data_path'])
