from main import simulate, fit, plot_heatmaps, interactive_game
import glob

# Set parameters
settings = {'run_on_cluster': False,
            'data_set': 'PS',  # 'PS' or 'Aliens'
            'fit_par_names': ['epsilon'],
            'learning_style': 'Bayes',
            'set_specific_parameters': True,
            'use_humans': False}

agent_ids = range(100, 150)

# Do things
# interactive_game(settings)

for agent_id in agent_ids:
    simulate(sets=settings, agent_id=agent_id,
             save_path='C:/Users/maria/MEGAsync/SLCN/' + settings['data_set'] + 'GenRec')

human_data_path = 'C:/Users/maria/MEGAsync/SLCNdata/PSResults/'
for file_name in glob.glob(human_data_path + '*.csv'):  # 'C:/Users/maria/MEGAsync/SLCN/PSGenRec/'
    fit(file_name=file_name, sets=settings,
        save_path=human_data_path + '/fit_par/')

for agent_id in agent_ids:
    plot_heatmaps(sets=settings, agent_id=agent_id)
