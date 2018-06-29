DATADIR="/home/bunge/maria/Desktop/PSGenRec"
DATASET="${SGE_TASK}"

export DISPLAY=""

cd /home/bunge/maria/Desktop/model && python -c "

from main import *
import glob


# Set and check parameters
sets = {'run_on_cluster': True,
        'data_set': 'PS',

	#########################
	##### ADJUST ME!!! ######
        'fit_par_names': ['epsilon'],
        'learning_style': 'Bayes',
	#########################
	#########################

        'set_specific_parameters': False,
        'use_humans': True,
        'n_agents': 2}
check_user_settings(sets)

# Get data paths, plot paths, etc.
paths = get_paths(sets['use_humans'], sets['data_set'], sets['run_on_cluster'])

# Fit data to files in the given directory (either human data or simulated agents)
fit(sets, '$DATADIR/$DATASET',
    paths['fitted_data_path'],
    paths['heatmap_data_path'],
    paths['prob_switch_randomized_sequences'])

# Simulate data based on fitted parameters
#for file_name in glob.glob(paths['fitted_data_path'] + '*.csv'):
#    simulate_based_on_data(sets=sets, file_name=file_name,
#                           simulation_data_path=paths['simulation_data_path'])"
