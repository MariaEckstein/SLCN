from alien_main import run

# for file_name in range(700, 800):
run(file_name='C:/Users/maria/MEGAsync/Berkeley/TaskSets/AlienGenRec/sim_400.csv',  # file_name,  #
    run_on_cluster=False, main_part=True, plot_heatmaps=False, interactive_game=False,
    fit_model=True, simulate_agents=False, use_humans=False)
