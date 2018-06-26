from alien_main import run

for file_name in range(700, 800):
    run(file_name=file_name,  #'C:/Users/maria/MEGAsync/Berkeley/TaskSets/AlienGenRec/sim_400.csv',  #
        run_on_cluster=False,
        main_part=False, fit_model=True, fit_par_names=['alpha', 'beta'],
        simulate_agents_post_fit=False, use_humans=False,
        plot_heatmaps=False, interactive_game=True)
