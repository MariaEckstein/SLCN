How to fit various models to the probabilistic switching task using maximum likelihood or sampling, how to simulate data using the same models and the fitted parameters, and how to plot everything:

1) PSAllModels.py fits the models.
It provides three classes of models: strategies, RL, and Bayes. It know which  model you want to run through the model_name you specify. E.g., "RLab" will run an alpha-beta RL model.
All model_names start with "RL" (for RL models), "B" (for Bayesian inference), or "WSLS" (for the two fixed strategies).
Specify the free parameters after the model_name. RL allows for parameters a (learning rate alpha), b (inverse softmax temperature beta), c (counter-factual value updating, will be multiplied with alpha to create calpha),
n (negative learning rate nalpha), x (counter-factual negative updating, will be multiplied with nalpha to create cnalpha), and persev (perseverance, i.e., prevalence to repeat the same trial, or undecision point in the softmax).
To run, make sure that load_data is loading the probswitch data correcty. After that, all specified models will be fitted and saved in the directory specified in save_dir.
You can choose you favorite fitting method (MAP + uniform priors gives you MLE and runs quickly; MCMC gives you awesome sampling results, you can specify beautiful hierarchical models, but it takes a long time).
For debugging, set load_data(n_subj='10') and create_model(n_trials=6, verbose=True) to see the values, probabilities, choices, and outcomes of each trial printed for a small number of participants and trials.
PSAllModels will save a pickly file for every fitted model that contains a dictionary with the fitted parameters, and the model fit.

2) Having fitted models, you can look at the results using MAPResultPlots.py
Specify the directory were PSAllModels saved its pickle results.
MAPResultPlots will plot the BICs, AICs, and NLLs of all fitted models, and created nice orderly .csv files of the fitted parameters (FOR NON-HIERARCHICAL MODELS ONLY; it crashes for hierarchical models because different parameters have different numbers of entries).
MAPResultPlots will also plot the distributions over fitted parameter values. This is a quick way to check if your model fits make sense (are parameters spread out nicely? Is the range correct?).

3) After getting the nice csvs, you are ready to simulate agent behavior using PSAllSimulation.py
Scroll down to the end and make sure data_dir is pointing to the directory in which MAPResultPlots stored the pretty csv files that contain the fitted parameters for each participant.
Run the file. You can specify how many simulations it will create per participant, and plot Q-values over time for each simulation if you wish.
PSAllSimulation will also calculate the correlation between fitted parameters and age, and plot average parameter values against age quantiles. If you already have the simulations and only want the plots, just comment out the lines with simulate_model_from_parameters and the one below.
PSAllSimulation will only work if you adhere to the model naming convention described above. I.e., some part of the file_names that contains the fitted parameters must contain a string of letters that starts with "RL", "B", or "WSLS". This will be the model_name to determine which model to simulate. 

4) After simulating, you can go back to MAPResultPlots and comment in the last section.
This will plot fitted NLLs against simulated NLLs. A good way to check that simulation and fitting are doing more or less the same thing.

Lastly, send me an email: maria.eckstein@berkeley.edu
