
read_in_files = function() {
  if (data %in% c("human", "fitted_human")) {
    data_dir = base_dir
  }
  all_files = data.frame()
  filenames = list.files(data_dir, pattern = ifelse(data == "human", "*.mat", "*PS_[0-9]*.csv"))
  filename = filenames[1] # debugging
  
  for(filename in filenames) {
    if (data == "human") {
      mat_file = readMat(file.path(data_dir, filename))$exp[,,1]$PROBSWITCHdata[,,1]
      subj_file = data.frame(RT = t(mat_file$RT))
      subj_file$selected_box = (t(mat_file$key) - 10) / 2
      subj_file$reward = t(mat_file$reward)
      subj_file$correct_box = 1 - t(mat_file$better.box.left)
      sID = as.numeric(strsplit(strsplit(filename, split = "PROBSWITCH_")[[1]][2], ".mat")[[1]])
      subj_file$sID = sID
      subj_file = subset(subj_file, !is.nan(reward) & selected_box %in% c(0, 1))
      subj_file = subj_file[47:nrow(subj_file),]  # needs to be changed for future participants - 51
    } else {
      subj_file = read.csv(file.path(data_dir, filename))
      subj_file$X = NULL
      subj_file$p_switch_rec = NULL
      # subj_file$learning_style = learning_style
      # subj_file$method = method
      # subj_file$fit_par = fit_par
    }
    subj_file$TrialID = 1:nrow(subj_file)
    subj_file$rewardversion = subj_file$sID %% 4
    subj_file$ACC = with(subj_file, selected_box == correct_box)
    
    # Get trials since last switch (forward pass -> numbers > 0)
    subj_file$switch_trial = NA
    box_prev_trial = subj_file$correct_box[1]
    block = 1
    trialsinceswitch = 0
    for (t in subj_file$TrialID) {
      # Find switch trials
      subj_file$switch_trial[t] = box_prev_trial != subj_file$correct_box[t]
      box_prev_trial = subj_file$correct_box[t]
      # Get blocks
      block = block + subj_file$switch_trial[t]
      subj_file$block[t] = block
      trialsinceswitch = ifelse(subj_file$switch_trial[t] == 1, 0, trialsinceswitch + 1)
      subj_file$trialsinceswitch[t] = trialsinceswitch
    }
    # Get trials since last switch (backward pass -> numbers < 0)
    block = 1
    trialsinceswitch = 0
    for (t in rev(subj_file$TrialID)) {
      trialsinceswitch = ifelse(subj_file$switch_trial[t] == 1, 0, trialsinceswitch - 1)
      if (trialsinceswitch > -4) {
        subj_file$trialsinceswitch[t] = trialsinceswitch
      }
    }
    
    outcome_1_back = c(NA, subj_file$reward[1:(nrow(subj_file) - 1)])
    subj_file$outcome_1_back = outcome_1_back
    outcome_2_back = c(NA, NA, subj_file$reward[1:(nrow(subj_file) - 2)])
    subj_file$outcome_2_back = outcome_2_back
    
    subj_file$choice_left = subj_file$selected_box == 0
    subj_file$choice_1_back = c(NA, subj_file$choice_left[1:(nrow(subj_file) - 1)])
    subj_file$choice_2_back = c(NA, NA, subj_file$choice_left[1:(nrow(subj_file) - 2)])
    
    if (data == "human") {
      write.csv(subj_file, paste(data_dir, "/PS_", sID, ".csv", sep = ""), row.names = F)
    }
    
    all_files = as.data.frame(rbind(all_files, subj_file))
  }
  
  all_files$outcome_12_back = paste(all_files$outcome_1_back, all_files$outcome_2_back)
  all_files$outcome_12_back = factor(all_files$outcome_12_back, levels = c("1 1", "1 0", "0 1", "0 0"))
  all_files$same_choice = all_files$choice_1_back == all_files$choice_2_back
  all_files$choice_12_back = ifelse(all_files$choice_1_back, "left", "right")
  all_files$choice_12_back[!all_files$same_choice | is.na(all_files$same_choice)] = NA
  all_files$reward_port = factor(all_files$correct_box, levels = c(0, 1), labels = c("Left", "Right"))
  all_files$age_group = "Children"
  all_files$age_group[all_files$sID >= 300] = "Adults"
  all_files$age_group[all_files$sID >= 1000] = "Simulated"
  
  return(all_files)
}


model_plots = function() {
  # Plot values for one example session
  ex_dat = subset(all_files, sID < 100)
  gg_example_values = ggplot(ex_dat) +
    geom_line(aes(TrialID, values_l_rec), color="red") +
    geom_line(aes(TrialID, values_r_rec), color="blue") +
    # geom_vline(xintercept = which(ex_dat$switch_trial)) +
    # geom_point(aes(TrialID, reward - 0.5)) +
    geom_point(aes(TrialID, 0.5, color = choice_left, size = factor(reward))) +
    geom_point(aes(TrialID, as.numeric(as.character(factor(switch_trial, labels = c(-5, 0.5))))), shape = 8, size = 3, color = "darkgreen") +
    scale_size_manual(values = c(0.5, 1)) +
    scale_color_manual(values = c("blue", "red")) +
    coord_cartesian(x = c(1, 80), y = c(0, 1)) +
    labs(x = "Trial", y = "Values (left: red; right: blue)") +
    theme(legend.position = 'none') +
    facet_wrap(~ sID, ncol = 2)
  
  # Plot action probs for one example session
  gg_example_probs = ggplot(ex_dat) +
    geom_line(aes(TrialID, p_action_l_rec), color="red") +
    geom_line(aes(TrialID, p_action_r_rec), color="blue") +
    # geom_vline(xintercept = which(ex_dat$switch_trial)) +
    # geom_point(aes(TrialID, reward - 0.5)) +
    geom_point(aes(TrialID, 0.5, color = choice_left, size = factor(reward))) +
    geom_point(aes(TrialID, as.numeric(as.character(factor(switch_trial, labels = c(-5, 0.5))))), shape = 8, size = 3, color = "darkgreen") +
    scale_size_manual(values = c(0.1, 1)) +
    scale_color_manual(values = c("blue", "red")) +
    coord_cartesian(x = c(1, 100), y = c(0, 1)) +
    labs(x = "Trial", y = "Action probs. (left: red; right: blue)") +
    theme(legend.position = 'none') +
    facet_wrap(~ sID, ncol = 2)
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_example_values.png"), gg_example_values, width = 12, height = 8)
    ggsave(file.path(plot_dir, "gg_example_probs.png"), gg_example_probs, width = 12, height = 8)
  }
  
  gg_switch_values = ggplot(all_files, aes(trialsinceswitch, values_l_rec, color = reward_port, group = reward_port)) +
    stat_summary(fun.data = "mean_se", geom = "smooth") +
    coord_cartesian(x = c(-3, 5))
  
  gg_trial_values = ggplot(all_files) +
    stat_summary(aes(TrialID, values_l_rec), color = "red", fun.data = "mean_se", geom = "smooth") +
    stat_summary(aes(TrialID, values_r_rec), color = "blue", fun.data = "mean_se", geom = "smooth") +
    facet_grid(~ rewardversion)
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "/gg_trial_values.png"), gg_trial_values)
    ggsave(file.path(plot_dir, "/gg_switch_values.png"), gg_switch_values)
  }
}

genrec_plots = function() {
  gen_plot = subset(genrec, learning_style != "estimate-switch")
  gg_genrec_a = ggplot(gen_plot, aes(alpha_gen, alpha_rec, color=fit_par, shape=fit_par)) +
    geom_abline(slope=1, linetype="dotted") +
    geom_point() +
    scale_shape_manual(values=1:length(unique(gen_plot$fit_par))) +
    facet_grid(learning_style ~ method)
  gg_genrec_b = gg_genrec_a + aes(beta_gen, beta_rec)
  gg_genrec_e = gg_genrec_a + aes(epsilon_gen, epsilon_rec)
  gg_genrec_p = gg_genrec_a + aes(perseverance_gen, perseverance_rec)
  gg_genrec_d = gg_genrec_a + aes(decay_gen, decay_rec)
  gg_genrec_wr = gg_genrec_a + aes(w_reward_gen, w_reward_rec)
  gg_genrec_we = gg_genrec_a + aes(w_explore_gen, w_explore_rec)
  gg_genrec_wn = gg_genrec_a + aes(w_noreward_gen, w_noreward_gen)
  
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "/gg_genrec_a.png"), gg_genrec_a, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_b.png"), gg_genrec_b, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_e.png"), gg_genrec_e, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_p.png"), gg_genrec_p, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_d.png"), gg_genrec_d, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_wr.png"), gg_genrec_wr, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_we.png"), gg_genrec_we, width = 11, height = 8)
    ggsave(file.path(plot_dir, "/gg_genrec_wn.png"), gg_genrec_wn, width = 11, height = 8)
  }
}

paper_plots = function() {
  # Response times
  gg_RT = ggplot(all_files, aes(TrialID, RT, fill = age_group)) +
    # geom_point() +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar", position = position_dodge(width = 0.9)) +
    facet_grid(~ age_group)
  
  gg_RTt = ggplot(all_files_sum2, aes(trialsinceswitch, RT, fill = age_group)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar", position = position_dodge(width = 0.9)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange", position = position_dodge(width = 0.9)) +
    # geom_point(alpha = .3, position = "jitter") +
    coord_cartesian(x = c(-3, 5)) +
    facet_grid(~ age_group)
  
  # Missed trials -> did not miss trials
  
  # ACC over trials
  gg_ACC = ggplot(all_files, aes(TrialID, 100 * as.numeric(ACC), fill = age_group)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
    labs(y = "% correct", fill = "Age group") +
    facet_grid(~ age_group)
  if (data == "fitted_sim") {
    gg_ACC = gg_ACC + facet_grid(learning_style ~ method)
  }
  
  # ACC over blocks
  gg_ACC_blocks = ggplot(subset(all_files, !is.na(choice_left)),
                         aes(trialsinceswitch, 100 * choice_left, color = reward_port, group = reward_port, shape = age_group)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "line") +
    geom_vline(xintercept = 0, linetype = "dotted") +
    coord_cartesian(x = c(-3, 7)) +
    labs(x = "Trials since switch", y = "% choice left", color = "Reward port", shape = "Age group") +
    facet_grid(~ age_group)
  if (data == "fitted_sim") {
    gg_ACC_blocks = gg_ACC_blocks + facet_grid(learning_style ~ method)
  }
  
  # Rewards over trials
  gg_rewards = ggplot(all_files, aes(TrialID, 100 * reward, fill = age_group)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
    labs(y = "% reward received", color = "Age group") +
    facet_grid(~ age_group)
  if (data != "human") {
    gg_rewards = gg_rewards + facet_grid(learning_style ~ method)
  }
  
  gg_rewards2 = ggplot(subset(all_files_sum, !is.na(choice_12_back)),
                       aes(outcome_12_back, 100 * choice, fill = choice_12_back, group = choice_12_back, shape = age_group)) +
    stat_summary(fun.data = mean_cl_normal, geom = "bar", fun.args = list(mult = 1), position = "dodge") +
    stat_summary(fun.data = mean_cl_normal, geom = "pointrange", fun.args = list(mult = 1), position = position_dodge(0.9)) +
    coord_cartesian(y = c(0, 100)) +
    labs(x = "Reward history (1 trial back, 2 trials back)", y = "% left choice", fill = "Previous two choices", shape = "Age group") +
    facet_grid(~ age_group)
  if (data == "fitted_sim") {
    gg_rewards2 = gg_rewards2 + facet_grid(learning_style ~ method)
  }
  
  # Simple win-stay loose-shift
  gg_wsls = ggplot(wsls, aes(reward == 1, stay, color = reward == 1, shape = age_group)) +
    stat_summary(fun.data = "mean_se", geom = "pointrange") +
    geom_point(alpha = 0.5, position = "jitter") +
    labs(x = "Reward", y = "% stay trials", color = "Reward", shape = "Age group") +
    facet_grid(~ age_group)
  if (data == "fitted_sim") {
    gg_wsls = gg_wsls + facet_grid(learning_style ~ method)
  }
  
  # Response times
  gg_RT_blocks = gg_ACC_blocks + aes(y = RT) + coord_cartesian(x = c(-3, 7), y = c(250, 550))
  
  if (gg_save) {
    if (data %in% c("human", "fitted_human")) {
      ggsave(file.path(plot_dir, "/gg_RT.png"), gg_RT)
      ggsave(file.path(plot_dir, "/gg_RTt.png"), gg_RTt)
      ggsave(file.path(plot_dir, "/gg_ACC.png"), gg_ACC)
      ggsave(file.path(plot_dir, "/gg_RT_blocks.png"), gg_RT_blocks, width = 8, height = 3)
    }
    ggsave(file.path(plot_dir, "/gg_wsls.png"), gg_wsls)
    ggsave(file.path(plot_dir, "/gg_ACC_blocks.png"), gg_ACC_blocks, width = 8, height = 3)
    ggsave(file.path(plot_dir, "/gg_rewards.png"), gg_rewards)
    ggsave(file.path(plot_dir, "/gg_rewards2.png"), gg_rewards2, width = 8, height = 3)
  }
}