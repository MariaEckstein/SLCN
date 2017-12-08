
read_in_files = function() {
  all_files = data.frame()
  filenames = list.files(data_dir, pattern = ifelse(data == "human", "*.mat", "PS_[0-9]*.csv"))
  filename = filenames[2] # debugging
  
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
  
  return(all_files)
}


model_plots = function() {
  # Plot values for one example session
  ex_dat = subset(all_files, sID < 11)
  gg_example_values = ggplot(ex_dat) +
    geom_line(aes(TrialID, values_l), color="red") +
    geom_line(aes(TrialID, values_r), color="blue") +
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
    geom_line(aes(TrialID, p_action_l), color="red") +
    geom_line(aes(TrialID, p_action_r), color="blue") +
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
    ggsave(file.path(plot_dir, paste("gg_example_values", learning_style, method, ".png", sep = "")), gg_example_values, width = 12, height = 8)
    ggsave(file.path(plot_dir, paste("gg_example_probs", learning_style, method, ".png", sep = "")), gg_example_probs, width = 12, height = 8)
  }
  
  gg_switch_values1 = ggplot(all_files) +
    stat_summary(aes(trialsinceswitch, values_l), color = "red", fun.data = "mean_se", geom = "smooth") +
    stat_summary(aes(trialsinceswitch, values_r), color = "blue", fun.data = "mean_se", geom = "smooth") +
    coord_cartesian(x = c(-3, 5))
  
  gg_switch_values2 = ggplot(all_files, aes(trialsinceswitch, values_l, color = reward_port, group = reward_port)) +
    stat_summary(fun.data = "mean_se", geom = "smooth") +
    coord_cartesian(x = c(-3, 5))
  
  gg_trial_values = ggplot(all_files) +
    stat_summary(aes(TrialID, values_l), color = "red", fun.data = "mean_se", geom = "smooth") +
    stat_summary(aes(TrialID, values_r), color = "blue", fun.data = "mean_se", geom = "smooth") +
    facet_grid(~ rewardversion)
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_trial_values", learning_style, method, ".png", sep = ""), gg_trial_values)
    ggsave(file.path(plot_dir, "gg_switch_values1", learning_style, method, ".png", sep = ""), gg_switch_values1)
    ggsave(file.path(plot_dir, "gg_switch_values2", learning_style, method, ".png", sep = ""), gg_switch_values2)
  }
}

genrec_plots = function() {
  genrec = read.csv(file.path(data_dir, "genrec.csv"))
  genrec$X = NULL
  gg_genrec_a = ggplot(genrec, aes(alpha_gen, alpha_rec)) +  # , color=epsilon_gen
    geom_point()
  gg_genrec_b = gg_genrec_a + aes(beta_gen, beta_rec)
  gg_genrec_e = gg_genrec_a + aes(epsilon_gen, epsilon_rec)
  gg_genrec_p = gg_genrec_a + aes(perseverance_gen, perseverance_rec)
  gg_genrec_d = gg_genrec_a + aes(decay_gen, decay_rec)
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_genrec_a", learning_style, method, ".png", sep = ""), gg_genrec_a)
    ggsave(file.path(plot_dir, "gg_genrec_b", learning_style, method, ".png", sep = ""), gg_genrec_b)
    ggsave(file.path(plot_dir, "gg_genrec_e", learning_style, method, ".png", sep = ""), gg_genrec_e)
    ggsave(file.path(plot_dir, "gg_genrec_p", learning_style, method, ".png", sep = ""), gg_genrec_p)
    ggsave(file.path(plot_dir, "gg_genrec_d", learning_style, method, ".png", sep = ""), gg_genrec_d)
  }
}

paper_plots = function() {
  # Response times
  gg_RT = ggplot(all_files, aes(TrialID, RT, color = sID)) +
    geom_point() +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
    facet_grid(~ age_group)
  
  gg_RTt = ggplot(all_files_sum2, aes(trialsinceswitch, RT, color = sID)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
    coord_cartesian(x = c(-3, 5)) +
    geom_point(alpha = .3, position = "jitter") +
    facet_grid(~ age_group)
  
  # Missed trials -> did not miss trials
  
  
  # ACC over trials
  gg_ACC = ggplot(all_files, aes(TrialID, ACC)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
    facet_grid(~ rewardversion) +
    facet_grid(~ age_group)
  
  # ACC over blocks
  gg_ACC_blocks = ggplot(subset(all_files, !is.na(choice_left)),
                         aes(trialsinceswitch, 100 * choice_left, color = reward_port, group = reward_port)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "line") +
    coord_cartesian(x = c(-3, 7)) +
    labs(x = "Trials since switch", y = "% choice left", color = "Reward port") +
    facet_grid(~ age_group)
  # gg_ACC_blocks + facet_wrap(~ sID)
  
  # Rewards over trials
  gg_rewards = ggplot(all_files, aes(TrialID, reward)) +
    stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
    geom_point() + 
    facet_grid(~ rewardversion) +
    facet_grid(~ age_group)
  
  gg_rewards2 = ggplot(subset(all_files_sum, !is.na(choice_12_back)),
                       aes(outcome_12_back, 100 * choice, fill = choice_12_back, group = choice_12_back)) +
    stat_summary(fun.data = mean_cl_normal, geom = "bar", fun.args = list(mult = 1), position = "dodge") +
    stat_summary(fun.data = mean_cl_normal, geom = "pointrange", fun.args = list(mult = 1), position = position_dodge(0.9)) +
    coord_cartesian(y = c(0, 100)) +
    labs(x = "Reward history (1 trial back, 2 trials back)", y = "% left choice", fill = "Previous two choices") +
    facet_grid(~ age_group)
  # gg_rewards2 + facet_wrap(~ sID)
  
  # Simple win-stay loose-shift
  gg_wsls = ggplot(wsls, aes(reward, stay, color = reward)) +
    stat_summary(fun.data = "mean_se", geom = "pointrange") +
    geom_point(alpha = 0.5, position = "jitter") +
    facet_grid(~ age_group)
  
  if (gg_save) {
    if (data %in% c("human", "fitted_human")) {
      ggsave(paste(plot_dir, "/gg_RT.png", sep = ""), gg_RT)
      ggsave(paste(plot_dir, "/gg_RTt.png", sep = ""), gg_RTt)
      ggsave(paste(plot_dir, "/gg_ACC.png", sep = ""), gg_ACC)
    }
    ggsave(paste(plot_dir, "/gg_wsls.png", sep = ""), gg_wsls)
    ggsave(paste(plot_dir, "/gg_ACC_blocks.png", sep = ""), gg_ACC_blocks)
    ggsave(paste(plot_dir, "/gg_rewards.png", sep = ""), gg_rewards)
    ggsave(paste(plot_dir, "/gg_rewards2.png", sep = ""), gg_rewards2)
  }
}