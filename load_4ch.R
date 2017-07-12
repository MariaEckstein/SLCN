#4choice/reversal files 

# Setting up parameters
data_dir = 'C:/Users/maria/Desktop/SLCN/reversal4choicegame/logs'
# plot(file_4ch$ResponseTime)
filenames=list.files(data_dir, pattern = "*.txt")
library("ggplot2"); theme_set(theme_bw())
library("plyr")

# Reading in all data files
all_files = data.frame()

for(filename in filenames) {
  file=read.table(file.path(data_dir, filename), header = T)
  all_files = rbind(all_files, file)
}

# Summary statistics
all_files$pID = factor(all_files$pID)
all_files$Reversal = factor(all_files$Reversal, levels = c(0, 1), labels = c("beforeReversal", "afterReversal"))
all_files$Choice = factor(all_files$Choice)
all_files$age_bin = 7
for (age_bin in seq(9, 30, 2)) {
  all_files$age_bin[all_files$Age >= age_bin] = age_bin
}
summary(all_files)
head(all_files)

# Do kids reach first criterion?
all_files_sum = ddply(all_files, .(pID, Age, age_bin, Gender, Reversal), summarize,
                      ReversalTrial = max(TrialNumPhase))
# Individual
ggplot(all_files_sum, aes(pID, ReversalTrial, color = Reversal)) +
  geom_point(size = 5) +
  coord_cartesian(y = c(0, 30)) +
  labs(x = "Participant ID", y = "# trials to / after reversal")
# Summary
ggplot(all_files_sum, aes(age_bin, ReversalTrial, color = Reversal)) +
  stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
  coord_cartesian(y = c(0, 30)) +
  labs(x = "Age", y = "# trials to / after reversal")

# Which boxes do they pick?
ggplot(all_files, aes(TrialNumPhase, Correct)) +
  stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
  labs(x = "Trial") +
  coord_cartesian(x = c(1, 9)) +
  facet_grid(~ Reversal)

# RTs
all_files$ResponseTime[all_files$ResponseTime > 50] = NA
ggplot(all_files, aes(TrialNumPhase, ResponseTime)) +
  stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
  stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
  coord_cartesian(x = c(1, 9), y = c(0, 50)) +
  geom_point(alpha = .4) +
  labs(x = "Trial", y = "Response Time (sec)") +
  facet_grid(~ Reversal)
