#4choice/reversal files 

# Setting up parameters
data_dir = ('C:/Users/prokofiev/Desktop/4choice_rawdata') # get 4choice data
plot(file_4ch$ResponseTime)
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
summary(all_files)
head(all_files)

# RTs
ggplot(all_files, aes(TrialNumPhase, ResponseTime)) +
  stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "bar") +
  stat_summary(fun.data = mean_cl_normal, fun.args = list(mult = 1), geom = "pointrange") +
  geom_point(alpha = .4) +
  labs(x = "Trial", y = "Response Time (sec)") +
  facet_grid(~ Reversal)

# Do kids reach first criterion?
all_files_sum = ddply(all_files, .(pID, Age, Gender, Reversal), summarize,
                      ReversalTrial = max(TrialNumPhase))

ggplot(all_files_sum, aes(pID, ReversalTrial, color = Reversal)) +
  geom_point(size=5) +
  labs(x = "Participant ID", y = "# trials to / after reversal")

# Which boxes do they pick?
ggplot(all_files, aes(TrialNumPhase, Correct)) +
  stat_summary(fun.y = mean, geom = "bar") +
  geom_point(alpha = .3) +
  labs(x = "Trial") +
  facet_grid(~ Reversal)
