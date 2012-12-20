library("plyr")
library("ggplot2")

aua <- 'epsilon_greedy_standard_results2' ## algorithm under analysis
results <- read.csv(paste0("../results/",aua,".csv"), header = FALSE)
names(results) <- c("Epsilon", "Sim", "T", "ChosenArm", "Reward", "CumulativeReward")
results <- transform(results, Epsilon = factor(Epsilon))

# Plot average reward as a function of time.
avg.reward.over.time <- ddply(results, c("Epsilon", "T"), function (df) {mean(df$Reward)})
p <- ggplot(avg.reward.over.time, aes(x = T, y = V1, group = Epsilon, color = Epsilon))
p <- p + geom_line() + ylim(0, 1) 
p <- p + xlab("Time") + ylab("Average Reward") 
p <- p + ggtitle("Performance of the Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_average_reward.pdf"), plot=p)

# Plot frequency of selecting correct arm as a function of time.
# In this instance, 5 is the correct arm.
stats <- ddply(results,
               c("Epsilon", "T"),
               function (df) {mean(df$ChosenArm == 5)})
ggplot(stats, aes(x = T, y = V1, group = Epsilon, color = Epsilon)) +
  geom_line() +
  ylim(0, 1) +
  xlab("Time") +
  ylab("Probability of Selecting Best Arm") +
  ggtitle("Accuracy of the Epsilon Greedy Algorithm")
ggsave("r/graphs/standard_epsilon_greedy_average_accuracy.pdf")

# Plot variance of chosen arms as a function of time.
stats <- ddply(results,
               c("Epsilon", "T"),
               function (df) {var(df$ChosenArm)})
ggplot(stats, aes(x = T, y = V1, group = Epsilon, color = Epsilon)) +
  geom_line() +
  xlab("Time") +
  ylab("Variance of Chosen Arm") +
  ggtitle("Variability of the Epsilon Greedy Algorithm")
ggsave("r/graphs/standard_epsilon_greedy_variance_choices.pdf")

# Plot cumulative reward as a function of time.
stats <- ddply(results,
               c("Epsilon", "T"),
               function (df) {mean(df$CumulativeReward)})
ggplot(stats, aes(x = T, y = V1, group = Epsilon, color = Epsilon)) +
  geom_line() +
  xlab("Time") +
  ylab("Cumulative Reward of Chosen Arm") +
  ggtitle("Cumulative Reward of the Epsilon Greedy Algorithm")
ggsave("r/graphs/standard_epsilon_greedy_cumulative_reward.pdf")
