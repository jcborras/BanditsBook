library("plyr")
library("ggplot2")

## algorithm under analysis
aua <- 'epsilon_greedy_standard_results2' 

results <- read.csv(paste0("../results/",aua,".csv"), header = TRUE)
#names(results) <- c("Epsilon", "Sim", "T", "ChosenArm", "Reward", "CumulativeReward")
results <- transform(results, epsilon = factor(epsilon))

# Plot average reward as a function of time.
t <- system.time (
  avg.reward.over.time <- ddply(results, .(epsilon, times), function (df) {mean(df$rewards)}, .progress='text'))
p <- ggplot(avg.reward.over.time, aes(x = times, y = V1, group = epsilon, color = epsilon))
p <- p + geom_line() + ylim(0, 1) 
p <- p + xlab("Time") + ylab("Average Reward") 
p <- p + ggtitle("Performance of the Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_average_reward.pdf"), plot=p)

# Plot frequency of selecting correct arm as a function of time.
# In this instance, 5 is the correct arm.
t <- system.time (
  stats.right.arm.frequency <- ddply(results, .(epsilon, times), 
   function (x) {c(right.arm.frequency=mean(x$chosen_arm == x$best_arm))}, .progress='text'))
p <- ggplot(stats.right.arm.frequency, aes(x = times, y = right.arm.frequency, group = epsilon, color = epsilon))
p <- p + geom_line() + ylim(0, 1)
p <- p + xlab("Time") + ylab("Probability of Selecting Best Arm") 
p <- p + ggtitle("Accuracy of the Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_right_arm_frequency.pdf"), plot=p)

# Plot variance of chosen arms as a function of time.
t <- system.time (
  stats.chosen.arm.variance <- ddply(results, .(epsilon, times), 
    function (x) {c(chosen.arm.variance=var(x$chosen_arm))}, .progress='text'))
p <- ggplot(stats.chosen.arm.variance, aes(x = times, y = chosen.arm.variance, group = epsilon, color = epsilon))
p <- p + geom_line() 
p <- p + xlab("Time") + ylab("Variance of Chosen Arm") 
p <- p + ggtitle("Variability of the Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_chosen_arm_variance.pdf"), plot=p)

# Plot cumulative reward as a function of time.
t <- system.time (
  stats.cum.rewards <- ddply(results, .(epsilon, times), 
     function (x) {c(cum.reward=mean(x$cumulative_rewards))}, .progress='text'))
p <- ggplot(stats.cum.rewards, aes(x = times, y = cum.reward, group = epsilon, color = epsilon))
p <- p + geom_line() +  xlab("Time") +  ylab("Cumulative Reward of Chosen Arm") 
p <- p + ggtitle("Cumulative Reward of the Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_cumulative_reward.pdf"), plot=p)
