library("plyr")
library("ggplot2")

## algorithm under analysis
aua <- 'epsilon_greedy_annealing_results' 

results <- read.csv(paste0("../results/",aua,".csv"), header = TRUE)
#results <- read.csv("julia/algorithms/epsilon_greedy/annealing_results.csv", header = FALSE)
#names(results) <- c("Sim", "T", "ChosenArm", "Reward", "CumulativeReward")

# Plot average reward as a function of time.
avg.reward.over.time <- ddply(results, .(times), function(x) {c(mean.reward=mean(x$rewards))})
p <- ggplot(avg.reward.over.time, aes(x = times, y = mean.reward)) + geom_line() + ylim(0, 1) 
p <- p + xlab("Time") + ylab("Average Reward") 
p <- p + ggtitle("Performance of the Annealing Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_average_reward.pdf"), plot=p)

# Plot frequency of selecting correct arm as a function of time.
# In this instance, 5 is the correct arm.
right.arm.frequency <- ddply(results, .(times), function(x) c(frequency=mean(x$chosen_arm == x$best_arm)))
p <- ggplot(right.arm.frequency, aes(x = times, y = frequency)) +  geom_line()
p <- p + ylim(0, 1) + xlab("Time") +ylab("Probability of Selecting Best Arm") 
p <- p + ggtitle("Accuracy of the Annealing Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_average_accuracy.pdf"), plot=p)

# Plot variance of chosen arms as a function of time.
chosen.arm.variance <- ddply(results, .(times), function(c) c(variance=var(c$chosen_arm)))
p <- p + ggplot(chose.arm.variance, aes(x = times, y = variance)) + geom_line() 
p <- p + xlab("Time") + ylab("Variance of Chosen Arm") 
p <- p + ggtitle("Variability of the Annealing Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_chosen_arm_variance.pdf"), plot=p)

# Plot cumulative reward as a function of time.
cumulative.reward <- ddply(results, .(times), function(x) c(cum.reward=mean(x$cumulative_rewards)))
p <- ggplot(cumulative.reward, aes(x = times, y = cum.reward)) +  geom_line()
p <- p + xlab("Time") + ylab("Cumulative Reward of Chosen Arm") 
p <- p + ggtitle("Cumulative Reward of the Annealing Epsilon Greedy Algorithm")
ggsave(filename=paste0("output/",aua,"_cumulative_reward.pdf"), plot=p)
