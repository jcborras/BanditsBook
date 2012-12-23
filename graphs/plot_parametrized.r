library("plyr")
library("ggplot2")

## algorithm under analysis
#aua <- 'epsilon_greedy_standard_results2'; parameter.name <- 'epsilon'
#aua <- 'softmax_standard_results2' ; parameter.name <- 'temperature'
#aua <- 'ucb2_results2' ; parameter.name <- 'alpha'
#aua <- 'exp3_results2' ; parameter.name <- 'exp3_gamma'
aua <- 'hedge_results2' ; parameter.name <- 'eta'

avg.reward.over.time <- function(x, p) {
  ddply(results, c(p, 'times'), function (x) c(avg.reward=mean(x$rewards)), .progress='text') }

right.arm.frequency.over.time <- function(x,p) {
  ddply(x, c(p, 'times'), function (x) {c(frequency=mean(x$chosen_arm == x$best_arm))}, .progress='text') }

arm.variance.over.time <- function(x,p) {
  ddply(x, c(p, 'times'), function (x) {c(arm.variance=var(x$chosen_arm))}, .progress='text') }

cum.reward.over.time <- function(x,p) {
  ddply(x, c(p, 'times'), function (x)  {c(cum.reward=mean(x$cumulative_rewards))}, .progress='text') }

# That's my problem with ggplot, I can't parametrize aesthetics
avg.reward.plot <- function(x, algorithm) {
  p <- ggplot(x, aes(x = times, y = variable, group = factor(parameter), color = factor(parameter)))
  p <- p + geom_line() + ylim(0, 1) 
  p <- p + xlab("Time") + ylab("Average Reward") 
  p + ggtitle(paste('Performance for', algorithm)) }

right.arm.frequency.plot <- function(x, algorithm) {
  p <- ggplot(x, aes(x = times, y = variable, group = factor(parameter), color = factor(parameter)))
  p <- p + geom_line() + ylim(0, 1)
  p <- p + xlab("Time") + ylab("Probability of Selecting Best Arm") 
  p + ggtitle(paste('Probability of optimal selection for', algorithm)) }

arm.variance.plot <- function(x, algorithm) {
  p <- ggplot(x, aes(x = times, y = variable, group = factor(parameter), color = factor(parameter)))
  p <- p + geom_line() + xlab("Time") + ylab("Variance of Chosen Arm")
  p + ggtitle(paste('Arm selection variability for', algorithm)) }

cum.reward.plot <- function(x, algorithm) {
  p <- ggplot(x, aes(x = times, y = variable, group = factor(parameter), color = factor(parameter)))
  p <- p + geom_line() +  xlab("Time") +  ylab("Cumulative Reward of Chosen Arm") 
  p + ggtitle(paste('Cumulative reward for', algorithm)) }

results <- read.csv(paste0("../results/",aua,".csv"), header = TRUE)

avg.reward <- avg.reward.over.time(results, parameter.name)
colnames(avg.reward) <- c('parameter', 'times', 'variable')
p1 <- avg.reward.plot(avg.reward, aua)
ggsave(filename=paste0("output/",aua,"_avg_reward.pdf"), plot=p1)

right.arm.frequency <- right.arm.frequency.over.time(results, parameter.name)
colnames(right.arm.frequency) <- c('parameter', 'times', 'variable')
p2 <- right.arm.frequency.plot(right.arm.frequency, aua)
ggsave(filename=paste0("output/",aua,"_right_selection_probability.pdf"), plot=p2)

arm.variance <- arm.variance.over.time(results, parameter.name)
colnames(arm.variance) <- c('parameter', 'times', 'variable')
p3 <- arm.variance.plot(arm.variance, aua)
ggsave(filename=paste0("output/",aua,"_arm_selection_variance.pdf"), plot=p3)

cum.reward <- cum.reward.over.time(results, parameter.name)
colnames(cum.reward) <- c('parameter', 'times', 'variable')
p4 <- cum.reward.plot(cum.reward, aua)
ggsave(filename=paste0("output/",aua,"_cumulative_reward.pdf"), plot=p4)



