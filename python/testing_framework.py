#!/usr/bin/env python

from random import random, seed, shuffle
from unittest import TestCase, main

from algorithms.epsilon_greedy import ind_max, EpsilonGreedy, AnnealingEpsilonGreedy
from algorithms.exp3 import Exp3
from algorithms.hedge import Hedge
from algorithms.softmax import AnnealingSoftmax, Softmax
from algorithms.ucb import UCB1, UCB2

from arms import BernoulliArm

RESULTS_DIR = '../results/'

def cumsum(it):
    total = 0
    for x in it:
        total += x
        yield total
        
def run_trial(alg, arms):
  chosen = alg.select_arm()
  reward = arms[chosen].draw()
  alg.update(chosen, reward)
  return {'chosen_arm':chosen, 'reward':reward}

def run_sumulation(alg,arms,hor):
  times = range(hor)
  x = [ run_trial(alg, arms) for i in times ]
  chosen_arm = [ z['chosen_arm'] for z in x ]
  rewards = [ z['reward'] for z in x ]
  cumulative_rewards = list(cumsum([ z['reward'] for z in x ]))
  assert len(times)==len(chosen_arm)==len(rewards)
  return {'times':times, 'chosen_arm':chosen_arm, 'rewards':rewards, 'cumulative_rewards':cumulative_rewards}

def f(d,n):
  d['num_sim'] = len(d['times'])*[n]
  return d
  
def run_montecarlo(alg, arms, n_sims, hor):
  alg.initialize(len(arms))
  return [f(run_sumulation(alg, arms, hor),i) for i in range(n_sims)]
          
def test_algorithm(algo, arms, num_sims, horizon):
  chosen_arms = [0.0 for i in range(num_sims * horizon)]
  rewards = [0.0 for i in range(num_sims * horizon)]
  cumulative_rewards = [0.0 for i in range(num_sims * horizon)]
  sim_nums = [0.0 for i in range(num_sims * horizon)]
  times = [0.0 for i in range(num_sims * horizon)]
  
  for sim in range(num_sims):
    sim = sim + 1
    algo.initialize(len(arms))
    
    for t in range(horizon):
      t = t + 1
      index = (sim - 1) * horizon + t - 1
      sim_nums[index] = sim
      times[index] = t
      
      chosen_arm = algo.select_arm()
      chosen_arms[index] = chosen_arm
      
      reward = arms[chosen_arms[index]].draw()
      rewards[index] = reward
      
      if t == 1:
        cumulative_rewards[index] = reward
      else:
        cumulative_rewards[index] = cumulative_rewards[index - 1] + reward
      
      algo.update(chosen_arm, reward)
  
  return [sim_nums, times, chosen_arms, rewards, cumulative_rewards]

class EpsilonGreedyTest(TestCase):
  def setUp(self):
    seed(1)
    self.means = [0.1, 0.1, 0.1, 0.1, 0.9]
    self.n_arms = len(self.means)
    shuffle(self.means)
    self.arms = map(lambda (mu): BernoulliArm(mu), self.means)
    print("Best arm is " + str(ind_max(self.means)))

  def tearDown(self):
    return
 
  def test_montecarlo(self):
    N_SIMS, HORIZON, EPSILON = 1, 10000, random()
    seed(1)
    algo = EpsilonGreedy(EPSILON, [], [])
    algo.initialize(self.n_arms)
    results = run_montecarlo(algo, self.arms, N_SIMS, HORIZON)
    seed(1)
    algo2 = EpsilonGreedy(EPSILON, [], [])
    algo2.initialize(self.n_arms)
    results2 = test_algorithm(algo2, self.arms, N_SIMS, HORIZON)
    self.assertEqual(results[0]['chosen_arm'], results2[2])
    self.assertEqual(results[0]['cumulative_rewards'], results2[4])
    
  def test_standard(self):
    f = open(RESULTS_DIR+"epsilon_greedy_standard_results.csv", "w")

    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
      algo = EpsilonGreedy(epsilon, [], [])
      algo.initialize(self.n_arms)
      results = test_algorithm(algo, self.arms, 5000, 250)
      for i in range(len(results[0])):
        f.write(str(epsilon) + ",")
        f.write(",".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

  def test_annealing(self):
    my_algo = AnnealingEpsilonGreedy([], [])
    my_algo.initialize(self.n_arms)
    results = test_algorithm(my_algo, self.arms, 5000, 250)

    f = open(RESULTS_DIR+"epsilon_greedy_annealing_results.tsv", "w")

    for i in range(len(results[0])):
      f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()


class SoftmaxTest(EpsilonGreedyTest):
  def test_standard(self):
    f = open(RESULTS_DIR+"softmax_standard_results.tsv", "w")
    for temperature in [0.1, 0.2, 0.3, 0.4, 0.5]:
      algo = Softmax(temperature, [], [])
      algo.initialize(self.n_arms)
      results = test_algorithm(algo, self.arms, 5000, 250)
      for i in range(len(results[0])):
        f.write(str(temperature) + "\t")
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

  def test_annealing(self):
    algo = AnnealingSoftmax([], [])
    algo.initialize(self.n_arms)
    results = test_algorithm(algo, self.arms, 5000, 250)

    f = open(RESULTS_DIR+"softmax_annealing_results.tsv", "w")
    for i in range(len(results[0])):
      f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

class UCBTest(TestCase):
  def test_UCB1(self):
    seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(means)
    shuffle(means)
    arms = map(lambda (mu): BernoulliArm(mu), means)
    print("Best arm is " + str(ind_max(means)))

    algo = UCB1([], [])
    algo.initialize(n_arms)
    results = test_algorithm(algo, arms, 5000, 250)

    f = open(RESULTS_DIR+"ucb1_results.tsv", "w")

    for i in range(len(results[0])):
      f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

  def test_UCB2(self):
    seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(means)
    shuffle(means)
    arms = map(lambda (mu): BernoulliArm(mu), means)
    print("Best arm is " + str(ind_max(means)))

    for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
      algo = UCB2(alpha, [], [])
      algo.initialize(n_arms)
      results = test_algorithm(algo, arms, 5000, 250)

      f = open(RESULTS_DIR+"ucb2_results_%s.tsv" % alpha, "w")

      for i in range(len(results[0])):
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]))
        f.write("\t%s\n" % alpha)

      f.close()

class Exp3Test(TestCase):
  def test_Exp3(self):
    seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(means)
    shuffle(means)
    arms = map(lambda (mu): BernoulliArm(mu), means)
    print("Best arm is " + str(ind_max(means)))

    f = open(RESULTS_DIR+"exp3_results.tsv", "w")
    for exp3_gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
      algo = Exp3(exp3_gamma, [])
      algo.initialize(n_arms)
      results = test_algorithm(algo, arms, 5000, 250)
      for i in range(len(results[0])):
        f.write(str(exp3_gamma) + "\t")
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

class HedgeTest(TestCase):
  def test_hedge(self):
    seed(1)
    means = [0.1, 0.1, 0.1, 0.1, 0.9]
    n_arms = len(means)
    shuffle(means)
    arms = map(lambda (mu): BernoulliArm(mu), means)
    print("Best arm is " + str(ind_max(means)))

    f = open(RESULTS_DIR+"hedge_results.tsv", "w")

    for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
      algo = Hedge(eta, [], [])
      algo.initialize(n_arms)
      results = test_algorithm(algo, arms, 5000, 250)
      for i in range(len(results[0])):
        f.write(str(temperature) + "\t")
        f.write("\t".join([str(results[j][i]) for j in range(len(results))]) + "\n")
    f.close()

if __name__ == '__main__':
  main()
