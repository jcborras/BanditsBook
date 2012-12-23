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
  return [chosen,reward]

def run_simulation(alg,arms,hor):
  times = range(1,hor+1)
  x = [ run_trial(alg, arms) for i in times ]
  chosen_arm = [ z[0] for z in x ]
  rewards = [ z[1] for z in x ]
  cumulative_rewards = list(cumsum(rewards))
  assert len(times)==len(chosen_arm)==len(rewards)==len(cumulative_rewards)
  return [times, chosen_arm, rewards, cumulative_rewards]

def run_montecarlo(alg, arms, n_sims, hor):
    alg.initialize(len(arms))
    L = n_sims*hor
    sim_num, times, chosen_arms, rewards, cumulative_rewards = L*[0.0], L*[0.0], L*[0.0], L*[0.0], L*[0.0]
    for i in range(n_sims):
        alg.initialize(len(arms))
        p = i*hor
        q = p+hor
        sim_num[p:q] = hor*[i+1]
        times[p:q],chosen_arms[p:q],rewards[p:q],cumulative_rewards[p:q] = run_simulation(alg, arms, hor)
    return [sim_num, times, chosen_arms, rewards, cumulative_rewards]
    
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

test_run = run_montecarlo
#test_run = test_algorithm

def csf(l):
    """Comma separated fields"""
    return reduce(lambda a,b: str(a)+','+str(b), l)

def results_to_file(f, r, fieldnames=None):
    if fieldnames is not None:
        [f.write(s) for s in [csf(fieldnames), '\n']]
    if r is not None:
        w = len(r)
        print w
        for l in range(len(r[0])):
            f.write(csf([r[j][l] for j in range(w)]) + '\n')

class BaseTestCase(TestCase):
    N_SIMS, HORIZON = 5000, 250

    def setUp(self):
        seed(1)
        self.means = [0.1, 0.1, 0.1, 0.1, 0.9]
        self.n_arms = len(self.means)
        shuffle(self.means)
        self.arms = map(lambda (mu): BernoulliArm(mu), self.means)
        print("Means shuffled " + str(self.means))
        print("Best arm is " + str(ind_max(self.means)))

    def tearDown(self):
        pass


class EpsilonGreedyTest(BaseTestCase):

    def montecarlo_alg(self, epsilon):
        seed(1)
        alg = EpsilonGreedy(epsilon, [], [])
        alg.initialize(self.n_arms)
        return alg
    
    def test_montecarlo(self):
        N_SIMS, HORIZON, EPSILON = 20, 500, random()
        results = run_montecarlo(self.montecarlo_alg(EPSILON), self.arms, N_SIMS, HORIZON)
        results_orig = test_algorithm(self.montecarlo_alg(EPSILON), self.arms, N_SIMS, HORIZON)
        [self.assertEqual(results[i], results_orig[i]) for i in range(1,5)]

    def test_standard(self):
        f = open(RESULTS_DIR+"epsilon_greedy_standard_results2.csv", "w")
        results_to_file(f, None, ['epsilon','best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards'])

        for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results = test_run(EpsilonGreedy(epsilon, [], []), self.arms, self.N_SIMS, self.HORIZON)
            results_to_file(f, [len(results[0])*[epsilon]] + [len(results[0])*[ind_max(self.means)]] + results)
        f.close()

    def test_annealing(self):
        results = test_run(AnnealingEpsilonGreedy([], []), self.arms, self.N_SIMS, self.HORIZON)

        f = open(RESULTS_DIR+"epsilon_greedy_annealing_results.csv", "w")
        head = ['best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, [len(results[0])*[ind_max(self.means)]]+results, head)
        f.close()


class SoftmaxTest(BaseTestCase):
    def test_standard(self):
        f = open(RESULTS_DIR+"softmax_standard_results2.csv", "w")
        headers = ['temperature', 'best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, None, headers)
        for temperature in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results = test_run(Softmax(temperature, [], []), self.arms, self.N_SIMS, self.HORIZON)
            results_to_file(f, [len(results[0])*[temperature]] + [len(results[0])*[ind_max(self.means)]] + results)
        f.close()

    def test_annealing(self):
        results = test_run(AnnealingSoftmax([], []), self.arms, self.N_SIMS, self.HORIZON)

        f = open(RESULTS_DIR+"softmax_annealing_results2.csv", "w")
        head = ['best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, [len(results[0])*[ind_max(self.means)]]+results, head)
        f.close()


class UCBTest(BaseTestCase):
    def test_UCB1(self):
        results = test_run( UCB1([], []), self.arms, self.N_SIMS, self.HORIZON)

        f = open(RESULTS_DIR+"ucb1_results2.csv", "w")
        head = ['best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, [len(results[0])*[ind_max(self.means)]]+results, head)
        f.close()

    def test_UCB2(self):
        f = open(RESULTS_DIR+"ucb2_results2.csv", "w")
        headers = ['alpha', 'best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, None, headers)
        for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
            results = test_run(UCB2(alpha, [], []), self.arms, self.N_SIMS, self.HORIZON)
            results_to_file(f, [len(results[0])*[alpha]] + [len(results[0])*[ind_max(self.means)]] + results)
        f.close()


class Exp3Test(BaseTestCase):
    def test_Exp3(self):
        f = open(RESULTS_DIR+"exp3_results2.csv", "w")
        headers = ['exp3_gamma', 'best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, None, headers)
        for exp3_gamma in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results = test_run( Exp3(exp3_gamma, []), self.arms, self.N_SIMS, self.HORIZON)
            results_to_file(f, [len(results[0])*[exp3_gamma]] + [len(results[0])*[ind_max(self.means)]] + results)
        f.close()


class HedgeTest(BaseTestCase):
    def test_hedge(self):
        return
        f = open(RESULTS_DIR+"hedge_results2.csv", "w")
        headers = ['eta', 'best_arm', 'num_sim', 'times', 'chosen_arm', 'rewards','cumulative_rewards']
        results_to_file(f, None, headers)
        for eta in [0.1, 0.2, 0.3, 0.4, 0.5]:
            results = test_run( Hedge(eta, [], []), self.arms, 5000, 250)
            results_to_file(f, [len(results[0])*[eta]] + [len(results[0])*[ind_max(self.means)]] + results)
        f.close()


if __name__ == '__main__':
  main()
