#!/usr/bin/env python

from random import random, randrange
from math import log

def ind_max(x):
  m = max(x)
  return x.index(m)

class AnnealingEpsilonGreedy(object):
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values

  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]

  def arm_selection(self):
    if random() > self.epsilon:
      return ind_max(self.values)
    else:
      return randrange(len(self.values))
    
  def select_arm(self):
    t = sum(self.counts) + 1
    self.epsilon = 1 / log(t + 0.0000001)
    return self.arm_selection()
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value


class EpsilonGreedy(AnnealingEpsilonGreedy):
  def __init__(self, epsilon, counts, values):
    self.epsilon = epsilon
    super(EpsilonGreedy,self).__init__(counts, values)

  def select_arm(self):
    return self.arm_selection()
  
