from math import exp
from random import random

def categorical_draw(probs):
  z = random()
  cum_prob = 0.0
  for i in range(len(probs)):
    prob = probs[i]
    cum_prob += prob
    if cum_prob > z:
      return i
  
  return len(probs) - 1

class Hedge(object):
  def __init__(self, eta, counts, values):
    self.eta = eta
    self.counts = counts
    self.values = values
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return
  
  def select_arm(self):
    zz = [exp(v / self.eta) for v in self.values]
    z = sum(zz)
    probs = [ i/z for i in zz]
    return categorical_draw(probs)
  
  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    
    value = self.values[chosen_arm]
    self.values[chosen_arm] = value + reward
    return
