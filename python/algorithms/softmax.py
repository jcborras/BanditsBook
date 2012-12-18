from math import exp, log
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

class AnnealingSoftmax(object):
  def __init__(self, counts, values):
    self.counts = counts
    self.values = values
    return
  
  def initialize(self, n_arms):
    self.counts = [0 for col in range(n_arms)]
    self.values = [0.0 for col in range(n_arms)]
    return

  def arm_selected(self):
    z = sum([exp(v / self.temperature) for v in self.values])
    probs = [exp(v / self.temperature) / z for v in self.values]
    return categorical_draw(probs)
    
  def select_arm(self):
    t = sum(self.counts) + 1
    self.temperature = 1 / log(t + 0.0000001)
    return self.arm_selected()

  def update(self, chosen_arm, reward):
    self.counts[chosen_arm] = self.counts[chosen_arm] + 1
    n = self.counts[chosen_arm]
    
    value = self.values[chosen_arm]
    new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
    self.values[chosen_arm] = new_value


class Softmax(AnnealingSoftmax):
  def __init__(self, temperature, counts, values):
    self.temperature = temperature
    super(Softmax,self).__init__(counts,values)
    
  def select_arm(self):
    return self.arm_selected()
  
