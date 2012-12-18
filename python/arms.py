from random import gauss, random

class AdversarialArm():
  def __init__(self, t, active_start, active_end):
    self.t = t
    self.active_start = active_start
    self.active_end = active_end
  
  def draw(self):
    self.t = self.t + 1
    if self.active_start <= self.t <= self.active_end:
      return 1.0
    else:
      return 0.0


class BernoulliArm():
  def __init__(self, p):
    self.p = p
  
  def draw(self):
    if random() > self.p:
      return 0.0
    else:
      return 1.0

class NormalArm():
  def __init__(self, mu, sigma):
    self.mu = mu
    self.sigma = sigma
  
  def draw(self):
    return gauss(self.mu, self.sigma)

  
