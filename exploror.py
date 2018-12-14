import numpy as np
import copy
import random

class exploror(object):

    def __init__(self, n_var, seed_size = 5):
        # n_var: number of forall vars to decide on
        self.n_var = n_var
        self.seed_size = seed_size

    def set_state(self, seeds, noise = 0.5, probs = None):
        # seeds: a small set of assignment for forall vars
        # probs: the probabilty for each var to mutate for exploration
        if len(seeds) == 0:
            self.seeds = set([tuple([random.randint(0, 1) for _ in range(self.n_var)]) for _ in range(self.seed_size)])
        else:
            self.seeds = set(seeds)
        same = np.ones(self.n_var, dtype = np.float32) / self.n_var
        if probs is None:
            self.probs = same
        else:
            self.probs = probs * (1 - noise) + same * noise

    def sample(self, n_out):
        n_start = len(self.seeds)
        while (n_start < n_out):
            self.double_to_at_most(n_out - n_start)
            n_start *= 2
        return self.seeds

    def double_to_at_most(self, count):
        i = 0
        for s in self.seeds.copy():
            self.add_new_permute(s)
            i += 1
            if i >= count:
                return

    def add_new_permute(self, seed):
        while True:
            ret = self.permute(seed)
            if ret in self.seeds:
                continue
            else:
                self.seeds.add(ret)
                return

    def permute(self, seed):
        # seed: a list of size n_var, representing the assignment
        # seed should not be changed by this function
        mut = np.random.choice(self.n_var, p=self.probs)
        seed1 = list(seed)
        seed1[mut] = 1 - seed1[mut]
        return tuple(seed1)
