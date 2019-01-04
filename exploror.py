import numpy as np
import copy
import random

class exploror(object):

    def __init__(self, n_var, seed_size=5, sample_noise=0.1):
        # n_var: number of forall vars to decide on
        self.n_var = n_var
        self.seed_size = seed_size
        self.sample_noise = sample_noise

    def set_state(self, seeds, probs = None, probs_noise=0.4):
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
            self.probs = probs * (1 - probs_noise) + same * probs_noise

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

    # TODO: need more efficient sampling method
    def add_new_permute(self, seed):
        while True:
            ret = self.permute(seed)
            if ret in self.seeds:
                continue
            else:
                self.seeds.add(ret)
                return

    def permute(self, seed):
        # seed: a tuple of size n_var, representing the assignment
        if np.random.uniform() < self.sample_noise:
            return tuple([random.randint(0, 1) for _ in range(self.n_var)])
        mut = np.random.choice(self.n_var, p=self.probs)
        seed1 = list(seed)
        seed1[mut] = 1 - seed1[mut]
        return tuple(seed1)
