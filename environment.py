from exploror import exploror
from feedbacker import feedbacker
from multiprocessing import Pool
from problems_loader import init_problems_loader
from options import add_neurosat_options
from nn_filter import NeuroSAT, dummy_filter
import sys, os, argparse, pickle, subprocess, random
import numpy as np
import pickle
import os

# a working unit (for one problem, no filter)
class env(object):

    def __init__(self, filename, specs, sizes, sample_size, keep_history=False):
        # set up exploror
        self.exp = exploror(sizes[0])
        self.exp.set_state([])
        self.sample_size = sample_size
        # set up feedbacker
        self.file_id = filename.split('/')[-1].split('_')[0]
        self.fb = feedbacker(filename, specs, sizes, keep_history=keep_history)
        # record result
        self.result = None
        self.steps = -1

    def data_sampler(self):
        return self.fb.data_sampler()

    def reset_state(self, clear_history=False):
        self.exp.set_state([])
        self.fb.reset_state(clear_history=clear_history)
        self.result = None
        self.steps = -1

    def sample_training_data(self, size):
        return self.fb.sample_training_data(size)

    def sample_candidates(self, candidate_size):
        # if result is not None, seed is not reset, sample is not doing any work, but just
        # returning the sample of last time
        return self.exp.sample(candidate_size)

    def evaluate_candidates(self, candidates):
        if self.result is None:
            self.fb.evaluate(candidates)

    def feedback_status(self):
        if self.result is not None:
            return False
        if self.fb.unsat:
            self.result = 'solve'
            self.steps = self.fb.steps
            self.witness = self.fb.witness
            return False
        elif self.fb.timeout:
            self.result = 'timeout'
            self.steps = self.fb.max_step
            return False
        else:
            seeds = self.fb.recommend_seed(self.exp.seed_size)
            self.exp.set_state(seeds)
            return True

    def run(self): 
        while self.feedback_status():
            trails = self.sample_candidates(self.sample_size)
            self.evaluate_candidates(trails)

    def run_for_data(self):
        self.run()
        while self.result != 'solve':
            self.reset_state(clear_history=True)
            self.run()
        return self.data_sampler()

    def report(self):
        if self.result == 'solve':
            print('file {}: solve in {} steps, witness is {}'.format(self.file_id, self.steps, self.witness))
        else:
            print('file {}: timeout in {} steps'.format(self.file_id, self.steps))

# define a runner for each file
def runner(filename, tracking=True):
    play = env(filename, [2,3], [8,10], 40, keep_history=True)
    sampler = play.run_for_data()
    if tracking:
        print(play.file_id, end=' ', flush=True)
    del play
    return sampler

def print_all_result(all_result):    
    print('got all_result of size {}'.format(len(all_result)))
    n_solve = len(list(filter(lambda x: x[1] == 'solve', all_result)))
    print('solved {} of them'.format(n_solve))
    all_steps = list(map(lambda x: x[2], all_result))
    print('average steps are {}'.format(sum(all_steps) / len(all_steps)))

def one_batch_eval(envs, problem, nn_filter):
    while any(list(map(lambda x: x.feedback_status(), envs))):
        trails = list(map(lambda x: x.sample_candidates(sample_size), envs))
        trails = np.array(list(map(lambda x: list(map(list, x)), trails)))
        candidates = nn_filter.filter(problem, trails, 40)
        candidates = list(map(lambda x: set(map(tuple, x)), candidates))
        list(map(lambda x: x[0].evaluate_candidates(x[1]), zip(envs, candidates)))
    all_result = list(map(lambda x: (x.file_id, x.result, x.steps), envs))
    print_all_result(all_result)

def one_batch_train(envs, problem, nn_filter, training_size):
    data = list(map(lambda x: x.sample_training_data(training_size), envs))
    candidates = list(map(lambda x: x[0], data))
    labels = list(map(lambda x: x[1], data))
    _, cost = nn_filter.train_one(problem, candidates, labels)
    print('training cost is {}'.format(cost))

if __name__ == '__main__':
    # get all filenames of dimacs files
    dimacs_dir = '/homes/wang603/QBF/train10_unsat/'
    filenames = sorted(os.listdir(dimacs_dir))
    filenames = [os.path.join(dimacs_dir, filename) for filename in filenames]
    # run in parallel
    p = Pool(50)
    all_sampler = p.map(runner, filenames)
    print('got all_sampler of size {}'.format(len(all_sampler)))
    all_steps = list(map(lambda x: x.N, all_sampler))
    print('average steps are {}'.format(sum(all_steps) / len(all_steps)))
    print('write all_sample to file')
    with open('all_sampler', 'wb') as f_dump:
        pickle.dump(all_sampler, f_dump, pickle.HIGHEST_PROTOCOL)

    ## RL style training (not working so far)
    ##
    exit(1)
    # basic set up
    specs = [2, 3]
    sizes = [8, 10]
    abs_filename = '/homes/wang603/QBF/train10_unsat/{}_unsat.qdimacs'
    file_num = 1000
    batch_size = 40
    sample_size = 40
    feedback_batch_size = 400
    parser = argparse.ArgumentParser()
    add_neurosat_options(parser)

    parser.add_argument('--train_dir', action='store', type=str, help='Directory with training data')
    parser.add_argument('--run_id', action='store', dest='run_id', type=int, default=None)
    parser.add_argument('--restore_id', action='store', dest='restore_id', type=int, default=None)
    parser.add_argument('--restore_epoch', action='store', dest='restore_epoch', type=int, default=None)
    parser.add_argument('--n_epochs', action='store', dest='n_epochs', type=int, default=100000, help='Number of epochs through data')
    parser.add_argument('--n_saves_to_keep', action='store', dest='n_saves_to_keep', type=int, default=4, help='Number of saved models to keep')

    opts = parser.parse_args()

    setattr(opts, 'commit', subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip())
    setattr(opts, 'hostname', subprocess.check_output(['hostname']).strip())

    if opts.run_id is None: opts.run_id = random.randrange(sys.maxsize)
    print(opts)

    if not os.path.exists("snapshots/"):
        os.mkdir("snapshots")

    problem_loader = init_problems_loader(opts.train_dir)
    problem = problem_loader.get_next()[0][0]
    nn_filter = NeuroSAT(opts)
    envs = [env(abs_filename, i, keep_history=True) for i in range(20)]
    for i in range(10):
        one_batch_eval(envs, problem, nn_filter)
        for j in range(i+1):
            one_batch_train(envs, problem, nn_filter, 40)
        list(map(lambda x: x.reset_state(), envs))
