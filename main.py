from exploror import exploror
from feedbacker import feedbacker
from multiprocessing import Pool

# basic set up
specs = [2, 3]
sizes = [8, 10]
abs_filename = '/homes/wang603/QBF/train10/{}_unsat.qdimacs'
file_num = 1000
batch_size = 50
sample_size = 40
feedback_batch_size = 400

# a working unit (for one problem)
class env(object):

    def __init__(self, abs_filename, file_id):
        self.exp = exploror(sizes[0])
        filename = abs_filename.format(file_id)
        self.file_id = file_id
        self.fb = feedbacker(filename, specs, sizes)
        self.exp.set_state([])
        self.result = None

    def run(self):
        while True:
            trials = self.exp.sample(sample_size)
            self.fb.evaluate(trials)
            if self.fb.unsat:
                self.result = 'solve'
                self.steps = self.fb.steps
                self.witness = self.fb.witness
                return
            elif self.fb.timeout:
                self.result = 'timeout'
                self.max_step = self.fb.max_step
                return
            else:
                seeds = self.fb.recommend_seed(self.exp.seed_size)
                self.exp.set_state(seeds) 

    def report(self):
        if self.result == 'solve':
            print('file {}: solve in {} steps, witness is {}'.format(self.file_id, self.steps, self.witness))
        else:
            print('file {}: timeout in {} steps'.format(self.file_id, self.max_step))

def run_exp(i):
    play = env(abs_filename, i)
    play.run()
    if play.result == 'solve':
        return (i, play.result, play.steps)
    else:
        return (i, play.result, play.max_step)

def run_exp_track(i):
    res = run_exp(i)
    print(i, end=' ', flush=True)
    return res

def main_batch():
    all_result = []
    for batch_id in range(file_num // batch_size):
        p = Pool(batch_size)
        result = p.map(run_exp, list(range(batch_id * batch_size, (batch_id+1) * batch_size)))
        for r in result:
            print('file {}: result {}: steps {}'.format(r[0], r[1], r[2]))
        all_result = all_result + result
    print_all_result(all_result)

def print_all_result(all_result):    
    print('got all_result if size {}'.format(len(all_result)))
    n_solve = len(list(filter(lambda x: x[1] == 'solve', all_result)))
    print('solved {} of them'.format(n_solve))
    all_steps = list(map(lambda x: x[2], all_result))
    print('average steps are {}'.format(sum(all_steps) / len(all_steps)))

def main():
    p = Pool(50)
    all_result = p.map(run_exp_track, list(range(file_num)))
    print_all_result(all_result)

# main()
# run_exp_track(275)
p = Pool(20)
all_result = p.map(run_exp_track, list(range(20)))
print_all_result(all_result)
