import subprocess
import os
import numpy as np

def parse_dimacs(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    i = 0
    while lines[i].strip().split(" ")[0] == "c":
        i += 1

    # the following line should be the "p cnf" line
    header = lines[i].strip().split(" ")
    assert(header[0] == "p")
    n_vars = int(header[2])

    # qdimacs file has 2 more lines (for 2QBF)
    # a 1 2 3 ... 0 << the forall variables
    # e 9 10 11 ... 0 << the exist variables
    i += 1
    while (lines[i].strip().split(" ")[0] == 'a' or lines[i].strip().split(" ")[0] == 'e'):
        i += 1
    iclauses = [[int(s) for s in line.strip().split(" ")[:-1]] for line in lines[i:]]
    return n_vars, iclauses

def nsat2score(nsat):
    if nsat <= 3: return 10.0 - nsat
    if nsat <= 5: return 6.0
    if nsat <= 8: return 5.0
    if nsat <= 12: return 4.0
    if nsat <= 16: return 3.0
    if nsat <= 21: return 2.0
    else: return 1.0

class Dsampler(object):
    # data sampler
    def __init__(self, history):
        all_data = list(set(history))
        self.N = len(all_data)
        self.all_data_X = np.array([i[0] for i in all_data])
        self.all_data_Y = np.array([nsat2score(i[1]) for i in all_data], dtype=np.float32)
        self.weight = self.all_data_Y / np.sum(self.all_data_Y) 

    def sample(self, sample_size):
        index = np.random.choice(self.N, size=sample_size, p=self.weight)
        return self.all_data_X[index], self.all_data_Y[index]

class feedbacker(object):

    def __init__(self, filename, specs, sizes, max_step = 400, keep_history=False):
        # filename: the filename of the 2QBF problem
        # specs: the spec of the 2QBF, i.e. (2, 3) means each clause has first 2 lits as forall vars, and following 3 lits as exists vars
        # sizes: the size of the 2QBF, i.e. (8, 10) means the problem has a total of 8 forall vars and 10 exists vars
        self.filename = filename
        self.get_temp_filename()
        self.n_vars, self.iclauses = parse_dimacs(filename)
        self.specs = specs
        self.sizes = sizes
        # state tracker
        self.unsat = False
        self.timeout = False
        self.steps = 0
        self.max_step = max_step
        self.witness = None
        # history for training
        self.keep_history = keep_history
        self.history = []
        self.last = []

    def get_temp_filename(self):
        paths = self.filename.split('/')
        index = paths.index('QBF')
        self.temp_filename = '/'.join(paths[:index+1] + ['temp'] + paths[index+1:])
        temp_path = '/'.join(paths[:index+1] + ['temp'] + paths[index+1:-1])
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)

    def reset_state(self, clear_history=False):
        self.unsat = False
        self.timeout = False
        self.steps = 0
        self.witness = None
        if clear_history:
            temp = self.history
            self.history = []
            del temp
            temp = self.last
            self.last = []
            del temp
        else:
            self.history = self.history + self.last
            self.last = []

    def data_sampler(self):
        return Dsampler(self.history + self.last)

    def sample_training_data(self, size):
        t_nsat_list = choices(self.history + self.last, k=size)
        X = [list(t_nsat[0]) for t_nsat in t_nsat_list]
        Y = [nsat2score(t_nsat[1]) for t_nsat in t_nsat_list]
        return X, Y

    def evaluate(self, trials):
        # trials: a set of assignement for forall vars to be tested
        if len(self.last) > 0:
            if self.keep_history:
                self.history = self.history + self.last
            self.last = []
        for t in trials:
            n_sat = self.evaluate_one(t)
            self.last.append((t, n_sat))
            self.steps += 1
            if self.steps >= self.max_step:
                self.timeout = True
                break
            if n_sat == 0:
                self.unsat = True
                self.witness = t
                break
        self.last.sort(key=lambda x: x[1])
        # os.remove(self.temp_filename)

    def recommend_seed(self, n_seed):
        if len(self.last) < n_seed:
            return []
        else:
            return [x[0] for x in self.last[:n_seed]]

    def evaluate_one(self, t):
        # t: one assignment of forall vars to be tested
        sat = []
        trues = set()
        for i in range(len(t)):
            trues.add(i+1 if t[i] == 1 else -(i+1))
        for c in self.iclauses:
            if self.trued(c, trues):
                continue
            else:
                sat.append(c[self.specs[0]:])
        # write sat to file
        fn = self.write_sat_to_file(sat)
        # run #sat and collect result
        n_sat = self.run_sharp_SAT(fn)
        return n_sat

    def trued(self, clause, trues):
        for i in range(self.specs[0]):
            if clause[i] in trues: return True
        return False

    def write_sat_to_file(self, sat):
        filename = self.temp_filename
        with open(filename, 'w') as f:
            f.write('p cnf {} {}\n'.format(self.sizes[1], len(sat)))
            for s in sat:
                s = self.shift(s)
                f.write('{} 0\n'.format(' '.join(map(str, s))))
        return filename

    def shift(self, s):
        return map(lambda x: (x + self.sizes[0]) if x < 0 else (x - self.sizes[0]), s)

    def run_sharp_SAT(self, filename, verbose=False):
        result = subprocess.run(['/homes/wang603/sharpSAT/build/Release/sharpSAT', filename], stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8').split('\n')
        try:
            idx1 = result.index('# solutions ')
            idx2 = result.index('# END')
            assert idx2 == idx1 + 2
            return int(result[idx1 + 1])
        except ValueError:
            print(result)
