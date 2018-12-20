import subprocess

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

class feedbacker(object):

    def __init__(self, filename, specs, sizes, keep_history=False):
        # filename: the filename of the 2QBF problem
        # specs: the spec of the 2QBF, i.e. (2, 3) means each clause has the first 2 lits as forall vars, and the following 3 lits from exists vars
        # sizes: the size of the 2QBF, i.e. (8, 10) means the problem has a total of 8 forall vars and 10 exists vars
        self.filename = filename
        self.n_vars, self.iclauses = parse_dimacs(filename)
        self.specs = specs
        self.sizes = sizes
        self.unsat = False
        self.history = []
        self.last = []
        self.witness = None
        self.keep_history = keep_history

    def evaluate(self, trials):
        # trials: a set of assignement for forall vars to be tested
        if len(self.last) > 0:
            if self.keep_history:
                self.history = self.history + self.last
            self.last = []
        trials = list(trials)
        for t in trials:
            n_sat = self.evaluate_one(t)
            self.last.append((t, n_sat))
            if n_sat == 0:
                self.unsat = True
                self.witness = t
                break
        self.last.sort(key=lambda x: x[1])

    def recommend_seed(self, n_seed):
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
        filename = self.filename + '_sat'
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
